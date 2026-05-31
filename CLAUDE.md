# CLAUDE.md — doc-agent

## 項目概述
基於 LangGraph 的文檔管理 Agent，私域文檔檢索分析場景。
技術棧：FastAPI, LangGraph, PostgreSQL, Redis, Milvus

## 架構決策（已定，不要在對話中重新討論）
- 數據庫：PostgreSQL + asyncpg（為 pgvector 預留）
  - 連接層：SQLAlchemy Core（`create_async_engine`）管理連接池，SQL 用 `text()` 裸執行
  - 禁止 ORM 模型層（`Base`、`Column`、`Session`、`relationship`）
  - 連接串格式：`postgresql+asyncpg://user:password@host:5432/dbname`，環境變量 POSTGRES_URL
  - pgvector 擴展在 M2 通過 `conn.execute(text("..."))` 直接調用
- MQ：Redis Streams 初版，M4 後可換 RabbitMQ
- MQ consumer name：自動生成 `f"{socket.gethostname()}-{os.getpid()}"`，禁止寫死固定名稱（多 worker 會衝突）；僅調試時可通過 MQ_CONSUMER_NAME 環境變量覆蓋
- Milvus 訪問：強制透過 alias，禁止直接傳 collection_name
- LLM 信號量：三類 interactive/background/audit，禁止共用單一 semaphore；默認限值 interactive=50 / background=20 / audit=10，通過 `llm_semaphore_limits` 配置覆蓋
- 緩存 key 格式：`{kb_version}:{namespace}:{hash}`，通過 `RedisClient.cache_key(namespace, *parts)` 生成
- Cache 自動審批：`cache_auto_approve=True` 時，`compute_quality()` 計算 query embedding 與 top-3 chunk cosine similarity 均值（0.0–1.0），≥ `cache_quality_threshold`（默認 0.6）→ APPROVED，否則進 PENDING_REVIEW；`cache_auto_approve_threshold`（默認 1）為觸發自動審批的最低命中次數
- Memory 自動壓縮：session turns 累計達 `memory_summary_threshold`（默認 15）自動 compact 為長期摘要並清空近期記錄；`memory_recent_max_turns=20`，Redis TTL=86400s
- chunk schema：pydantic.BaseModel + frozen=True

## 模塊邊界（worktree 隔離邊界）
- M0: app/core/（config, exceptions, logging_config）, app/clients/, app/middleware/, app/models/, app/routers/health
- M1: app/knowledge_base/
- M2: app/retrieval/
- M3: app/cache/
- M4: app/agent/
- M5: app/memory/
- M6: app/consistency/
- M7: app/skills/

| 模組 | 主要職責 | 依賴 |
|------|---------|------|
| M0 基礎設施 | FastAPI 骨架、配置、DB 連接（PostgreSQL/Redis/Milvus）、日誌、異常處理 | 無 |
| M1 知識庫 | 文檔解析（pdf/txt）、清洗、去重、分塊、向量化、HNSW 索引、線上更新 | M0 |
| M2 混合檢索 | BM25 + HNSW 向量檢索 + RRF 融合 + LLM 重排序；`POST /retrieval/search` 調試接口 | M1 |
| M3 RAG 緩存 | 兩層緩存：Layer 1 答案緩存（embed query → cosine 近鄰查找，閾值 `cache_semantic_threshold=0.92` → 命中直接推流答案，0 次 LLM）；Layer 2 chunk 緩存（normalized query hash → APPROVED chunk list）；`compute_quality()` embedding 質量評分自動審批（均值 ≥ `cache_quality_threshold=0.6` → APPROVED，否則 PENDING_REVIEW）；人工審查隊列；`GET/POST /cache/*` 管理接口 | M2 |
| M4 Agent 編排 | LangGraph 圖（入口 `cache_lookup`，Layer 1 命中走 `stream_cached → cache_write`，miss 走 `query_rewrite → retrieval → entity_extraction → rerank → generate → cache_write`）；MQ 協程消費；全局信號量限流（interactive=50/background=20/audit=10）；流式輸出 | M2, M3 |
| M5 分層記憶 | 近期對話 + 長期摘要 + 靜態知識向量化；`/memory/*` 管理接口 | M0, M1 |
| M6 一致性 | 知識庫更新 → Redis 緩存失效/重算 | M1, M3 |
| M7 Skill 封裝 | 問答 / 工作總結封裝為可編排 Skill | M4 |
| F0 前端界面 | Vanilla JS 對話頁面（文件上傳 + SSE 流式問答 + 文檔刪除），FastAPI 直接托管 `app/static/` | M4 |
| F1 SSE 流式輸出 | `/agent/stream/{job_id}` 後端實現，EventSourceResponse | M4 |
| F2 工程化腳本 | start.bat/stop.bat 一鍵啟停 | 無 |
| F3 緩存管理面板 | Cache 頁籤：stats（hits/misses/auto_approved/pending）、待審列表（僅 PENDING_REVIEW 條目）、inline reviewer-id approve/reject/delete，操作後即時移除條目 | M3 |
| F4 記憶管理面板 | Memory 頁籤：會話記憶展示、手動觸發摘要、靜態知識增刪 | M5 |
| F5 檢索調試面板 | Retrieval 頁籤：輸入 query + top_k，直接返回原始 chunk 列表（不經 LLM），用於驗證索引質量 | M2 |

## 測試規範
- M0 覆蓋率 ≥ 90%，其他模塊 ≥ 80%
- 所有網絡層用 AsyncMock，不發真實請求
- 集成測試標記 @pytest.mark.integration，單獨跑

## 禁止事項
- 不要在業務層直接 import pymilvus，必須透過 MilvusClient
- 不要寫死 collection_name，只用 alias
- 不要在 API response 中暴露連接字符串或內部錯誤詳情
- 不要使用 SQLite（已決定用 PostgreSQL）

## M0 前瞻性預留（M1-M7 開發前必讀，勿重新實現）

### ChunkSchema（app/models/chunk.py）
`frozen=True`，字段：`doc_id`, `section_id`, `chunk_index`, `parent_chunk_id`（層級檢索/GraphRAG預留）, `content_hash`（去重）, `version`, `content`, `embedding`

### RetrievalStrategy Protocol（app/models/retrieval.py）
M2 實現時繼承此 Protocol，禁止改簽名：
```python
async def retrieve(self, query: str, top_k: int, **kwargs) -> list[ChunkSchema]: ...
```
`HybridRetriever` 已聲明接口，M2 中補充 RRF 融合排序實現。

### RedisClient 分布式工具（app/clients/redis.py）
M3/M6 直接調用，不要重新實現：
- `cache_key(namespace, *parts)` → `{kb_version}:{namespace}:{parts}`
- `increment_with_ttl(key, ttl_seconds, amount)` → 原子 INCR+EXPIRE（Lua），用於限流
- `acquire_lock(key, ttl_seconds)` → `(acquired: bool, token: str)`，SET NX EX
- `release_lock(key, token)` → Lua 驗證 token 後 DEL，防誤釋放

多 key 操作需使用 hash tag（`{user:u123}:rate_limit`），避免 Cluster CROSSSLOT 錯誤。

## ECC 工作流
單模塊標準流水線：/ecc:plan → /tdd → /code-review → /security-scan → /e2e → merge