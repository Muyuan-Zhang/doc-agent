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
- LLM 信號量：三類 interactive/background/audit，禁止共用單一 semaphore
- 緩存 key 格式：`{kb_version}:{namespace}:{hash}`，通過 `RedisClient.cache_key(namespace, *parts)` 生成
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
| M2 混合檢索 | BM25 + HNSW 向量檢索 + RRF 融合 + LLM 重排序 | M1 |
| M3 RAG 緩存 | Redis 跨用戶緩存、查詢重寫、命中加速、用戶介入審查 | M2 |
| M4 Agent 編排 | LangGraph 圖、MQ 協程消費、全局信號量限流、流式輸出 | M2, M3 |
| M5 分層記憶 | 近期對話 + 長期摘要 + 靜態知識向量化 | M0, M1 |
| M6 一致性 | 知識庫更新 → Redis 緩存失效/重算 | M1, M3 |
| M7 Skill 封裝 | 問答 / 工作總結封裝為可編排 Skill | M4 |
| F0 前端界面 | Vanilla JS 對話頁面（文件上傳 + SSE 流式問答），FastAPI 直接托管 | M4 |
| F1 SSE 流式輸出 | /agent/stream/{job_id} 後端實現，EventSourceResponse | M4 |
| F2 工程化腳本 | Makefile + start.bat/stop.bat 一鍵啟停 | 無 |

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

## 前端待實現功能（後端已就緒）

### F0 擴展：文檔管理
- **刪除文檔**：文件列表每項加刪除按鈕，調用 `DELETE /knowledge-base/documents/{doc_id}`（204），成功後從列表移除

### F3 緩存管理面板（對應 M3）
後端路由：`app/routers/cache.py`
- 顯示緩存統計：`GET /cache/stats` → hits / misses / pending 數字展示
- 待審核列表：`GET /cache/review` → 列出 pending 條目（query 預覽 + query_hash）
- 審核操作：每條目配 approve / reject 按鈕
  - `POST /cache/review/{query_hash}/approve`（需 reviewer_id）
  - `POST /cache/review/{query_hash}/reject`（204）
- 刪除緩存：`DELETE /cache/{query_hash}`（204）
- 認證：請求頭帶 `X-API-Key`，值來自 `.env` 的 `CACHE_API_KEY`

### F4 記憶管理面板（對應 M5）
後端路由：`app/routers/memory.py`
- 當前會話記憶展示：`GET /memory/context/{session_id}?user_id=` → 顯示 turns + summary + static_facts
- 手動觸發摘要：`POST /memory/summarize/{session_id}?user_id=` → 壓縮近期對話為長期摘要
- 靜態知識管理：
  - 添加：`POST /memory/static`，body `{user_id, content}`
  - 刪除：`DELETE /memory/static/{fact_id}?user_id=`（204）

### F5 直接檢索調試視圖（對應 M2）
後端路由：`app/routers/retrieval.py`（`POST /retrieval/search`）
- 輸入查詢 + top_k，直接返回原始 chunk 列表（不經過 LLM）
- 用途：調試檢索質量，驗證文檔是否被正確索引

## ECC 工作流
單模塊標準流水線：/ecc:plan → /tdd → /code-review → /security-scan → /e2e → merge