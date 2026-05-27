# CLAUDE.md — doc-agent

## 項目概述
基於 LangGraph 的文檔管理 Agent，私域文檔檢索分析場景。
技術棧：FastAPI, LangGraph, PostgreSQL, Redis, Milvus

## 架構決策（已定，不要在對話中重新討論）
- 數據庫：PostgreSQL + asyncpg（為 pgvector 預留）
- MQ：Redis Streams 初版，M4 後可換 RabbitMQ
- Milvus 訪問：強制透過 alias，禁止直接傳 collection_name
- LLM 信號量：三類 interactive/background/audit，禁止共用單一 semaphore
- 緩存 key 格式：{kb_version}:{namespace}:{hash}
- chunk schema：pydantic.BaseModel + frozen=True

## 模塊邊界（worktree 隔離邊界）
- M0: app/core/, app/clients/, app/middleware/
- M1: app/knowledge_base/
- M2: app/retrieval/
- M3: app/cache/
- M4: app/agent/
- M5: app/memory/
- M6: app/consistency/
- M7: app/skills/

## 測試規範
- M0 覆蓋率 ≥ 90%，其他模塊 ≥ 80%
- 所有網絡層用 AsyncMock，不發真實請求
- 集成測試標記 @pytest.mark.integration，單獨跑

## 禁止事項
- 不要在業務層直接 import pymilvus，必須透過 MilvusClient
- 不要寫死 collection_name，只用 alias
- 不要在 API response 中暴露連接字符串或內部錯誤詳情
- 不要使用 SQLite（已決定用 PostgreSQL）

## ECC 工作流
單模塊標準流水線：/ecc:plan → /tdd → /code-review → /security-scan → /e2e → merge