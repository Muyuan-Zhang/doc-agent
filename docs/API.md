# API Reference

<!-- AUTO-GENERATED from app/routers/ — do not hand-edit this section -->

Base URL: `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs` (Swagger UI)

---

## Health

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe — returns `{"status":"ok","service":"<app_name>"}` |
| `GET` | `/health/ready` | Readiness probe — pings all 5 backends (postgres, redis, milvus, mq, llm); returns `503` if any fail |

**Liveness response schema:**
```json
{ "status": "ok", "service": "doc-agent" }
```

**Readiness response schema:**
```json
{
  "status": "ok | degraded",
  "kb_version": "v1",
  "checks": {
    "postgres": "ok",
    "redis": "ok",
    "milvus": "ok",
    "mq": "ok",
    "llm": "ok"
  }
}
```

---

## Knowledge Base (M1)

| Method | Path | Status | Description |
|--------|------|--------|-------------|
| `POST` | `/knowledge-base/documents` | `202` | Upload a document (pdf/txt, max 50 MB); ingestion runs in the background |
| `GET` | `/knowledge-base/documents/{doc_id}/status` | `200` | Poll ingest status for a document |
| `DELETE` | `/knowledge-base/documents/{doc_id}` | `204` | Delete a document and its vectors from Milvus |

`doc_id` must be a UUID v4.

**Upload response:**
```json
{ "doc_id": "550e8400-e29b-41d4-a716-446655440000" }
```

**Status response:**
```json
{
  "doc_id": "550e8400-...",
  "filename": "report.pdf",
  "status": "pending | processing | indexed | failed",
  "chunk_count": 42,
  "version": "v1"
}
```

---

## Memory (M5)

Manages three-tier conversation memory per user/session.

| Method | Path | Status | Description |
|--------|------|--------|-------------|
| `POST` | `/memory/turns` | `204` | Append a conversation turn; auto-compacts to PG summary when turn count ≥ threshold |
| `GET` | `/memory/context/{session_id}?user_id=` | `200` | Retrieve merged memory context (recent turns + latest summary + static facts) |
| `POST` | `/memory/summarize/{session_id}?user_id=` | `200` | Manually trigger LLM summarization and clear recent turns |
| `POST` | `/memory/static` | `201` | Add a permanent static knowledge fact (embedded into Milvus) |
| `DELETE` | `/memory/static/{fact_id}?user_id=` | `204` | Remove a static fact from PG and Milvus |

`user_id` is a **required** query parameter on context, summarize, and delete-fact endpoints.

**Append turn request:**
```json
{
  "session_id": "sess-abc",
  "user_id": "user-123",
  "role": "user | assistant | system",
  "content": "What documents are available?"
}
```

**Context response:**
```json
{
  "turns": [
    { "session_id": "sess-abc", "role": "user", "content": "...", "ts": 1717000000.0 }
  ],
  "summary": {
    "summary_id": "...",
    "user_id": "user-123",
    "session_id": "sess-abc",
    "summary_text": "User asked about available documents...",
    "content_hash": "abc123..."
  },
  "static_facts": [
    { "fact_id": "...", "user_id": "user-123", "content": "Prefers concise answers", "content_hash": "..." }
  ]
}
```

**Add static fact request:**
```json
{ "user_id": "user-123", "content": "I prefer Python examples over Java." }
```

---

## Agent (M4 — placeholder)

| Method | Path | Description |
|--------|------|-------------|
| — | `/agent/*` | LangGraph pipeline endpoints; implemented in M4 |

<!-- END AUTO-GENERATED -->
