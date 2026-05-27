# Environment Variables

<!-- AUTO-GENERATED from app/core/config.py and .env.example — do not hand-edit this section -->

Copy `.env.example` to `.env` and fill in required values before running the service.

## Application

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `APP_NAME` | No | `doc-agent` | Service name shown in health responses |
| `DEBUG` | No | `false` | Enables verbose SQL and pool logging |

## PostgreSQL

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `POSTGRES_URL` | **Yes** | — | Full asyncpg connection string.<br>Format: `postgresql+asyncpg://user:pass@host:5432/db` |

## Redis

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `REDIS_URL` | **Yes** | — | Redis connection URL.<br>Example: `redis://localhost:6379/0` |

## Milvus

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MILVUS_HOST` | No | `localhost` | Milvus server hostname |
| `MILVUS_PORT` | No | `19530` | Milvus gRPC port |
| `MILVUS_ALIAS` | No | `default` | Connection alias (all collection ops route through this) |

## Knowledge Base (M1)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `KNOWLEDGE_BASE_VERSION` | No | `v1` | Version prefix embedded in every cache key |
| `CHUNK_SIZE` | No | `512` | Target token size per chunk |
| `CHUNK_OVERLAP` | No | `64` | Token overlap between adjacent chunks |
| `EMBEDDING_DIM` | No | `1536` | Embedding vector dimension (must match model) |
| `EMBEDDING_BATCH_SIZE` | No | `100` | Texts per OpenAI embedding API call |
| `MILVUS_KB_COLLECTION` | No | `knowledge_base` | Milvus collection name for KB chunks |

## LLM / OpenAI

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | **Yes** | — | OpenAI secret key |
| `OPENAI_EMBEDDING_MODEL` | No | `text-embedding-3-small` | Model for chunk and fact embeddings |
| `OPENAI_CHAT_MODEL` | No | `gpt-4o-mini` | Model for conversation summarization (M5) |

## LLM Concurrency Semaphores

Uses `__` as nested delimiter (pydantic-settings convention).

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_SEMAPHORE_LIMITS__INTERACTIVE` | No | `5` | Max concurrent interactive (user-facing) LLM calls |
| `LLM_SEMAPHORE_LIMITS__BACKGROUND` | No | `2` | Max concurrent background (embedding, indexing) calls |
| `LLM_SEMAPHORE_LIMITS__AUDIT` | No | `1` | Max concurrent audit calls (reserved for M7) |

## Memory (M5)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MEMORY_RECENT_MAX_TURNS` | No | `20` | Hard cap on turns kept in Redis per session |
| `MEMORY_RECENT_TTL_SECONDS` | No | `86400` | Redis key TTL (seconds) reset on each append; default 24 h |
| `MEMORY_SUMMARY_THRESHOLD` | No | `15` | Turn count that triggers automatic LLM compaction |
| `MEMORY_MILVUS_COLLECTION` | No | `memory_vectors` | Milvus collection for static-knowledge embeddings |

## Redis Streams MQ

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MQ_STREAM_NAME` | No | `doc-agent:tasks` | Redis Stream key for task messages |
| `MQ_CONSUMER_GROUP` | No | `doc-agent-workers` | Consumer group name |
| `MQ_CONSUMER_NAME` | No | `{hostname}-{pid}` | Auto-generated per worker; override only for debugging |

<!-- END AUTO-GENERATED -->
