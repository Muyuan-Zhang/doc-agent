# Plan: M3 RAG Cache

**Source**: CLAUDE.md — M3 module boundary  
**Selected Milestone**: M3 RAG Cache — Redis cross-user cache, query rewrite, hit acceleration, user review  
**Complexity**: Large

---

## Summary

M3 adds a Redis-backed caching layer between the query entrypoint and the M2 HybridRetriever. Incoming queries are semantically normalised (optionally via LLM), hashed to a cache key, and looked up in Redis. On a miss the full retrieval runs and the result is stored in a `PENDING_REVIEW` state; a reviewer API lets humans approve or reject entries before they are served to other users. Approved entries are returned immediately on subsequent identical (or semantically equivalent) queries, bypassing M2 entirely.

---

## Patterns to Mirror

| Category | Source | Pattern |
|---|---|---|
| Schemas | `app/models/chunk.py:1` | `frozen=True` Pydantic BaseModel for all cache data objects |
| Service DI | `app/routers/knowledge_base.py:~30` | `_get_service(request)` factory pulls clients from `request.app.state` |
| Cache keys | `app/clients/redis.py` | Always via `RedisClient.cache_key(namespace, *parts)` → `{kb_version}:{ns}:{parts}` |
| Router style | `app/routers/knowledge_base.py` | `APIRouter(prefix=..., tags=[...])`, explicit status codes, no business logic in router |
| Exceptions | `app/core/exceptions.py` | Raise `AppError` subclass; never leak internals |
| Tests — unit | `tests/test_redis_client.py` | `_connected_client()` helper, group by `class Test<Feature>` |
| Tests — E2E | `tests/e2e/conftest.py` | `make_app()` injects mocks into `app.state`; `AsyncClient` via `ASGITransport` |
| LLM semaphore | `app/core/config.py:LLMSemaphoreLimits` | Use `background` semaphore for query-rewrite LLM calls (non-interactive) |
| Logging | `app/knowledge_base/service.py` | `logger = logging.getLogger(__name__)`, structured key=value pairs |

---

## Architecture: Request Flow

```
Query (string)
  │
  ▼
QueryRewriter.rewrite()          ← LLM call (background semaphore, optional)
  │ normalized_query + hash
  ▼
RagCacheStore.get(hash)
  ├─ HIT / APPROVED     ──────────────────────────────► Return CacheEntry.chunks  ✅ fast path
  ├─ HIT / PENDING      ──► bypass cache, run M2
  ├─ HIT / REJECTED     ──► bypass cache, run M2
  └─ MISS               ──► run M2 HybridRetriever
                               │
                               ▼
                        RagCacheStore.set(hash, chunks, status=PENDING_REVIEW)
                        ReviewQueue.enqueue(hash)
                               │
                               ▼
                        Return M2 chunks to caller

Reviewer API (async, human)
  GET  /cache/review              ← list PENDING_REVIEW entries
  POST /cache/review/{key}/approve  ← status → APPROVED
  POST /cache/review/{key}/reject   ← status → REJECTED
  DELETE /cache/{key}             ← manual eviction
  GET  /cache/stats               ← hit rate, pending count
```

---

## Cache Key Conventions

| Purpose | Call | Redis key |
|---|---|---|
| Cached result | `redis.cache_key("rag_cache", query_hash)` | `v1:rag_cache:{sha256}` |
| Review queue (list) | `redis.cache_key("review", "pending")` | `v1:review:pending` |
| Entry status | `redis.cache_key("review", "status", query_hash)` | `v1:review:status:{sha256}` |
| Stats counter | `redis.cache_key("stats", "hits")` / `"misses"` | `v1:stats:hits` / `v1:stats:misses` |

---

## Files to Create / Modify

| File | Action | Why |
|---|---|---|
| `app/cache/__init__.py` | CREATE | Module marker |
| `app/cache/schemas.py` | CREATE | `CacheStatus` enum, `CacheEntry` (frozen), `ReviewEntry` |
| `app/cache/query_rewriter.py` | CREATE | `QueryRewriter` — normalize + LLM rewrite + sha256 hash |
| `app/cache/store.py` | CREATE | `RagCacheStore` — Redis get/set/delete/invalidate |
| `app/cache/review.py` | CREATE | `ReviewQueue` — enqueue, list pending, approve, reject |
| `app/cache/service.py` | CREATE | `RagCacheService.get_or_retrieve()` — orchestrates all above |
| `app/routers/cache.py` | CREATE | REST endpoints for review workflow + stats |
| `app/core/config.py` | UPDATE | Add `cache_ttl_seconds`, `cache_auto_approve_threshold`, `cache_similarity_threshold`, `cache_rewrite_enabled` |
| `app/__init__.py` | UPDATE | `app.include_router(cache_router)` + lifespan wires no new clients (reuses redis) |
| `tests/cache/__init__.py` | CREATE | Package marker |
| `tests/cache/test_schemas.py` | CREATE | Schema validation, frozen invariants |
| `tests/cache/test_query_rewriter.py` | CREATE | normalize, hash stability, LLM mock |
| `tests/cache/test_store.py` | CREATE | get/set/delete/invalidate with AsyncMock redis |
| `tests/cache/test_review.py` | CREATE | enqueue, approve, reject edge cases |
| `tests/cache/test_service.py` | CREATE | Full orchestration — miss/hit/pending/rejected paths |
| `tests/test_cache_router.py` | CREATE | HTTP-level router tests |
| `tests/e2e/test_cache_pipeline.py` | CREATE | End-to-end cache hit/miss pipeline |
| `tests/e2e/conftest.py` | UPDATE | Add `make_cache_service_mock()` factory |

---

## Tasks

### Task 1 — Schemas & Config
- **Action**: Create `app/cache/schemas.py` with `CacheStatus(str, Enum)`, `CacheEntry(BaseModel, frozen=True)`, `ReviewEntry(BaseModel, frozen=True)`. Add `cache_ttl_seconds: int = 3600`, `cache_auto_approve_threshold: int = 3`, `cache_rewrite_enabled: bool = True` to `Settings` in `app/core/config.py`.
- **Mirror**: `app/models/chunk.py` — `frozen=True`, Optional fields, `model_config = ConfigDict(frozen=True)`
- **Validate**: `python -m pytest tests/cache/test_schemas.py -x`

```python
class CacheStatus(str, Enum):
    PENDING_REVIEW = "pending_review"
    APPROVED       = "approved"
    REJECTED       = "rejected"

class CacheEntry(BaseModel):
    model_config = ConfigDict(frozen=True)
    query_hash:        str
    original_query:    str
    normalized_query:  str
    chunks:            list[ChunkSchema]
    status:            CacheStatus = CacheStatus.PENDING_REVIEW
    approval_count:    int = 0
    created_at:        datetime
    approved_by:       list[str] = Field(default_factory=list)
```

---

### Task 2 — Query Rewriter
- **Action**: Create `app/cache/query_rewriter.py`. `QueryRewriter` holds a reference to `OpenAILLMClient` and `settings`. Two public methods: `normalize(query) -> str` (deterministic, no I/O) and `async rewrite(query) -> tuple[str, str]` (normalized, sha256_hex). LLM call uses `background` semaphore and is guarded by `settings.cache_rewrite_enabled`.
- **Mirror**: `app/knowledge_base/embedder.py` for semaphore usage pattern; `hashlib.sha256` for stable hash
- **Validate**: `python -m pytest tests/cache/test_query_rewriter.py -x`

**normalize()** steps:
1. Strip whitespace, lowercase
2. Remove punctuation (keep CJK characters)
3. Collapse internal spaces

**rewrite()** steps:
1. `normalize(query)` → `norm`
2. If `cache_rewrite_enabled`: LLM prompt → canonical form → `normalize(llm_out)` → `norm`
3. `sha256(norm.encode()).hexdigest()[:16]` → hash (16-char prefix, collision risk acceptable)

---

### Task 3 — Cache Store
- **Action**: Create `app/cache/store.py`. `RagCacheStore` takes a `RedisClient`. All key generation via `self._redis.cache_key(...)`.
- **Mirror**: `app/clients/redis.py` — `eval()` for atomics, JSON serialization
- **Validate**: `python -m pytest tests/cache/test_store.py -x`

```python
class RagCacheStore:
    async def get(self, query_hash: str) -> CacheEntry | None: ...
    async def set(self, entry: CacheEntry, ttl: int) -> None: ...
    async def update_status(self, query_hash: str, status: CacheStatus) -> None: ...
    async def delete(self, query_hash: str) -> None: ...
    async def invalidate_all(self) -> int:
        # SCAN + DEL all keys matching v1:rag_cache:*
        # Used by M6 on KB version bump
    async def get_stats(self) -> dict[str, int]: ...  # hits, misses, pending count
```

Serialization: `entry.model_dump_json()` → Redis string value. Retrieve: `CacheEntry.model_validate_json(raw)`.

---

### Task 4 — Review Queue
- **Action**: Create `app/cache/review.py`. `ReviewQueue` takes `RedisClient` and `Settings`.
- **Mirror**: `app/clients/redis.py:acquire_lock` — Lua for atomic status update
- **Validate**: `python -m pytest tests/cache/test_review.py -x`

```python
class ReviewQueue:
    async def enqueue(self, query_hash: str) -> None:
        # LPUSH v1:review:pending query_hash (deduplicated)
    async def list_pending(self, limit: int = 20) -> list[str]:
        # LRANGE v1:review:pending 0 limit-1
    async def approve(self, query_hash: str, reviewer_id: str) -> CacheStatus:
        # Increment approval_count in store; if >= threshold → APPROVED
        # Return resulting status
    async def reject(self, query_hash: str) -> None:
        # update_status REJECTED; LREM from pending list
```

---

### Task 5 — Cache Service (Orchestrator)
- **Action**: Create `app/cache/service.py`. `RagCacheService` orchestrates rewriter → store → review → retriever fallback.
- **Mirror**: `app/knowledge_base/service.py` — structured logging, explicit error handling
- **Validate**: `python -m pytest tests/cache/test_service.py -x`

```python
class RagCacheService:
    def __init__(self, redis: RedisClient, llm: OpenAILLMClient, settings: Settings) -> None:
        self._store = RagCacheStore(redis)
        self._rewriter = QueryRewriter(llm, settings)
        self._review = ReviewQueue(redis, settings)

    async def get_or_retrieve(
        self,
        query: str,
        retriever: RetrievalStrategy,
        top_k: int = 5,
    ) -> tuple[list[ChunkSchema], bool]:  # (chunks, cache_hit)
```

Logic:
1. `normalized, query_hash = await self._rewriter.rewrite(query)`
2. `entry = await self._store.get(query_hash)`
3. `if entry and entry.status == APPROVED: return entry.chunks, True`
4. `chunks = await retriever.retrieve(query, top_k)` — M2
5. `entry = CacheEntry(query_hash=..., chunks=chunks, status=PENDING_REVIEW, ...)`
6. `await self._store.set(entry, ttl=settings.cache_ttl_seconds)`
7. `await self._review.enqueue(query_hash)`
8. `return chunks, False`

---

### Task 6 — API Router
- **Action**: Create `app/routers/cache.py`. Register in `app/__init__.py`.
- **Mirror**: `app/routers/knowledge_base.py` — `_get_service(request)` DI pattern
- **Validate**: `python -m pytest tests/test_cache_router.py -x`

Endpoints:

| Method | Path | Status | Body / Response |
|---|---|---|---|
| GET | `/cache/review` | 200 | `?limit=20` → `list[ReviewEntry]` |
| POST | `/cache/review/{key}/approve` | 200 | `{"reviewer_id": str}` → `{"status": str}` |
| POST | `/cache/review/{key}/reject` | 204 | — |
| DELETE | `/cache/{key}` | 204 | manual eviction |
| GET | `/cache/stats` | 200 | `{"hits": int, "misses": int, "pending": int}` |

---

### Task 7 — Tests (Unit)
- **Action**: Write `tests/cache/test_*.py` for all components. All Redis/LLM calls mocked with `AsyncMock`. Minimum 80% coverage per file.
- **Mirror**: `tests/test_redis_client.py` — `class Test<Feature>:`, helper factories, parametric edge cases
- **Validate**: `python -m pytest tests/cache/ -x --cov=app/cache --cov-report=term-missing`

Key test cases per component:
- `test_schemas.py`: frozen enforcement, enum values, field defaults
- `test_query_rewriter.py`: normalize strips correctly, hash stable across calls, LLM disabled when `cache_rewrite_enabled=False`
- `test_store.py`: get returns `None` on miss, set serialises + TTL, `invalidate_all` scans correct pattern
- `test_review.py`: enqueue idempotent, approve increments count, auto-approve at threshold, reject removes from list
- `test_service.py`: APPROVED hit returns cached chunks, PENDING hit falls through, REJECTED falls through, MISS runs retriever + enqueues

---

### Task 8 — E2E Tests
- **Action**: Add `make_cache_service_mock()` to `tests/e2e/conftest.py`. Write `tests/e2e/test_cache_pipeline.py`.
- **Mirror**: `tests/e2e/conftest.py` — `make_app()` factory, `AsyncClient` + `ASGITransport`
- **Validate**: `python -m pytest tests/e2e/test_cache_pipeline.py -x`

E2E scenarios:
1. First query → miss → retrieve → 202 enqueued for review
2. Second identical query before approval → still runs retriever (PENDING)
3. Approve via POST → third query → APPROVED hit (retriever NOT called)
4. Reject → query → retriever called again
5. GET /cache/stats reflects hit/miss counters

---

## Validation

```bash
# All M3 unit tests with coverage
python -m pytest tests/cache/ tests/test_cache_router.py -x --cov=app/cache --cov-report=term-missing

# E2E pipeline
python -m pytest tests/e2e/test_cache_pipeline.py -x -v

# Full suite (no regressions)
python -m pytest --ignore=tests/e2e -x -q
python -m pytest tests/e2e/ -x -q

# Coverage gate (80% minimum)
python -m pytest tests/cache/ --cov=app/cache --cov-fail-under=80
```

---

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Hash collision (16-char prefix) | Low | Upgrade to full sha256 hex (64 chars) if collision observed in prod |
| LLM rewrite latency added to every cold miss | Medium | Run rewrite concurrently with retriever (asyncio.gather) for MISS path; skip if `cache_rewrite_enabled=False` |
| Redis cluster CROSSSLOT on multi-key ops | Medium | Use hash-tag keys `{v1:rag_cache}:hash` for any MULTI/EVAL touching multiple cache entries |
| Large embedding vectors bloating Redis | Medium | Strip `embedding` field from cached chunks (embeddings already in Milvus); restore on return |
| Review queue grows unbounded | Low | Cap via `cache_max_pending_reviews` setting; LLEN check before LPUSH |
| M6 invalidation race (KB update mid-query) | Low | `invalidate_all` uses SCAN+DEL; use `acquire_lock` guard during KB version bump |
| M2 not yet fully implemented | Medium | `RagCacheService.get_or_retrieve` accepts `RetrievalStrategy` Protocol (M0 already defines it); M2 can be stubbed for M3 tests |

---

## Acceptance

- [ ] All 8 tasks complete
- [ ] `python -m pytest tests/cache/ tests/test_cache_router.py --cov=app/cache --cov-fail-under=80` passes
- [ ] `python -m pytest tests/e2e/test_cache_pipeline.py -x` passes
- [ ] Full test suite green (`python -m pytest -x -q`)
- [ ] No new `AppError` subclass leaks internals to API responses
- [ ] All cache keys generated via `RedisClient.cache_key()` — no raw string construction
- [ ] `frozen=True` on all new Pydantic models
- [ ] `embedding` field stripped from chunks before Redis storage (size guard)
- [ ] `invalidate_all()` in `RagCacheStore` is callable by M6 without importing internal details
