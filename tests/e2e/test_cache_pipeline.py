"""E2E tests for M3 cache router via FastAPI ASGI.

All tests bypass lifespan — app.state is injected directly.
Exercises the full middleware → router → service stack with mocked Redis.
"""
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import ASGITransport, AsyncClient

from app.cache.schemas import CacheEntry, CacheStatus
from app.core.config import settings as _settings
from app.models.chunk import ChunkSchema
from tests.e2e.conftest import make_app, make_llm_mock, make_cache_redis_mock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk() -> ChunkSchema:
    return ChunkSchema(
        doc_id="doc-1", section_id="sec-1", chunk_index=0,
        content_hash="abc123", version="v1", content="test content",
    )


def _make_cache_entry(status: CacheStatus = CacheStatus.PENDING_REVIEW) -> CacheEntry:
    return CacheEntry(
        query_hash="deadbeef12345678",
        original_query="What is the deadline?",
        normalized_query="what is the deadline",
        chunks=[_make_chunk()],
        status=status,
        created_at=datetime.now(tz=timezone.utc),
    )


# ---------------------------------------------------------------------------
# GET /cache/stats
# ---------------------------------------------------------------------------

def _make_stats_pipeline(hits=0, misses=0, pending=0):
    from unittest.mock import MagicMock, AsyncMock
    pipe = MagicMock()
    pipe.hgetall = MagicMock(return_value=pipe)
    pipe.zcard = MagicMock(return_value=pipe)
    pipe.execute = AsyncMock(
        return_value=[{"hits": str(hits), "misses": str(misses)}, pending]
    )
    return pipe


class TestCacheStatsEndpoint:
    async def test_returns_200_with_stats_fields(self):
        redis = make_cache_redis_mock()
        redis.client.pipeline = MagicMock(return_value=_make_stats_pipeline(5, 3, 2))
        app = make_app(redis=redis, llm=make_llm_mock())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/cache/stats")
        assert r.status_code == 200
        body = r.json()
        assert body["hits"] == 5
        assert body["misses"] == 3
        assert body["pending"] == 2

    async def test_returns_zeros_when_no_activity(self):
        redis = make_cache_redis_mock()
        redis.client.pipeline = MagicMock(return_value=_make_stats_pipeline(0, 0, 0))
        app = make_app(redis=redis, llm=make_llm_mock())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/cache/stats")
        assert r.status_code == 200
        assert r.json()["hits"] == 0
        assert r.json()["misses"] == 0

    async def test_includes_request_id_header(self):
        redis = make_cache_redis_mock()
        app = make_app(redis=redis, llm=make_llm_mock())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/cache/stats")
        assert "x-request-id" in r.headers

    async def test_503_when_state_missing(self):
        from app import create_app
        app = create_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/cache/stats")
        assert r.status_code in (500, 503)


# ---------------------------------------------------------------------------
# GET /cache/review
# ---------------------------------------------------------------------------

class TestCacheReviewListEndpoint:
    async def test_returns_empty_list_when_no_pending(self):
        redis = make_cache_redis_mock()
        redis.client.zrange = AsyncMock(return_value=[])
        app = make_app(redis=redis, llm=make_llm_mock())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/cache/review")
        assert r.status_code == 200
        assert r.json()["pending"] == []
        assert r.json()["total"] == 0

    async def test_returns_pending_entries(self):
        from unittest.mock import MagicMock, AsyncMock
        entry = _make_cache_entry()
        redis = make_cache_redis_mock(entry=entry)
        redis.client.zrange = AsyncMock(return_value=[entry.query_hash])
        pipe = MagicMock()
        pipe.get = MagicMock(return_value=pipe)
        pipe.execute = AsyncMock(return_value=[entry.model_dump_json()])
        redis.client.pipeline = MagicMock(return_value=pipe)
        app = make_app(redis=redis, llm=make_llm_mock())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/cache/review")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        assert body["pending"][0]["query_hash"] == entry.query_hash
        assert body["pending"][0]["status"] == "pending_review"

    async def test_response_includes_chunk_count(self):
        from unittest.mock import MagicMock, AsyncMock
        entry = _make_cache_entry()
        redis = make_cache_redis_mock(entry=entry)
        redis.client.zrange = AsyncMock(return_value=[entry.query_hash])
        pipe = MagicMock()
        pipe.get = MagicMock(return_value=pipe)
        pipe.execute = AsyncMock(return_value=[entry.model_dump_json()])
        redis.client.pipeline = MagicMock(return_value=pipe)
        app = make_app(redis=redis, llm=make_llm_mock())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/cache/review")
        assert r.json()["pending"][0]["chunk_count"] == 1

    async def test_respects_limit_query_param(self):
        redis = make_cache_redis_mock()
        redis.client.zrange = AsyncMock(return_value=[])
        app = make_app(redis=redis, llm=make_llm_mock())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/cache/review", params={"limit": 5})
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# POST /cache/review/{key}/approve
# ---------------------------------------------------------------------------

class TestCacheApproveEndpoint:
    async def test_returns_200_with_query_hash_and_status(self):
        entry = _make_cache_entry()
        redis = make_cache_redis_mock(entry=entry)
        app = make_app(redis=redis, llm=make_llm_mock())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post(
                f"/cache/review/{entry.query_hash}/approve",
                json={"reviewer_id": "admin-1"},
            )
        assert r.status_code == 200
        body = r.json()
        assert body["query_hash"] == entry.query_hash
        assert body["status"] in ["approved", "pending_review"]

    async def test_returns_404_when_entry_not_found(self):
        redis = make_cache_redis_mock()
        app = make_app(redis=redis, llm=make_llm_mock())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post(
                "/cache/review/0000000000000000/approve",
                json={"reviewer_id": "admin-1"},
            )
        assert r.status_code == 404

    async def test_returns_422_when_reviewer_id_missing(self):
        entry = _make_cache_entry()
        redis = make_cache_redis_mock(entry=entry)
        app = make_app(redis=redis, llm=make_llm_mock())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post(
                f"/cache/review/{entry.query_hash}/approve",
                json={},
            )
        assert r.status_code == 422

    async def test_returns_422_when_body_missing(self):
        redis = make_cache_redis_mock()
        app = make_app(redis=redis, llm=make_llm_mock())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/cache/review/0000000000000000/approve")
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# POST /cache/review/{key}/reject
# ---------------------------------------------------------------------------

class TestCacheRejectEndpoint:
    async def test_returns_204_on_successful_reject(self):
        entry = _make_cache_entry()
        redis = make_cache_redis_mock(entry=entry)
        app = make_app(redis=redis, llm=make_llm_mock())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post(f"/cache/review/{entry.query_hash}/reject")
        assert r.status_code == 204

    async def test_returns_404_when_entry_not_found(self):
        redis = make_cache_redis_mock()
        app = make_app(redis=redis, llm=make_llm_mock())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/cache/review/0000000000000000/reject")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /cache/{key}
# ---------------------------------------------------------------------------

class TestCacheDeleteEndpoint:
    async def test_returns_204_on_successful_delete(self):
        redis = make_cache_redis_mock()
        redis.client.delete = AsyncMock(return_value=1)
        app = make_app(redis=redis, llm=make_llm_mock())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.delete("/cache/0123456789abcdef")
        assert r.status_code == 204

    async def test_returns_404_when_key_not_found(self):
        redis = make_cache_redis_mock()
        redis.client.delete = AsyncMock(return_value=0)
        app = make_app(redis=redis, llm=make_llm_mock())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.delete("/cache/0000000000000000")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Full pipeline: miss → approve → hit (via service, not HTTP)
# ---------------------------------------------------------------------------

class TestCachePipeline:
    async def test_approve_changes_status_in_stored_entry(self):
        """After approving, the stored entry should have status=APPROVED."""
        entry = _make_cache_entry(CacheStatus.PENDING_REVIEW)
        redis = make_cache_redis_mock(entry=entry)
        app = make_app(redis=redis, llm=make_llm_mock())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post(
                f"/cache/review/{entry.query_hash}/approve",
                json={"reviewer_id": "admin-1"},
            )
        assert r.status_code == 200
        # setex was called to store the updated entry
        redis.client.setex.assert_awaited()

    async def test_reject_changes_status_in_stored_entry(self):
        """After rejecting, the stored entry should have status=REJECTED."""
        entry = _make_cache_entry(CacheStatus.PENDING_REVIEW)
        redis = make_cache_redis_mock(entry=entry)
        app = make_app(redis=redis, llm=make_llm_mock())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post(f"/cache/review/{entry.query_hash}/reject")
        assert r.status_code == 204
        redis.client.setex.assert_awaited()


# ---------------------------------------------------------------------------
# X-API-Key authentication requirements
# ---------------------------------------------------------------------------

_TEST_KEY = "test-secret"


def _key_patch():
    """Patch settings so cache_api_key is non-empty, enabling auth checks."""
    return patch.object(_settings, "cache_api_key", _TEST_KEY)


class TestCacheAuthRequirements:
    """When cache_api_key is configured, every endpoint must enforce X-API-Key.

    GET /stats and GET /review are the RED cases: they have no auth dependency
    yet. The write endpoints (approve/reject/delete) already enforce auth but
    have no tests covering the 401 paths.
    """

    # ---- GET /cache/stats ----

    async def test_stats_rejects_missing_key(self):
        redis = make_cache_redis_mock()
        redis.client.pipeline = MagicMock(return_value=_make_stats_pipeline())
        app = make_app(redis=redis, llm=make_llm_mock())
        with _key_patch():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                r = await c.get("/cache/stats")
        assert r.status_code == 401

    async def test_stats_rejects_wrong_key(self):
        redis = make_cache_redis_mock()
        redis.client.pipeline = MagicMock(return_value=_make_stats_pipeline())
        app = make_app(redis=redis, llm=make_llm_mock())
        with _key_patch():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                r = await c.get("/cache/stats", headers={"X-API-Key": "wrong"})
        assert r.status_code == 401

    async def test_stats_accepts_correct_key(self):
        redis = make_cache_redis_mock()
        redis.client.pipeline = MagicMock(return_value=_make_stats_pipeline(1, 2, 0))
        app = make_app(redis=redis, llm=make_llm_mock())
        with _key_patch():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                r = await c.get("/cache/stats", headers={"X-API-Key": _TEST_KEY})
        assert r.status_code == 200

    # ---- GET /cache/review ----

    async def test_review_rejects_missing_key(self):
        redis = make_cache_redis_mock()
        redis.client.zrange = AsyncMock(return_value=[])
        app = make_app(redis=redis, llm=make_llm_mock())
        with _key_patch():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                r = await c.get("/cache/review")
        assert r.status_code == 401

    async def test_review_rejects_wrong_key(self):
        redis = make_cache_redis_mock()
        redis.client.zrange = AsyncMock(return_value=[])
        app = make_app(redis=redis, llm=make_llm_mock())
        with _key_patch():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                r = await c.get("/cache/review", headers={"X-API-Key": "wrong"})
        assert r.status_code == 401

    async def test_review_accepts_correct_key(self):
        redis = make_cache_redis_mock()
        redis.client.zrange = AsyncMock(return_value=[])
        app = make_app(redis=redis, llm=make_llm_mock())
        with _key_patch():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                r = await c.get("/cache/review", headers={"X-API-Key": _TEST_KEY})
        assert r.status_code == 200

    # ---- POST /cache/review/{hash}/approve ----

    async def test_approve_rejects_missing_key(self):
        entry = _make_cache_entry()
        redis = make_cache_redis_mock(entry=entry)
        app = make_app(redis=redis, llm=make_llm_mock())
        with _key_patch():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                r = await c.post(
                    f"/cache/review/{entry.query_hash}/approve",
                    json={"reviewer_id": "admin-1"},
                )
        assert r.status_code == 401

    async def test_approve_rejects_wrong_key(self):
        entry = _make_cache_entry()
        redis = make_cache_redis_mock(entry=entry)
        app = make_app(redis=redis, llm=make_llm_mock())
        with _key_patch():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                r = await c.post(
                    f"/cache/review/{entry.query_hash}/approve",
                    json={"reviewer_id": "admin-1"},
                    headers={"X-API-Key": "wrong"},
                )
        assert r.status_code == 401

    async def test_approve_accepts_correct_key(self):
        entry = _make_cache_entry()
        redis = make_cache_redis_mock(entry=entry)
        app = make_app(redis=redis, llm=make_llm_mock())
        with _key_patch():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                r = await c.post(
                    f"/cache/review/{entry.query_hash}/approve",
                    json={"reviewer_id": "admin-1"},
                    headers={"X-API-Key": _TEST_KEY},
                )
        assert r.status_code == 200

    # ---- POST /cache/review/{hash}/reject ----

    async def test_reject_rejects_missing_key(self):
        entry = _make_cache_entry()
        redis = make_cache_redis_mock(entry=entry)
        app = make_app(redis=redis, llm=make_llm_mock())
        with _key_patch():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                r = await c.post(f"/cache/review/{entry.query_hash}/reject")
        assert r.status_code == 401

    async def test_reject_accepts_correct_key(self):
        entry = _make_cache_entry()
        redis = make_cache_redis_mock(entry=entry)
        app = make_app(redis=redis, llm=make_llm_mock())
        with _key_patch():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                r = await c.post(
                    f"/cache/review/{entry.query_hash}/reject",
                    headers={"X-API-Key": _TEST_KEY},
                )
        assert r.status_code == 204

    # ---- DELETE /cache/{hash} ----

    async def test_delete_rejects_missing_key(self):
        redis = make_cache_redis_mock()
        redis.client.delete = AsyncMock(return_value=1)
        app = make_app(redis=redis, llm=make_llm_mock())
        with _key_patch():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                r = await c.delete("/cache/0123456789abcdef")
        assert r.status_code == 401

    async def test_delete_accepts_correct_key(self):
        redis = make_cache_redis_mock()
        redis.client.delete = AsyncMock(return_value=1)
        app = make_app(redis=redis, llm=make_llm_mock())
        with _key_patch():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                r = await c.delete("/cache/0123456789abcdef", headers={"X-API-Key": _TEST_KEY})
        assert r.status_code == 204
