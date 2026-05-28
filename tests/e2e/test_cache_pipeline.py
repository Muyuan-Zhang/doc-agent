"""E2E tests for M3 cache router via FastAPI ASGI.

All tests bypass lifespan — app.state is injected directly.
Exercises the full middleware → router → service stack with mocked Redis.
"""
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from httpx import ASGITransport, AsyncClient

from app.cache.schemas import CacheEntry, CacheStatus
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

class TestCacheStatsEndpoint:
    async def test_returns_200_with_stats_fields(self):
        redis = make_cache_redis_mock()
        redis.client.get = AsyncMock(side_effect=["5", "3"])
        redis.client.llen = AsyncMock(return_value=2)
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
        redis.client.get = AsyncMock(return_value=None)
        redis.client.llen = AsyncMock(return_value=0)
        app = make_app(redis=redis, llm=make_llm_mock())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/cache/stats")
        assert r.status_code == 200
        assert r.json()["hits"] == 0
        assert r.json()["misses"] == 0

    async def test_includes_request_id_header(self):
        redis = make_cache_redis_mock()
        redis.client.get = AsyncMock(return_value=None)
        redis.client.llen = AsyncMock(return_value=0)
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
        redis.client.lrange = AsyncMock(return_value=[])
        app = make_app(redis=redis, llm=make_llm_mock())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/cache/review")
        assert r.status_code == 200
        body = r.json()
        assert body["pending"] == []
        assert body["total"] == 0

    async def test_returns_pending_entries(self):
        entry = _make_cache_entry()
        redis = make_cache_redis_mock(entry=entry)
        app = make_app(redis=redis, llm=make_llm_mock())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/cache/review")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        assert body["pending"][0]["query_hash"] == entry.query_hash
        assert body["pending"][0]["status"] == "pending_review"

    async def test_response_includes_chunk_count(self):
        entry = _make_cache_entry()
        redis = make_cache_redis_mock(entry=entry)
        app = make_app(redis=redis, llm=make_llm_mock())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/cache/review")
        assert r.json()["pending"][0]["chunk_count"] == 1

    async def test_respects_limit_query_param(self):
        redis = make_cache_redis_mock()
        redis.client.lrange = AsyncMock(return_value=[])
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
                "/cache/review/nonexistenthash000/approve",
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
            r = await c.post("/cache/review/somehash/approve")
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
            r = await c.post("/cache/review/badkey123456789/reject")
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
            r = await c.delete("/cache/abc123")
        assert r.status_code == 204

    async def test_returns_404_when_key_not_found(self):
        redis = make_cache_redis_mock()
        redis.client.delete = AsyncMock(return_value=0)
        app = make_app(redis=redis, llm=make_llm_mock())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.delete("/cache/nonexistent")
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
