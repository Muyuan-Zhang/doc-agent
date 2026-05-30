"""Cache admin panel journey tests.

Each test simulates a complete admin workflow — multiple requests sent through a
single persistent AsyncClient, the way a real browser session would behave.

These complement the per-endpoint tests in test_cache_pipeline.py.  The goal
here is to verify that chained calls produce coherent results and that shared
response shapes are stable across a session.
"""
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.cache.schemas import CacheEntry, CacheStatus
from app.core.config import settings as _settings
from app.models.chunk import ChunkSchema
from tests.e2e.conftest import make_app, make_cache_redis_mock, make_llm_mock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk() -> ChunkSchema:
    return ChunkSchema(
        doc_id="doc-1", section_id="sec-1", chunk_index=0,
        content_hash="abc123", version="v1", content="test content",
    )


def _entry(status: CacheStatus = CacheStatus.PENDING_REVIEW) -> CacheEntry:
    return CacheEntry(
        query_hash="deadbeef12345678",
        original_query="What is the deadline?",
        normalized_query="what is the deadline",
        chunks=[_chunk()],
        status=status,
        created_at=datetime.now(tz=timezone.utc),
    )


def _stats_pipe(hits: int = 0, misses: int = 0, pending: int = 0):
    pipe = MagicMock()
    pipe.hgetall = MagicMock(return_value=pipe)
    pipe.zcard = MagicMock(return_value=pipe)
    pipe.execute = AsyncMock(
        return_value=[{"hits": str(hits), "misses": str(misses)}, pending]
    )
    return pipe


def _get_many_pipe(entry: CacheEntry):
    pipe = MagicMock()
    pipe.get = MagicMock(return_value=pipe)
    pipe.execute = AsyncMock(return_value=[entry.model_dump_json()])
    return pipe


# ---------------------------------------------------------------------------
# Journey 1: stats → review list → approve
# ---------------------------------------------------------------------------

class TestApprovalJourney:
    """Admin loads the panel, views the pending list, and approves an entry."""

    async def test_stats_then_review_then_approve(self):
        e = _entry()
        redis = make_cache_redis_mock(entry=e)
        redis.client.zrange = AsyncMock(return_value=[e.query_hash])
        redis.client.pipeline = MagicMock(side_effect=[
            _stats_pipe(hits=3, misses=7, pending=1),  # get_stats call
            _get_many_pipe(e),                          # get_many call in list_pending
        ])
        app = make_app(redis=redis, llm=make_llm_mock())

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            # Step 1: admin checks stats
            stats = await c.get("/cache/stats")
            assert stats.status_code == 200
            body = stats.json()
            assert body["hits"] == 3
            assert body["misses"] == 7
            assert body["pending"] == 1

            # Step 2: admin opens review queue
            review = await c.get("/cache/review")
            assert review.status_code == 200
            items = review.json()["pending"]
            assert len(items) == 1
            assert items[0]["query_hash"] == e.query_hash
            assert items[0]["chunk_count"] == 1

            # Step 3: admin approves the entry
            approve = await c.post(
                f"/cache/review/{e.query_hash}/approve",
                json={"reviewer_id": "reviewer-1"},
            )
            assert approve.status_code == 200
            result = approve.json()
            assert result["query_hash"] == e.query_hash
            assert result["status"] in ("pending_review", "approved")

    async def test_multi_reviewer_approval_reaches_approved(self):
        """Three reviewers approving the same entry flips status to APPROVED
        (default cache_auto_approve_threshold = 3).
        """
        e = _entry()
        redis = make_cache_redis_mock(entry=e)

        # Track approval state across requests with a mutable container.
        state: dict = {"entry": e}

        async def _get_side(key: str):
            return state["entry"].model_dump_json()

        async def _setex_side(key: str, ttl: int, value: str):
            state["entry"] = CacheEntry.model_validate_json(value)

        redis.client.get = AsyncMock(side_effect=_get_side)
        redis.client.setex = AsyncMock(side_effect=_setex_side)

        app = make_app(redis=redis, llm=make_llm_mock())

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            for reviewer in ("alice", "bob", "carol"):
                r = await c.post(
                    f"/cache/review/{e.query_hash}/approve",
                    json={"reviewer_id": reviewer},
                )
                assert r.status_code == 200

        assert state["entry"].status == CacheStatus.APPROVED
        assert state["entry"].approval_count == 3


# ---------------------------------------------------------------------------
# Journey 2: stats → review list → reject
# ---------------------------------------------------------------------------

class TestRejectionJourney:
    """Admin views the queue and rejects an entry."""

    async def test_stats_then_review_then_reject(self):
        e = _entry()
        redis = make_cache_redis_mock(entry=e)
        redis.client.zrange = AsyncMock(return_value=[e.query_hash])
        redis.client.pipeline = MagicMock(side_effect=[
            _stats_pipe(hits=0, misses=4, pending=1),
            _get_many_pipe(e),
        ])
        app = make_app(redis=redis, llm=make_llm_mock())

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            # Stats show one pending
            stats = await c.get("/cache/stats")
            assert stats.json()["pending"] == 1

            # Review list contains the entry
            review = await c.get("/cache/review")
            assert review.json()["pending"][0]["status"] == "pending_review"

            # Reject it
            reject = await c.post(f"/cache/review/{e.query_hash}/reject")
            assert reject.status_code == 204

        # Rejection persisted: setex was called with REJECTED status
        stored = CacheEntry.model_validate_json(redis.client.setex.call_args[0][2])
        assert stored.status == CacheStatus.REJECTED


# ---------------------------------------------------------------------------
# Journey 3: delete an entry
# ---------------------------------------------------------------------------

class TestDeleteJourney:
    """Admin deletes a cache entry; subsequent lookup returns 404."""

    async def test_delete_then_404_on_approve(self):
        e = _entry()
        redis = make_cache_redis_mock(entry=e)
        # delete() uses redis.client.delete; approve() uses redis.client.get.
        # Simulating post-deletion state: get() always returns None so that the
        # approve router sees a missing entry and returns 404.
        redis.client.get = AsyncMock(return_value=None)
        redis.client.delete = AsyncMock(return_value=1)
        app = make_app(redis=redis, llm=make_llm_mock())

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            delete = await c.delete(f"/cache/{e.query_hash}")
            assert delete.status_code == 204

            approve = await c.post(
                f"/cache/review/{e.query_hash}/approve",
                json={"reviewer_id": "admin"},
            )
            assert approve.status_code == 404


# ---------------------------------------------------------------------------
# Response shape contract
# ---------------------------------------------------------------------------

class TestResponseShapeContract:
    """Verify the response payloads match the documented schema."""

    async def test_review_summary_has_required_fields(self):
        e = _entry()
        redis = make_cache_redis_mock(entry=e)
        redis.client.zrange = AsyncMock(return_value=[e.query_hash])
        redis.client.pipeline = MagicMock(return_value=_get_many_pipe(e))
        app = make_app(redis=redis, llm=make_llm_mock())

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/cache/review")

        item = r.json()["pending"][0]
        required = {
            "query_hash", "original_query", "normalized_query",
            "chunk_count", "status", "approval_count", "created_at",
        }
        assert required.issubset(item.keys()), f"Missing fields: {required - item.keys()}"
        assert "approved_by" not in item  # reviewer identity not exposed in list

    async def test_stats_response_has_all_three_fields(self):
        redis = make_cache_redis_mock()
        redis.client.pipeline = MagicMock(return_value=_stats_pipe(1, 2, 3))
        app = make_app(redis=redis, llm=make_llm_mock())

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/cache/stats")

        body = r.json()
        assert set(body.keys()) == {"hits", "misses", "pending"}
        assert all(isinstance(v, int) for v in body.values())

    async def test_approve_response_shape(self):
        e = _entry()
        redis = make_cache_redis_mock(entry=e)
        app = make_app(redis=redis, llm=make_llm_mock())

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post(
                f"/cache/review/{e.query_hash}/approve",
                json={"reviewer_id": "admin"},
            )

        body = r.json()
        assert set(body.keys()) == {"query_hash", "status"}
        assert body["query_hash"] == e.query_hash


# ---------------------------------------------------------------------------
# Security headers on cache management routes
# ---------------------------------------------------------------------------

class TestCacheRouteSecurityHeaders:
    """CSP and other security headers must be present on all cache endpoints."""

    _REQUIRED = {"content-security-policy", "x-frame-options", "x-content-type-options"}

    async def test_stats_has_security_headers(self):
        redis = make_cache_redis_mock()
        redis.client.pipeline = MagicMock(return_value=_stats_pipe())
        app = make_app(redis=redis, llm=make_llm_mock())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/cache/stats")
        assert self._REQUIRED.issubset(resp.headers)

    async def test_review_list_has_security_headers(self):
        redis = make_cache_redis_mock()
        redis.client.zrange = AsyncMock(return_value=[])
        app = make_app(redis=redis, llm=make_llm_mock())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/cache/review")
        assert self._REQUIRED.issubset(resp.headers)

    async def test_401_response_has_security_headers(self):
        """Even rejected (401) responses carry security headers."""
        e = _entry()
        redis = make_cache_redis_mock(entry=e)
        app = make_app(redis=redis, llm=make_llm_mock())
        with patch.object(_settings, "cache_api_key", "secret"):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                resp = await c.post(
                    f"/cache/review/{e.query_hash}/approve",
                    json={"reviewer_id": "admin"},
                )
        assert resp.status_code == 401
        assert self._REQUIRED.issubset(resp.headers)


# ---------------------------------------------------------------------------
# Dev-mode: empty cache_api_key disables auth
# ---------------------------------------------------------------------------

class TestDevModeAuthBypass:
    """When cache_api_key is empty (default), all endpoints respond without a key."""

    async def test_all_endpoints_accessible_without_key(self):
        e = _entry()
        redis = make_cache_redis_mock(entry=e)
        redis.client.zrange = AsyncMock(return_value=[])
        redis.client.pipeline = MagicMock(return_value=_stats_pipe())
        app = make_app(redis=redis, llm=make_llm_mock())

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            assert (await c.get("/cache/stats")).status_code == 200
            assert (await c.get("/cache/review")).status_code == 200
            assert (await c.post(
                f"/cache/review/{e.query_hash}/approve",
                json={"reviewer_id": "admin"},
            )).status_code == 200
            assert (await c.post(
                f"/cache/review/{e.query_hash}/reject",
            )).status_code == 204
