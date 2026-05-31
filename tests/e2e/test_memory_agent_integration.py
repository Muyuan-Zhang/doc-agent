"""E2E tests for M5 Memory ↔ M4 Agent integration.

Covers the end-to-end user_id flow:
  POST /agent/query (with/without user_id) → MQ publish → consumer state
  Validation of user_id in QueryRequest
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from httpx import ASGITransport, AsyncClient

from tests.e2e.conftest import make_app, make_redis_mock


def _make_mq_mock() -> MagicMock:
    mq = MagicMock()
    mq.publish = AsyncMock(return_value="1-0")
    mq.ping = AsyncMock(return_value=True)
    mq.connect = AsyncMock()
    mq.disconnect = AsyncMock()
    return mq


def _make_redis_with_job(status: str = "queued", answer: str = "", error: str = "") -> MagicMock:
    redis = make_redis_mock()
    redis.client.hset = AsyncMock()
    redis.client.expire = AsyncMock()
    redis.client.hgetall = AsyncMock(
        return_value={"status": status, "answer": answer, "error": error}
    )
    return redis


# ---------------------------------------------------------------------------
# POST /agent/query — user_id field
# ---------------------------------------------------------------------------

class TestAgentQueryWithUserId:
    """user_id flows from HTTP request → MQ publish payload."""

    async def test_includes_user_id_in_published_message(self):
        mq = _make_mq_mock()
        redis = _make_redis_with_job()
        app = make_app(mq=mq, redis=redis)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            await c.post("/agent/query", json={
                "session_id": "sess-1",
                "query": "explain async",
                "user_id": "user-42",
            })

        mq.publish.assert_awaited_once()
        published = mq.publish.call_args[0][0]
        assert published["user_id"] == "user-42"
        assert published["query"] == "explain async"

    async def test_user_id_defaults_to_empty_string_when_not_provided(self):
        mq = _make_mq_mock()
        redis = _make_redis_with_job()
        app = make_app(mq=mq, redis=redis)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            await c.post("/agent/query", json={
                "session_id": "sess-1",
                "query": "hello",
            })

        published = mq.publish.call_args[0][0]
        assert published["user_id"] == ""

    async def test_user_id_with_underscore_accepted(self):
        redis = _make_redis_with_job()
        app = make_app(mq=_make_mq_mock(), redis=redis)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post("/agent/query", json={
                "session_id": "sess-1",
                "query": "test",
                "user_id": "user_42",
            })

        assert resp.status_code == 202

    async def test_user_id_with_hyphen_accepted(self):
        redis = _make_redis_with_job()
        app = make_app(mq=_make_mq_mock(), redis=redis)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post("/agent/query", json={
                "session_id": "sess-1",
                "query": "test",
                "user_id": "user-42-admin",
            })

        assert resp.status_code == 202

    async def test_user_id_64_chars_accepted(self):
        redis = _make_redis_with_job()
        app = make_app(mq=_make_mq_mock(), redis=redis)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post("/agent/query", json={
                "session_id": "sess-1",
                "query": "test",
                "user_id": "a" * 64,
            })

        assert resp.status_code == 202

    async def test_empty_user_id_rejected(self):
        app = make_app()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post("/agent/query", json={
                "session_id": "sess-1",
                "query": "test",
                "user_id": "",
            })

        assert resp.status_code == 422

    async def test_user_id_too_long_rejected(self):
        app = make_app()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post("/agent/query", json={
                "session_id": "sess-1",
                "query": "test",
                "user_id": "a" * 65,
            })

        assert resp.status_code == 422

    async def test_user_id_with_special_chars_rejected(self):
        app = make_app()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post("/agent/query", json={
                "session_id": "sess-1",
                "query": "test",
                "user_id": "user@evil",
            })

        assert resp.status_code == 422

    async def test_user_id_null_accepted_and_coerced(self):
        """user_id: null in JSON → None → coerced to '' in MQ."""
        mq = _make_mq_mock()
        redis = _make_redis_with_job()
        app = make_app(mq=mq, redis=redis)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post("/agent/query", json={
                "session_id": "sess-1",
                "query": "test",
                "user_id": None,
            })

        assert resp.status_code == 202
        published = mq.publish.call_args[0][0]
        assert published["user_id"] == ""


# ---------------------------------------------------------------------------
# Full pipeline: all fields in MQ publish
# ---------------------------------------------------------------------------

class TestMemoryAgentPipeline:
    """End-to-end: API query with user_id → MQ publish includes all consumer fields."""

    async def test_publish_includes_all_required_fields(self):
        mq = _make_mq_mock()
        redis = _make_redis_with_job()
        app = make_app(mq=mq, redis=redis)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post("/agent/query", json={
                "session_id": "sess-e2e",
                "query": "end-to-end test",
                "top_k": 7,
                "user_id": "u-e2e",
            })

        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        published = mq.publish.call_args[0][0]
        assert published["job_id"] == job_id
        assert published["session_id"] == "sess-e2e"
        assert published["query"] == "end-to-end test"
        assert published["top_k"] == "7"
        assert published["user_id"] == "u-e2e"

    async def test_stream_includes_security_headers(self):
        redis = _make_redis_with_job(status="done", answer="ok")
        app = make_app(mq=_make_mq_mock(), redis=redis)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            qresp = await c.post("/agent/query", json={
                "session_id": "sess-sec",
                "query": "test",
                "user_id": "u-sec",
            })
            job_id = qresp.json()["job_id"]
            sresp = await c.get(f"/agent/stream/{job_id}")

        assert "content-security-policy" in sresp.headers
        assert "x-frame-options" in sresp.headers
