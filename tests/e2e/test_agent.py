"""E2E tests for M4 Agent API endpoints.

Exercises the full ASGI stack (middleware → router) with mocked client state.
No real Redis, MQ, or LLM connections are made.
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
# POST /agent/query
# ---------------------------------------------------------------------------

class TestEnqueueQuery:
    async def test_returns_202_with_job_id_and_queued_status(self):
        redis = _make_redis_with_job()
        app = make_app(mq=_make_mq_mock(), redis=redis)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post("/agent/query", json={
                "session_id": "sess-1",
                "query": "what is fastapi?",
            })

        assert resp.status_code == 202
        data = resp.json()
        assert "job_id" in data
        assert isinstance(data["job_id"], str)
        assert len(data["job_id"]) > 0
        assert data["status"] == "queued"

    async def test_publishes_to_mq(self):
        mq = _make_mq_mock()
        redis = _make_redis_with_job()
        app = make_app(mq=mq, redis=redis)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            await c.post("/agent/query", json={
                "session_id": "sess-2",
                "query": "explain python",
            })

        mq.publish.assert_awaited_once()
        published = mq.publish.call_args[0][0]
        assert published["query"] == "explain python"
        assert published["session_id"] == "sess-2"

    async def test_uses_custom_top_k(self):
        mq = _make_mq_mock()
        redis = _make_redis_with_job()
        app = make_app(mq=mq, redis=redis)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            await c.post("/agent/query", json={
                "session_id": "sess-3",
                "query": "deep search",
                "top_k": 10,
            })

        published = mq.publish.call_args[0][0]
        assert published["top_k"] == "10"

    async def test_rejects_missing_session_id(self):
        app = make_app()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post("/agent/query", json={"query": "hello"})

        assert resp.status_code == 422

    async def test_rejects_missing_query(self):
        app = make_app()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post("/agent/query", json={"session_id": "s1"})

        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /agent/jobs/{job_id}
# ---------------------------------------------------------------------------

class TestGetJob:
    async def test_returns_queued_status(self):
        redis = _make_redis_with_job(status="queued")
        app = make_app(redis=redis)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/agent/jobs/job-abc")

        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == "job-abc"
        assert data["status"] == "queued"

    async def test_returns_done_status_with_answer(self):
        redis = _make_redis_with_job(status="done", answer="FastAPI is fast.")
        app = make_app(redis=redis)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/agent/jobs/job-done")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "done"
        assert data["answer"] == "FastAPI is fast."

    async def test_returns_404_for_unknown_job(self):
        redis = make_redis_mock()
        redis.client.hgetall = AsyncMock(return_value={})
        app = make_app(redis=redis)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/agent/jobs/nonexistent-job")

        assert resp.status_code == 404

    async def test_returns_error_status_with_message(self):
        redis = _make_redis_with_job(status="error", error="LLM timeout")
        app = make_app(redis=redis)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/agent/jobs/job-err")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"
        assert data["error"] == "LLM timeout"


# ---------------------------------------------------------------------------
# GET /agent/stream/{job_id}
# ---------------------------------------------------------------------------

class TestStreamAnswer:
    async def test_returns_event_stream_content_type(self):
        redis = _make_redis_with_job(status="done", answer="hello world")
        app = make_app(redis=redis)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/agent/stream/job-stream-1")

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    async def test_streams_done_sentinel_at_end(self):
        redis = _make_redis_with_job(status="done", answer="hello world")
        app = make_app(redis=redis)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/agent/stream/job-stream-2")

        assert "[DONE]" in resp.text

    async def test_returns_404_for_unknown_job_in_stream(self):
        redis = make_redis_mock()
        redis.client.hgetall = AsyncMock(return_value={})
        app = make_app(redis=redis)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/agent/stream/nonexistent-job")

        assert resp.status_code == 404
