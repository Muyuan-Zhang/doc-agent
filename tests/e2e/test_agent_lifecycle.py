"""E2E lifecycle test: full job cycle across all M4 endpoints.

Uses _StatefulRedisInner so that writes from _init_job (via POST /query)
and _set_job_status (via _process_message) are visible to subsequent
GET /jobs and GET /stream calls — testing real state transitions.
"""
from collections import defaultdict
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.agent.consumer import _process_message
from app.clients.mq import MQMessage
from tests.e2e.conftest import make_app


class _StatefulRedisInner:
    """In-memory Redis stand-in that maintains hash and list state across calls."""

    def __init__(self):
        self._hashes: dict[str, dict] = defaultdict(dict)
        self._lists: dict[str, list] = defaultdict(list)

    async def hset(self, key: str, mapping: dict) -> None:
        self._hashes[key].update(mapping)

    async def hgetall(self, key: str) -> dict:
        return dict(self._hashes[key])

    async def expire(self, key: str, ttl: int) -> None:
        pass

    async def setex(self, key: str, ttl: int, value) -> None:
        pass

    async def rpush(self, key: str, *values) -> int:
        self._lists[key].extend(values)
        return len(self._lists[key])

    async def ltrim(self, *args) -> None:
        pass

    async def llen(self, key: str) -> int:
        return len(self._lists[key])

    async def lrange(self, key: str, start: int, end: int) -> list:
        data = self._lists[key]
        return data[start:] if end == -1 else data[start: end + 1]

    async def delete(self, *keys) -> None:
        for k in keys:
            self._hashes.pop(k, None)
            self._lists.pop(k, None)


def _make_stateful_redis() -> MagicMock:
    m = MagicMock()
    m.ping = AsyncMock(return_value=True)
    m.connect = AsyncMock()
    m.disconnect = AsyncMock()
    m.cache_key = MagicMock(return_value="v1:test:hash")
    m.increment_with_ttl = AsyncMock(return_value=1)
    m.client = _StatefulRedisInner()
    return m


def _make_mq(ack: AsyncMock | None = None) -> MagicMock:
    mq = MagicMock()
    mq.ping = AsyncMock(return_value=True)
    mq.connect = AsyncMock()
    mq.disconnect = AsyncMock()
    mq.publish = AsyncMock(return_value="1-0")
    mq.ack = ack or AsyncMock()
    return mq


def _graph_ok(answer: str = "FastAPI is fast.") -> MagicMock:
    g = MagicMock()
    g.ainvoke = AsyncMock(return_value={"answer": answer, "error": None})
    return g


def _graph_fail(exc: Exception) -> MagicMock:
    g = MagicMock()
    g.ainvoke = AsyncMock(side_effect=exc)
    return g


@pytest.mark.asyncio
class TestAgentJobLifecycle:
    async def test_success_lifecycle(self):
        redis = _make_stateful_redis()
        mq = _make_mq()
        app = make_app(redis=redis, mq=mq)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            # 1. Enqueue
            resp = await c.post("/agent/query", json={
                "session_id": "lc-sess-1",
                "query": "What is FastAPI?",
            })
            assert resp.status_code == 202
            job_id = resp.json()["job_id"]
            assert resp.json()["status"] == "queued"

            # 2. Simulate consumer (graph succeeds)
            msg = MQMessage(id="1-0", stream="agent", data={
                "job_id": job_id,
                "session_id": "lc-sess-1",
                "query": "What is FastAPI?",
                "top_k": "5",
            })
            await _process_message(msg, _graph_ok("FastAPI is fast and async."), redis, mq)
            mq.ack.assert_awaited_once_with("1-0")

            # 3. Poll job status
            resp = await c.get(f"/agent/jobs/{job_id}")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "done"
            assert data["answer"] == "FastAPI is fast and async."
            assert data["job_id"] == job_id

            # 4. SSE stream delivers event: done
            # (no token events — the graph mock bypasses generate, so no RPUSH occurs)
            resp = await c.get(f"/agent/stream/{job_id}")
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            assert "event: done" in resp.text

    async def test_failure_lifecycle(self):
        redis = _make_stateful_redis()
        mq = _make_mq()
        app = make_app(redis=redis, mq=mq)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            # 1. Enqueue
            resp = await c.post("/agent/query", json={
                "session_id": "lc-sess-2",
                "query": "What is LangGraph?",
            })
            assert resp.status_code == 202
            job_id = resp.json()["job_id"]

            # 2. Simulate consumer (graph raises)
            msg = MQMessage(id="1-1", stream="agent", data={
                "job_id": job_id,
                "session_id": "lc-sess-2",
                "query": "What is LangGraph?",
                "top_k": "5",
            })
            await _process_message(msg, _graph_fail(ValueError("embedding down")), redis, mq)
            mq.ack.assert_awaited_once_with("1-1")

            # 3. Job status shows error
            resp = await c.get(f"/agent/jobs/{job_id}")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "error"
            assert "ValueError" in (data["error"] or "")

            # 4. SSE stream delivers event: error with opaque code (raw error is server-only)
            resp = await c.get(f"/agent/stream/{job_id}")
            assert resp.status_code == 200
            assert "event: error" in resp.text
            assert "job_failed" in resp.text
            assert "ValueError" not in resp.text
