"""E2E tests for M5 memory pipeline via FastAPI ASGI.

Uses shared fixtures from conftest.py — no lifespan patching needed.
The memory router reads only from app.state.*, so direct state injection
is sufficient to exercise the full middleware → router → service stack.
"""
import json

import pytest
from httpx import ASGITransport, AsyncClient

from tests.e2e.conftest import make_app, make_llm_mock, make_pg_mock, make_redis_mock, make_milvus_mock
from app.core.exceptions import NotFoundError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _app_with_memory():
    redis = make_redis_mock()
    pg = make_pg_mock()
    milvus = make_milvus_mock()
    llm = make_llm_mock()
    return make_app(redis=redis, postgres=pg, milvus=milvus, llm=llm), redis, pg, milvus, llm


def _turn_json(content: str = "Hello") -> str:
    from app.memory.schemas import ConversationTurn
    return ConversationTurn(
        session_id="sess-1", role="user", content=content, ts=1000.0
    ).model_dump_json()


# ---------------------------------------------------------------------------
# Middleware — x-request-id must propagate on all memory endpoints
# ---------------------------------------------------------------------------

class TestMemoryMiddleware:
    async def test_request_id_on_append_turn(self):
        app, redis, *_ = _app_with_memory()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/memory/turns", json={
                "session_id": "s1", "user_id": "u1", "role": "user", "content": "hi"
            })
        assert "x-request-id" in r.headers

    async def test_request_id_on_get_context(self):
        app, redis, pg, *_ = _app_with_memory()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/memory/context/s1", params={"user_id": "u1"})
        assert "x-request-id" in r.headers

    async def test_503_when_state_missing_on_memory(self):
        from app import create_app
        app = create_app()  # no state set
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/memory/turns", json={
                "session_id": "s1", "user_id": "u1", "role": "user", "content": "hi"
            })
        assert r.status_code in (500, 503)


# ---------------------------------------------------------------------------
# POST /memory/turns
# ---------------------------------------------------------------------------

class TestAppendTurnEndpoint:
    async def test_returns_204(self):
        app, *_ = _app_with_memory()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/memory/turns", json={
                "session_id": "sess-1", "user_id": "user-1",
                "role": "user", "content": "Hello world",
            })
        assert r.status_code == 204

    async def test_invalid_body_returns_422(self):
        app, *_ = _app_with_memory()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/memory/turns", json={"session_id": "s1"})
        assert r.status_code == 422

    async def test_missing_body_returns_422(self):
        app, *_ = _app_with_memory()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/memory/turns")
        assert r.status_code == 422

    async def test_session_id_with_colon_returns_422(self):
        app, *_ = _app_with_memory()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/memory/turns", json={
                "session_id": "sess:inject", "user_id": "u1", "role": "user", "content": "hi",
            })
        assert r.status_code == 422

    async def test_user_id_with_quotes_returns_422(self):
        app, *_ = _app_with_memory()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/memory/turns", json={
                "session_id": "s1", "user_id": 'user"evil"', "role": "user", "content": "hi",
            })
        assert r.status_code == 422

    async def test_invalid_role_returns_422(self):
        app, *_ = _app_with_memory()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/memory/turns", json={
                "session_id": "s1", "user_id": "u1", "role": "admin", "content": "hi",
            })
        assert r.status_code == 422

    async def test_content_exceeds_max_length_returns_422(self):
        app, *_ = _app_with_memory()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/memory/turns", json={
                "session_id": "s1", "user_id": "u1", "role": "user", "content": "x" * 32769,
            })
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# GET /memory/context/{session_id}
# ---------------------------------------------------------------------------

class TestGetContextEndpoint:
    async def test_returns_200_with_empty_context(self):
        app, *_ = _app_with_memory()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/memory/context/sess-1", params={"user_id": "user-1"})
        assert r.status_code == 200
        body = r.json()
        assert body["turns"] == []
        assert body["summary"] is None
        assert body["static_facts"] == []

    async def test_returns_turns_from_redis(self):
        app, redis, *_ = _app_with_memory()
        redis.client.lrange = __import__("unittest.mock", fromlist=["AsyncMock"]).AsyncMock(
            return_value=[_turn_json("hi there")]
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/memory/context/sess-1", params={"user_id": "user-1"})
        assert r.status_code == 200
        assert len(r.json()["turns"]) == 1
        assert r.json()["turns"][0]["content"] == "hi there"

    async def test_missing_user_id_returns_422(self):
        app, *_ = _app_with_memory()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/memory/context/sess-1")
        assert r.status_code == 422

    async def test_user_id_with_special_chars_returns_422(self):
        app, *_ = _app_with_memory()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/memory/context/sess-1", params={"user_id": "user@example.com"})
        assert r.status_code == 422

    async def test_summary_included_when_pg_has_row(self):
        row = ("sum-1", "user-1", "sess-1", "Prior context.", "a" * 64)
        pg = make_pg_mock(fetchone=row)
        app = make_app(postgres=pg)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/memory/context/sess-1", params={"user_id": "user-1"})
        assert r.status_code == 200
        assert r.json()["summary"]["summary_text"] == "Prior context."


# ---------------------------------------------------------------------------
# POST /memory/summarize/{session_id}
# ---------------------------------------------------------------------------

class TestSummarizeEndpoint:
    async def test_returns_200_with_summary(self):
        app, redis, pg, milvus, llm = _app_with_memory()
        llm.complete = __import__("unittest.mock", fromlist=["AsyncMock"]).AsyncMock(
            return_value="Key decisions: A and B."
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/memory/summarize/sess-1", params={"user_id": "user-1"})
        assert r.status_code == 200
        body = r.json()
        assert "summary_id" in body
        assert body["summary_text"] == "Key decisions: A and B."
        assert body["user_id"] == "user-1"
        assert body["session_id"] == "sess-1"

    async def test_missing_user_id_returns_422(self):
        app, *_ = _app_with_memory()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/memory/summarize/sess-1")
        assert r.status_code == 422

    async def test_user_id_too_long_returns_422(self):
        app, *_ = _app_with_memory()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/memory/summarize/sess-1", params={"user_id": "u" * 65})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# POST /memory/static
# ---------------------------------------------------------------------------

class TestAddStaticFactEndpoint:
    async def test_returns_201_with_fact(self):
        app, *_ = _app_with_memory()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/memory/static", json={
                "user_id": "user-1",
                "content": "I prefer Python over Java.",
            })
        assert r.status_code == 201
        body = r.json()
        assert body["content"] == "I prefer Python over Java."
        assert body["user_id"] == "user-1"
        assert "fact_id" in body
        assert "content_hash" in body

    async def test_missing_content_returns_422(self):
        app, *_ = _app_with_memory()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/memory/static", json={"user_id": "user-1"})
        assert r.status_code == 422

    async def test_missing_user_id_returns_422(self):
        app, *_ = _app_with_memory()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/memory/static", json={"content": "some fact"})
        assert r.status_code == 422

    async def test_user_id_with_at_sign_returns_422(self):
        app, *_ = _app_with_memory()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/memory/static", json={"user_id": "u@bad", "content": "fact"})
        assert r.status_code == 422

    async def test_fact_content_exceeds_max_length_returns_422(self):
        app, *_ = _app_with_memory()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/memory/static", json={"user_id": "u1", "content": "x" * 32769})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# DELETE /memory/static/{fact_id}
# ---------------------------------------------------------------------------

_FACT_ID = "a1b2c3d4-e5f6-4a7b-8c9d-e0f1a2b3c4d5"
_FACT_ID_2 = "b2c3d4e5-f6a7-4b8c-9d0e-f1a2b3c4d5e6"


class TestDeleteStaticFactEndpoint:
    async def test_returns_204_on_success(self):
        app, redis, pg, milvus, llm = _app_with_memory()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.delete(f"/memory/static/{_FACT_ID}", params={"user_id": "user-1"})
        assert r.status_code == 204

    async def test_returns_404_when_fact_missing(self):
        pg = make_pg_mock(rowcount=0)
        app = make_app(postgres=pg)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.delete(f"/memory/static/{_FACT_ID_2}", params={"user_id": "user-1"})
        assert r.status_code == 404

    async def test_missing_user_id_returns_422(self):
        app, *_ = _app_with_memory()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.delete(f"/memory/static/{_FACT_ID}")
        assert r.status_code == 422

    async def test_non_uuid_fact_id_returns_422(self):
        app, *_ = _app_with_memory()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.delete("/memory/static/not-a-uuid", params={"user_id": "u1"})
        assert r.status_code == 422

    async def test_uuid_v1_fact_id_returns_422(self):
        app, *_ = _app_with_memory()
        uuid_v1 = "a1b2c3d4-e5f6-1a7b-8c9d-e0f1a2b3c4d5"  # version 1, not 4
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.delete(f"/memory/static/{uuid_v1}", params={"user_id": "u1"})
        assert r.status_code == 422

    async def test_user_id_with_colon_returns_422(self):
        app, *_ = _app_with_memory()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.delete(f"/memory/static/{_FACT_ID}", params={"user_id": "u:bad"})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class TestMemoryRateLimiting:
    async def test_turns_returns_429_when_rate_limit_exceeded(self):
        from unittest.mock import AsyncMock
        app, *_ = _app_with_memory()
        app.state.redis.increment_with_ttl = AsyncMock(return_value=61)  # over limit of 60
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/memory/turns", json={
                "session_id": "s1", "user_id": "u1", "role": "user", "content": "hi",
            })
        assert r.status_code == 429
        assert "Rate limit exceeded" in r.json()["detail"]

    async def test_summarize_returns_429_when_rate_limit_exceeded(self):
        from unittest.mock import AsyncMock
        app, *_ = _app_with_memory()
        app.state.redis.increment_with_ttl = AsyncMock(return_value=6)  # over limit of 5
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/memory/summarize/sess-1", params={"user_id": "u1"})
        assert r.status_code == 429
        assert "Rate limit exceeded" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Full pipeline — append → retrieve
# ---------------------------------------------------------------------------

class TestFullPipeline:
    async def test_append_then_retrieve_shows_turn(self):
        app, redis, pg, *_ = _app_with_memory()
        turn_raw = _turn_json("hello pipeline")
        redis.client.lrange = __import__("unittest.mock", fromlist=["AsyncMock"]).AsyncMock(
            return_value=[turn_raw]
        )

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            post_r = await c.post("/memory/turns", json={
                "session_id": "sess-1", "user_id": "user-1",
                "role": "user", "content": "hello pipeline",
            })
            assert post_r.status_code == 204

            get_r = await c.get("/memory/context/sess-1", params={"user_id": "user-1"})
        assert get_r.status_code == 200
        body = get_r.json()
        assert len(body["turns"]) == 1
        assert body["turns"][0]["content"] == "hello pipeline"

    async def test_add_fact_then_present_in_milvus(self):
        app, redis, pg, milvus, llm = _app_with_memory()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/memory/static", json={
                "user_id": "user-1",
                "content": "Prefers concise answers.",
            })
        assert r.status_code == 201
        milvus.memory_insert.assert_awaited_once()
        entity = milvus.memory_insert.call_args[0][0][0]
        assert entity["content"] == "Prefers concise answers."
        assert entity["user_id"] == "user-1"
