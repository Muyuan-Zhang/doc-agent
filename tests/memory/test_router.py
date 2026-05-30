"""Unit tests for memory router endpoints.

Follows the same ASGI-via-httpx pattern as tests/retrieval/test_router.py.
All external I/O (PostgreSQL, Redis, Milvus, LLM) is replaced by in-memory mocks.
"""
from unittest.mock import AsyncMock

from httpx import ASGITransport, AsyncClient

from app.memory.schemas import ConversationTurn
from tests.e2e.conftest import make_app, make_llm_mock, make_pg_mock, make_redis_mock

_VALID_UUID = "a1b2c3d4-e5f6-4a7b-8c9d-e0f1a2b3c4d5"
_UUID_V1 = "a1b2c3d4-e5f6-1a7b-8c9d-e0f1a2b3c4d5"


def _summary_row(text: str = "Previous summary.", summary_id: str = "sum-1") -> tuple:
    return (summary_id, "user-1", "sess-1", text, "a" * 64, 1.0, {})


def _turn_json(content: str = "hello") -> str:
    return ConversationTurn(
        session_id="sess-1", role="user", content=content, ts=1000.0
    ).model_dump_json()


# ---------------------------------------------------------------------------
# GET /memory/context/{session_id}
# ---------------------------------------------------------------------------

class TestGetContextEndpoint:
    async def test_returns_200_with_empty_context(self):
        app = make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/memory/context/sess-1", params={"user_id": "user-1"})
        assert r.status_code == 200
        body = r.json()
        assert body["turns"] == []
        assert body["summary"] is None
        assert body["static_facts"] == []

    async def test_returns_turns_from_redis(self):
        redis = make_redis_mock()
        redis.client.lrange = AsyncMock(return_value=[_turn_json("hello context")])
        app = make_app(redis=redis)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/memory/context/sess-1", params={"user_id": "user-1"})
        assert r.status_code == 200
        turns = r.json()["turns"]
        assert len(turns) == 1
        assert turns[0]["content"] == "hello context"

    async def test_returns_summary_when_pg_has_row(self):
        pg = make_pg_mock(fetchone=_summary_row("Prior context."))
        app = make_app(postgres=pg)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/memory/context/sess-1", params={"user_id": "user-1"})
        assert r.status_code == 200
        assert r.json()["summary"]["summary_text"] == "Prior context."

    async def test_missing_user_id_returns_422(self):
        app = make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/memory/context/sess-1")
        assert r.status_code == 422

    async def test_user_id_with_special_chars_returns_422(self):
        app = make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/memory/context/sess-1", params={"user_id": "user@example.com"})
        assert r.status_code == 422

    async def test_missing_state_returns_503(self):
        from app import create_app
        app = create_app()  # no state injected
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/memory/context/sess-1", params={"user_id": "user-1"})
        assert r.status_code == 503


# ---------------------------------------------------------------------------
# POST /memory/summarize/{session_id}
# ---------------------------------------------------------------------------

class TestSummarizeEndpoint:
    async def test_returns_200_with_llm_summary(self):
        redis = make_redis_mock()
        redis.client.lrange = AsyncMock(return_value=[_turn_json("discuss design")])
        llm = make_llm_mock()
        llm.complete = AsyncMock(return_value="Design decisions: X and Y.")
        app = make_app(redis=redis, llm=llm)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/memory/summarize/sess-1", params={"user_id": "user-1"})
        assert r.status_code == 200
        body = r.json()
        assert body["summary_text"] == "Design decisions: X and Y."
        assert body["session_id"] == "sess-1"
        assert body["user_id"] == "user-1"

    async def test_returns_400_when_no_turns_and_no_prior_summary(self):
        app = make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/memory/summarize/sess-1", params={"user_id": "user-1"})
        assert r.status_code == 400

    async def test_returns_prior_summary_when_no_new_turns(self):
        pg = make_pg_mock(fetchone=_summary_row("Existing summary.", "sum-prev"))
        app = make_app(postgres=pg)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/memory/summarize/sess-1", params={"user_id": "user-1"})
        assert r.status_code == 200
        assert r.json()["summary_id"] == "sum-prev"
        assert r.json()["summary_text"] == "Existing summary."

    async def test_missing_user_id_returns_422(self):
        app = make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/memory/summarize/sess-1")
        assert r.status_code == 422

    async def test_user_id_too_long_returns_422(self):
        app = make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/memory/summarize/sess-1", params={"user_id": "u" * 65})
        assert r.status_code == 422

    async def test_rate_limit_returns_429(self):
        app = make_app()
        app.state.redis.increment_with_ttl = AsyncMock(return_value=6)  # over limit of 5
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/memory/summarize/sess-1", params={"user_id": "user-1"})
        assert r.status_code == 429


# ---------------------------------------------------------------------------
# POST /memory/static
# ---------------------------------------------------------------------------

class TestAddStaticFactEndpoint:
    async def test_returns_201_with_fact(self):
        app = make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/memory/static", json={"user_id": "user-1", "content": "I prefer Python."})
        assert r.status_code == 201
        body = r.json()
        assert body["content"] == "I prefer Python."
        assert body["user_id"] == "user-1"
        assert "fact_id" in body
        assert "content_hash" in body

    async def test_missing_content_returns_422(self):
        app = make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/memory/static", json={"user_id": "user-1"})
        assert r.status_code == 422

    async def test_missing_user_id_returns_422(self):
        app = make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/memory/static", json={"content": "some fact"})
        assert r.status_code == 422

    async def test_user_id_with_at_sign_returns_422(self):
        app = make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/memory/static", json={"user_id": "u@bad", "content": "fact"})
        assert r.status_code == 422

    async def test_content_exceeds_max_length_returns_422(self):
        app = make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/memory/static", json={"user_id": "u1", "content": "x" * 32769})
        assert r.status_code == 422

    async def test_embed_failure_returns_500(self):
        llm = make_llm_mock()
        llm.embed = AsyncMock(side_effect=RuntimeError("embed service down"))
        app = make_app(llm=llm)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/memory/static", json={"user_id": "user-1", "content": "fact"})
        assert r.status_code == 500


# ---------------------------------------------------------------------------
# DELETE /memory/static/{fact_id}
# ---------------------------------------------------------------------------

class TestDeleteStaticFactEndpoint:
    async def test_returns_204_on_success(self):
        app = make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.delete(f"/memory/static/{_VALID_UUID}", params={"user_id": "user-1"})
        assert r.status_code == 204

    async def test_returns_404_when_fact_not_found(self):
        pg = make_pg_mock(rowcount=0)
        app = make_app(postgres=pg)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.delete(f"/memory/static/{_VALID_UUID}", params={"user_id": "user-1"})
        assert r.status_code == 404

    async def test_missing_user_id_returns_422(self):
        app = make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.delete(f"/memory/static/{_VALID_UUID}")
        assert r.status_code == 422

    async def test_non_uuid_fact_id_returns_422(self):
        app = make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.delete("/memory/static/not-a-uuid", params={"user_id": "user-1"})
        assert r.status_code == 422

    async def test_uuid_v1_returns_422(self):
        app = make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.delete(f"/memory/static/{_UUID_V1}", params={"user_id": "user-1"})
        assert r.status_code == 422

    async def test_user_id_with_colon_returns_422(self):
        app = make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.delete(f"/memory/static/{_VALID_UUID}", params={"user_id": "u:bad"})
        assert r.status_code == 422
