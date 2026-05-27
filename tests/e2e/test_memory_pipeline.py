"""E2E tests for M5 memory pipeline via FastAPI ASGI."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app import create_app


def _make_state():
    pg = MagicMock()
    redis = MagicMock()
    milvus = MagicMock()
    llm = MagicMock()
    mq = MagicMock()
    return pg, redis, milvus, llm, mq


@pytest.fixture
def app_with_memory(monkeypatch):
    """FastAPI app with all clients pre-patched so no real infra is needed."""
    pg, redis, milvus, llm, mq = _make_state()

    with (
        patch("app.PostgreSQLClient") as MockPG,
        patch("app.RedisClient") as MockRedis,
        patch("app.MilvusClient") as MockMilvus,
        patch("app.RedisStreamsMQClient") as MockMQ,
        patch("app.OpenAILLMClient") as MockLLM,
    ):
        pg_inst = MagicMock()
        pg_inst.connect = AsyncMock()
        pg_inst.disconnect = AsyncMock()
        redis_inst = MagicMock()
        redis_inst.connect = AsyncMock()
        redis_inst.disconnect = AsyncMock()
        milvus_inst = MagicMock()
        milvus_inst.connect = AsyncMock()
        milvus_inst.disconnect = AsyncMock()
        mq_inst = MagicMock()
        mq_inst.connect = AsyncMock()
        mq_inst.disconnect = AsyncMock()
        llm_inst = MagicMock()
        llm_inst.connect = AsyncMock()
        llm_inst.disconnect = AsyncMock()
        llm_inst.complete = AsyncMock(return_value="Summarized conversation.")
        llm_inst.embed = AsyncMock(return_value=[0.1] * 10)

        MockPG.return_value = pg_inst
        MockRedis.return_value = redis_inst
        MockMilvus.return_value = milvus_inst
        MockMQ.return_value = mq_inst
        MockLLM.return_value = llm_inst

        app = create_app()
        app.state.postgres = pg_inst
        app.state.redis = redis_inst
        app.state.milvus = milvus_inst
        app.state.mq = mq_inst
        app.state.llm = llm_inst

        yield app, redis_inst, pg_inst, milvus_inst, llm_inst


@pytest.fixture
async def client(app_with_memory):
    app, *_ = app_with_memory
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c, app_with_memory


def _patch_recent_store(redis_inst, llen_val: int = 1, lrange_val: list | None = None):
    redis_inst.client = MagicMock()
    redis_inst.client.rpush = AsyncMock(return_value=llen_val)
    redis_inst.client.ltrim = AsyncMock()
    redis_inst.client.llen = AsyncMock(return_value=llen_val)
    redis_inst.client.lrange = AsyncMock(return_value=lrange_val or [])
    redis_inst.client.expire = AsyncMock()
    redis_inst.client.delete = AsyncMock()
    redis_inst.cache_key = MagicMock(return_value="v1:memory:recent:sess-1")


def _patch_pg_summary(pg_inst, fetchone=None):
    mock_conn = AsyncMock()
    mock_result = MagicMock()
    mock_result.fetchone = MagicMock(return_value=fetchone)
    mock_result.rowcount = 1
    mock_conn.execute = AsyncMock(return_value=mock_result)
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    ctx.__aexit__ = AsyncMock(return_value=None)
    pg_inst.engine = MagicMock()
    pg_inst.engine.begin = MagicMock(return_value=ctx)
    pg_inst.engine.connect = MagicMock(return_value=ctx)
    return mock_conn


def _patch_milvus_memory(milvus_inst, search_hits: list | None = None):
    milvus_inst.memory_insert = AsyncMock()
    milvus_inst.memory_search = AsyncMock(return_value=search_hits or [])
    milvus_inst.memory_delete = AsyncMock()


class TestAppendTurnEndpoint:
    async def test_returns_204(self, client):
        c, (app, redis, pg, milvus, llm) = client
        _patch_recent_store(redis, llen_val=1)
        resp = await c.post("/memory/turns", json={
            "session_id": "sess-1",
            "user_id": "user-1",
            "role": "user",
            "content": "Hello world",
        })
        assert resp.status_code == 204

    async def test_invalid_body_returns_422(self, client):
        c, _ = client
        resp = await c.post("/memory/turns", json={"session_id": "sess-1"})
        assert resp.status_code == 422


class TestGetContextEndpoint:
    async def test_returns_200_with_context(self, client):
        c, (app, redis, pg, milvus, llm) = client
        _patch_recent_store(redis, lrange_val=[])
        _patch_pg_summary(pg, fetchone=None)
        resp = await c.get("/memory/context/sess-1", params={"user_id": "user-1"})
        assert resp.status_code == 200
        body = resp.json()
        assert "turns" in body
        assert "static_facts" in body

    async def test_empty_context_when_no_history(self, client):
        c, (app, redis, pg, milvus, llm) = client
        _patch_recent_store(redis, lrange_val=[])
        _patch_pg_summary(pg, fetchone=None)
        resp = await c.get("/memory/context/sess-1", params={"user_id": "user-1"})
        body = resp.json()
        assert body["turns"] == []
        assert body["summary"] is None
        assert body["static_facts"] == []

    async def test_missing_user_id_returns_422(self, client):
        c, _ = client
        resp = await c.get("/memory/context/sess-1")
        assert resp.status_code == 422


class TestSummarizeEndpoint:
    async def test_returns_200_with_summary(self, client):
        c, (app, redis, pg, milvus, llm) = client
        _patch_recent_store(redis, lrange_val=[])
        conn = _patch_pg_summary(pg)
        resp = await c.post("/memory/summarize/sess-1", params={"user_id": "user-1"})
        assert resp.status_code == 200
        body = resp.json()
        assert "summary_id" in body
        assert "summary_text" in body

    async def test_summary_text_from_llm(self, client):
        c, (app, redis, pg, milvus, llm) = client
        llm.complete = AsyncMock(return_value="Key points: A, B, C.")
        _patch_recent_store(redis, lrange_val=[])
        _patch_pg_summary(pg)
        resp = await c.post("/memory/summarize/sess-1", params={"user_id": "user-1"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["summary_text"] == "Key points: A, B, C."


class TestStaticFactEndpoints:
    async def test_add_fact_returns_201(self, client):
        c, (app, redis, pg, milvus, llm) = client
        _patch_pg_summary(pg)
        _patch_milvus_memory(milvus)
        resp = await c.post("/memory/static", json={
            "user_id": "user-1",
            "content": "I prefer Python over Java.",
        })
        assert resp.status_code == 201
        body = resp.json()
        assert body["content"] == "I prefer Python over Java."
        assert "fact_id" in body

    async def test_add_fact_missing_content_returns_422(self, client):
        c, _ = client
        resp = await c.post("/memory/static", json={"user_id": "user-1"})
        assert resp.status_code == 422

    async def test_delete_fact_returns_204(self, client):
        c, (app, redis, pg, milvus, llm) = client
        conn = _patch_pg_summary(pg)
        _patch_milvus_memory(milvus)
        resp = await c.delete("/memory/static/fact-1", params={"user_id": "user-1"})
        assert resp.status_code == 204


class TestFullPipeline:
    async def test_append_then_retrieve_context(self, client):
        """Full flow: append turn → retrieve context shows turns."""
        import json as _json
        c, (app, redis, pg, milvus, llm) = client

        turn_json = '{"session_id":"sess-1","role":"user","content":"Hello","ts":1000.0}'
        _patch_recent_store(redis, llen_val=1, lrange_val=[turn_json])
        _patch_pg_summary(pg, fetchone=None)

        resp = await c.post("/memory/turns", json={
            "session_id": "sess-1", "user_id": "user-1",
            "role": "user", "content": "Hello",
        })
        assert resp.status_code == 204

        resp = await c.get("/memory/context/sess-1", params={"user_id": "user-1"})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["turns"]) == 1
        assert body["turns"][0]["content"] == "Hello"
