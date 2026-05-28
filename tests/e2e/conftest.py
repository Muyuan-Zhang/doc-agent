"""Shared fixtures for E2E tests.

All tests bypass the lifespan entirely — `create_app()` is called, then
`app.state.*` is set directly before any request is made. This keeps
tests fast and infrastructure-free while still exercising the full
ASGI stack (middleware → router → service).
"""
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient


def make_client_mock(ping_ok: bool = True) -> MagicMock:
    """Return a mock that satisfies AbstractClient.ping()."""
    m = MagicMock()
    m.ping = AsyncMock(return_value=ping_ok)
    m.connect = AsyncMock()
    m.disconnect = AsyncMock()
    return m


def make_llm_mock() -> MagicMock:
    """LLM mock with embed + complete stubs."""
    m = make_client_mock()
    m.complete = AsyncMock(return_value="mock summary")
    m.embed = AsyncMock(return_value=[0.1] * 10)
    return m


def make_redis_mock() -> MagicMock:
    """Redis mock with all memory-store and M4 agent operations pre-stubbed."""
    m = make_client_mock()
    m.cache_key = MagicMock(return_value="v1:memory:recent:test-session")
    m.increment_with_ttl = AsyncMock(return_value=1)
    inner = MagicMock()
    # M5 memory operations
    inner.rpush = AsyncMock(return_value=1)
    inner.ltrim = AsyncMock()
    inner.llen = AsyncMock(return_value=1)
    inner.lrange = AsyncMock(return_value=[])
    inner.expire = AsyncMock()
    inner.delete = AsyncMock()
    # M4 agent job operations
    inner.hset = AsyncMock()
    inner.hgetall = AsyncMock(return_value={})
    inner.setex = AsyncMock()
    m.client = inner
    return m


def make_pg_mock(fetchone=None, rowcount: int = 1) -> MagicMock:
    """PostgreSQL mock with a pre-wired async context-manager engine."""
    conn = AsyncMock()
    result = MagicMock()
    result.fetchone = MagicMock(return_value=fetchone)
    result.rowcount = rowcount
    conn.execute = AsyncMock(return_value=result)
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=None)
    engine = MagicMock()
    engine.begin = MagicMock(return_value=ctx)
    engine.connect = MagicMock(return_value=ctx)
    m = make_client_mock()
    m.engine = engine
    return m


def make_milvus_mock() -> MagicMock:
    """Milvus mock with memory + KB stubs."""
    m = make_client_mock()
    m.memory_insert = AsyncMock()
    m.memory_search = AsyncMock(return_value=[])
    m.memory_delete = AsyncMock()
    m.ensure_kb_collection = AsyncMock()
    m.insert = AsyncMock(return_value=[])
    m.delete_by_doc_id = AsyncMock()
    return m


def make_app(*, postgres=None, redis=None, milvus=None, mq=None, llm=None):
    """Create the FastAPI app with all client state injected (no lifespan)."""
    from app import create_app
    app = create_app()
    app.state.postgres = postgres or make_pg_mock()
    app.state.redis = redis or make_redis_mock()
    app.state.milvus = milvus or make_milvus_mock()
    app.state.mq = mq or make_client_mock()
    app.state.llm = llm or make_llm_mock()
    return app


@pytest.fixture
def base_app():
    """App with all healthy mocked clients."""
    return make_app()


@pytest.fixture
async def http(base_app):
    """Ready-to-use AsyncClient against the base app."""
    async with AsyncClient(
        transport=ASGITransport(app=base_app), base_url="http://test"
    ) as c:
        yield c
