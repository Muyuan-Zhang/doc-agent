"""
Tests for FastAPI app lifespan: startup/shutdown success and failure paths.

Priority area 1:
- All clients connect successfully on startup → state attributes are set.
- One client raises during connect → startup propagates the exception.
- Shutdown calls disconnect on every client that was already connected,
  even when a later client's connect raised (partial startup cleanup).
- Shutdown is idempotent when called after a clean startup.
"""
import pytest
from unittest.mock import AsyncMock, patch, call
from contextlib import asynccontextmanager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client_mock(name: str) -> AsyncMock:
    """Return an AsyncMock that behaves like AbstractClient."""
    m = AsyncMock()
    m.connect = AsyncMock()
    m.disconnect = AsyncMock()
    m.ping = AsyncMock(return_value=True)
    m.__repr__ = lambda self: name  # noqa: E731
    return m


# ---------------------------------------------------------------------------
# Test: happy-path lifespan
# ---------------------------------------------------------------------------

class TestLifespanStartupSuccess:
    async def test_all_clients_connect_called_on_startup(self):
        """connect() is awaited on all four clients during lifespan startup."""
        from app import create_app  # will fail (RED) until app/__init__.py is written

        mysql = _make_client_mock("mysql")
        redis = _make_client_mock("redis")
        milvus = _make_client_mock("milvus")
        mq = _make_client_mock("mq")

        with (
            patch("app.MySQLClient", return_value=mysql),
            patch("app.RedisClient", return_value=redis),
            patch("app.MilvusClient", return_value=milvus),
            patch("app.RedisStreamsMQClient", return_value=mq),
        ):
            app = create_app()
            async with app.router.lifespan_context(app):
                mysql.connect.assert_awaited_once()
                redis.connect.assert_awaited_once()
                milvus.connect.assert_awaited_once()
                mq.connect.assert_awaited_once()

    async def test_all_clients_attached_to_app_state_on_startup(self):
        """After startup, app.state exposes .mysql, .redis, .milvus, .mq."""
        from app import create_app

        mysql = _make_client_mock("mysql")
        redis = _make_client_mock("redis")
        milvus = _make_client_mock("milvus")
        mq = _make_client_mock("mq")

        with (
            patch("app.MySQLClient", return_value=mysql),
            patch("app.RedisClient", return_value=redis),
            patch("app.MilvusClient", return_value=milvus),
            patch("app.RedisStreamsMQClient", return_value=mq),
        ):
            app = create_app()
            async with app.router.lifespan_context(app):
                assert app.state.mysql is mysql
                assert app.state.redis is redis
                assert app.state.milvus is milvus
                assert app.state.mq is mq

    async def test_all_clients_disconnect_called_on_shutdown(self):
        """disconnect() is awaited on all four clients when lifespan exits."""
        from app import create_app

        mysql = _make_client_mock("mysql")
        redis = _make_client_mock("redis")
        milvus = _make_client_mock("milvus")
        mq = _make_client_mock("mq")

        with (
            patch("app.MySQLClient", return_value=mysql),
            patch("app.RedisClient", return_value=redis),
            patch("app.MilvusClient", return_value=milvus),
            patch("app.RedisStreamsMQClient", return_value=mq),
        ):
            app = create_app()
            async with app.router.lifespan_context(app):
                pass  # trigger shutdown

        mysql.disconnect.assert_awaited_once()
        redis.disconnect.assert_awaited_once()
        milvus.disconnect.assert_awaited_once()
        mq.disconnect.assert_awaited_once()


# ---------------------------------------------------------------------------
# Test: startup failure paths
# ---------------------------------------------------------------------------

class TestLifespanStartupFailure:
    async def test_startup_raises_when_first_client_fails(self):
        """If the first client's connect() raises, lifespan propagates the error."""
        from app import create_app

        mysql = _make_client_mock("mysql")
        mysql.connect = AsyncMock(side_effect=RuntimeError("DB unreachable"))
        redis = _make_client_mock("redis")
        milvus = _make_client_mock("milvus")
        mq = _make_client_mock("mq")

        with (
            patch("app.MySQLClient", return_value=mysql),
            patch("app.RedisClient", return_value=redis),
            patch("app.MilvusClient", return_value=milvus),
            patch("app.RedisStreamsMQClient", return_value=mq),
        ):
            app = create_app()
            with pytest.raises(RuntimeError, match="DB unreachable"):
                async with app.router.lifespan_context(app):
                    pass

    async def test_startup_raises_when_middle_client_fails(self):
        """If redis connect() raises after mysql succeeded, error propagates."""
        from app import create_app

        mysql = _make_client_mock("mysql")
        redis = _make_client_mock("redis")
        redis.connect = AsyncMock(side_effect=ConnectionError("Redis down"))
        milvus = _make_client_mock("milvus")
        mq = _make_client_mock("mq")

        with (
            patch("app.MySQLClient", return_value=mysql),
            patch("app.RedisClient", return_value=redis),
            patch("app.MilvusClient", return_value=milvus),
            patch("app.RedisStreamsMQClient", return_value=mq),
        ):
            app = create_app()
            with pytest.raises(ConnectionError, match="Redis down"):
                async with app.router.lifespan_context(app):
                    pass

    async def test_shutdown_still_disconnects_after_partial_startup(self):
        """
        If milvus.connect() fails, the lifespan must still disconnect
        the clients that already connected (mysql, redis) during cleanup.

        This test is intentionally lenient about *which* clients get
        disconnected — it only asserts that mysql and redis (the ones
        that succeeded) have disconnect called, and that the error is
        re-raised.  The exact cleanup strategy (disconnect all / only
        connected) is enforced by the implementation; this test pins the
        minimum safety guarantee.
        """
        from app import create_app

        mysql = _make_client_mock("mysql")
        redis = _make_client_mock("redis")
        milvus = _make_client_mock("milvus")
        milvus.connect = AsyncMock(side_effect=RuntimeError("Milvus offline"))
        mq = _make_client_mock("mq")

        with (
            patch("app.MySQLClient", return_value=mysql),
            patch("app.RedisClient", return_value=redis),
            patch("app.MilvusClient", return_value=milvus),
            patch("app.RedisStreamsMQClient", return_value=mq),
        ):
            app = create_app()
            with pytest.raises(RuntimeError, match="Milvus offline"):
                async with app.router.lifespan_context(app):
                    pass

        # Clients that connected before the failure must be cleaned up.
        mysql.disconnect.assert_awaited_once()
        redis.disconnect.assert_awaited_once()
