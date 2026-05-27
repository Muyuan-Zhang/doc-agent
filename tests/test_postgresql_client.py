"""Unit tests for PostgreSQLClient."""
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.clients.postgresql import PostgreSQLClient


class TestPostgreSQLClientProperty:
    def test_engine_raises_before_connect(self):
        with pytest.raises(RuntimeError, match="not connected"):
            _ = PostgreSQLClient().engine


class TestPostgreSQLClientConnect:
    async def test_connect_creates_engine(self):
        client = PostgreSQLClient()
        mock_engine = MagicMock()
        with patch("app.clients.postgresql.create_async_engine", return_value=mock_engine):
            await client.connect()
        assert client._engine is mock_engine

    async def test_connect_passes_pool_settings(self):
        client = PostgreSQLClient()
        with patch("app.clients.postgresql.create_async_engine", return_value=MagicMock()) as mock_create:
            await client.connect()
        kw = mock_create.call_args.kwargs
        assert kw["pool_size"] == 10
        assert kw["max_overflow"] == 20
        assert kw["pool_pre_ping"] is True

    async def test_connect_uses_postgres_url_from_settings(self):
        client = PostgreSQLClient()
        with patch("app.clients.postgresql.create_async_engine", return_value=MagicMock()) as mock_create:
            await client.connect()
        assert "postgresql" in mock_create.call_args.args[0]


class TestPostgreSQLClientDisconnect:
    async def test_disconnect_calls_dispose(self):
        client = PostgreSQLClient()
        mock_engine = AsyncMock()
        client._engine = mock_engine
        await client.disconnect()
        mock_engine.dispose.assert_awaited_once()

    async def test_disconnect_clears_engine_reference(self):
        client = PostgreSQLClient()
        client._engine = AsyncMock()
        await client.disconnect()
        assert client._engine is None

    async def test_disconnect_when_not_connected_does_not_raise(self):
        await PostgreSQLClient().disconnect()


class TestPostgreSQLClientPing:
    @staticmethod
    def _engine_with_conn(conn_mock: AsyncMock) -> MagicMock:
        engine = MagicMock()

        @asynccontextmanager
        async def fake_connect():
            yield conn_mock

        engine.connect = fake_connect
        return engine

    async def test_ping_returns_true_on_success(self):
        client = PostgreSQLClient()
        client._engine = self._engine_with_conn(AsyncMock())
        assert await client.ping() is True

    async def test_ping_executes_select_1(self):
        client = PostgreSQLClient()
        mock_conn = AsyncMock()
        client._engine = self._engine_with_conn(mock_conn)
        await client.ping()
        mock_conn.execute.assert_awaited_once()

    async def test_ping_returns_false_when_connect_raises(self):
        client = PostgreSQLClient()
        engine = MagicMock()
        engine.connect.side_effect = RuntimeError("connection refused")
        client._engine = engine
        assert await client.ping() is False

    async def test_ping_returns_false_when_execute_raises(self):
        client = PostgreSQLClient()
        mock_conn = AsyncMock()
        mock_conn.execute.side_effect = Exception("query failed")
        client._engine = self._engine_with_conn(mock_conn)
        assert await client.ping() is False
