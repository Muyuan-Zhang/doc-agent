"""
Tests for FastAPI app lifespan: startup/shutdown success and failure paths.
"""
from unittest.mock import AsyncMock, patch

import pytest


def _make_client_mock(name: str) -> AsyncMock:
    m = AsyncMock()
    m.connect = AsyncMock()
    m.disconnect = AsyncMock()
    m.ping = AsyncMock(return_value=True)
    m.__repr__ = lambda self: name
    return m


class TestLifespanStartupSuccess:
    async def test_all_clients_connect_called_on_startup(self):
        from app import create_app

        postgres = _make_client_mock("postgres")
        redis = _make_client_mock("redis")
        milvus = _make_client_mock("milvus")
        mq = _make_client_mock("mq")
        llm = _make_client_mock("llm")

        with (
            patch("app.PostgreSQLClient", return_value=postgres),
            patch("app.RedisClient", return_value=redis),
            patch("app.MilvusClient", return_value=milvus),
            patch("app.RedisStreamsMQClient", return_value=mq),
            patch("app.OpenAILLMClient", return_value=llm),
        ):
            app = create_app()
            async with app.router.lifespan_context(app):
                postgres.connect.assert_awaited_once()
                redis.connect.assert_awaited_once()
                milvus.connect.assert_awaited_once()
                mq.connect.assert_awaited_once()
                llm.connect.assert_awaited_once()

    async def test_all_clients_attached_to_app_state_on_startup(self):
        from app import create_app

        postgres = _make_client_mock("postgres")
        redis = _make_client_mock("redis")
        milvus = _make_client_mock("milvus")
        mq = _make_client_mock("mq")
        llm = _make_client_mock("llm")

        with (
            patch("app.PostgreSQLClient", return_value=postgres),
            patch("app.RedisClient", return_value=redis),
            patch("app.MilvusClient", return_value=milvus),
            patch("app.RedisStreamsMQClient", return_value=mq),
            patch("app.OpenAILLMClient", return_value=llm),
        ):
            app = create_app()
            async with app.router.lifespan_context(app):
                assert app.state.postgres is postgres
                assert app.state.redis is redis
                assert app.state.milvus is milvus
                assert app.state.mq is mq
                assert app.state.llm is llm

    async def test_all_clients_disconnect_called_on_shutdown(self):
        from app import create_app

        postgres = _make_client_mock("postgres")
        redis = _make_client_mock("redis")
        milvus = _make_client_mock("milvus")
        mq = _make_client_mock("mq")
        llm = _make_client_mock("llm")

        with (
            patch("app.PostgreSQLClient", return_value=postgres),
            patch("app.RedisClient", return_value=redis),
            patch("app.MilvusClient", return_value=milvus),
            patch("app.RedisStreamsMQClient", return_value=mq),
            patch("app.OpenAILLMClient", return_value=llm),
        ):
            app = create_app()
            async with app.router.lifespan_context(app):
                pass

        postgres.disconnect.assert_awaited_once()
        redis.disconnect.assert_awaited_once()
        milvus.disconnect.assert_awaited_once()
        mq.disconnect.assert_awaited_once()
        llm.disconnect.assert_awaited_once()


class TestLifespanStartupFailure:
    async def test_startup_raises_when_first_client_fails(self):
        from app import create_app

        postgres = _make_client_mock("postgres")
        postgres.connect = AsyncMock(side_effect=RuntimeError("DB unreachable"))
        redis = _make_client_mock("redis")
        milvus = _make_client_mock("milvus")
        mq = _make_client_mock("mq")
        llm = _make_client_mock("llm")

        with (
            patch("app.PostgreSQLClient", return_value=postgres),
            patch("app.RedisClient", return_value=redis),
            patch("app.MilvusClient", return_value=milvus),
            patch("app.RedisStreamsMQClient", return_value=mq),
            patch("app.OpenAILLMClient", return_value=llm),
        ):
            app = create_app()
            with pytest.raises(RuntimeError, match="DB unreachable"):
                async with app.router.lifespan_context(app):
                    pass

    async def test_startup_raises_when_middle_client_fails(self):
        from app import create_app

        postgres = _make_client_mock("postgres")
        redis = _make_client_mock("redis")
        redis.connect = AsyncMock(side_effect=ConnectionError("Redis down"))
        milvus = _make_client_mock("milvus")
        mq = _make_client_mock("mq")
        llm = _make_client_mock("llm")

        with (
            patch("app.PostgreSQLClient", return_value=postgres),
            patch("app.RedisClient", return_value=redis),
            patch("app.MilvusClient", return_value=milvus),
            patch("app.RedisStreamsMQClient", return_value=mq),
            patch("app.OpenAILLMClient", return_value=llm),
        ):
            app = create_app()
            with pytest.raises(ConnectionError, match="Redis down"):
                async with app.router.lifespan_context(app):
                    pass

    async def test_shutdown_still_disconnects_after_partial_startup(self):
        """
        If milvus.connect() fails, already-connected clients (postgres, redis)
        must still be disconnected during cleanup.
        """
        from app import create_app

        postgres = _make_client_mock("postgres")
        redis = _make_client_mock("redis")
        milvus = _make_client_mock("milvus")
        milvus.connect = AsyncMock(side_effect=RuntimeError("Milvus offline"))
        mq = _make_client_mock("mq")
        llm = _make_client_mock("llm")

        with (
            patch("app.PostgreSQLClient", return_value=postgres),
            patch("app.RedisClient", return_value=redis),
            patch("app.MilvusClient", return_value=milvus),
            patch("app.RedisStreamsMQClient", return_value=mq),
            patch("app.OpenAILLMClient", return_value=llm),
        ):
            app = create_app()
            with pytest.raises(RuntimeError, match="Milvus offline"):
                async with app.router.lifespan_context(app):
                    pass

        postgres.disconnect.assert_awaited_once()
        redis.disconnect.assert_awaited_once()
