"""Unit tests for RedisStreamsMQClient — P0."""
from unittest.mock import AsyncMock, patch

import pytest
import redis.asyncio as aioredis

from app.clients.mq import MQMessage, RedisStreamsMQClient


class TestMQClientProperty:
    def test_client_raises_before_connect(self):
        client = RedisStreamsMQClient()
        with pytest.raises(RuntimeError, match="not connected"):
            _ = client.client


class TestMQClientConnect:
    async def test_connect_stores_redis_client(self):
        client = RedisStreamsMQClient()
        mock_redis = AsyncMock()
        mock_redis.xgroup_create = AsyncMock(
            side_effect=aioredis.ResponseError("BUSYGROUP Consumer Group name already exists")
        )
        with patch("app.clients.mq.aioredis.from_url", return_value=mock_redis):
            await client.connect()
        assert client._client is mock_redis

    async def test_connect_calls_ensure_group(self):
        client = RedisStreamsMQClient()
        mock_redis = AsyncMock()
        mock_redis.xgroup_create = AsyncMock()
        with patch("app.clients.mq.aioredis.from_url", return_value=mock_redis):
            await client.connect()
        mock_redis.xgroup_create.assert_awaited_once()


class TestMQClientDisconnect:
    async def test_disconnect_calls_aclose(self):
        client = RedisStreamsMQClient()
        mock_redis = AsyncMock()
        client._client = mock_redis
        await client.disconnect()
        mock_redis.aclose.assert_awaited_once()

    async def test_disconnect_clears_client(self):
        client = RedisStreamsMQClient()
        client._client = AsyncMock()
        await client.disconnect()
        assert client._client is None

    async def test_disconnect_when_not_connected_does_not_raise(self):
        client = RedisStreamsMQClient()
        await client.disconnect()


class TestMQClientPing:
    async def test_ping_returns_true_on_success(self):
        client = RedisStreamsMQClient()
        mock_redis = AsyncMock()
        mock_redis.xlen = AsyncMock(return_value=0)
        client._client = mock_redis
        assert await client.ping() is True

    async def test_ping_calls_xlen(self):
        client = RedisStreamsMQClient()
        mock_redis = AsyncMock()
        mock_redis.xlen = AsyncMock(return_value=5)
        client._client = mock_redis
        await client.ping()
        mock_redis.xlen.assert_awaited_once()

    async def test_ping_returns_false_on_exception(self):
        client = RedisStreamsMQClient()
        mock_redis = AsyncMock()
        mock_redis.xlen = AsyncMock(side_effect=RuntimeError("redis down"))
        client._client = mock_redis
        assert await client.ping() is False


class TestEnsureGroup:
    async def test_ensure_group_creates_group_with_mkstream(self):
        client = RedisStreamsMQClient()
        mock_redis = AsyncMock()
        client._client = mock_redis
        await client.ensure_group()
        mock_redis.xgroup_create.assert_awaited_once_with(
            client._stream, client._group, id="0", mkstream=True
        )

    async def test_ensure_group_swallows_busygroup_error(self):
        client = RedisStreamsMQClient()
        mock_redis = AsyncMock()
        mock_redis.xgroup_create = AsyncMock(
            side_effect=aioredis.ResponseError("BUSYGROUP Consumer Group name already exists")
        )
        client._client = mock_redis
        await client.ensure_group()  # must not raise

    async def test_ensure_group_reraises_non_busygroup_error(self):
        client = RedisStreamsMQClient()
        mock_redis = AsyncMock()
        mock_redis.xgroup_create = AsyncMock(
            side_effect=aioredis.ResponseError("some other redis error")
        )
        client._client = mock_redis
        with pytest.raises(aioredis.ResponseError):
            await client.ensure_group()


class TestPublish:
    async def test_publish_returns_message_id(self):
        client = RedisStreamsMQClient()
        mock_redis = AsyncMock()
        mock_redis.xadd = AsyncMock(return_value="1234567890-0")
        client._client = mock_redis
        result = await client.publish({"key": "value"})
        assert result == "1234567890-0"

    async def test_publish_calls_xadd_with_stream_from_settings(self):
        client = RedisStreamsMQClient()
        mock_redis = AsyncMock()
        mock_redis.xadd = AsyncMock(return_value="123-0")
        client._client = mock_redis
        await client.publish({"action": "process"})
        mock_redis.xadd.assert_awaited_once_with(client._stream, {"action": "process"})


class TestConsume:
    async def test_consume_yields_mq_messages(self):
        client = RedisStreamsMQClient()
        mock_redis = AsyncMock()
        mock_redis.xreadgroup = AsyncMock(return_value=[
            (client._stream, [("123-0", {"k": "v"}), ("124-0", {"k": "v2"})])
        ])
        client._client = mock_redis

        results = [msg async for msg in client.consume()]

        assert len(results) == 2
        assert results[0].id == "123-0"
        assert results[0].data == {"k": "v"}
        assert results[0].stream == client._stream
        assert results[1].id == "124-0"

    async def test_consume_yields_nothing_on_empty_response(self):
        client = RedisStreamsMQClient()
        mock_redis = AsyncMock()
        mock_redis.xreadgroup = AsyncMock(return_value=None)
        client._client = mock_redis

        results = [msg async for msg in client.consume()]
        assert results == []

    async def test_consume_passes_correct_args_to_xreadgroup(self):
        client = RedisStreamsMQClient()
        mock_redis = AsyncMock()
        mock_redis.xreadgroup = AsyncMock(return_value=None)
        client._client = mock_redis

        async for _ in client.consume(count=5, block_ms=1000):
            pass

        call_kwargs = mock_redis.xreadgroup.call_args.kwargs
        assert call_kwargs["groupname"] == client._group
        assert call_kwargs["consumername"] == client._consumer
        assert call_kwargs["count"] == 5
        assert call_kwargs["block"] == 1000


class TestAck:
    async def test_ack_calls_xack(self):
        client = RedisStreamsMQClient()
        mock_redis = AsyncMock()
        client._client = mock_redis
        await client.ack("123-0")
        mock_redis.xack.assert_awaited_once_with(client._stream, client._group, "123-0")
