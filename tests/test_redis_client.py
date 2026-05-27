"""
Tests for RedisClient distributed lock and increment_with_ttl.

Priority area 3:
- acquire_lock: returns (True, token) when SET NX succeeds.
- acquire_lock: returns (False, token) when SET NX fails (lock already held).
- acquire_lock: uses the caller-supplied token when one is provided.
- acquire_lock: generates a UUID token when none is supplied.
- release_lock: returns True when the token matches (Lua script returns 1).
- release_lock: returns False when the token does NOT match (Lua returns 0).
- release_lock (double release): second call returns False.
- increment_with_ttl: returns the value the Lua script returns.
- ping: returns True when underlying Redis responds.
- ping: returns False when underlying Redis raises an exception.
- client property: raises RuntimeError when not connected.
- cache_key: formats correctly.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.clients.redis import RedisClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _connected_client() -> tuple[RedisClient, MagicMock]:
    """Return (RedisClient, mock_aioredis_instance) with _client already set."""
    client = RedisClient()
    mock_redis = MagicMock()
    # Make async methods return coroutines via AsyncMock attributes
    mock_redis.ping = AsyncMock(return_value=True)
    mock_redis.set = AsyncMock(return_value=True)
    mock_redis.eval = AsyncMock(return_value=1)
    mock_redis.aclose = AsyncMock()
    client._client = mock_redis
    return client, mock_redis


# ---------------------------------------------------------------------------
# Test: client property guard
# ---------------------------------------------------------------------------

class TestClientProperty:
    async def test_raises_runtime_error_when_not_connected(self):
        client = RedisClient()
        with pytest.raises(RuntimeError, match="RedisClient not connected"):
            _ = client.client


# ---------------------------------------------------------------------------
# Test: cache_key
# ---------------------------------------------------------------------------

class TestCacheKey:
    async def test_cache_key_format(self):
        client = RedisClient()
        key = client.cache_key("docs", "d123", "lock")
        # Format: {kb_version}:{namespace}:{parts joined by :}
        from app.core.config import settings
        assert key == f"{settings.knowledge_base_version}:docs:d123:lock"

    async def test_cache_key_single_part(self):
        client = RedisClient()
        key = client.cache_key("sessions", "abc")
        from app.core.config import settings
        assert key == f"{settings.knowledge_base_version}:sessions:abc"


# ---------------------------------------------------------------------------
# Test: ping
# ---------------------------------------------------------------------------

class TestPing:
    async def test_ping_returns_true_when_redis_responds(self):
        client, mock_redis = _connected_client()
        mock_redis.ping = AsyncMock(return_value=b"PONG")  # aioredis returns bytes/True
        result = await client.ping()
        assert result is True

    async def test_ping_returns_false_when_redis_raises(self):
        client, mock_redis = _connected_client()
        mock_redis.ping = AsyncMock(side_effect=ConnectionError("refused"))
        result = await client.ping()
        assert result is False

    async def test_ping_calls_underlying_redis_ping(self):
        client, mock_redis = _connected_client()
        await client.ping()
        mock_redis.ping.assert_awaited_once()


# ---------------------------------------------------------------------------
# Test: acquire_lock
# ---------------------------------------------------------------------------

class TestAcquireLock:
    async def test_acquire_returns_true_and_token_when_set_nx_succeeds(self):
        client, mock_redis = _connected_client()
        mock_redis.set = AsyncMock(return_value=True)  # SET NX succeeded

        acquired, token = await client.acquire_lock("{doc:d1}:lock", ttl_seconds=30)

        assert acquired is True
        assert isinstance(token, str)
        assert len(token) > 0

    async def test_acquire_returns_false_and_token_when_set_nx_fails(self):
        client, mock_redis = _connected_client()
        mock_redis.set = AsyncMock(return_value=None)  # SET NX failed — key exists

        acquired, token = await client.acquire_lock("{doc:d1}:lock", ttl_seconds=30)

        assert acquired is False
        assert isinstance(token, str)  # token is still returned even on failure

    async def test_acquire_uses_provided_token(self):
        client, mock_redis = _connected_client()
        mock_redis.set = AsyncMock(return_value=True)
        fixed_token = "my-fixed-token-123"

        acquired, returned_token = await client.acquire_lock(
            "{doc:d1}:lock", ttl_seconds=30, token=fixed_token
        )

        assert returned_token == fixed_token
        mock_redis.set.assert_awaited_once_with(
            "{doc:d1}:lock", fixed_token, nx=True, ex=30
        )

    async def test_acquire_generates_uuid_token_when_none_provided(self):
        import uuid
        client, mock_redis = _connected_client()
        mock_redis.set = AsyncMock(return_value=True)

        _, token = await client.acquire_lock("{doc:d1}:lock", ttl_seconds=10)

        # Must be a valid UUID4 string
        parsed = uuid.UUID(token, version=4)
        assert str(parsed) == token

    async def test_acquire_passes_nx_and_ex_to_redis_set(self):
        client, mock_redis = _connected_client()
        mock_redis.set = AsyncMock(return_value=True)

        _, token = await client.acquire_lock("{doc:d1}:lock", ttl_seconds=60)

        mock_redis.set.assert_awaited_once_with(
            "{doc:d1}:lock", token, nx=True, ex=60
        )


# ---------------------------------------------------------------------------
# Test: release_lock
# ---------------------------------------------------------------------------

class TestReleaseLock:
    async def test_release_returns_true_when_token_matches(self):
        """Lua script returns 1 (DEL succeeded) → release_lock returns True."""
        client, mock_redis = _connected_client()
        mock_redis.eval = AsyncMock(return_value=1)

        result = await client.release_lock("{doc:d1}:lock", "correct-token")

        assert result is True

    async def test_release_returns_false_when_token_does_not_match(self):
        """Lua script returns 0 (GET != token) → release_lock returns False."""
        client, mock_redis = _connected_client()
        mock_redis.eval = AsyncMock(return_value=0)

        result = await client.release_lock("{doc:d1}:lock", "wrong-token")

        assert result is False

    async def test_release_calls_eval_with_correct_script_args(self):
        """eval must be called with exactly 1 key and the token as ARGV[1]."""
        from app.clients.redis import _RELEASE_LOCK_SCRIPT

        client, mock_redis = _connected_client()
        mock_redis.eval = AsyncMock(return_value=1)

        await client.release_lock("{doc:d1}:lock", "my-token")

        mock_redis.eval.assert_awaited_once_with(
            _RELEASE_LOCK_SCRIPT, 1, "{doc:d1}:lock", "my-token"
        )

    async def test_double_release_second_call_returns_false(self):
        """
        After a successful release (eval returns 1), a second release with the
        same token returns False because the key no longer exists (eval → 0).
        """
        client, mock_redis = _connected_client()
        mock_redis.eval = AsyncMock(side_effect=[1, 0])

        first = await client.release_lock("{doc:d1}:lock", "token-abc")
        second = await client.release_lock("{doc:d1}:lock", "token-abc")

        assert first is True
        assert second is False


# ---------------------------------------------------------------------------
# Test: increment_with_ttl
# ---------------------------------------------------------------------------

class TestIncrementWithTtl:
    async def test_returns_integer_value_from_lua_script(self):
        client, mock_redis = _connected_client()
        mock_redis.eval = AsyncMock(return_value=3)

        result = await client.increment_with_ttl("{user:u1}:rate_limit", ttl_seconds=60)

        assert result == 3

    async def test_passes_correct_args_to_eval(self):
        from app.clients.redis import _INCR_WITH_TTL_SCRIPT

        client, mock_redis = _connected_client()
        mock_redis.eval = AsyncMock(return_value=1)

        await client.increment_with_ttl("{user:u1}:rate_limit", ttl_seconds=30, amount=2)

        mock_redis.eval.assert_awaited_once_with(
            _INCR_WITH_TTL_SCRIPT, 1, "{user:u1}:rate_limit", 2, 30
        )

    async def test_default_amount_is_1(self):
        from app.clients.redis import _INCR_WITH_TTL_SCRIPT

        client, mock_redis = _connected_client()
        mock_redis.eval = AsyncMock(return_value=1)

        await client.increment_with_ttl("{user:u1}:counter", ttl_seconds=10)

        mock_redis.eval.assert_awaited_once_with(
            _INCR_WITH_TTL_SCRIPT, 1, "{user:u1}:counter", 1, 10
        )

    async def test_returns_int_even_when_redis_returns_bytes_like(self):
        """Redis can return integers as various Python types; result must be int."""
        client, mock_redis = _connected_client()
        mock_redis.eval = AsyncMock(return_value=5)  # aioredis returns int

        result = await client.increment_with_ttl("{user:u1}:x", ttl_seconds=60)

        assert isinstance(result, int)
        assert result == 5


# ---------------------------------------------------------------------------
# Test: connect / disconnect lifecycle
# ---------------------------------------------------------------------------

class TestConnectDisconnect:
    async def test_connect_sets_internal_client(self):
        client = RedisClient()
        mock_redis_instance = MagicMock()
        mock_redis_instance.aclose = AsyncMock()

        with patch("app.clients.redis.aioredis.from_url", return_value=mock_redis_instance):
            await client.connect()

        assert client._client is mock_redis_instance

    async def test_disconnect_closes_and_clears_client(self):
        client, mock_redis = _connected_client()

        await client.disconnect()

        mock_redis.aclose.assert_awaited_once()
        assert client._client is None

    async def test_disconnect_is_safe_when_not_connected(self):
        """Calling disconnect on an unconnected client must not raise."""
        client = RedisClient()
        await client.disconnect()  # should not raise
