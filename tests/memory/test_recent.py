"""Unit tests for RecentMemoryStore."""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.memory.recent import RecentMemoryStore
from app.memory.schemas import ConversationTurn


def _make_redis(llen_return: int = 1, lrange_return: list[str] | None = None) -> MagicMock:
    redis = MagicMock()
    redis.cache_key = MagicMock(return_value="v1:memory:recent:sess-1")
    client = MagicMock()
    client.rpush = AsyncMock(return_value=1)
    client.ltrim = AsyncMock()
    client.llen = AsyncMock(return_value=llen_return)
    client.lrange = AsyncMock(return_value=lrange_return or [])
    client.expire = AsyncMock()
    client.delete = AsyncMock()
    redis.client = client
    return redis


def _make_turn(role: str = "user", content: str = "hello") -> ConversationTurn:
    return ConversationTurn(session_id="sess-1", role=role, content=content, ts=1000.0)


class TestAppendTurn:
    async def test_returns_count_from_llen(self):
        redis = _make_redis(llen_return=3)
        store = RecentMemoryStore()
        count = await store.append_turn(redis, "sess-1", _make_turn())
        assert count == 3

    async def test_rpush_called_with_serialized_turn(self):
        redis = _make_redis()
        turn = _make_turn(content="test message")
        store = RecentMemoryStore()
        await store.append_turn(redis, "sess-1", turn)
        redis.client.rpush.assert_awaited_once()
        pushed_key, pushed_val = redis.client.rpush.call_args[0]
        assert pushed_key == "v1:memory:recent:sess-1"
        data = json.loads(pushed_val)
        assert data["content"] == "test message"

    async def test_ltrim_trims_to_max_turns(self):
        redis = _make_redis()
        store = RecentMemoryStore()
        await store.append_turn(redis, "sess-1", _make_turn())
        redis.client.ltrim.assert_awaited_once()
        key, start, stop = redis.client.ltrim.call_args[0]
        assert start < 0  # keeps last N items

    async def test_expire_refreshes_ttl(self):
        redis = _make_redis()
        store = RecentMemoryStore()
        await store.append_turn(redis, "sess-1", _make_turn())
        redis.client.expire.assert_awaited_once()

    async def test_cache_key_uses_session_id(self):
        redis = _make_redis()
        store = RecentMemoryStore()
        await store.append_turn(redis, "sess-abc", _make_turn())
        redis.cache_key.assert_called_once_with("memory:recent", "sess-abc")


class TestGetTurns:
    async def test_empty_session_returns_empty_list(self):
        redis = _make_redis(lrange_return=[])
        store = RecentMemoryStore()
        result = await store.get_turns(redis, "sess-1")
        assert result == []

    async def test_deserializes_turns_in_order(self):
        turn = _make_turn(content="hello")
        raw = [turn.model_dump_json()]
        redis = _make_redis(lrange_return=raw)
        store = RecentMemoryStore()
        result = await store.get_turns(redis, "sess-1")
        assert len(result) == 1
        assert result[0].content == "hello"
        assert result[0].role == "user"

    async def test_multiple_turns_preserves_order(self):
        turns = [
            ConversationTurn(session_id="s", role="user", content=f"msg{i}", ts=float(i))
            for i in range(3)
        ]
        raw = [t.model_dump_json() for t in turns]
        redis = _make_redis(lrange_return=raw)
        store = RecentMemoryStore()
        result = await store.get_turns(redis, "s")
        assert [r.content for r in result] == ["msg0", "msg1", "msg2"]

    async def test_lrange_full_list(self):
        redis = _make_redis()
        store = RecentMemoryStore()
        await store.get_turns(redis, "sess-1")
        redis.client.lrange.assert_awaited_once_with("v1:memory:recent:sess-1", 0, -1)


class TestCount:
    async def test_returns_llen_value(self):
        redis = _make_redis(llen_return=7)
        store = RecentMemoryStore()
        result = await store.count(redis, "sess-1")
        assert result == 7

    async def test_zero_for_missing_key(self):
        redis = _make_redis(llen_return=0)
        store = RecentMemoryStore()
        result = await store.count(redis, "sess-1")
        assert result == 0


class TestClear:
    async def test_deletes_key(self):
        redis = _make_redis()
        store = RecentMemoryStore()
        await store.clear(redis, "sess-1")
        redis.client.delete.assert_awaited_once_with("v1:memory:recent:sess-1")
