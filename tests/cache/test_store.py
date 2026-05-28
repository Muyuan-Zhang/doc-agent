"""
Tests for app/cache/store.py — RagCacheStore.

Covers:
- get(): miss (None), hit (deserialised entry), corrupt JSON → None
- set(): calls setex with correct TTL and key
- update_status(): updates status field, returns False on miss
- delete(): returns True/False based on Redis DEL result
- invalidate_all(): scans and deletes matching keys, returns count
- get_stats(): returns hits/misses/pending as integers
"""
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from app.cache.schemas import CacheEntry, CacheStatus
from app.cache.store import RagCacheStore
from app.clients.redis import RedisClient
from app.models.chunk import ChunkSchema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_redis() -> tuple[RedisClient, MagicMock]:
    client = RedisClient()
    inner = MagicMock()
    inner.get = AsyncMock(return_value=None)
    inner.setex = AsyncMock(return_value=True)
    inner.set = AsyncMock(return_value=True)   # acquire_lock
    inner.eval = AsyncMock(return_value=1)     # release_lock Lua script
    inner.delete = AsyncMock(return_value=1)
    inner.ttl = AsyncMock(return_value=3600)
    inner.scan = AsyncMock(return_value=(0, []))
    inner.llen = AsyncMock(return_value=0)
    inner.incr = AsyncMock(return_value=1)
    client._client = inner
    return client, inner


def _make_chunk() -> ChunkSchema:
    return ChunkSchema(
        doc_id="doc-1",
        section_id="sec-1",
        chunk_index=0,
        content_hash="abc",
        version="v1",
        content="test",
    )


def _make_entry(**overrides) -> CacheEntry:
    defaults = dict(
        query_hash="deadbeefcafe0000",
        original_query="test query",
        normalized_query="test query",
        chunks=[_make_chunk()],
        created_at=datetime.now(tz=timezone.utc),
    )
    return CacheEntry(**(defaults | overrides))


# ---------------------------------------------------------------------------
# get()
# ---------------------------------------------------------------------------

class TestRagCacheStoreGet:
    async def test_returns_none_on_cache_miss(self):
        redis, inner = _make_redis()
        inner.get = AsyncMock(return_value=None)
        store = RagCacheStore(redis)
        result = await store.get("deadbeefcafe0000")
        assert result is None

    async def test_returns_entry_on_cache_hit(self):
        redis, inner = _make_redis()
        entry = _make_entry()
        inner.get = AsyncMock(return_value=entry.model_dump_json())
        store = RagCacheStore(redis)
        result = await store.get("deadbeefcafe0000")
        assert result is not None
        assert result.query_hash == entry.query_hash
        assert result.status == entry.status

    async def test_returns_none_on_corrupt_json(self):
        redis, inner = _make_redis()
        inner.get = AsyncMock(return_value="{not valid json")
        store = RagCacheStore(redis)
        result = await store.get("deadbeefcafe0000")
        assert result is None

    async def test_uses_rag_cache_namespace_in_key(self):
        redis, inner = _make_redis()
        store = RagCacheStore(redis)
        await store.get("abc123")
        key = inner.get.call_args.args[0]
        assert "rag_cache" in key
        assert "abc123" in key

    async def test_round_trips_chunk_content(self):
        redis, inner = _make_redis()
        chunk = ChunkSchema(
            doc_id="d1", section_id="s1", chunk_index=0,
            content_hash="h1", version="v1", content="hello round-trip",
        )
        entry = _make_entry(chunks=[chunk])
        inner.get = AsyncMock(return_value=entry.model_dump_json())
        store = RagCacheStore(redis)
        result = await store.get("deadbeefcafe0000")
        assert result.chunks[0].content == "hello round-trip"


# ---------------------------------------------------------------------------
# set()
# ---------------------------------------------------------------------------

class TestRagCacheStoreSet:
    async def test_calls_setex_with_correct_ttl(self):
        redis, inner = _make_redis()
        store = RagCacheStore(redis)
        entry = _make_entry()
        await store.set(entry, ttl=600)
        inner.setex.assert_awaited_once()
        assert inner.setex.call_args.args[1] == 600

    async def test_serialises_entry_as_json_string(self):
        redis, inner = _make_redis()
        store = RagCacheStore(redis)
        entry = _make_entry()
        await store.set(entry, ttl=300)
        raw = inner.setex.call_args.args[2]
        restored = CacheEntry.model_validate_json(raw)
        assert restored.query_hash == entry.query_hash

    async def test_key_contains_rag_cache_and_hash(self):
        redis, inner = _make_redis()
        store = RagCacheStore(redis)
        entry = _make_entry(query_hash="0123456789abcdef")
        await store.set(entry, ttl=100)
        key = inner.setex.call_args.args[0]
        assert "rag_cache" in key
        assert "0123456789abcdef" in key


# ---------------------------------------------------------------------------
# update_status()
# ---------------------------------------------------------------------------

class TestRagCacheStoreUpdateStatus:
    async def test_returns_true_and_updates_status(self):
        redis, inner = _make_redis()
        entry = _make_entry(status=CacheStatus.PENDING_REVIEW)
        inner.get = AsyncMock(return_value=entry.model_dump_json())
        inner.ttl = AsyncMock(return_value=1800)
        store = RagCacheStore(redis)
        result = await store.update_status("deadbeefcafe0000", CacheStatus.APPROVED)
        assert result is True
        raw = inner.setex.call_args.args[2]
        updated = CacheEntry.model_validate_json(raw)
        assert updated.status == CacheStatus.APPROVED

    async def test_returns_false_when_entry_not_found(self):
        redis, inner = _make_redis()
        inner.get = AsyncMock(return_value=None)
        store = RagCacheStore(redis)
        result = await store.update_status("nonexistent", CacheStatus.APPROVED)
        assert result is False
        inner.setex.assert_not_awaited()

    async def test_updates_approval_count_when_provided(self):
        redis, inner = _make_redis()
        entry = _make_entry(approval_count=0)
        inner.get = AsyncMock(return_value=entry.model_dump_json())
        store = RagCacheStore(redis)
        await store.update_status("deadbeefcafe0000", CacheStatus.APPROVED, approval_count=3)
        raw = inner.setex.call_args.args[2]
        updated = CacheEntry.model_validate_json(raw)
        assert updated.approval_count == 3

    async def test_updates_approved_by_when_provided(self):
        redis, inner = _make_redis()
        entry = _make_entry()
        inner.get = AsyncMock(return_value=entry.model_dump_json())
        store = RagCacheStore(redis)
        await store.update_status(
            "deadbeefcafe0000",
            CacheStatus.APPROVED,
            approved_by=["alice", "bob"],
        )
        raw = inner.setex.call_args.args[2]
        updated = CacheEntry.model_validate_json(raw)
        assert updated.approved_by == ["alice", "bob"]

    async def test_uses_fallback_ttl_when_ttl_is_negative(self):
        redis, inner = _make_redis()
        entry = _make_entry()
        inner.get = AsyncMock(return_value=entry.model_dump_json())
        inner.ttl = AsyncMock(return_value=-1)
        store = RagCacheStore(redis)
        await store.update_status("deadbeefcafe0000", CacheStatus.REJECTED)
        ttl_used = inner.setex.call_args.args[1]
        assert ttl_used == 3600

    async def test_returns_false_when_lock_not_acquired(self):
        redis, inner = _make_redis()
        entry = _make_entry(status=CacheStatus.PENDING_REVIEW)
        inner.get = AsyncMock(return_value=entry.model_dump_json())
        inner.set = AsyncMock(return_value=None)  # lock not acquired
        store = RagCacheStore(redis)
        result = await store.update_status("deadbeefcafe0000", CacheStatus.APPROVED)
        assert result is False
        inner.setex.assert_not_awaited()

    async def test_returns_false_when_entry_expired_between_get_and_ttl(self):
        redis, inner = _make_redis()
        entry = _make_entry()
        inner.get = AsyncMock(return_value=entry.model_dump_json())
        inner.ttl = AsyncMock(return_value=-2)  # key gone by TTL check
        store = RagCacheStore(redis)
        result = await store.update_status("deadbeefcafe0000", CacheStatus.APPROVED)
        assert result is False
        inner.setex.assert_not_awaited()


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------

class TestRagCacheStoreDelete:
    async def test_returns_true_when_key_existed(self):
        redis, inner = _make_redis()
        inner.delete = AsyncMock(return_value=1)
        store = RagCacheStore(redis)
        result = await store.delete("abc123")
        assert result is True

    async def test_returns_false_when_key_not_found(self):
        redis, inner = _make_redis()
        inner.delete = AsyncMock(return_value=0)
        store = RagCacheStore(redis)
        result = await store.delete("abc123")
        assert result is False

    async def test_deletes_correct_key(self):
        redis, inner = _make_redis()
        store = RagCacheStore(redis)
        await store.delete("myhash")
        key = inner.delete.call_args.args[0]
        assert "rag_cache" in key
        assert "myhash" in key


# ---------------------------------------------------------------------------
# invalidate_all()
# ---------------------------------------------------------------------------

class TestRagCacheStoreInvalidateAll:
    async def test_returns_count_of_deleted_keys(self):
        redis, inner = _make_redis()
        inner.scan = AsyncMock(return_value=(0, ["key1", "key2", "key3"]))
        inner.delete = AsyncMock(return_value=3)
        store = RagCacheStore(redis)
        count = await store.invalidate_all()
        assert count == 3

    async def test_returns_zero_when_no_keys_match(self):
        redis, inner = _make_redis()
        inner.scan = AsyncMock(return_value=(0, []))
        store = RagCacheStore(redis)
        count = await store.invalidate_all()
        assert count == 0
        inner.delete.assert_not_awaited()

    async def test_handles_pagination_via_cursor(self):
        redis, inner = _make_redis()
        inner.scan = AsyncMock(side_effect=[
            (1, ["key1", "key2"]),
            (0, ["key3"]),
        ])
        inner.delete = AsyncMock(return_value=1)
        store = RagCacheStore(redis)
        count = await store.invalidate_all()
        assert count == 3
        assert inner.scan.await_count == 2


# ---------------------------------------------------------------------------
# get_stats()
# ---------------------------------------------------------------------------

class TestRagCacheStoreStats:
    async def test_returns_hits_misses_and_pending(self):
        redis, inner = _make_redis()
        inner.get = AsyncMock(side_effect=["42", "10"])
        inner.llen = AsyncMock(return_value=5)
        store = RagCacheStore(redis)
        stats = await store.get_stats()
        assert stats["hits"] == 42
        assert stats["misses"] == 10
        assert stats["pending"] == 5

    async def test_returns_zeros_when_no_stats_keys(self):
        redis, inner = _make_redis()
        inner.get = AsyncMock(return_value=None)
        inner.llen = AsyncMock(return_value=0)
        store = RagCacheStore(redis)
        stats = await store.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["pending"] == 0

    async def test_returns_integer_types(self):
        redis, inner = _make_redis()
        inner.get = AsyncMock(return_value="7")
        inner.llen = AsyncMock(return_value=3)
        store = RagCacheStore(redis)
        stats = await store.get_stats()
        assert isinstance(stats["hits"], int)
        assert isinstance(stats["misses"], int)
        assert isinstance(stats["pending"], int)
