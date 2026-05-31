"""
Tests for app/cache/store.py — RagCacheStore.

Covers:
- get() / get_many(): miss, hit, corrupt JSON, pipeline batch
- set(): setex TTL and key
- update_status(): state machine validation, TTL fallback, lock failure
- delete() / invalidate_all()
- increment_stat() / get_stats(): Redis Hash + pipeline
"""
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from app.cache.schemas import CacheEntry, CacheStatus
from app.cache.store import RagCacheStore
from app.clients.redis import RedisClient
from app.core.exceptions import ValidationError
from app.models.chunk import ChunkSchema


def _make_pipeline(*return_values):
    pipe = MagicMock()
    pipe.get = MagicMock(return_value=pipe)
    pipe.hgetall = MagicMock(return_value=pipe)
    pipe.hincrby = MagicMock(return_value=pipe)
    pipe.zcard = MagicMock(return_value=pipe)
    pipe.execute = AsyncMock(return_value=list(return_values))
    return pipe


def _make_redis(pipeline_returns=None) -> tuple[RedisClient, MagicMock]:
    client = RedisClient()
    inner = MagicMock()
    inner.get = AsyncMock(return_value=None)
    inner.setex = AsyncMock(return_value=True)
    inner.set = AsyncMock(return_value=True)    # acquire_lock
    inner.eval = AsyncMock(return_value=1)      # release_lock Lua
    inner.delete = AsyncMock(return_value=1)
    inner.ttl = AsyncMock(return_value=3600)
    inner.scan = AsyncMock(return_value=(0, []))
    inner.hincrby = AsyncMock(return_value=1)
    pipe = _make_pipeline(*(pipeline_returns or []))
    inner.pipeline = MagicMock(return_value=pipe)
    client._client = inner
    return client, inner


def _make_chunk() -> ChunkSchema:
    return ChunkSchema(
        doc_id="doc-1", section_id="sec-1", chunk_index=0,
        content_hash="abc", version="v1", content="test",
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
        assert await RagCacheStore(redis).get("deadbeefcafe0000") is None

    async def test_returns_entry_on_cache_hit(self):
        redis, inner = _make_redis()
        entry = _make_entry()
        inner.get = AsyncMock(return_value=entry.model_dump_json())
        result = await RagCacheStore(redis).get("deadbeefcafe0000")
        assert result is not None
        assert result.query_hash == entry.query_hash

    async def test_returns_none_on_corrupt_json(self):
        redis, inner = _make_redis()
        inner.get = AsyncMock(return_value="{not valid json")
        assert await RagCacheStore(redis).get("deadbeefcafe0000") is None

    async def test_uses_rag_cache_namespace_in_key(self):
        redis, inner = _make_redis()
        await RagCacheStore(redis).get("abc123")
        key = inner.get.call_args.args[0]
        assert "rag_cache" in key and "abc123" in key

    async def test_round_trips_chunk_content(self):
        redis, inner = _make_redis()
        chunk = ChunkSchema(
            doc_id="d1", section_id="s1", chunk_index=0,
            content_hash="h1", version="v1", content="hello round-trip",
        )
        inner.get = AsyncMock(return_value=_make_entry(chunks=[chunk]).model_dump_json())
        result = await RagCacheStore(redis).get("deadbeefcafe0000")
        assert result.chunks[0].content == "hello round-trip"


# ---------------------------------------------------------------------------
# get_many() — pipeline batch
# ---------------------------------------------------------------------------

class TestRagCacheStoreGetMany:
    async def test_returns_empty_list_for_empty_input(self):
        redis, _ = _make_redis()
        assert await RagCacheStore(redis).get_many([]) == []

    async def test_returns_entries_in_order(self):
        entry1 = _make_entry(query_hash="aaaaaaaaaaaaaaaa")
        entry2 = _make_entry(query_hash="bbbbbbbbbbbbbbbb")
        pipe = _make_pipeline(entry1.model_dump_json(), entry2.model_dump_json())
        redis, inner = _make_redis()
        inner.pipeline = MagicMock(return_value=pipe)
        result = await RagCacheStore(redis).get_many(["aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb"])
        assert len(result) == 2
        assert result[0].query_hash == "aaaaaaaaaaaaaaaa"
        assert result[1].query_hash == "bbbbbbbbbbbbbbbb"

    async def test_returns_none_for_missing_entries(self):
        pipe = _make_pipeline(None, None)
        redis, inner = _make_redis()
        inner.pipeline = MagicMock(return_value=pipe)
        result = await RagCacheStore(redis).get_many(["h1", "h2"])
        assert result == [None, None]

    async def test_uses_single_pipeline_round_trip(self):
        pipe = _make_pipeline("{}")
        redis, inner = _make_redis()
        inner.pipeline = MagicMock(return_value=pipe)
        await RagCacheStore(redis).get_many(["deadbeefcafe0000"])
        inner.pipeline.assert_called_once()
        pipe.execute.assert_awaited_once()

    async def test_handles_corrupt_entry_gracefully(self):
        pipe = _make_pipeline("{bad json}")
        redis, inner = _make_redis()
        inner.pipeline = MagicMock(return_value=pipe)
        result = await RagCacheStore(redis).get_many(["h1"])
        assert result == [None]


# ---------------------------------------------------------------------------
# set()
# ---------------------------------------------------------------------------

class TestRagCacheStoreSet:
    async def test_calls_setex_with_correct_ttl(self):
        redis, inner = _make_redis()
        await RagCacheStore(redis).set(_make_entry(), ttl=600)
        assert inner.setex.call_args.args[1] == 600

    async def test_serialises_entry_as_json(self):
        redis, inner = _make_redis()
        entry = _make_entry()
        await RagCacheStore(redis).set(entry, ttl=300)
        restored = CacheEntry.model_validate_json(inner.setex.call_args.args[2])
        assert restored.query_hash == entry.query_hash

    async def test_key_contains_rag_cache_and_hash(self):
        redis, inner = _make_redis()
        await RagCacheStore(redis).set(_make_entry(query_hash="0123456789abcdef"), ttl=100)
        key = inner.setex.call_args.args[0]
        assert "rag_cache" in key and "0123456789abcdef" in key


# ---------------------------------------------------------------------------
# update_status() — includes state machine validation
# ---------------------------------------------------------------------------

class TestRagCacheStoreUpdateStatus:
    async def test_returns_true_and_updates_status(self):
        redis, inner = _make_redis()
        inner.get = AsyncMock(return_value=_make_entry().model_dump_json())
        result = await RagCacheStore(redis).update_status("deadbeefcafe0000", CacheStatus.APPROVED)
        assert result is True
        updated = CacheEntry.model_validate_json(inner.setex.call_args.args[2])
        assert updated.status == CacheStatus.APPROVED

    async def test_returns_false_when_entry_not_found(self):
        redis, inner = _make_redis()
        result = await RagCacheStore(redis).update_status("nonexistent", CacheStatus.APPROVED)
        assert result is False
        inner.setex.assert_not_awaited()

    async def test_raises_for_invalid_transition_approved_to_rejected(self):
        redis, inner = _make_redis()
        inner.get = AsyncMock(
            return_value=_make_entry(status=CacheStatus.APPROVED).model_dump_json()
        )
        with pytest.raises(ValidationError):
            await RagCacheStore(redis).update_status("deadbeefcafe0000", CacheStatus.REJECTED)

    async def test_updates_approval_count_when_provided(self):
        redis, inner = _make_redis()
        inner.get = AsyncMock(return_value=_make_entry(approval_count=0).model_dump_json())
        await RagCacheStore(redis).update_status(
            "deadbeefcafe0000", CacheStatus.APPROVED, approval_count=3
        )
        updated = CacheEntry.model_validate_json(inner.setex.call_args.args[2])
        assert updated.approval_count == 3

    async def test_updates_approved_by_when_provided(self):
        redis, inner = _make_redis()
        inner.get = AsyncMock(return_value=_make_entry().model_dump_json())
        await RagCacheStore(redis).update_status(
            "deadbeefcafe0000", CacheStatus.APPROVED, approved_by=["alice", "bob"]
        )
        updated = CacheEntry.model_validate_json(inner.setex.call_args.args[2])
        assert updated.approved_by == ["alice", "bob"]

    async def test_uses_fallback_ttl_when_ttl_is_negative(self):
        redis, inner = _make_redis()
        inner.get = AsyncMock(return_value=_make_entry().model_dump_json())
        inner.ttl = AsyncMock(return_value=-1)
        await RagCacheStore(redis).update_status("deadbeefcafe0000", CacheStatus.REJECTED)
        assert inner.setex.call_args.args[1] == 3600

    async def test_returns_false_when_lock_not_acquired(self):
        redis, inner = _make_redis()
        inner.set = AsyncMock(return_value=None)  # lock fails
        result = await RagCacheStore(redis).update_status("deadbeefcafe0000", CacheStatus.APPROVED)
        assert result is False
        inner.setex.assert_not_awaited()

    async def test_returns_false_when_entry_expired_between_get_and_ttl(self):
        redis, inner = _make_redis()
        inner.get = AsyncMock(return_value=_make_entry().model_dump_json())
        inner.ttl = AsyncMock(return_value=-2)
        result = await RagCacheStore(redis).update_status("deadbeefcafe0000", CacheStatus.APPROVED)
        assert result is False
        inner.setex.assert_not_awaited()


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------

class TestRagCacheStoreDelete:
    async def test_returns_true_when_key_existed(self):
        redis, inner = _make_redis()
        inner.delete = AsyncMock(return_value=1)
        assert await RagCacheStore(redis).delete("abc123") is True

    async def test_returns_false_when_key_not_found(self):
        redis, inner = _make_redis()
        inner.delete = AsyncMock(return_value=0)
        assert await RagCacheStore(redis).delete("abc123") is False

    async def test_deletes_correct_key(self):
        redis, inner = _make_redis()
        await RagCacheStore(redis).delete("myhash")
        key = inner.delete.call_args.args[0]
        assert "rag_cache" in key and "myhash" in key


# ---------------------------------------------------------------------------
# invalidate_all()
# ---------------------------------------------------------------------------

class TestRagCacheStoreInvalidateAll:
    async def test_returns_count_of_deleted_keys(self):
        redis, inner = _make_redis()
        inner.scan = AsyncMock(return_value=(0, ["key1", "key2", "key3"]))
        inner.delete = AsyncMock(return_value=3)
        assert await RagCacheStore(redis).invalidate_all() == 3

    async def test_returns_zero_when_no_keys_match(self):
        redis, inner = _make_redis()
        inner.scan = AsyncMock(return_value=(0, []))
        count = await RagCacheStore(redis).invalidate_all()
        assert count == 0
        inner.delete.assert_not_awaited()

    async def test_handles_pagination_via_cursor(self):
        redis, inner = _make_redis()
        inner.scan = AsyncMock(side_effect=[(1, ["key1", "key2"]), (0, ["key3"])])
        inner.delete = AsyncMock(return_value=1)
        assert await RagCacheStore(redis).invalidate_all() == 3
        assert inner.scan.await_count == 2


# ---------------------------------------------------------------------------
# Stats — Redis Hash via HINCRBY / HGETALL + pipeline
# ---------------------------------------------------------------------------

class TestRagCacheStoreStats:
    async def test_increment_stat_calls_hincrby(self):
        redis, inner = _make_redis()
        await RagCacheStore(redis).increment_stat("hits")
        inner.hincrby.assert_awaited_once()
        assert "hits" in inner.hincrby.call_args.args

    async def test_increment_stat_error_does_not_propagate(self):
        redis, inner = _make_redis()
        inner.hincrby = AsyncMock(side_effect=ConnectionError("Redis down"))
        await RagCacheStore(redis).increment_stat("hits")  # must not raise

    async def test_get_stats_returns_hits_misses_and_pending(self):
        pipe = _make_pipeline({"hits": "42", "misses": "10"}, 5)
        redis, inner = _make_redis()
        inner.pipeline = MagicMock(return_value=pipe)
        stats = await RagCacheStore(redis).get_stats()
        assert stats["hits"] == 42
        assert stats["misses"] == 10
        assert stats["pending"] == 5

    async def test_get_stats_returns_zeros_when_no_data(self):
        pipe = _make_pipeline({}, 0)
        redis, inner = _make_redis()
        inner.pipeline = MagicMock(return_value=pipe)
        stats = await RagCacheStore(redis).get_stats()
        assert stats == {"hits": 0, "misses": 0, "auto_approved": 0, "pending": 0}

    async def test_get_stats_uses_single_pipeline_round_trip(self):
        pipe = _make_pipeline({}, 0)
        redis, inner = _make_redis()
        inner.pipeline = MagicMock(return_value=pipe)
        await RagCacheStore(redis).get_stats()
        inner.pipeline.assert_called_once()
        pipe.execute.assert_awaited_once()

    async def test_get_stats_values_are_integers(self):
        pipe = _make_pipeline({"hits": "7", "misses": "3"}, 2)
        redis, inner = _make_redis()
        inner.pipeline = MagicMock(return_value=pipe)
        stats = await RagCacheStore(redis).get_stats()
        assert all(isinstance(v, int) for v in stats.values())


# ---------------------------------------------------------------------------
# search_by_embedding()  — Layer 1 semantic cache lookup
# ---------------------------------------------------------------------------

class TestRagCacheStoreSearchByEmbedding:
    _SENTINEL = object()

    def _make_approved_entry(self, query_embedding=_SENTINEL, answer="cached answer") -> CacheEntry:
        from datetime import datetime, timezone
        emb = [1.0, 0.0] if query_embedding is self._SENTINEL else query_embedding
        return CacheEntry(
            query_hash="deadbeefcafe0001",
            original_query="what is fastapi?",
            normalized_query="what is fastapi",
            chunks=[_make_chunk()],
            status=CacheStatus.APPROVED,
            created_at=datetime.now(tz=timezone.utc),
            query_embedding=emb,
            answer=answer,
        )

    async def test_returns_none_when_no_approved_entries(self):
        redis, inner = _make_redis()
        inner.scan = AsyncMock(return_value=(0, []))
        result = await RagCacheStore(redis).search_by_embedding([1.0, 0.0], threshold=0.9)
        assert result is None

    async def test_returns_entry_when_similarity_above_threshold(self):
        redis, inner = _make_redis()
        entry = self._make_approved_entry(query_embedding=[1.0, 0.0])
        key = "v1:rag_cache:deadbeefcafe0001"
        pipe = _make_pipeline(entry.model_dump_json())
        inner.pipeline = MagicMock(return_value=pipe)
        inner.scan = AsyncMock(return_value=(0, [key.encode()]))
        result = await RagCacheStore(redis).search_by_embedding([1.0, 0.0], threshold=0.9)
        assert result is not None
        assert result.answer == "cached answer"

    async def test_returns_none_when_similarity_below_threshold(self):
        redis, inner = _make_redis()
        entry = self._make_approved_entry(query_embedding=[0.0, 1.0])  # orthogonal
        key = "v1:rag_cache:deadbeefcafe0001"
        pipe = _make_pipeline(entry.model_dump_json())
        inner.pipeline = MagicMock(return_value=pipe)
        inner.scan = AsyncMock(return_value=(0, [key.encode()]))
        result = await RagCacheStore(redis).search_by_embedding([1.0, 0.0], threshold=0.9)
        assert result is None

    async def test_skips_entries_without_query_embedding(self):
        redis, inner = _make_redis()
        entry = self._make_approved_entry(query_embedding=None)
        key = "v1:rag_cache:deadbeefcafe0001"
        pipe = _make_pipeline(entry.model_dump_json())
        inner.pipeline = MagicMock(return_value=pipe)
        inner.scan = AsyncMock(return_value=(0, [key.encode()]))
        result = await RagCacheStore(redis).search_by_embedding([1.0, 0.0], threshold=0.5)
        assert result is None

    async def test_skips_non_approved_entries(self):
        redis, inner = _make_redis()
        from datetime import datetime, timezone
        entry = CacheEntry(
            query_hash="deadbeefcafe0002",
            original_query="q", normalized_query="q",
            chunks=[_make_chunk()],
            status=CacheStatus.PENDING_REVIEW,
            created_at=datetime.now(tz=timezone.utc),
            query_embedding=[1.0, 0.0],
            answer="",
        )
        key = "v1:rag_cache:deadbeefcafe0002"
        pipe = _make_pipeline(entry.model_dump_json())
        inner.pipeline = MagicMock(return_value=pipe)
        inner.scan = AsyncMock(return_value=(0, [key.encode()]))
        result = await RagCacheStore(redis).search_by_embedding([1.0, 0.0], threshold=0.5)
        assert result is None

    async def test_returns_best_match_when_multiple_candidates(self):
        redis, inner = _make_redis()
        from datetime import datetime, timezone
        entry_low = CacheEntry(
            query_hash="aabbccddeeff0001",
            original_query="q1", normalized_query="q1",
            chunks=[_make_chunk()], status=CacheStatus.APPROVED,
            created_at=datetime.now(tz=timezone.utc),
            query_embedding=[0.8, 0.6], answer="low match",
        )
        entry_high = CacheEntry(
            query_hash="aabbccddeeff0002",
            original_query="q2", normalized_query="q2",
            chunks=[_make_chunk()], status=CacheStatus.APPROVED,
            created_at=datetime.now(tz=timezone.utc),
            query_embedding=[1.0, 0.0], answer="high match",
        )
        pipe = _make_pipeline(entry_low.model_dump_json(), entry_high.model_dump_json())
        inner.pipeline = MagicMock(return_value=pipe)
        inner.scan = AsyncMock(return_value=(0, [b"key1", b"key2"]))
        result = await RagCacheStore(redis).search_by_embedding([1.0, 0.0], threshold=0.5)
        assert result is not None
        assert result.answer == "high match"
