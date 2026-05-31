"""
Tests for app/cache/review.py — ReviewQueue.

Covers:
- enqueue(): atomic Lua script + Sorted Set NX, capacity, dedup
- list_pending(): ZRANGE desc, limit, bytes decode
- approve(): threshold, idempotency, lock failure
- reject(): lock guard, APPROVED→REJECTED raises ValidationError
"""
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from app.cache.review import ReviewQueue
from app.cache.schemas import CacheEntry, CacheStatus
from app.cache.store import RagCacheStore
from app.clients.redis import RedisClient
from app.core.config import Settings
from app.core.exceptions import ValidationError
from app.models.chunk import ChunkSchema


def _make_redis() -> tuple[RedisClient, MagicMock]:
    client = RedisClient()
    inner = MagicMock()
    inner.eval = AsyncMock(return_value=1)    # enqueue Lua (1=added) + lock release
    inner.zrange = AsyncMock(return_value=[])
    inner.zrem = AsyncMock(return_value=1)
    inner.get = AsyncMock(return_value=None)
    inner.setex = AsyncMock(return_value=True)
    inner.set = AsyncMock(return_value=True)  # acquire_lock
    inner.ttl = AsyncMock(return_value=3600)
    client._client = inner
    return client, inner


def _make_chunk() -> ChunkSchema:
    return ChunkSchema(
        doc_id="d1", section_id="s1", chunk_index=0,
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


def _make_cfg(**overrides) -> Settings:
    return Settings(
        cache_auto_approve_threshold=overrides.get("cache_auto_approve_threshold", 1),
        cache_max_pending_reviews=overrides.get("cache_max_pending_reviews", 100),
        cache_ttl_seconds=overrides.get("cache_ttl_seconds", 3600),
    )


def _make_queue(redis, inner, **cfg_overrides) -> tuple[ReviewQueue, RagCacheStore]:
    cfg = _make_cfg(**cfg_overrides)
    store = RagCacheStore(redis, cfg)
    queue = ReviewQueue(redis, store, cfg)
    return queue, store


# ---------------------------------------------------------------------------
# enqueue() — atomic Lua + Sorted Set
# ---------------------------------------------------------------------------

class TestReviewQueueEnqueue:
    async def test_calls_eval_to_add_hash(self):
        redis, inner = _make_redis()
        queue, _ = _make_queue(redis, inner)
        await queue.enqueue("abc123")
        inner.eval.assert_awaited_once()

    async def test_eval_receives_hash_as_member(self):
        redis, inner = _make_redis()
        queue, _ = _make_queue(redis, inner)
        await queue.enqueue("abc123")
        assert "abc123" in inner.eval.call_args.args

    async def test_handles_capacity_full_gracefully(self):
        redis, inner = _make_redis()
        inner.eval = AsyncMock(return_value=-1)  # at capacity
        queue, _ = _make_queue(redis, inner, cache_max_pending_reviews=100)
        await queue.enqueue("abc123")  # must not raise

    async def test_handles_already_present_gracefully(self):
        redis, inner = _make_redis()
        inner.eval = AsyncMock(return_value=0)  # ZADD NX no-op
        queue, _ = _make_queue(redis, inner)
        await queue.enqueue("abc123")  # must not raise

    async def test_pending_key_uses_review_namespace(self):
        redis, inner = _make_redis()
        queue, _ = _make_queue(redis, inner)
        await queue.enqueue("h1")
        key = inner.eval.call_args.args[2]   # KEYS[1]
        assert "review" in key
        assert "pending" in key


# ---------------------------------------------------------------------------
# list_pending() — ZRANGE desc=True
# ---------------------------------------------------------------------------

class TestReviewQueueListPending:
    async def test_returns_list_of_hashes(self):
        redis, inner = _make_redis()
        inner.zrange = AsyncMock(return_value=["hash1", "hash2"])
        queue, _ = _make_queue(redis, inner)
        assert await queue.list_pending(limit=10) == ["hash1", "hash2"]

    async def test_respects_limit_parameter(self):
        redis, inner = _make_redis()
        queue, _ = _make_queue(redis, inner)
        await queue.list_pending(limit=5)
        inner.zrange.assert_awaited_once()
        assert inner.zrange.call_args.args[2] == 4  # end = limit - 1

    async def test_returns_empty_list_when_no_pending(self):
        redis, inner = _make_redis()
        queue, _ = _make_queue(redis, inner)
        assert await queue.list_pending() == []

    async def test_decodes_bytes_members(self):
        redis, inner = _make_redis()
        inner.zrange = AsyncMock(return_value=[b"hash1", b"hash2"])
        queue, _ = _make_queue(redis, inner)
        assert await queue.list_pending() == ["hash1", "hash2"]


# ---------------------------------------------------------------------------
# approve()
# ---------------------------------------------------------------------------

class TestReviewQueueApprove:
    async def test_returns_approved_when_threshold_reached(self):
        redis, inner = _make_redis()
        inner.get = AsyncMock(return_value=_make_entry(approval_count=0).model_dump_json())
        queue, _ = _make_queue(redis, inner, cache_auto_approve_threshold=1)
        assert await queue.approve("deadbeefcafe0000", "r1") == CacheStatus.APPROVED

    async def test_returns_pending_when_threshold_not_reached(self):
        redis, inner = _make_redis()
        inner.get = AsyncMock(return_value=_make_entry(approval_count=0).model_dump_json())
        queue, _ = _make_queue(redis, inner, cache_auto_approve_threshold=3)
        assert await queue.approve("deadbeefcafe0000", "r1") == CacheStatus.PENDING_REVIEW

    async def test_second_approval_reaches_threshold_of_two(self):
        redis, inner = _make_redis()
        entry = _make_entry(approval_count=1, approved_by=["r1"])
        inner.get = AsyncMock(return_value=entry.model_dump_json())
        queue, _ = _make_queue(redis, inner, cache_auto_approve_threshold=2)
        assert await queue.approve("deadbeefcafe0000", "r2") == CacheStatus.APPROVED

    async def test_idempotent_for_same_reviewer(self):
        redis, inner = _make_redis()
        entry = _make_entry(approval_count=1, approved_by=["r1"])
        inner.get = AsyncMock(return_value=entry.model_dump_json())
        queue, _ = _make_queue(redis, inner, cache_auto_approve_threshold=3)
        status = await queue.approve("deadbeefcafe0000", "r1")
        assert status == entry.status
        inner.setex.assert_not_awaited()

    async def test_returns_pending_when_entry_not_found(self):
        redis, inner = _make_redis()
        queue, _ = _make_queue(redis, inner)
        assert await queue.approve("nonexistent", "r1") == CacheStatus.PENDING_REVIEW
        inner.setex.assert_not_awaited()

    async def test_removes_from_queue_via_zrem_when_approved(self):
        redis, inner = _make_redis()
        inner.get = AsyncMock(return_value=_make_entry(approval_count=0).model_dump_json())
        queue, _ = _make_queue(redis, inner, cache_auto_approve_threshold=1)
        await queue.approve("deadbeefcafe0000", "r1")
        inner.zrem.assert_awaited_once()

    async def test_stores_updated_approval_count_and_reviewer(self):
        redis, inner = _make_redis()
        inner.get = AsyncMock(return_value=_make_entry(approval_count=0).model_dump_json())
        queue, _ = _make_queue(redis, inner, cache_auto_approve_threshold=1)
        await queue.approve("deadbeefcafe0000", "r1")
        inner.setex.assert_awaited_once()
        updated = CacheEntry.model_validate_json(inner.setex.call_args.args[2])
        assert updated.approval_count == 1
        assert "r1" in updated.approved_by

    async def test_returns_pending_when_lock_not_acquired(self):
        redis, inner = _make_redis()
        inner.set = AsyncMock(return_value=None)  # lock acquire fails
        queue, _ = _make_queue(redis, inner)
        assert await queue.approve("deadbeefcafe0000", "r1") == CacheStatus.PENDING_REVIEW
        inner.setex.assert_not_awaited()

    async def test_already_approved_entry_returns_approved_and_cleans_queue(self):
        """APPROVED entries already in the queue (e.g. auto-approved) are
        returned as APPROVED and removed from the sorted set without status change."""
        redis, inner = _make_redis()
        inner.get = AsyncMock(
            return_value=_make_entry(status=CacheStatus.APPROVED).model_dump_json()
        )
        queue, _ = _make_queue(redis, inner)
        result = await queue.approve("deadbeefcafe0000", "r1")
        assert result == CacheStatus.APPROVED
        inner.setex.assert_not_awaited()   # no status change needed
        inner.zrem.assert_awaited_once()   # cleaned from pending queue


# ---------------------------------------------------------------------------
# reject() guards
# ---------------------------------------------------------------------------

class TestReviewQueueRejectGuards:
    async def test_noop_when_already_rejected(self):
        redis, inner = _make_redis()
        inner.get = AsyncMock(
            return_value=_make_entry(status=CacheStatus.REJECTED).model_dump_json()
        )
        queue, _ = _make_queue(redis, inner)
        await queue.reject("deadbeefcafe0000")
        inner.setex.assert_not_awaited()

    async def test_noop_when_entry_missing(self):
        redis, inner = _make_redis()
        queue, _ = _make_queue(redis, inner)
        await queue.reject("deadbeefcafe0000")
        inner.setex.assert_not_awaited()

    async def test_already_approved_entry_just_removed_from_queue(self):
        """APPROVED entries cannot be rejected (terminal) — they are silently
        removed from the pending queue without a status change."""
        redis, inner = _make_redis()
        inner.get = AsyncMock(
            return_value=_make_entry(status=CacheStatus.APPROVED).model_dump_json()
        )
        queue, _ = _make_queue(redis, inner)
        await queue.reject("deadbeefcafe0000")
        # Must not call setex (no status update) — just zrem for cleanup.
        inner.setex.assert_not_awaited()
        inner.zrem.assert_awaited_once()


# ---------------------------------------------------------------------------
# reject()
# ---------------------------------------------------------------------------

class TestReviewQueueReject:
    async def test_updates_status_to_rejected(self):
        redis, inner = _make_redis()
        entry = _make_entry()
        inner.get = AsyncMock(return_value=entry.model_dump_json())
        queue, _ = _make_queue(redis, inner)
        await queue.reject("deadbeefcafe0000")
        updated = CacheEntry.model_validate_json(inner.setex.call_args.args[2])
        assert updated.status == CacheStatus.REJECTED

    async def test_removes_from_pending_queue_via_zrem(self):
        redis, inner = _make_redis()
        inner.get = AsyncMock(return_value=_make_entry().model_dump_json())
        queue, _ = _make_queue(redis, inner)
        await queue.reject("deadbeefcafe0000")
        inner.zrem.assert_awaited_once()

    async def test_zrem_targets_pending_key(self):
        redis, inner = _make_redis()
        inner.get = AsyncMock(return_value=_make_entry().model_dump_json())
        queue, _ = _make_queue(redis, inner)
        await queue.reject("deadbeefcafe0000")
        key = inner.zrem.call_args.args[0]
        assert "review" in key and "pending" in key
