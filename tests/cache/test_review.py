"""
Tests for app/cache/review.py — ReviewQueue.

Covers:
- enqueue(): pushes hash, respects max capacity, deduplicates
- list_pending(): returns hashes, respects limit
- approve(): increments count, auto-approves at threshold, idempotent for same reviewer
- reject(): updates status to REJECTED, removes from queue
"""
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from app.cache.review import ReviewQueue
from app.cache.schemas import CacheEntry, CacheStatus
from app.cache.store import RagCacheStore
from app.clients.redis import RedisClient
from app.core.config import Settings
from app.models.chunk import ChunkSchema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_redis() -> tuple[RedisClient, MagicMock]:
    client = RedisClient()
    inner = MagicMock()
    inner.llen = AsyncMock(return_value=0)
    inner.lrange = AsyncMock(return_value=[])
    inner.lpush = AsyncMock(return_value=1)
    inner.lrem = AsyncMock(return_value=1)
    inner.get = AsyncMock(return_value=None)
    inner.setex = AsyncMock(return_value=True)
    inner.set = AsyncMock(return_value=True)   # acquire_lock
    inner.eval = AsyncMock(return_value=1)     # release_lock Lua script
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
# enqueue()
# ---------------------------------------------------------------------------

class TestReviewQueueEnqueue:
    async def test_pushes_hash_to_pending_list(self):
        redis, inner = _make_redis()
        inner.llen = AsyncMock(return_value=0)
        inner.lrange = AsyncMock(return_value=[])
        queue, _ = _make_queue(redis, inner)
        await queue.enqueue("abc123")
        inner.lpush.assert_awaited_once()
        assert inner.lpush.call_args.args[1] == "abc123"

    async def test_does_not_push_when_queue_at_capacity(self):
        redis, inner = _make_redis()
        inner.llen = AsyncMock(return_value=100)
        queue, _ = _make_queue(redis, inner, cache_max_pending_reviews=100)
        await queue.enqueue("abc123")
        inner.lpush.assert_not_awaited()

    async def test_does_not_duplicate_existing_hash(self):
        redis, inner = _make_redis()
        inner.llen = AsyncMock(return_value=1)
        inner.lrange = AsyncMock(return_value=["abc123"])
        queue, _ = _make_queue(redis, inner)
        await queue.enqueue("abc123")
        inner.lpush.assert_not_awaited()

    async def test_pushes_when_hash_is_new(self):
        redis, inner = _make_redis()
        inner.llen = AsyncMock(return_value=1)
        inner.lrange = AsyncMock(return_value=["other_hash"])
        queue, _ = _make_queue(redis, inner)
        await queue.enqueue("new_hash")
        inner.lpush.assert_awaited_once()

    async def test_pending_key_uses_review_namespace(self):
        redis, inner = _make_redis()
        inner.llen = AsyncMock(return_value=0)
        inner.lrange = AsyncMock(return_value=[])
        queue, _ = _make_queue(redis, inner)
        await queue.enqueue("h1")
        key = inner.lpush.call_args.args[0]
        assert "review" in key
        assert "pending" in key


# ---------------------------------------------------------------------------
# list_pending()
# ---------------------------------------------------------------------------

class TestReviewQueueListPending:
    async def test_returns_list_of_hashes(self):
        redis, inner = _make_redis()
        inner.lrange = AsyncMock(return_value=["hash1", "hash2"])
        queue, _ = _make_queue(redis, inner)
        result = await queue.list_pending(limit=10)
        assert result == ["hash1", "hash2"]

    async def test_respects_limit_parameter(self):
        redis, inner = _make_redis()
        queue, _ = _make_queue(redis, inner)
        await queue.list_pending(limit=5)
        inner.lrange.assert_awaited_once_with(
            redis.cache_key("review", "pending"), 0, 4
        )

    async def test_returns_empty_list_when_no_pending(self):
        redis, inner = _make_redis()
        inner.lrange = AsyncMock(return_value=[])
        queue, _ = _make_queue(redis, inner)
        result = await queue.list_pending()
        assert result == []


# ---------------------------------------------------------------------------
# approve()
# ---------------------------------------------------------------------------

class TestReviewQueueApprove:
    async def test_returns_approved_when_threshold_reached(self):
        redis, inner = _make_redis()
        entry = _make_entry(approval_count=0)
        inner.get = AsyncMock(return_value=entry.model_dump_json())
        queue, _ = _make_queue(redis, inner, cache_auto_approve_threshold=1)
        status = await queue.approve("deadbeefcafe0000", "reviewer-1")
        assert status == CacheStatus.APPROVED

    async def test_returns_pending_when_threshold_not_reached(self):
        redis, inner = _make_redis()
        entry = _make_entry(approval_count=0)
        inner.get = AsyncMock(return_value=entry.model_dump_json())
        queue, _ = _make_queue(redis, inner, cache_auto_approve_threshold=3)
        status = await queue.approve("deadbeefcafe0000", "reviewer-1")
        assert status == CacheStatus.PENDING_REVIEW

    async def test_second_approval_reaches_threshold_of_two(self):
        redis, inner = _make_redis()
        entry = _make_entry(approval_count=1, approved_by=["reviewer-1"])
        inner.get = AsyncMock(return_value=entry.model_dump_json())
        queue, _ = _make_queue(redis, inner, cache_auto_approve_threshold=2)
        status = await queue.approve("deadbeefcafe0000", "reviewer-2")
        assert status == CacheStatus.APPROVED

    async def test_idempotent_for_same_reviewer(self):
        redis, inner = _make_redis()
        entry = _make_entry(approval_count=1, approved_by=["reviewer-1"])
        inner.get = AsyncMock(return_value=entry.model_dump_json())
        queue, _ = _make_queue(redis, inner, cache_auto_approve_threshold=3)
        status = await queue.approve("deadbeefcafe0000", "reviewer-1")
        assert status == entry.status
        inner.setex.assert_not_awaited()

    async def test_returns_pending_when_entry_not_found(self):
        redis, inner = _make_redis()
        inner.get = AsyncMock(return_value=None)
        queue, _ = _make_queue(redis, inner)
        status = await queue.approve("nonexistent", "reviewer-1")
        assert status == CacheStatus.PENDING_REVIEW
        inner.setex.assert_not_awaited()

    async def test_removes_from_queue_when_approved(self):
        redis, inner = _make_redis()
        entry = _make_entry(approval_count=0)
        inner.get = AsyncMock(return_value=entry.model_dump_json())
        queue, _ = _make_queue(redis, inner, cache_auto_approve_threshold=1)
        await queue.approve("deadbeefcafe0000", "reviewer-1")
        inner.lrem.assert_awaited_once()

    async def test_stores_updated_entry_on_approval(self):
        redis, inner = _make_redis()
        entry = _make_entry(approval_count=0)
        inner.get = AsyncMock(return_value=entry.model_dump_json())
        queue, _ = _make_queue(redis, inner, cache_auto_approve_threshold=1)
        await queue.approve("deadbeefcafe0000", "reviewer-1")
        inner.setex.assert_awaited_once()
        raw = inner.setex.call_args.args[2]
        updated = CacheEntry.model_validate_json(raw)
        assert updated.approval_count == 1
        assert "reviewer-1" in updated.approved_by


# ---------------------------------------------------------------------------
# reject()
# ---------------------------------------------------------------------------

class TestReviewQueueReject:
    async def test_updates_status_to_rejected(self):
        redis, inner = _make_redis()
        entry = _make_entry()
        inner.get = AsyncMock(return_value=entry.model_dump_json())
        inner.ttl = AsyncMock(return_value=3600)
        queue, _ = _make_queue(redis, inner)
        await queue.reject("deadbeefcafe0000")
        raw = inner.setex.call_args.args[2]
        updated = CacheEntry.model_validate_json(raw)
        assert updated.status == CacheStatus.REJECTED

    async def test_removes_from_pending_queue(self):
        redis, inner = _make_redis()
        entry = _make_entry()
        inner.get = AsyncMock(return_value=entry.model_dump_json())
        queue, _ = _make_queue(redis, inner)
        await queue.reject("deadbeefcafe0000")
        inner.lrem.assert_awaited_once()

    async def test_lrem_targets_pending_key(self):
        redis, inner = _make_redis()
        entry = _make_entry()
        inner.get = AsyncMock(return_value=entry.model_dump_json())
        queue, _ = _make_queue(redis, inner)
        await queue.reject("deadbeefcafe0000")
        lrem_key = inner.lrem.call_args.args[0]
        assert "review" in lrem_key
        assert "pending" in lrem_key
