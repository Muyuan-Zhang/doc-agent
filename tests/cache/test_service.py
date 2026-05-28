"""
Tests for app/cache/service.py — RagCacheService.

Covers:
- APPROVED hit: returns cached chunks, skips retriever, increments hit stat (HINCRBY)
- PENDING/REJECTED hit: runs retriever, no cache write, no re-enqueue
- MISS: runs retriever, stores PENDING_REVIEW, enqueues via Lua eval
- Stat resilience: HINCRBY failure must not propagate
"""
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from app.cache.schemas import CacheEntry, CacheStatus
from app.cache.service import RagCacheService
from app.clients.redis import RedisClient
from app.core.config import Settings
from app.models.chunk import ChunkSchema


def _make_pipeline(*return_values):
    pipe = MagicMock()
    pipe.get = MagicMock(return_value=pipe)
    pipe.hgetall = MagicMock(return_value=pipe)
    pipe.zcard = MagicMock(return_value=pipe)
    pipe.execute = AsyncMock(return_value=list(return_values))
    return pipe


def _make_redis() -> tuple[RedisClient, MagicMock]:
    client = RedisClient()
    inner = MagicMock()
    inner.get = AsyncMock(return_value=None)
    inner.setex = AsyncMock(return_value=True)
    inner.set = AsyncMock(return_value=True)    # acquire_lock
    inner.eval = AsyncMock(return_value=1)      # enqueue Lua + lock release
    inner.delete = AsyncMock(return_value=1)
    inner.ttl = AsyncMock(return_value=3600)
    inner.hincrby = AsyncMock(return_value=1)   # stats HINCRBY
    inner.pipeline = MagicMock(return_value=_make_pipeline())
    client._client = inner
    return client, inner


def _make_llm() -> MagicMock:
    m = MagicMock()
    m.complete = AsyncMock(return_value="normalized query")
    return m


def _make_cfg(**overrides) -> Settings:
    return Settings(
        cache_rewrite_enabled=overrides.get("cache_rewrite_enabled", False),
        cache_ttl_seconds=overrides.get("cache_ttl_seconds", 3600),
        cache_auto_approve_threshold=overrides.get("cache_auto_approve_threshold", 1),
        cache_max_pending_reviews=overrides.get("cache_max_pending_reviews", 100),
    )


def _make_chunk(content: str = "cached content") -> ChunkSchema:
    return ChunkSchema(
        doc_id="d1", section_id="s1", chunk_index=0,
        content_hash="abc", version="v1", content=content,
    )


def _make_retriever(chunks: list[ChunkSchema] | None = None) -> MagicMock:
    m = MagicMock()
    m.retrieve = AsyncMock(return_value=chunks or [_make_chunk("retrieved")])
    return m


def _make_cache_entry(status: CacheStatus = CacheStatus.PENDING_REVIEW) -> CacheEntry:
    return CacheEntry(
        query_hash="abc1234567890000",
        original_query="test query",
        normalized_query="test query",
        chunks=[_make_chunk()],
        status=status,
        created_at=datetime.now(tz=timezone.utc),
    )


# ---------------------------------------------------------------------------
# APPROVED hit
# ---------------------------------------------------------------------------

class TestRagCacheServiceApprovedHit:
    async def test_returns_cached_chunks_without_calling_retriever(self):
        redis, inner = _make_redis()
        inner.get = AsyncMock(return_value=_make_cache_entry(CacheStatus.APPROVED).model_dump_json())
        svc = RagCacheService(redis, _make_llm(), _make_cfg())
        retriever = _make_retriever()
        chunks, hit = await svc.get_or_retrieve("test query", retriever)
        assert hit is True
        retriever.retrieve.assert_not_awaited()

    async def test_returns_correct_chunk_content_from_cache(self):
        redis, inner = _make_redis()
        entry = CacheEntry(
            query_hash="abc1234567890000", original_query="test",
            normalized_query="test", chunks=[_make_chunk("cached text")],
            status=CacheStatus.APPROVED, created_at=datetime.now(tz=timezone.utc),
        )
        inner.get = AsyncMock(return_value=entry.model_dump_json())
        svc = RagCacheService(redis, _make_llm(), _make_cfg())
        chunks, hit = await svc.get_or_retrieve("test query", _make_retriever())
        assert hit is True
        assert chunks[0].content == "cached text"

    async def test_does_not_write_to_redis_on_hit(self):
        redis, inner = _make_redis()
        inner.get = AsyncMock(return_value=_make_cache_entry(CacheStatus.APPROVED).model_dump_json())
        await RagCacheService(redis, _make_llm(), _make_cfg()).get_or_retrieve(
            "test query", _make_retriever()
        )
        inner.setex.assert_not_awaited()

    async def test_increments_hit_stat_via_hincrby(self):
        redis, inner = _make_redis()
        inner.get = AsyncMock(return_value=_make_cache_entry(CacheStatus.APPROVED).model_dump_json())
        await RagCacheService(redis, _make_llm(), _make_cfg()).get_or_retrieve(
            "test query", _make_retriever()
        )
        inner.hincrby.assert_awaited_once()


# ---------------------------------------------------------------------------
# MISS
# ---------------------------------------------------------------------------

class TestRagCacheServiceMiss:
    async def test_runs_retriever_on_miss(self):
        redis, inner = _make_redis()
        retriever = _make_retriever()
        chunks, hit = await RagCacheService(redis, _make_llm(), _make_cfg()).get_or_retrieve(
            "test query", retriever
        )
        assert hit is False
        retriever.retrieve.assert_awaited_once_with("test query", 5)

    async def test_stores_new_entry_as_pending_review(self):
        redis, inner = _make_redis()
        await RagCacheService(redis, _make_llm(), _make_cfg()).get_or_retrieve(
            "test query", _make_retriever()
        )
        inner.setex.assert_awaited_once()
        stored = CacheEntry.model_validate_json(inner.setex.call_args.args[2])
        assert stored.status == CacheStatus.PENDING_REVIEW

    async def test_enqueues_for_review_via_eval(self):
        redis, inner = _make_redis()
        await RagCacheService(redis, _make_llm(), _make_cfg()).get_or_retrieve(
            "test query", _make_retriever()
        )
        inner.eval.assert_awaited_once()  # Lua enqueue script

    async def test_returns_retriever_chunks_on_miss(self):
        redis, inner = _make_redis()
        retrieved = [_make_chunk("retrieved text")]
        chunks, _ = await RagCacheService(redis, _make_llm(), _make_cfg()).get_or_retrieve(
            "test query", _make_retriever(retrieved)
        )
        assert chunks[0].content == "retrieved text"

    async def test_increments_miss_stat_via_hincrby(self):
        redis, inner = _make_redis()
        await RagCacheService(redis, _make_llm(), _make_cfg()).get_or_retrieve(
            "test query", _make_retriever()
        )
        inner.hincrby.assert_awaited_once()

    async def test_stored_entry_contains_original_query(self):
        redis, inner = _make_redis()
        await RagCacheService(redis, _make_llm(), _make_cfg()).get_or_retrieve(
            "What is the deadline?", _make_retriever()
        )
        stored = CacheEntry.model_validate_json(inner.setex.call_args.args[2])
        assert stored.original_query == "What is the deadline?"

    async def test_uses_configured_ttl_for_storage(self):
        redis, inner = _make_redis()
        await RagCacheService(redis, _make_llm(), _make_cfg(cache_ttl_seconds=1800)).get_or_retrieve(
            "test", _make_retriever()
        )
        assert inner.setex.call_args.args[1] == 1800


# ---------------------------------------------------------------------------
# PENDING bypass
# ---------------------------------------------------------------------------

class TestRagCacheServicePendingBypass:
    async def test_runs_retriever_when_entry_is_pending(self):
        redis, inner = _make_redis()
        inner.get = AsyncMock(
            return_value=_make_cache_entry(CacheStatus.PENDING_REVIEW).model_dump_json()
        )
        retriever = _make_retriever()
        _, hit = await RagCacheService(redis, _make_llm(), _make_cfg()).get_or_retrieve(
            "test query", retriever
        )
        assert hit is False
        retriever.retrieve.assert_awaited_once()

    async def test_does_not_overwrite_pending_entry(self):
        redis, inner = _make_redis()
        inner.get = AsyncMock(
            return_value=_make_cache_entry(CacheStatus.PENDING_REVIEW).model_dump_json()
        )
        await RagCacheService(redis, _make_llm(), _make_cfg()).get_or_retrieve(
            "test query", _make_retriever()
        )
        inner.setex.assert_not_awaited()

    async def test_does_not_re_enqueue_pending_entry(self):
        redis, inner = _make_redis()
        inner.get = AsyncMock(
            return_value=_make_cache_entry(CacheStatus.PENDING_REVIEW).model_dump_json()
        )
        await RagCacheService(redis, _make_llm(), _make_cfg()).get_or_retrieve(
            "test query", _make_retriever()
        )
        inner.eval.assert_not_awaited()


# ---------------------------------------------------------------------------
# REJECTED bypass
# ---------------------------------------------------------------------------

class TestRagCacheServiceRejectedBypass:
    async def test_runs_retriever_when_entry_is_rejected(self):
        redis, inner = _make_redis()
        inner.get = AsyncMock(
            return_value=_make_cache_entry(CacheStatus.REJECTED).model_dump_json()
        )
        retriever = _make_retriever()
        _, hit = await RagCacheService(redis, _make_llm(), _make_cfg()).get_or_retrieve(
            "test query", retriever
        )
        assert hit is False
        retriever.retrieve.assert_awaited_once()

    async def test_does_not_overwrite_rejected_entry(self):
        redis, inner = _make_redis()
        inner.get = AsyncMock(
            return_value=_make_cache_entry(CacheStatus.REJECTED).model_dump_json()
        )
        await RagCacheService(redis, _make_llm(), _make_cfg()).get_or_retrieve(
            "test query", _make_retriever()
        )
        inner.setex.assert_not_awaited()


# ---------------------------------------------------------------------------
# Stat resilience
# ---------------------------------------------------------------------------

class TestRagCacheServiceStatResilience:
    async def test_hincrby_failure_does_not_propagate(self):
        redis, inner = _make_redis()
        inner.hincrby = AsyncMock(side_effect=ConnectionError("Redis down"))
        chunks, hit = await RagCacheService(redis, _make_llm(), _make_cfg()).get_or_retrieve(
            "test query", _make_retriever()
        )
        assert hit is False
        assert len(chunks) > 0


# ---------------------------------------------------------------------------
# top_k forwarding
# ---------------------------------------------------------------------------

class TestRagCacheServiceTopK:
    async def test_forwards_top_k_to_retriever(self):
        redis, inner = _make_redis()
        retriever = _make_retriever()
        await RagCacheService(redis, _make_llm(), _make_cfg()).get_or_retrieve(
            "test query", retriever, top_k=10
        )
        retriever.retrieve.assert_awaited_once_with("test query", 10)
