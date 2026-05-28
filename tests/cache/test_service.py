"""
Tests for app/cache/service.py — RagCacheService.

Covers:
- APPROVED cache hit: returns cached chunks, skips retriever
- PENDING hit: runs retriever, does NOT overwrite cache
- REJECTED hit: runs retriever, does NOT overwrite cache
- MISS: runs retriever, stores PENDING_REVIEW entry, enqueues for review
"""
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from app.cache.schemas import CacheEntry, CacheStatus
from app.cache.service import RagCacheService
from app.clients.redis import RedisClient
from app.core.config import Settings
from app.models.chunk import ChunkSchema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_redis() -> tuple[RedisClient, MagicMock]:
    client = RedisClient()
    inner = MagicMock()
    inner.get = AsyncMock(return_value=None)
    inner.setex = AsyncMock(return_value=True)
    inner.delete = AsyncMock(return_value=1)
    inner.ttl = AsyncMock(return_value=3600)
    inner.llen = AsyncMock(return_value=0)
    inner.lrange = AsyncMock(return_value=[])
    inner.lpush = AsyncMock(return_value=1)
    inner.lrem = AsyncMock(return_value=1)
    inner.incr = AsyncMock(return_value=1)
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
    chunks = chunks or [_make_chunk("retrieved")]
    m = MagicMock()
    m.retrieve = AsyncMock(return_value=chunks)
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
        entry = _make_cache_entry(CacheStatus.APPROVED)
        inner.get = AsyncMock(return_value=entry.model_dump_json())
        svc = RagCacheService(redis, _make_llm(), _make_cfg())
        retriever = _make_retriever()
        chunks, hit = await svc.get_or_retrieve("test query", retriever)
        assert hit is True
        retriever.retrieve.assert_not_awaited()

    async def test_returns_correct_chunk_content_from_cache(self):
        redis, inner = _make_redis()
        cached_chunk = _make_chunk("cached text")
        entry = CacheEntry(
            query_hash="abc1234567890000",
            original_query="test",
            normalized_query="test",
            chunks=[cached_chunk],
            status=CacheStatus.APPROVED,
            created_at=datetime.now(tz=timezone.utc),
        )
        inner.get = AsyncMock(return_value=entry.model_dump_json())
        svc = RagCacheService(redis, _make_llm(), _make_cfg())
        chunks, hit = await svc.get_or_retrieve("test query", _make_retriever())
        assert hit is True
        assert chunks[0].content == "cached text"

    async def test_does_not_write_to_redis_on_hit(self):
        redis, inner = _make_redis()
        entry = _make_cache_entry(CacheStatus.APPROVED)
        inner.get = AsyncMock(return_value=entry.model_dump_json())
        svc = RagCacheService(redis, _make_llm(), _make_cfg())
        await svc.get_or_retrieve("test query", _make_retriever())
        inner.setex.assert_not_awaited()

    async def test_increments_hit_stat(self):
        redis, inner = _make_redis()
        entry = _make_cache_entry(CacheStatus.APPROVED)
        inner.get = AsyncMock(return_value=entry.model_dump_json())
        svc = RagCacheService(redis, _make_llm(), _make_cfg())
        await svc.get_or_retrieve("test query", _make_retriever())
        inner.incr.assert_awaited_once()


# ---------------------------------------------------------------------------
# MISS
# ---------------------------------------------------------------------------

class TestRagCacheServiceMiss:
    async def test_runs_retriever_on_miss(self):
        redis, inner = _make_redis()
        inner.get = AsyncMock(return_value=None)
        svc = RagCacheService(redis, _make_llm(), _make_cfg())
        retriever = _make_retriever()
        chunks, hit = await svc.get_or_retrieve("test query", retriever)
        assert hit is False
        retriever.retrieve.assert_awaited_once_with("test query", 5)

    async def test_stores_new_entry_as_pending_review(self):
        redis, inner = _make_redis()
        inner.get = AsyncMock(return_value=None)
        svc = RagCacheService(redis, _make_llm(), _make_cfg())
        await svc.get_or_retrieve("test query", _make_retriever())
        inner.setex.assert_awaited_once()
        raw = inner.setex.call_args.args[2]
        stored = CacheEntry.model_validate_json(raw)
        assert stored.status == CacheStatus.PENDING_REVIEW

    async def test_enqueues_for_review_on_miss(self):
        redis, inner = _make_redis()
        inner.get = AsyncMock(return_value=None)
        svc = RagCacheService(redis, _make_llm(), _make_cfg())
        await svc.get_or_retrieve("test query", _make_retriever())
        inner.lpush.assert_awaited_once()

    async def test_returns_retriever_chunks_on_miss(self):
        redis, inner = _make_redis()
        inner.get = AsyncMock(return_value=None)
        svc = RagCacheService(redis, _make_llm(), _make_cfg())
        retrieved = [_make_chunk("retrieved text")]
        chunks, _ = await svc.get_or_retrieve("test query", _make_retriever(retrieved))
        assert chunks[0].content == "retrieved text"

    async def test_increments_miss_stat(self):
        redis, inner = _make_redis()
        inner.get = AsyncMock(return_value=None)
        svc = RagCacheService(redis, _make_llm(), _make_cfg())
        await svc.get_or_retrieve("test query", _make_retriever())
        inner.incr.assert_awaited_once()

    async def test_stored_entry_contains_original_query(self):
        redis, inner = _make_redis()
        inner.get = AsyncMock(return_value=None)
        svc = RagCacheService(redis, _make_llm(), _make_cfg())
        await svc.get_or_retrieve("What is the deadline?", _make_retriever())
        raw = inner.setex.call_args.args[2]
        stored = CacheEntry.model_validate_json(raw)
        assert stored.original_query == "What is the deadline?"

    async def test_uses_configured_ttl_for_storage(self):
        redis, inner = _make_redis()
        inner.get = AsyncMock(return_value=None)
        cfg = _make_cfg(cache_ttl_seconds=1800)
        svc = RagCacheService(redis, _make_llm(), cfg)
        await svc.get_or_retrieve("test", _make_retriever())
        ttl = inner.setex.call_args.args[1]
        assert ttl == 1800


# ---------------------------------------------------------------------------
# PENDING bypass
# ---------------------------------------------------------------------------

class TestRagCacheServicePendingBypass:
    async def test_runs_retriever_when_entry_is_pending(self):
        redis, inner = _make_redis()
        entry = _make_cache_entry(CacheStatus.PENDING_REVIEW)
        inner.get = AsyncMock(return_value=entry.model_dump_json())
        svc = RagCacheService(redis, _make_llm(), _make_cfg())
        retriever = _make_retriever()
        _, hit = await svc.get_or_retrieve("test query", retriever)
        assert hit is False
        retriever.retrieve.assert_awaited_once()

    async def test_does_not_overwrite_pending_entry(self):
        redis, inner = _make_redis()
        entry = _make_cache_entry(CacheStatus.PENDING_REVIEW)
        inner.get = AsyncMock(return_value=entry.model_dump_json())
        svc = RagCacheService(redis, _make_llm(), _make_cfg())
        await svc.get_or_retrieve("test query", _make_retriever())
        inner.setex.assert_not_awaited()

    async def test_does_not_enqueue_again_when_already_pending(self):
        redis, inner = _make_redis()
        entry = _make_cache_entry(CacheStatus.PENDING_REVIEW)
        inner.get = AsyncMock(return_value=entry.model_dump_json())
        svc = RagCacheService(redis, _make_llm(), _make_cfg())
        await svc.get_or_retrieve("test query", _make_retriever())
        inner.lpush.assert_not_awaited()


# ---------------------------------------------------------------------------
# REJECTED bypass
# ---------------------------------------------------------------------------

class TestRagCacheServiceRejectedBypass:
    async def test_runs_retriever_when_entry_is_rejected(self):
        redis, inner = _make_redis()
        entry = _make_cache_entry(CacheStatus.REJECTED)
        inner.get = AsyncMock(return_value=entry.model_dump_json())
        svc = RagCacheService(redis, _make_llm(), _make_cfg())
        retriever = _make_retriever()
        _, hit = await svc.get_or_retrieve("test query", retriever)
        assert hit is False
        retriever.retrieve.assert_awaited_once()

    async def test_does_not_overwrite_rejected_entry(self):
        redis, inner = _make_redis()
        entry = _make_cache_entry(CacheStatus.REJECTED)
        inner.get = AsyncMock(return_value=entry.model_dump_json())
        svc = RagCacheService(redis, _make_llm(), _make_cfg())
        await svc.get_or_retrieve("test query", _make_retriever())
        inner.setex.assert_not_awaited()


# ---------------------------------------------------------------------------
# Stat increment resilience
# ---------------------------------------------------------------------------

class TestRagCacheServiceStatResilience:
    async def test_stat_error_does_not_propagate(self):
        """incr() failure must not surface to the caller."""
        redis, inner = _make_redis()
        inner.get = AsyncMock(return_value=None)
        inner.incr = AsyncMock(side_effect=ConnectionError("Redis down"))
        svc = RagCacheService(redis, _make_llm(), _make_cfg())
        chunks, hit = await svc.get_or_retrieve("test query", _make_retriever())
        assert hit is False
        assert len(chunks) > 0


# ---------------------------------------------------------------------------
# top_k forwarding
# ---------------------------------------------------------------------------

class TestRagCacheServiceTopK:
    async def test_forwards_top_k_to_retriever(self):
        redis, inner = _make_redis()
        inner.get = AsyncMock(return_value=None)
        svc = RagCacheService(redis, _make_llm(), _make_cfg())
        retriever = _make_retriever()
        await svc.get_or_retrieve("test query", retriever, top_k=10)
        retriever.retrieve.assert_awaited_once_with("test query", 10)
