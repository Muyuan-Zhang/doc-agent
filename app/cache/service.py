import logging
import time
from datetime import datetime, timezone

from app.cache.quality import compute_quality
from app.cache.query_rewriter import QueryRewriter
from app.cache.review import ReviewQueue
from app.cache.schemas import CacheEntry, CacheStatus
from app.cache.store import RagCacheStore
from app.clients.llm import AbstractLLMClient
from app.clients.redis import RedisClient
from app.core.config import Settings, settings as _default_settings
from app.models.chunk import ChunkSchema
from app.models.retrieval import RetrievalStrategy

logger = logging.getLogger(__name__)


class RagCacheService:
    def __init__(
        self,
        redis: RedisClient,
        llm: AbstractLLMClient,
        cfg: Settings | None = None,
    ) -> None:
        self._cfg = cfg or _default_settings
        self._llm = llm
        self._store = RagCacheStore(redis, self._cfg)
        self._rewriter = QueryRewriter(llm, self._cfg)
        self._review = ReviewQueue(redis, self._store, self._cfg)

    @property
    def store(self) -> RagCacheStore:
        return self._store

    @property
    def review(self) -> ReviewQueue:
        return self._review

    async def get_or_retrieve(
        self,
        query: str,
        retriever: RetrievalStrategy,
        top_k: int = 5,
    ) -> tuple[list[ChunkSchema], bool, str]:
        """Return (chunks, cache_hit, query_hash).

        query_hash is the normalized hash used to key this entry — callers that
        need to write back (e.g. save_answer) should use this value rather than
        recomputing it from the raw query string.
        """
        t0 = time.perf_counter()
        normalized, query_hash = await self._rewriter.rewrite(query)
        rewrite_ms = (time.perf_counter() - t0) * 1000

        entry = await self._store.get(query_hash)
        lookup_ms = (time.perf_counter() - t0) * 1000 - rewrite_ms

        if entry is not None and entry.status == CacheStatus.APPROVED:
            await self._inc_stat("hits")
            logger.info(
                "cache=get_or_retrieve result=hit hash=%s chunks=%d lookup_ms=%.1f",
                query_hash, len(entry.chunks), lookup_ms,
            )
            return list(entry.chunks), True, query_hash

        miss_reason = "no_entry" if entry is None else f"status={entry.status.value}"
        await self._inc_stat("misses")
        logger.info(
            "cache=get_or_retrieve result=miss hash=%s reason=%s lookup_ms=%.1f",
            query_hash, miss_reason, lookup_ms,
        )
        chunks = await retriever.retrieve(query, top_k)
        retrieve_ms = (time.perf_counter() - t0) * 1000 - rewrite_ms - lookup_ms
        logger.info(
            "cache=get_or_retrieve retrieve=done hash=%s chunks=%d retrieve_ms=%.1f",
            query_hash, len(chunks), retrieve_ms,
        )

        if entry is None:
            status = await self._decide_status(normalized, chunks)
            new_entry = CacheEntry(
                query_hash=query_hash,
                original_query=query,
                normalized_query=normalized,
                chunks=chunks,
                status=status,
                created_at=datetime.now(tz=timezone.utc),
            )
            await self._store.set(new_entry, self._cfg.cache_ttl_seconds)
            if status == CacheStatus.PENDING_REVIEW:
                await self._review.enqueue(query_hash)
            elif status == CacheStatus.APPROVED:
                await self._inc_stat("auto_approved")
            logger.info(
                "cache=get_or_retrieve entry=created hash=%s status=%s",
                query_hash, status.value,
            )

        total_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "cache=get_or_retrieve done hash=%s chunks=%d cache_hit=%s total_ms=%.1f",
            query_hash, len(chunks), False, total_ms,
        )
        return chunks, False, query_hash

    async def lookup_by_embedding(
        self,
        query_embedding: list[float],
        threshold: float,
    ) -> "CacheEntry | None":
        """Return an approved cache entry whose query_embedding is within threshold."""
        t0 = time.perf_counter()
        result = await self._store.search_by_embedding(query_embedding, threshold=threshold)
        elapsed = time.perf_counter() - t0
        if result is not None:
            logger.info(
                "cache=lookup_by_embedding result=hit hash=%s elapsed=%.3fs",
                result.query_hash, elapsed,
            )
        else:
            logger.info("cache=lookup_by_embedding result=miss elapsed=%.3fs", elapsed)
        return result

    async def save_answer(
        self,
        query_hash: str,
        answer: str,
        query_embedding: list[float],
    ) -> None:
        """Write the generated answer and query embedding back into an existing entry."""
        t0 = time.perf_counter()
        entry = await self._store.get(query_hash)
        load_ms = (time.perf_counter() - t0) * 1000

        if entry is None:
            logger.warning("cache=save_answer hash=%s result=no_entry load_ms=%.1f", query_hash, load_ms)
            return

        logger.info(
            "cache=save_answer hash=%s result=writing status=%s answer_len=%d load_ms=%.1f",
            query_hash, entry.status.value, len(answer), load_ms,
        )
        updated = entry.model_copy(update={"answer": answer, "query_embedding": query_embedding})
        ttl = await self._store.get_ttl(query_hash)
        effective_ttl = ttl if ttl > 0 else self._cfg.cache_ttl_seconds
        await self._store.set(updated, effective_ttl)
        total_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "cache=save_answer hash=%s result=done ttl=%d total_ms=%.1f",
            query_hash, effective_ttl, total_ms,
        )

    async def _decide_status(
        self, query: str, chunks: list[ChunkSchema],
    ) -> CacheStatus:
        """Determine the initial cache status for a new entry.

        - cache_auto_approve=False → PENDING_REVIEW (strict, manual review)
        - cache_auto_approve=True, threshold=0 → APPROVED (auto, no quality check)
        - cache_auto_approve=True, threshold>0 → compute quality score:
            >= threshold → APPROVED, < threshold → PENDING_REVIEW
        """
        if not self._cfg.cache_auto_approve:
            logger.info("cache=decide_status result=PENDING_REVIEW reason=auto_approve_disabled")
            return CacheStatus.PENDING_REVIEW

        if not chunks:
            logger.info("cache=decide_status result=PENDING_REVIEW reason=empty_chunks")
            return CacheStatus.PENDING_REVIEW

        threshold = self._cfg.cache_quality_threshold
        if threshold <= 0.0:
            logger.info("cache=decide_status result=APPROVED reason=threshold_zero")
            return CacheStatus.APPROVED

        embed_t0 = time.perf_counter()
        try:
            query_embedding = await self._llm.embed(query)
        except Exception as exc:
            logger.warning(
                "cache=decide_status result=PENDING_REVIEW reason=embed_failed query=%s error=%s",
                query[:80], exc,
            )
            return CacheStatus.PENDING_REVIEW
        embed_ms = (time.perf_counter() - embed_t0) * 1000

        try:
            score = compute_quality(query_embedding, chunks)
        except Exception as exc:
            logger.warning(
                "cache=decide_status result=PENDING_REVIEW reason=quality_failed query=%s error=%s",
                query[:80], exc,
            )
            return CacheStatus.PENDING_REVIEW

        if score >= threshold:
            logger.info(
                "cache=decide_status result=APPROVED reason=quality score=%.4f threshold=%.2f embed_ms=%.1f",
                score, threshold, embed_ms,
            )
            return CacheStatus.APPROVED

        logger.info(
            "cache=decide_status result=PENDING_REVIEW reason=low_quality score=%.4f threshold=%.2f embed_ms=%.1f chunks=%d",
            score, threshold, embed_ms, len(chunks),
        )
        return CacheStatus.PENDING_REVIEW

    async def _inc_stat(self, stat: str) -> None:
        await self._store.increment_stat(stat)
