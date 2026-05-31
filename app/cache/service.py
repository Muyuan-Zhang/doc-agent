import logging
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
        normalized, query_hash = await self._rewriter.rewrite(query)
        entry = await self._store.get(query_hash)

        if entry is not None and entry.status == CacheStatus.APPROVED:
            await self._inc_stat("hits")
            logger.info("cache=hit hash=%s", query_hash)
            return list(entry.chunks), True, query_hash

        await self._inc_stat("misses")
        logger.info(
            "cache=miss hash=%s status=%s",
            query_hash, entry.status.value if entry else "none",
        )
        chunks = await retriever.retrieve(query, top_k)

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

        return chunks, False, query_hash

    async def lookup_by_embedding(
        self,
        query_embedding: list[float],
        threshold: float,
    ) -> "CacheEntry | None":
        """Return an approved cache entry whose query_embedding is within threshold."""
        return await self._store.search_by_embedding(query_embedding, threshold=threshold)

    async def save_answer(
        self,
        query_hash: str,
        answer: str,
        query_embedding: list[float],
    ) -> None:
        """Write the generated answer and query embedding back into an existing entry."""
        entry = await self._store.get(query_hash)
        if entry is None:
            return
        updated = entry.model_copy(update={"answer": answer, "query_embedding": query_embedding})
        ttl = await self._store.get_ttl(query_hash)
        await self._store.set(updated, ttl if ttl > 0 else self._cfg.cache_ttl_seconds)

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
            return CacheStatus.PENDING_REVIEW

        if not chunks:
            logger.info("cache=empty_chunks — PENDING_REVIEW")
            return CacheStatus.PENDING_REVIEW

        threshold = self._cfg.cache_quality_threshold
        if threshold <= 0.0:
            return CacheStatus.APPROVED

        try:
            query_embedding = await self._llm.embed(query)
        except Exception as exc:
            logger.warning(
                "cache=embed_failed query=%s error=%s — falling back to PENDING_REVIEW",
                query[:80], exc,
            )
            return CacheStatus.PENDING_REVIEW

        try:
            score = compute_quality(query_embedding, chunks)
        except Exception as exc:
            logger.warning(
                "cache=quality_failed query=%s error=%s — falling back to PENDING_REVIEW",
                query[:80], exc,
            )
            return CacheStatus.PENDING_REVIEW

        if score >= threshold:
            logger.info(
                "cache=auto_approved score=%.4f threshold=%.2f",
                score, threshold,
            )
            return CacheStatus.APPROVED

        logger.info(
            "cache=quality_below_threshold score=%.4f threshold=%.2f — PENDING_REVIEW",
            score, threshold,
        )
        return CacheStatus.PENDING_REVIEW

    async def _inc_stat(self, stat: str) -> None:
        await self._store.increment_stat(stat)
