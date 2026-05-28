import logging
from datetime import datetime, timezone

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
    ) -> tuple[list[ChunkSchema], bool]:
        """Return (chunks, cache_hit). cache_hit=True means served from approved cache."""
        normalized, query_hash = await self._rewriter.rewrite(query)
        entry = await self._store.get(query_hash)

        if entry is not None and entry.status == CacheStatus.APPROVED:
            await self._inc_stat("hits")
            logger.info("cache=hit hash=%s", query_hash)
            return list(entry.chunks), True

        await self._inc_stat("misses")
        logger.info(
            "cache=miss hash=%s status=%s",
            query_hash, entry.status.value if entry else "none",
        )
        chunks = await retriever.retrieve(query, top_k)

        if entry is None:
            new_entry = CacheEntry(
                query_hash=query_hash,
                original_query=query,
                normalized_query=normalized,
                chunks=chunks,
                status=CacheStatus.PENDING_REVIEW,
                created_at=datetime.now(tz=timezone.utc),
            )
            await self._store.set(new_entry, self._cfg.cache_ttl_seconds)
            await self._review.enqueue(query_hash)

        return chunks, False

    async def _inc_stat(self, stat: str) -> None:
        await self._store.increment_stat(stat)
