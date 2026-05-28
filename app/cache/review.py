import logging

from app.cache.schemas import CacheStatus
from app.cache.store import RagCacheStore
from app.clients.redis import RedisClient
from app.core.config import Settings, settings as _default_settings

logger = logging.getLogger(__name__)


class ReviewQueue:
    def __init__(
        self,
        redis: RedisClient,
        store: RagCacheStore,
        cfg: Settings | None = None,
    ) -> None:
        self._redis = redis
        self._store = store
        self._cfg = cfg or _default_settings

    def _pending_key(self) -> str:
        return self._redis.cache_key("review", "pending")

    async def enqueue(self, query_hash: str) -> None:
        key = self._pending_key()
        current_len = await self._redis.client.llen(key)
        if current_len >= self._cfg.cache_max_pending_reviews:
            logger.warning(
                "review_queue=full capacity=%d hash=%s",
                self._cfg.cache_max_pending_reviews, query_hash,
            )
            return
        existing = await self._redis.client.lrange(key, 0, -1)
        if query_hash not in existing:
            await self._redis.client.lpush(key, query_hash)
            logger.info("review_queue=enqueued hash=%s", query_hash)

    async def list_pending(self, limit: int = 20) -> list[str]:
        return await self._redis.client.lrange(self._pending_key(), 0, limit - 1)

    async def approve(self, query_hash: str, reviewer_id: str) -> CacheStatus:
        entry = await self._store.get(query_hash)
        if entry is None:
            return CacheStatus.PENDING_REVIEW
        if reviewer_id in entry.approved_by:
            return entry.status
        new_count = entry.approval_count + 1
        new_approved_by = entry.approved_by + [reviewer_id]
        if new_count >= self._cfg.cache_auto_approve_threshold:
            new_status = CacheStatus.APPROVED
            await self._remove_from_queue(query_hash)
        else:
            new_status = CacheStatus.PENDING_REVIEW
        await self._store.update_status(
            query_hash,
            new_status,
            approval_count=new_count,
            approved_by=new_approved_by,
        )
        return new_status

    async def reject(self, query_hash: str) -> None:
        await self._store.update_status(query_hash, CacheStatus.REJECTED)
        await self._remove_from_queue(query_hash)
        logger.info("review_queue=rejected hash=%s", query_hash)

    async def _remove_from_queue(self, query_hash: str) -> None:
        await self._redis.client.lrem(self._pending_key(), 0, query_hash)
