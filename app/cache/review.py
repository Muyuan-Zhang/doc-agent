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
        # NOTE: llen + lrange + lpush is non-atomic. Under concurrent load the
        # queue may briefly exceed cache_max_pending_reviews by up to the number
        # of concurrent writers, and the same hash may be enqueued twice.
        # Acceptable for the review-queue use-case; a Lua-script fix is needed
        # only if strict capacity or strict dedup guarantees become required.
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
        # Acquire the same lock as update_status() so the full
        # read-check-compute-write sequence is atomic.
        lock_key = self._store._lock_key(query_hash)
        acquired, token = await self._redis.acquire_lock(lock_key, ttl_seconds=10)
        if not acquired:
            logger.warning("review_queue=approve_lock_failed hash=%s", query_hash)
            return CacheStatus.PENDING_REVIEW
        try:
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
            await self._store.update_status_under_lock(
                query_hash, new_status,
                approval_count=new_count,
                approved_by=new_approved_by,
            )
            return new_status
        finally:
            await self._redis.release_lock(lock_key, token)

    async def reject(self, query_hash: str) -> None:
        entry = await self._store.get(query_hash)
        if entry is None or entry.status == CacheStatus.REJECTED:
            return
        if entry.status == CacheStatus.APPROVED:
            logger.warning("review_queue=reject_denied_approved hash=%s", query_hash)
            return
        await self._store.update_status(query_hash, CacheStatus.REJECTED)
        await self._remove_from_queue(query_hash)
        logger.info("review_queue=rejected hash=%s", query_hash)

    async def _remove_from_queue(self, query_hash: str) -> None:
        await self._redis.client.lrem(self._pending_key(), 0, query_hash)
