import logging
import time

from app.cache.schemas import CacheStatus, validate_transition
from app.cache.store import RagCacheStore
from app.clients.redis import RedisClient
from app.core.config import Settings, settings as _default_settings
from app.core.exceptions import ValidationError

logger = logging.getLogger(__name__)

# Atomically check capacity then ZADD NX.
# Returns 1 if added, 0 if already present (NX no-op), -1 if at capacity.
_ENQUEUE_LUA = """
local cap = tonumber(ARGV[1])
local card = redis.call('ZCARD', KEYS[1])
if card >= cap then return -1 end
return redis.call('ZADD', KEYS[1], 'NX', ARGV[2], ARGV[3])
"""


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
        """Atomically add query_hash to the pending sorted set (score = timestamp).

        Lua script makes capacity check + ZADD NX a single round-trip with
        no race between concurrent writers.
        """
        score = time.time()
        result = await self._redis.client.eval(
            _ENQUEUE_LUA,
            1,
            self._pending_key(),
            str(self._cfg.cache_max_pending_reviews),
            score,
            query_hash,
        )
        if result == -1:
            logger.warning(
                "review_queue=full capacity=%d hash=%s",
                self._cfg.cache_max_pending_reviews, query_hash,
            )
        elif result == 1:
            logger.info("review_queue=enqueued hash=%s", query_hash)
        # result == 0: already present, ZADD NX was a no-op

    async def list_pending(self, limit: int = 20) -> list[str]:
        """Return up to `limit` hashes ordered newest-first (highest score first)."""
        members = await self._redis.client.zrange(
            self._pending_key(), 0, limit - 1, desc=True
        )
        return [m.decode() if isinstance(m, bytes) else m for m in members]

    async def approve(self, query_hash: str, reviewer_id: str) -> CacheStatus:
        lock_key = self._store._lock_key(query_hash)
        acquired, token = await self._redis.acquire_lock(lock_key, ttl_seconds=10)
        if not acquired:
            logger.warning("review_queue=approve_lock_failed hash=%s", query_hash)
            return CacheStatus.PENDING_REVIEW
        try:
            entry = await self._store.get(query_hash)
            if entry is None:
                return CacheStatus.PENDING_REVIEW

            # Already-approved entries (e.g. auto-approved): just clean up the
            # pending queue and return the current status — no transition needed.
            if entry.status == CacheStatus.APPROVED:
                await self.remove_from_queue(query_hash)
                return CacheStatus.APPROVED

            if reviewer_id in entry.approved_by:
                return entry.status
            new_count = entry.approval_count + 1
            new_approved_by = entry.approved_by + [reviewer_id]
            if new_count >= self._cfg.cache_auto_approve_threshold:
                new_status = CacheStatus.APPROVED
                await self.remove_from_queue(query_hash)
            else:
                new_status = CacheStatus.PENDING_REVIEW
            try:
                await self._store.update_status_under_lock(
                    query_hash, new_status,
                    approval_count=new_count,
                    approved_by=new_approved_by,
                )
            except ValidationError:
                # Race: entry became APPROVED (TTL-expiry + re-creation) between
                # the two store.get() calls inside update_status_under_lock.
                await self.remove_from_queue(query_hash)
                return CacheStatus.APPROVED
            return new_status
        finally:
            await self._redis.release_lock(lock_key, token)

    async def reject(self, query_hash: str) -> None:
        """Reject under lock. Already-APPROVED entries are just removed from queue."""
        lock_key = self._store._lock_key(query_hash)
        acquired, token = await self._redis.acquire_lock(lock_key, ttl_seconds=10)
        if not acquired:
            logger.warning("review_queue=reject_lock_failed hash=%s", query_hash)
            return
        try:
            entry = await self._store.get(query_hash)
            if entry is None or entry.status == CacheStatus.REJECTED:
                return

            # Auto-approved entries are terminal — just clean up the queue.
            if entry.status == CacheStatus.APPROVED:
                await self.remove_from_queue(query_hash)
                logger.info("review_queue=skipped_approved hash=%s", query_hash)
                return

            validate_transition(entry.status, CacheStatus.REJECTED)
            await self._store.update_status_under_lock(query_hash, CacheStatus.REJECTED)
            await self.remove_from_queue(query_hash)
            logger.info("review_queue=rejected hash=%s", query_hash)
        finally:
            await self._redis.release_lock(lock_key, token)

    async def remove_from_queue(self, query_hash: str) -> None:
        """Remove a hash from the pending sorted set (idempotent)."""
        await self._redis.client.zrem(self._pending_key(), query_hash)
