import logging

from app.cache.schemas import CacheEntry, CacheStatus
from app.clients.redis import RedisClient

logger = logging.getLogger(__name__)

_SCAN_COUNT = 100
_FALLBACK_TTL = 3600


class RagCacheStore:
    def __init__(self, redis: RedisClient) -> None:
        self._redis = redis

    def _entry_key(self, query_hash: str) -> str:
        return self._redis.cache_key("rag_cache", query_hash)

    async def get(self, query_hash: str) -> CacheEntry | None:
        raw = await self._redis.client.get(self._entry_key(query_hash))
        if raw is None:
            return None
        try:
            return CacheEntry.model_validate_json(raw)
        except Exception as exc:
            logger.warning("cache_store=deserialize_failed hash=%s error=%s", query_hash, exc)
            return None

    async def set(self, entry: CacheEntry, ttl: int) -> None:
        await self._redis.client.setex(
            self._entry_key(entry.query_hash), ttl, entry.model_dump_json()
        )
        logger.info(
            "cache_store=set hash=%s status=%s ttl=%d",
            entry.query_hash, entry.status.value, ttl,
        )

    async def update_status(
        self,
        query_hash: str,
        new_status: CacheStatus,
        approval_count: int | None = None,
        approved_by: list[str] | None = None,
    ) -> bool:
        existing = await self.get(query_hash)
        if existing is None:
            return False
        data = existing.model_dump()
        data["status"] = new_status
        if approval_count is not None:
            data["approval_count"] = approval_count
        if approved_by is not None:
            data["approved_by"] = approved_by
        updated = CacheEntry.model_validate(data)
        ttl = await self._redis.client.ttl(self._entry_key(query_hash))
        await self.set(updated, ttl if ttl > 0 else _FALLBACK_TTL)
        return True

    async def delete(self, query_hash: str) -> bool:
        result = await self._redis.client.delete(self._entry_key(query_hash))
        return bool(result)

    async def invalidate_all(self) -> int:
        """Delete all rag_cache entries. Called by M6 on KB version bump."""
        pattern = self._redis.cache_key("rag_cache", "*")
        deleted = 0
        cursor = 0
        while True:
            cursor, keys = await self._redis.client.scan(
                cursor, match=pattern, count=_SCAN_COUNT
            )
            if keys:
                await self._redis.client.delete(*keys)
                deleted += len(keys)
            if cursor == 0:
                break
        logger.info("cache_store=invalidate_all deleted=%d", deleted)
        return deleted

    async def get_stats(self) -> dict:
        hits_key = self._redis.cache_key("stats", "hits")
        misses_key = self._redis.cache_key("stats", "misses")
        pending_key = self._redis.cache_key("review", "pending")
        hits = await self._redis.client.get(hits_key) or "0"
        misses = await self._redis.client.get(misses_key) or "0"
        pending = await self._redis.client.llen(pending_key)
        return {
            "hits": int(hits),
            "misses": int(misses),
            "pending": int(pending),
        }
