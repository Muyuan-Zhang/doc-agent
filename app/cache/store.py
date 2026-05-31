import logging

from app.cache.quality import cosine_similarity
from app.cache.schemas import CacheEntry, CacheStatus, validate_transition
from app.clients.redis import RedisClient
from app.core.config import Settings, settings as _default_settings

logger = logging.getLogger(__name__)

_SCAN_COUNT = 100
_STATS_HASH_KEY = "stats"
_THRESHOLD_EPSILON = 1e-9  # strict-greater comparison tolerance


class RagCacheStore:
    def __init__(self, redis: RedisClient, cfg: Settings | None = None) -> None:
        self._redis = redis
        self._cfg = cfg or _default_settings

    def _entry_key(self, query_hash: str) -> str:
        return self._redis.cache_key("rag_cache", query_hash)

    def _lock_key(self, query_hash: str) -> str:
        return self._redis.cache_key("rag_cache_lock", query_hash)

    def _stats_key(self) -> str:
        return self._redis.cache_key(_STATS_HASH_KEY, "counters")

    async def get(self, query_hash: str) -> CacheEntry | None:
        raw = await self._redis.client.get(self._entry_key(query_hash))
        if raw is None:
            return None
        try:
            return CacheEntry.model_validate_json(raw)
        except Exception as exc:
            logger.warning("cache_store=deserialize_failed hash=%s error=%s", query_hash, exc)
            return None

    async def get_many(self, hashes: list[str]) -> list[CacheEntry | None]:
        """Fetch multiple entries in a single pipeline round-trip."""
        if not hashes:
            return []
        pipe = self._redis.client.pipeline()
        for h in hashes:
            pipe.get(self._entry_key(h))
        raws = await pipe.execute()
        results: list[CacheEntry | None] = []
        for h, raw in zip(hashes, raws):
            if raw is None:
                results.append(None)
                continue
            try:
                results.append(CacheEntry.model_validate_json(raw))
            except Exception as exc:
                logger.warning("cache_store=deserialize_failed hash=%s error=%s", h, exc)
                results.append(None)
        return results

    async def set(self, entry: CacheEntry, ttl: int) -> None:
        await self._redis.client.setex(
            self._entry_key(entry.query_hash), ttl, entry.model_dump_json()
        )
        logger.info(
            "cache_store=set hash=%s status=%s ttl=%d",
            entry.query_hash, entry.status.value, ttl,
        )

    async def update_status_under_lock(
        self,
        query_hash: str,
        new_status: CacheStatus,
        approval_count: int | None = None,
        approved_by: list[str] | None = None,
    ) -> bool:
        """Apply a status update without acquiring the lock. Caller MUST hold the lock."""
        existing = await self.get(query_hash)
        if existing is None:
            return False
        validate_transition(existing.status, new_status)
        data = existing.model_dump()
        data["status"] = new_status
        if approval_count is not None:
            data["approval_count"] = approval_count
        if approved_by is not None:
            data["approved_by"] = approved_by
        updated = CacheEntry.model_validate(data)
        ttl = await self._redis.client.ttl(self._entry_key(query_hash))
        if ttl == -2:
            # Entry expired between get() and TTL lookup — do not reanimate
            return False
        await self.set(updated, ttl if ttl > 0 else self._cfg.cache_ttl_seconds)
        return True

    async def update_status(
        self,
        query_hash: str,
        new_status: CacheStatus,
        approval_count: int | None = None,
        approved_by: list[str] | None = None,
    ) -> bool:
        acquired, token = await self._redis.acquire_lock(self._lock_key(query_hash), ttl_seconds=10)
        if not acquired:
            logger.warning("cache_store=update_status_lock_failed hash=%s", query_hash)
            return False
        try:
            return await self.update_status_under_lock(
                query_hash, new_status, approval_count, approved_by
            )
        finally:
            await self._redis.release_lock(self._lock_key(query_hash), token)

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

    async def increment_stat(self, stat: str) -> None:
        try:
            await self._redis.client.hincrby(self._stats_key(), stat, 1)
        except Exception as exc:
            logger.warning("cache_store=stat_failed stat=%s error=%s", stat, exc)

    async def get_stats(self) -> dict:
        pending_key = self._redis.cache_key("review", "pending")
        pipe = self._redis.client.pipeline()
        pipe.hgetall(self._stats_key())
        pipe.zcard(pending_key)
        raw_stats, pending = await pipe.execute()
        return {
            "hits": int(raw_stats.get("hits", 0)),
            "misses": int(raw_stats.get("misses", 0)),
            "auto_approved": int(raw_stats.get("auto_approved", 0)),
            "pending": int(pending),
        }

    async def search_by_embedding(
        self,
        query_embedding: list[float],
        threshold: float,
    ) -> "CacheEntry | None":
        """Scan approved entries and return the best cosine-similarity match.

        Returns None when no entry exceeds the threshold or has a stored
        query_embedding.  Iterates via SCAN to avoid blocking Redis.
        """
        pattern = self._redis.cache_key("rag_cache", "*")
        best_entry: CacheEntry | None = None
        best_score: float = threshold - _THRESHOLD_EPSILON
        cursor = 0
        while True:
            cursor, keys = await self._redis.client.scan(
                cursor, match=pattern, count=_SCAN_COUNT
            )
            if keys:
                pipe = self._redis.client.pipeline()
                for raw_key in keys:
                    pipe.get(raw_key)
                raws = await pipe.execute()
                for raw in raws:
                    if raw is None:
                        continue
                    try:
                        entry = CacheEntry.model_validate_json(raw)
                    except Exception:
                        continue
                    if entry.status != CacheStatus.APPROVED:
                        continue
                    if not entry.query_embedding:
                        continue
                    try:
                        score = cosine_similarity(query_embedding, entry.query_embedding)
                    except ValueError:
                        continue
                    if score > best_score:
                        best_score = score
                        best_entry = entry
            if cursor == 0:
                break
        return best_entry

    async def get_ttl(self, query_hash: str) -> int:
        """Return the remaining TTL in seconds for a cache entry (-2 if missing)."""
        return await self._redis.client.ttl(self._entry_key(query_hash))
