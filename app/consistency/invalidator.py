import logging

from app.clients.redis import RedisClient
from app.core.config import settings

logger = logging.getLogger(__name__)


class CacheInvalidator:
    def __init__(
        self,
        redis: RedisClient,
        namespace: str,
        batch: int | None = None,
    ) -> None:
        self._redis = redis
        self._namespace = namespace
        self._batch = batch if batch is not None else settings.cache_invalidation_scan_batch

    async def invalidate(self, version: str) -> int:
        """SCAN {version}:{namespace}:* and UNLINK matches. Returns count of deleted keys."""
        pattern = f"{version}:{self._namespace}:*"
        cursor = 0
        total = 0
        while True:
            cursor, keys = await self._redis.client.scan(
                cursor, match=pattern, count=self._batch
            )
            if keys:
                await self._redis.client.unlink(*keys)
                total += len(keys)
            if not cursor:
                break
        logger.info("Cache invalidated namespace=%s version=%s deleted=%d", self._namespace, version, total)
        return total
