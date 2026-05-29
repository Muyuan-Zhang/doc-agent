import asyncio
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
        max_iterations: int | None = None,
    ) -> None:
        self._redis = redis
        self._namespace = namespace
        self._batch = batch if batch is not None else settings.cache_invalidation_scan_batch
        self._max_iterations = (
            max_iterations if max_iterations is not None
            else settings.cache_invalidation_max_iterations
        )

    async def invalidate(self, version: str) -> int:
        """SCAN {version}:{namespace}:* and UNLINK matches. Returns count of deleted keys."""
        pattern = f"{version}:{self._namespace}:*"
        cursor = 0
        total = 0
        for _ in range(self._max_iterations):
            cursor, keys = await self._redis.client.scan(
                cursor, match=pattern, count=self._batch
            )
            if keys:
                await self._redis.client.unlink(*keys)
                total += len(keys)
            if not cursor:
                break
            await asyncio.sleep(0)  # yield event loop between non-final batches
        else:
            logger.error(
                "invalidate: exceeded max_iterations=%d namespace=%s version=%s "
                "— cache partially invalidated",
                self._max_iterations, self._namespace, version,
            )
        logger.info(
            "Cache invalidated namespace=%s version=%s deleted=%d",
            self._namespace, version, total,
        )
        return total
