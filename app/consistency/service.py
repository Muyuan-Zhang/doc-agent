import asyncio
import logging

from app.consistency.consumer import ConsistencyConsumer

logger = logging.getLogger(__name__)

_BASE_BACKOFF = 1.0
_MAX_BACKOFF = 60.0
_MAX_CONSECUTIVE_ERRORS = 10


class ConsistencyService:
    def __init__(self, consumer: ConsistencyConsumer) -> None:
        self._consumer = consumer
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def _loop(self) -> None:
        backoff = _BASE_BACKOFF
        consecutive = 0
        while True:
            try:
                await self._consumer.run_once()
                backoff = _BASE_BACKOFF
                consecutive = 0
            except Exception as exc:
                consecutive += 1
                if consecutive >= _MAX_CONSECUTIVE_ERRORS:
                    logger.critical(
                        "ConsistencyService: %d consecutive failures, last: %s",
                        consecutive, exc,
                    )
                else:
                    logger.error("ConsistencyService loop error: %s", exc)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF)
