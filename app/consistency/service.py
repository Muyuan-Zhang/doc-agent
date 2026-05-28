import asyncio
import logging

from app.consistency.consumer import ConsistencyConsumer

logger = logging.getLogger(__name__)


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
        while True:
            try:
                await self._consumer.run_once()
            except Exception as exc:
                logger.error("ConsistencyService loop error: %s", exc)
