import logging

from app.clients.mq import AbstractMQClient
from app.consistency.invalidator import CacheInvalidator

logger = logging.getLogger(__name__)

_INVALIDATE_EVENTS = frozenset({"kb_updated", "kb_deleted"})


class ConsistencyConsumer:
    def __init__(
        self,
        mq: AbstractMQClient,
        invalidator: CacheInvalidator,
        version: str,
    ) -> None:
        self._mq = mq
        self._invalidator = invalidator
        self._version = version

    async def run_once(self) -> int:
        """Process one batch of MQ messages. Returns count of invalidation events handled."""
        count = 0
        async for msg in self._mq.consume():
            event = msg.data.get("event")
            if event in _INVALIDATE_EVENTS:
                version = msg.data.get("version", self._version)
                await self._invalidator.invalidate(version)
                count += 1
            await self._mq.ack(msg.id)
        return count
