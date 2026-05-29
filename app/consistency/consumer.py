import logging
import re

from app.clients.mq import AbstractMQClient
from app.consistency.invalidator import CacheInvalidator

logger = logging.getLogger(__name__)

_INVALIDATE_EVENTS = frozenset({"kb_updated", "kb_deleted"})
_VERSION_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")


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
                version_raw = msg.data.get("version", self._version)
                if not _VERSION_RE.fullmatch(version_raw):
                    logger.warning(
                        "ConsistencyConsumer: rejected invalid version %r in message %s",
                        version_raw, msg.id,
                    )
                else:
                    await self._invalidator.invalidate(version_raw)
                    count += 1
            else:
                logger.warning(
                    "ConsistencyConsumer: unknown event type %r in message %s — skipping",
                    event, msg.id,
                )
            await self._mq.ack(msg.id)
        return count
