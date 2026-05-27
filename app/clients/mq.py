import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator

import redis.asyncio as aioredis

from app.clients.base import AbstractClient
from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class MQMessage:
    id: str
    data: dict
    stream: str


class AbstractMQClient(AbstractClient):
    @abstractmethod
    async def publish(self, stream: str, data: dict) -> str:
        """发布消息，返回 message id。"""

    @abstractmethod
    async def consume(
        self,
        stream: str,
        group: str,
        consumer: str,
        count: int = 10,
        block_ms: int = 2000,
    ) -> AsyncIterator[MQMessage]: ...

    @abstractmethod
    async def ack(self, stream: str, group: str, message_id: str) -> None: ...

    @abstractmethod
    async def ensure_group(self, stream: str, group: str) -> None:
        """创建 Consumer Group（幂等）。"""


class RedisStreamsMQClient(AbstractMQClient):
    """
    Redis Streams 初版 MQ 实现。
    M4 可透明替换为 RabbitMQ / Kafka 实现，只需实现 AbstractMQClient 接口。
    """

    def __init__(self) -> None:
        self._client: aioredis.Redis | None = None
        self._stream = settings.mq_stream_name
        self._group = settings.mq_consumer_group
        self._consumer = settings.mq_consumer_name

    @property
    def client(self) -> aioredis.Redis:
        if self._client is None:
            raise RuntimeError("RedisStreamsMQClient not connected")
        return self._client

    async def connect(self) -> None:
        self._client = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        await self.ensure_group(self._stream, self._group)
        logger.info(
            "MQ connected stream=%s group=%s consumer=%s",
            self._stream, self._group, self._consumer,
        )

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def ping(self) -> bool:
        try:
            await self.client.xlen(self._stream)
            return True
        except Exception as exc:
            logger.warning("MQ ping failed: %s", exc)
            return False

    async def ensure_group(self, stream: str, group: str) -> None:
        try:
            await self.client.xgroup_create(stream, group, id="0", mkstream=True)
        except aioredis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def publish(self, stream: str, data: dict) -> str:
        msg_id = await self.client.xadd(stream, data)
        return msg_id

    async def consume(
        self,
        stream: str,
        group: str,
        consumer: str,
        count: int = 10,
        block_ms: int = 2000,
    ) -> AsyncIterator[MQMessage]:
        entries = await self.client.xreadgroup(
            groupname=group,
            consumername=consumer,
            streams={stream: ">"},
            count=count,
            block=block_ms,
        )
        if not entries:
            return
        for _, messages in entries:
            for msg_id, fields in messages:
                yield MQMessage(id=msg_id, data=fields, stream=stream)

    async def ack(self, stream: str, group: str, message_id: str) -> None:
        await self.client.xack(stream, group, message_id)
