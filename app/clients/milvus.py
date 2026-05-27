import asyncio
import logging
from typing import Any, Callable, TypeVar

from pymilvus import connections, utility

from app.clients.base import AbstractClient
from app.core.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


class MilvusClient(AbstractClient):
    """
    Milvus 客户端。所有 collection 操作通过 alias 路由，
    调用方不得绕过此类直接传入 collection_name。
    """

    def __init__(self) -> None:
        self._alias = settings.milvus_alias

    async def _run_sync(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """所有 pymilvus 同步调用的唯一入口。禁止在此类其他方法中直接调用 asyncio.to_thread。"""
        return await asyncio.to_thread(fn, *args, **kwargs)

    async def connect(self) -> None:
        await self._run_sync(
            connections.connect,
            alias=self._alias,
            host=settings.milvus_host,
            port=settings.milvus_port,
        )
        logger.info("Milvus connected via alias=%s", self._alias)

    async def disconnect(self) -> None:
        await self._run_sync(connections.disconnect, alias=self._alias)
        logger.info("Milvus disconnected alias=%s", self._alias)

    async def ping(self) -> bool:
        try:
            await self._run_sync(utility.get_server_version, using=self._alias)
            return True
        except Exception as exc:
            logger.warning("Milvus ping failed: %s", exc)
            return False
