import logging

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy import text

from app.clients.base import AbstractClient
from app.config import settings

logger = logging.getLogger(__name__)


class MySQLClient(AbstractClient):
    def __init__(self) -> None:
        self._engine: AsyncEngine | None = None

    @property
    def engine(self) -> AsyncEngine:
        if self._engine is None:
            raise RuntimeError("MySQLClient not connected")
        return self._engine

    async def connect(self) -> None:
        self._engine = create_async_engine(
            settings.mysql_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=settings.debug,
        )
        logger.info("MySQL connection pool created")

    async def disconnect(self) -> None:
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            logger.info("MySQL connection pool disposed")

    async def ping(self) -> bool:
        try:
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception as exc:
            logger.warning("MySQL ping failed: %s", exc)
            return False
