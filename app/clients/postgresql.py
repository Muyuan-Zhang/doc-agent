import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.clients.base import AbstractClient
from app.core.config import settings

logger = logging.getLogger(__name__)


class PostgreSQLClient(AbstractClient):
    def __init__(self) -> None:
        self._engine: AsyncEngine | None = None

    @property
    def engine(self) -> AsyncEngine:
        if self._engine is None:
            raise RuntimeError("PostgreSQLClient not connected")
        return self._engine

    async def connect(self) -> None:
        self._engine = create_async_engine(
            settings.postgres_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=settings.debug,
        )
        logger.info("PostgreSQL connection pool created")

    async def disconnect(self) -> None:
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            logger.info("PostgreSQL connection pool disposed")

    async def ping(self) -> bool:
        try:
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception as exc:
            logger.warning("PostgreSQL ping failed: %s", exc)
            return False
