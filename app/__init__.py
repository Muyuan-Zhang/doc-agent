import contextlib
import logging

from fastapi import FastAPI

from app.clients.milvus import MilvusClient
from app.clients.mq import RedisStreamsMQClient
from app.clients.mysql import MySQLClient
from app.clients.redis import RedisClient
from app.exceptions import register_exception_handlers
from app.logging_config import setup_logging
from app.middleware.registry import register_middlewares
from app.routers.agent import router as agent_router
from app.routers.health import router as health_router

logger = logging.getLogger(__name__)


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    setup_logging()

    # Sequential connect so partial-startup cleanup is deterministic.
    # Each connected client is tracked; on failure, already-connected
    # clients are disconnected in reverse order before re-raising.
    named_clients = [
        ("mysql",  MySQLClient()),
        ("redis",  RedisClient()),
        ("milvus", MilvusClient()),
        ("mq",     RedisStreamsMQClient()),
    ]

    connected: list = []
    try:
        for name, client in named_clients:
            await client.connect()
            setattr(app.state, name, client)
            connected.append(client)
    except Exception:
        for client in reversed(connected):
            try:
                await client.disconnect()
            except Exception as exc:
                logger.warning("Error during startup-failure cleanup: %s", exc)
        raise

    yield

    for client in reversed(connected):
        try:
            await client.disconnect()
        except Exception as exc:
            logger.warning("Error during shutdown: %s", exc)


def create_app() -> FastAPI:
    app = FastAPI(title="doc-agent", lifespan=_lifespan)
    register_middlewares(app)
    register_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(agent_router)
    return app
