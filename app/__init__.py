import contextlib
import logging

from fastapi import FastAPI

from app.clients.llm import OpenAILLMClient
from app.clients.milvus import MilvusClient
from app.clients.mq import RedisStreamsMQClient
from app.clients.postgresql import PostgreSQLClient
from app.clients.redis import RedisClient
from app.core.exceptions import register_exception_handlers
from app.core.logging_config import setup_logging
from app.middleware.registry import register_middlewares
from app.retrieval.bm25 import BM25Strategy
from app.retrieval.hybrid import ConcreteHybridRetriever
from app.retrieval.reranker import LLMReranker
from app.retrieval.router import router as retrieval_router
from app.retrieval.vector import VectorStrategy
from app.routers.agent import router as agent_router
from app.routers.health import router as health_router
from app.routers.knowledge_base import router as kb_router

logger = logging.getLogger(__name__)


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    setup_logging()

    # Sequential connect so partial-startup cleanup is deterministic.
    # Each connected client is tracked; on failure, already-connected
    # clients are disconnected in reverse order before re-raising.
    named_clients = [
        ("postgres", PostgreSQLClient()),
        ("redis",    RedisClient()),
        ("milvus",   MilvusClient()),
        ("mq",       RedisStreamsMQClient()),
        ("llm",      OpenAILLMClient()),
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

    bm25 = BM25Strategy(pg=app.state.postgres)
    vector = VectorStrategy(milvus=app.state.milvus, llm=app.state.llm)
    reranker = LLMReranker(llm=app.state.llm)
    app.state.retriever = ConcreteHybridRetriever(bm25=bm25, vector=vector, reranker=reranker)

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
    app.include_router(kb_router)
    app.include_router(retrieval_router)
    return app
