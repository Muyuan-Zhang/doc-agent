import asyncio
import contextlib
import logging

from fastapi import FastAPI

from app.agent.consumer import run_consumer
from app.agent.graph import build_graph
from app.cache.service import RagCacheService
from app.clients.llm import OpenAILLMClient
from app.clients.milvus import MilvusClient
from app.clients.mq import ConsistencyMQClient, RedisStreamsMQClient
from app.clients.postgresql import PostgreSQLClient
from app.clients.redis import RedisClient
from app.consistency.consumer import ConsistencyConsumer
from app.consistency.invalidator import CacheInvalidator
from app.consistency.service import ConsistencyService
from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.logging_config import setup_logging
from app.middleware.registry import register_middlewares
from app.retrieval.bm25 import BM25Strategy
from app.retrieval.hybrid import ConcreteHybridRetriever
from app.retrieval.reranker import LLMReranker
from app.retrieval.router import router as retrieval_router
from app.retrieval.vector import VectorStrategy
from app.routers.agent import router as agent_router
from app.routers.cache import router as cache_router
from app.routers.health import router as health_router
from app.routers.knowledge_base import router as kb_router
from app.routers.memory import router as memory_router

logger = logging.getLogger(__name__)


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    setup_logging()

    # Sequential connect so partial-startup cleanup is deterministic.
    # Each connected client is tracked; on failure, already-connected
    # clients are disconnected in reverse order before re-raising.
    named_clients = [
        ("postgres",        PostgreSQLClient()),
        ("redis",           RedisClient()),
        ("milvus",          MilvusClient()),
        ("mq",              RedisStreamsMQClient()),
        ("consistency_mq",  ConsistencyMQClient()),
        ("llm",             OpenAILLMClient()),
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

    invalidator = CacheInvalidator(app.state.redis, settings.cache_rag_namespace)
    consumer = ConsistencyConsumer(
        app.state.consistency_mq, invalidator, settings.knowledge_base_version
    )
    consistency_service = ConsistencyService(consumer)
    app.state.consistency_service = consistency_service
    await consistency_service.start()

    await app.state.milvus.ensure_kb_collection()
    await app.state.milvus.ensure_memory_collection()

    bm25 = BM25Strategy(pg=app.state.postgres)
    vector = VectorStrategy(milvus=app.state.milvus, llm=app.state.llm)
    reranker = LLMReranker(llm=app.state.llm)
    app.state.retriever = ConcreteHybridRetriever(bm25=bm25, vector=vector, reranker=reranker)
    app.state.cache_svc = RagCacheService(redis=app.state.redis, llm=app.state.llm)

    graph = build_graph(
        llm=app.state.llm,
        retriever=app.state.retriever,
        redis=app.state.redis,
    )
    consumer_task = asyncio.create_task(
        run_consumer(app.state.mq, graph, app.state.redis)
    )
    app.state.consumer_task = consumer_task

    yield

    consumer_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await consumer_task

    await app.state.consistency_service.stop()

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
    app.include_router(memory_router)
    app.include_router(cache_router)
    return app
