import asyncio
import logging
import time

from app.core.config import Settings, settings as _default_settings
from app.models.chunk import ChunkSchema
from app.retrieval.bm25 import BM25Strategy
from app.retrieval.reranker import LLMReranker
from app.retrieval.rrf import rrf_fuse
from app.retrieval.vector import VectorStrategy

logger = logging.getLogger(__name__)


class ConcreteHybridRetriever:
    """BM25 + HNSW → RRF → LLM reranking pipeline.

    Satisfies RetrievalStrategy Protocol (duck-typed; no import cycle with models/).
    """

    def __init__(
        self,
        bm25: BM25Strategy,
        vector: VectorStrategy,
        reranker: LLMReranker,
        settings: Settings | None = None,
    ) -> None:
        self._bm25 = bm25
        self._vector = vector
        self._reranker = reranker
        self._settings = settings if settings is not None else _default_settings

    async def retrieve(self, query: str, top_k: int, **kwargs) -> list[ChunkSchema]:
        t0 = time.perf_counter()
        logger.info("hybrid_retriever=enter query=%.80s top_k=%d", query, top_k)

        results = await asyncio.gather(
            self._bm25.retrieve(query, self._settings.bm25_top_k),
            self._vector.retrieve(query, self._settings.vector_top_k),
            return_exceptions=True,
        )
        retrieve_ms = (time.perf_counter() - t0) * 1000

        ranked_lists: list[list[ChunkSchema]] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                name = "BM25" if i == 0 else "Vector"
                logger.error("hybrid_retriever=%s_failed error=%s", name.lower(), result)
            else:
                ranked_lists.append(result)
                logger.info(
                    "hybrid_retriever=%s_ok chunks=%d",
                    "bm25" if i == 0 else "vector",
                    len(result),
                )

        if not ranked_lists:
            raise RuntimeError("All retrieval strategies failed")

        effective_top_k = min(top_k, self._settings.final_top_k)
        fused = rrf_fuse(ranked_lists, k=self._settings.rrf_k)
        logger.info(
            "hybrid_retriever=fused fused=%d rrf_ms=%.1f",
            len(fused), (time.perf_counter() - t0) * 1000 - retrieve_ms,
        )

        candidates = fused[: self._settings.rerank_top_n]
        reranked = await self._reranker.rerank(query, candidates, top_n=effective_top_k)

        elapsed = time.perf_counter() - t0
        logger.info(
            "hybrid_retriever=done result=%d retrieved_ms=%.1f total_ms=%.1f",
            len(reranked), retrieve_ms, elapsed * 1000,
        )
        return reranked[:effective_top_k]
