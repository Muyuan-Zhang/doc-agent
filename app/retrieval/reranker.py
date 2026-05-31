import asyncio
import json
import logging
import time

from app.clients.llm import AbstractLLMClient
from app.core.config import settings
from app.models.chunk import ChunkSchema

logger = logging.getLogger(__name__)

_RERANK_SYSTEM = (
    "You are a relevance ranking assistant.\n"
    "Rank the following document chunks by relevance to the query.\n"
    'Return ONLY a JSON object: {"ranked_indices": [2, 0, 1, ...]}\n'
    "Indices are 0-based and refer to the chunks listed below.\n"
)


class LLMReranker:
    def __init__(self, llm: AbstractLLMClient) -> None:
        self._llm = llm
        self._sem = asyncio.Semaphore(settings.llm_semaphore_limits.interactive)

    async def rerank(
        self, query: str, chunks: list[ChunkSchema], top_n: int
    ) -> list[ChunkSchema]:
        t0 = time.perf_counter()
        logger.info("reranker=enter chunks=%d top_n=%d query=%.80s", len(chunks), top_n, query)

        if not chunks:
            logger.info("reranker=skip reason=no_chunks elapsed=%.3fs", time.perf_counter() - t0)
            return []
        if len(chunks) == 1:
            logger.info("reranker=skip reason=single_chunk elapsed=%.3fs", time.perf_counter() - t0)
            return chunks[:top_n]

        chunks_text = "\n".join(
            f"[{i}] <<<{c.content[:500]}>>>" for i, c in enumerate(chunks)
        )
        prompt = (
            _RERANK_SYSTEM
            + f"\nQuery: <<<{query}>>>\n\nChunks:\n{chunks_text}\n\nReturn JSON only."
        )

        llm_t0 = time.perf_counter()
        try:
            async with self._sem:
                response = await self._llm.complete(prompt)
        except Exception as exc:
            llm_elapsed = time.perf_counter() - llm_t0
            logger.warning("reranker=llm_failed error=%s elapsed=%.3fs — falling back", exc, llm_elapsed)
            return chunks[:top_n]
        llm_elapsed = time.perf_counter() - llm_t0

        try:
            data = json.loads(response)
            indices = data["ranked_indices"]
            if not isinstance(indices, list):
                raise ValueError("ranked_indices must be a list")
            valid = [i for i in indices if isinstance(i, int) and 0 <= i < len(chunks)]
            reranked = [chunks[i] for i in valid]
            mentioned = set(valid)
            for i, c in enumerate(chunks):
                if i not in mentioned:
                    reranked.append(c)
            result = reranked[:top_n]
            elapsed = time.perf_counter() - t0
            logger.info(
                "reranker=done input=%d output=%d llm_ms=%.1f elapsed=%.3fs",
                len(chunks), len(result), llm_elapsed * 1000, elapsed,
            )
            return result
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.warning("reranker=parse_failed error=%s elapsed=%.3fs — falling back", exc, elapsed)
            return chunks[:top_n]
