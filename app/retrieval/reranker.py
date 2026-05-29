import asyncio
import json
import logging

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
        if not chunks:
            return []
        if len(chunks) == 1:
            return chunks[:top_n]

        chunks_text = "\n".join(
            f"[{i}] <<<{c.content[:500]}>>>" for i, c in enumerate(chunks)
        )
        prompt = (
            _RERANK_SYSTEM
            + f"\nQuery: <<<{query}>>>\n\nChunks:\n{chunks_text}\n\nReturn JSON only."
        )

        try:
            async with self._sem:
                response = await self._llm.complete(prompt)
        except Exception as exc:
            logger.warning("LLM reranker call failed, falling back to original order: %s", exc)
            return chunks[:top_n]

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
            return reranked[:top_n]
        except Exception as exc:
            logger.warning("LLM reranker parsing failed, falling back to original order: %s", exc)
            return chunks[:top_n]
