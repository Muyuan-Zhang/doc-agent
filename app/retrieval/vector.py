import asyncio
import logging

from app.clients.llm import AbstractLLMClient
from app.clients.milvus import MilvusClient
from app.core.config import settings
from app.models.chunk import ChunkSchema

logger = logging.getLogger(__name__)

_OUTPUT_FIELDS = [
    "chunk_id", "doc_id", "section_id", "chunk_index",
    "version", "content", "content_hash",
]


class VectorStrategy:
    def __init__(self, milvus: MilvusClient, llm: AbstractLLMClient) -> None:
        self._milvus = milvus
        self._llm = llm
        self._sem = asyncio.Semaphore(settings.llm_semaphore_limits.interactive)

    async def retrieve(self, query: str, top_k: int, **kwargs) -> list[ChunkSchema]:
        async with self._sem:
            embedding = await self._llm.embed(query)
        hits = await self._milvus.search(
            embedding=embedding,
            top_k=top_k,
            output_fields=_OUTPUT_FIELDS,
        )
        return [
            ChunkSchema(
                doc_id=hit["doc_id"],
                section_id=hit["section_id"],
                chunk_index=hit["chunk_index"],
                content_hash=hit["content_hash"],
                version=hit["version"],
                content=hit["content"],
            )
            for hit in hits
        ]
