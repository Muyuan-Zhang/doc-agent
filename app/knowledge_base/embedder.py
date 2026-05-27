from app.clients.llm import AbstractLLMClient
from app.models.chunk import ChunkSchema


class ChunkEmbedder:
    def __init__(self, llm: AbstractLLMClient, batch_size: int) -> None:
        self._llm = llm
        self._batch_size = batch_size

    async def embed(self, chunks: list[ChunkSchema]) -> list[ChunkSchema]:
        if not chunks:
            return []
        results: list[ChunkSchema] = []
        for i in range(0, len(chunks), self._batch_size):
            batch = chunks[i : i + self._batch_size]
            texts = [c.content for c in batch]
            embeddings = await self._llm.embed_batch(texts)
            for chunk, vec in zip(batch, embeddings):
                results.append(chunk.model_copy(update={"embedding": vec}))
        return results
