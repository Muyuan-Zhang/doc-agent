"""Unit tests for ChunkEmbedder."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.knowledge_base.embedder import ChunkEmbedder
from app.models.chunk import ChunkSchema


def _chunk(content: str, idx: int = 0) -> ChunkSchema:
    return ChunkSchema(
        doc_id="d1",
        section_id="s0000",
        chunk_index=idx,
        content_hash=f"hash{idx}",
        version="v1",
        content=content,
    )


def _make_llm(embeddings: list[list[float]] | None = None) -> MagicMock:
    llm = MagicMock()
    llm.embed_batch = AsyncMock(return_value=embeddings or [[0.1, 0.2]])
    return llm


class TestChunkEmbedderEmbed:
    async def test_empty_input_returns_empty(self):
        result = await ChunkEmbedder(llm=_make_llm(), batch_size=10).embed([])
        assert result == []

    async def test_embedding_is_populated(self):
        vec = [0.1, 0.2, 0.3]
        llm = _make_llm([[vec[0], vec[1], vec[2]]])
        chunks = [_chunk("hello")]
        result = await ChunkEmbedder(llm=llm, batch_size=10).embed(chunks)
        assert result[0].embedding == vec

    async def test_returns_new_frozen_instances(self):
        llm = _make_llm([[0.1]])
        chunks = [_chunk("hello")]
        result = await ChunkEmbedder(llm=llm, batch_size=10).embed(chunks)
        assert result[0] is not chunks[0]

    async def test_original_chunks_unchanged(self):
        llm = _make_llm([[0.1]])
        chunks = [_chunk("hello")]
        await ChunkEmbedder(llm=llm, batch_size=10).embed(chunks)
        assert chunks[0].embedding is None

    async def test_all_chunks_get_embeddings(self):
        vecs = [[float(i)] for i in range(5)]
        llm = _make_llm(vecs)
        chunks = [_chunk(f"text{i}", i) for i in range(5)]
        result = await ChunkEmbedder(llm=llm, batch_size=10).embed(chunks)
        assert all(c.embedding is not None for c in result)

    async def test_batching_splits_correctly(self):
        llm = MagicMock()
        call_sizes: list[int] = []

        async def fake_embed_batch(texts):
            call_sizes.append(len(texts))
            return [[0.1]] * len(texts)

        llm.embed_batch = fake_embed_batch
        chunks = [_chunk(f"t{i}", i) for i in range(7)]
        await ChunkEmbedder(llm=llm, batch_size=3).embed(chunks)
        assert call_sizes == [3, 3, 1]

    async def test_single_batch_calls_embed_batch_once(self):
        llm = _make_llm([[0.1], [0.2]])
        chunks = [_chunk("a", 0), _chunk("b", 1)]
        await ChunkEmbedder(llm=llm, batch_size=10).embed(chunks)
        assert llm.embed_batch.await_count == 1

    async def test_content_is_passed_to_llm(self):
        llm = _make_llm([[0.1]])
        chunks = [_chunk("specific content")]
        await ChunkEmbedder(llm=llm, batch_size=10).embed(chunks)
        texts = llm.embed_batch.call_args[0][0]
        assert texts == ["specific content"]

    async def test_other_chunk_fields_preserved(self):
        llm = _make_llm([[0.1, 0.2]])
        original = _chunk("hello", idx=3)
        result = await ChunkEmbedder(llm=llm, batch_size=10).embed([original])
        out = result[0]
        assert out.doc_id == original.doc_id
        assert out.chunk_index == original.chunk_index
        assert out.content == original.content
        assert out.version == original.version
