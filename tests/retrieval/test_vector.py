"""Unit tests for VectorStrategy — HNSW vector retrieval via Milvus."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.chunk import ChunkSchema
from app.retrieval.vector import VectorStrategy


def _make_milvus(hits=None):
    m = MagicMock()
    m.search = AsyncMock(return_value=hits or [])
    return m


def _make_llm(embedding=None):
    llm = MagicMock()
    llm.embed = AsyncMock(return_value=embedding or [0.1, 0.2, 0.3])
    return llm


def _hit(
    doc_id="d1",
    section_id="s0",
    chunk_index=0,
    version="v1",
    content="hello world",
    content_hash="h0",
    chunk_id="d1:s0:000000",
):
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "section_id": section_id,
        "chunk_index": chunk_index,
        "version": version,
        "content": content,
        "content_hash": content_hash,
    }


class TestVectorStrategyRetrieve:
    async def test_calls_llm_embed_with_query(self):
        milvus = _make_milvus()
        llm = _make_llm()
        await VectorStrategy(milvus=milvus, llm=llm).retrieve("search query", top_k=5)
        llm.embed.assert_awaited_once_with("search query")

    async def test_calls_milvus_search(self):
        milvus = _make_milvus()
        llm = _make_llm()
        await VectorStrategy(milvus=milvus, llm=llm).retrieve("query", top_k=3)
        milvus.search.assert_awaited_once()

    async def test_passes_embedding_to_milvus(self):
        embedding = [0.5, 0.6, 0.7]
        milvus = _make_milvus()
        llm = _make_llm(embedding=embedding)
        await VectorStrategy(milvus=milvus, llm=llm).retrieve("query", top_k=3)
        _, kwargs = milvus.search.call_args
        assert kwargs["embedding"] == embedding

    async def test_passes_top_k_to_milvus(self):
        milvus = _make_milvus()
        llm = _make_llm()
        await VectorStrategy(milvus=milvus, llm=llm).retrieve("query", top_k=7)
        _, kwargs = milvus.search.call_args
        assert kwargs["top_k"] == 7

    async def test_empty_hits_returns_empty(self):
        milvus = _make_milvus(hits=[])
        llm = _make_llm()
        result = await VectorStrategy(milvus=milvus, llm=llm).retrieve("query", top_k=5)
        assert result == []

    async def test_returns_chunk_schemas(self):
        milvus = _make_milvus(hits=[_hit()])
        llm = _make_llm()
        result = await VectorStrategy(milvus=milvus, llm=llm).retrieve("query", top_k=5)
        assert len(result) == 1
        assert isinstance(result[0], ChunkSchema)

    async def test_doc_id_populated_from_hit(self):
        milvus = _make_milvus(hits=[_hit(doc_id="doc-abc")])
        llm = _make_llm()
        result = await VectorStrategy(milvus=milvus, llm=llm).retrieve("query", top_k=5)
        assert result[0].doc_id == "doc-abc"

    async def test_content_populated_from_hit(self):
        milvus = _make_milvus(hits=[_hit(content="vector content")])
        llm = _make_llm()
        result = await VectorStrategy(milvus=milvus, llm=llm).retrieve("query", top_k=5)
        assert result[0].content == "vector content"

    async def test_content_hash_populated_from_hit(self):
        milvus = _make_milvus(hits=[_hit(content_hash="abc123")])
        llm = _make_llm()
        result = await VectorStrategy(milvus=milvus, llm=llm).retrieve("query", top_k=5)
        assert result[0].content_hash == "abc123"

    async def test_multiple_hits_all_returned(self):
        hits = [_hit(content_hash=f"h{i}", content=f"text{i}") for i in range(4)]
        milvus = _make_milvus(hits=hits)
        llm = _make_llm()
        result = await VectorStrategy(milvus=milvus, llm=llm).retrieve("query", top_k=10)
        assert len(result) == 4
