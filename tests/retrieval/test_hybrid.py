"""Unit tests for ConcreteHybridRetriever — BM25 + Vector → RRF → LLM reranking."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.chunk import ChunkSchema
from app.models.retrieval import RetrievalStrategy
from app.retrieval.hybrid import ConcreteHybridRetriever


def _chunk(hash_: str, content: str = "text", embedding=None) -> ChunkSchema:
    return ChunkSchema(
        doc_id="d1",
        section_id="s0",
        chunk_index=0,
        content_hash=hash_,
        version="v1",
        content=content,
        embedding=embedding,
    )


def _make_strategy(results: list[ChunkSchema] | None = None, fail: bool = False) -> MagicMock:
    s = MagicMock()
    if fail:
        s.retrieve = AsyncMock(side_effect=RuntimeError("strategy failed"))
    else:
        s.retrieve = AsyncMock(return_value=results or [])
    return s


def _make_reranker(results: list[ChunkSchema] | None = None) -> MagicMock:
    r = MagicMock()
    r.rerank = AsyncMock(return_value=results or [])
    return r


def _settings(
    bm25_top_k=5,
    vector_top_k=5,
    rrf_k=60,
    rerank_top_n=3,
    final_top_k=10,
) -> MagicMock:
    s = MagicMock()
    s.bm25_top_k = bm25_top_k
    s.vector_top_k = vector_top_k
    s.rrf_k = rrf_k
    s.rerank_top_n = rerank_top_n
    s.final_top_k = final_top_k
    return s


class TestConcreteHybridRetrieverProtocol:
    def test_satisfies_retrieval_strategy_protocol(self):
        retriever = ConcreteHybridRetriever(
            bm25=_make_strategy(),
            vector=_make_strategy(),
            reranker=_make_reranker(),
            settings=_settings(),
        )
        assert isinstance(retriever, RetrievalStrategy)


class TestConcreteHybridRetrieverRetrieve:
    async def test_calls_bm25_strategy(self):
        bm25 = _make_strategy()
        retriever = ConcreteHybridRetriever(
            bm25=bm25,
            vector=_make_strategy(),
            reranker=_make_reranker(),
            settings=_settings(),
        )
        await retriever.retrieve("query", top_k=3)
        bm25.retrieve.assert_awaited_once()

    async def test_calls_vector_strategy(self):
        vector = _make_strategy()
        retriever = ConcreteHybridRetriever(
            bm25=_make_strategy(),
            vector=vector,
            reranker=_make_reranker(),
            settings=_settings(),
        )
        await retriever.retrieve("query", top_k=3)
        vector.retrieve.assert_awaited_once()

    async def test_passes_bm25_top_k_to_bm25(self):
        bm25 = _make_strategy()
        retriever = ConcreteHybridRetriever(
            bm25=bm25,
            vector=_make_strategy(),
            reranker=_make_reranker(),
            settings=_settings(bm25_top_k=12),
        )
        await retriever.retrieve("q", top_k=3)
        bm25.retrieve.assert_awaited_once_with("q", 12)

    async def test_passes_vector_top_k_to_vector(self):
        vector = _make_strategy()
        retriever = ConcreteHybridRetriever(
            bm25=_make_strategy(),
            vector=vector,
            reranker=_make_reranker(),
            settings=_settings(vector_top_k=8),
        )
        await retriever.retrieve("q", top_k=3)
        vector.retrieve.assert_awaited_once_with("q", 8)

    async def test_calls_reranker_after_fusion(self):
        bm25 = _make_strategy(results=[_chunk("a")])
        vector = _make_strategy(results=[_chunk("b")])
        reranker = _make_reranker(results=[_chunk("a")])
        retriever = ConcreteHybridRetriever(
            bm25=bm25, vector=vector, reranker=reranker, settings=_settings(rerank_top_n=5)
        )
        await retriever.retrieve("query", top_k=2)
        reranker.rerank.assert_awaited_once()

    async def test_reranker_receives_fused_query(self):
        bm25 = _make_strategy(results=[_chunk("a")])
        vector = _make_strategy(results=[_chunk("b")])
        reranker = _make_reranker(results=[_chunk("a")])
        retriever = ConcreteHybridRetriever(
            bm25=bm25, vector=vector, reranker=reranker, settings=_settings()
        )
        await retriever.retrieve("my query", top_k=2)
        call_args = reranker.rerank.call_args[0]
        assert call_args[0] == "my query"

    async def test_returns_top_k_results(self):
        chunks = [_chunk(f"h{i}") for i in range(5)]
        bm25 = _make_strategy(results=chunks)
        reranker = _make_reranker(results=chunks[:2])
        retriever = ConcreteHybridRetriever(
            bm25=bm25,
            vector=_make_strategy(),
            reranker=reranker,
            settings=_settings(rerank_top_n=5),
        )
        result = await retriever.retrieve("query", top_k=2)
        assert len(result) == 2

    async def test_continues_when_bm25_fails(self):
        vector = _make_strategy(results=[_chunk("v1")])
        reranker = _make_reranker(results=[_chunk("v1")])
        retriever = ConcreteHybridRetriever(
            bm25=_make_strategy(fail=True),
            vector=vector,
            reranker=reranker,
            settings=_settings(),
        )
        await retriever.retrieve("query", top_k=1)
        vector.retrieve.assert_awaited_once()

    async def test_continues_when_vector_fails(self):
        bm25 = _make_strategy(results=[_chunk("b1")])
        reranker = _make_reranker(results=[_chunk("b1")])
        retriever = ConcreteHybridRetriever(
            bm25=bm25,
            vector=_make_strategy(fail=True),
            reranker=reranker,
            settings=_settings(),
        )
        await retriever.retrieve("query", top_k=1)
        bm25.retrieve.assert_awaited_once()

    async def test_raises_when_all_strategies_fail(self):
        reranker = _make_reranker()
        retriever = ConcreteHybridRetriever(
            bm25=_make_strategy(fail=True),
            vector=_make_strategy(fail=True),
            reranker=reranker,
            settings=_settings(),
        )
        with pytest.raises(RuntimeError, match="All retrieval strategies failed"):
            await retriever.retrieve("query", top_k=3)
        reranker.rerank.assert_not_awaited()

    async def test_reranker_receives_at_most_rerank_top_n_candidates(self):
        chunks = [_chunk(f"h{i}") for i in range(10)]
        bm25 = _make_strategy(results=chunks)
        reranker = _make_reranker(results=chunks[:3])
        retriever = ConcreteHybridRetriever(
            bm25=bm25,
            vector=_make_strategy(),
            reranker=reranker,
            settings=_settings(rerank_top_n=3),
        )
        await retriever.retrieve("q", top_k=5)
        candidates = reranker.rerank.call_args[0][1]
        assert len(candidates) <= 3

    async def test_deduplicates_shared_content_hash_across_strategies(self):
        shared = _chunk("shared-hash", content="same content")
        bm25 = _make_strategy(results=[shared])
        vector = _make_strategy(results=[shared])
        reranker = _make_reranker(results=[shared])
        retriever = ConcreteHybridRetriever(
            bm25=bm25,
            vector=vector,
            reranker=reranker,
            settings=_settings(rerank_top_n=10),
        )
        await retriever.retrieve("q", top_k=5)
        candidates = reranker.rerank.call_args[0][1]
        hashes = [c.content_hash for c in candidates]
        assert hashes.count("shared-hash") == 1

    async def test_final_top_k_caps_results(self):
        chunks = [_chunk(f"h{i}") for i in range(10)]
        bm25 = _make_strategy(results=chunks)
        reranker = _make_reranker(results=chunks[:3])
        retriever = ConcreteHybridRetriever(
            bm25=bm25,
            vector=_make_strategy(),
            reranker=reranker,
            settings=_settings(rerank_top_n=10, final_top_k=3),
        )
        result = await retriever.retrieve("q", top_k=10)
        assert len(result) <= 3
