"""Tests for M4 LangGraph graph compilation and wiring."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.agent.graph import build_graph
from app.models.chunk import ChunkSchema


def _chunk() -> ChunkSchema:
    return ChunkSchema(
        doc_id="d1", section_id="s1", chunk_index=0,
        content_hash="abc", version="v1", content="FastAPI docs",
    )


class TestBuildGraph:
    def test_graph_compiles_without_error(self):
        llm = MagicMock()
        retriever = MagicMock()
        redis = MagicMock()
        redis.cache_key = MagicMock(return_value="v1:rag:key")
        redis.client = MagicMock()
        cache_svc = MagicMock()

        graph = build_graph(llm=llm, retriever=retriever, redis=redis, cache_svc=cache_svc)

        assert graph is not None

    def test_graph_has_six_nodes(self):
        graph = build_graph(llm=MagicMock(), retriever=MagicMock(), redis=MagicMock(), cache_svc=MagicMock())
        node_names = set(graph.get_graph().nodes.keys())
        expected = {"query_rewrite", "retrieval", "entity_extraction", "rerank", "generate", "cache_write"}
        assert expected.issubset(node_names)

    async def test_graph_invocation_end_to_end(self):
        chunk = _chunk()

        async def _mock_stream(prompt, **kwargs):
            yield "A great answer."

        llm = MagicMock()
        llm.complete = AsyncMock(return_value="rewritten query")
        llm.stream_complete = _mock_stream
        retriever = MagicMock()
        retriever.retrieve = AsyncMock(return_value=[chunk])
        redis = MagicMock()
        redis.cache_key = MagicMock(return_value="v1:rag:sess1:abc")
        inner = MagicMock()
        inner.setex = AsyncMock()
        inner.rpush = AsyncMock()
        redis.client = inner
        cache_svc = MagicMock()
        cache_svc.get_or_retrieve = AsyncMock(return_value=([chunk], False))

        graph = build_graph(llm=llm, retriever=retriever, redis=redis, cache_svc=cache_svc)
        result = await graph.ainvoke({
            "session_id": "sess1",
            "job_id": "j1",
            "query": "what is fastapi?",
            "top_k": 5,
            "rewritten_query": "",
            "chunks": [],
            "reranked_chunks": [],
            "answer": "",
            "cache_hit": False,
            "error": None,
        })

        assert result["answer"] == "A great answer."
        cache_svc.get_or_retrieve.assert_awaited_once()
        inner.setex.assert_awaited_once()
        inner.rpush.assert_awaited_once()
