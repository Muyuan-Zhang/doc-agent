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

    def test_graph_has_required_nodes(self):
        graph = build_graph(llm=MagicMock(), retriever=MagicMock(), redis=MagicMock(), cache_svc=MagicMock())
        node_names = set(graph.get_graph().nodes.keys())
        expected = {"cache_lookup", "stream_cached", "query_rewrite", "retrieval",
                    "entity_extraction", "rerank", "generate", "cache_write"}
        assert expected.issubset(node_names)

    def test_graph_has_retrieve_memory_node_when_memory_svc_provided(self):
        graph = build_graph(llm=MagicMock(), retriever=MagicMock(), redis=MagicMock(),
                          cache_svc=MagicMock(), memory_svc=MagicMock())
        node_names = set(graph.get_graph().nodes.keys())
        assert "retrieve_memory" in node_names

    def test_graph_includes_retrieve_memory_in_cache_miss_path(self):
        """When memory_svc is provided, retrieve_memory should be between cache_lookup and query_rewrite."""
        graph = build_graph(llm=MagicMock(), retriever=MagicMock(), redis=MagicMock(),
                          cache_svc=MagicMock(), memory_svc=MagicMock())
        edges = graph.get_graph().edges
        # retrieve_memory should have an incoming edge from cache_lookup (on miss path)
        sources_to_retrieve_memory = {e.source for e in edges if e.target == "retrieve_memory"}
        assert "cache_lookup" in sources_to_retrieve_memory

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
        cache_svc.get_or_retrieve = AsyncMock(return_value=([chunk], False, "aabb112233440000"))

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
            "chunk_cache_hit": False,
            "cached_answer": "",
            "query_embedding": None,
            "rag_cache_hash": None,
            "error": None,
            "user_id": "",
            "memory_context": None,
        })

        assert result["answer"] == "A great answer."
        cache_svc.get_or_retrieve.assert_awaited_once()
        inner.setex.assert_awaited_once()
        inner.rpush.assert_awaited_once()


# ---------------------------------------------------------------------------
# New graph topology: cache_lookup as entry, stream_cached on hit
# ---------------------------------------------------------------------------

class TestBuildGraphNewTopology:
    def test_graph_has_cache_lookup_and_stream_cached_nodes(self):
        graph = build_graph(llm=MagicMock(), retriever=MagicMock(), redis=MagicMock(), cache_svc=MagicMock())
        node_names = set(graph.get_graph().nodes.keys())
        assert "cache_lookup" in node_names
        assert "stream_cached" in node_names

    def test_cache_lookup_is_entry_point(self):
        graph = build_graph(llm=MagicMock(), retriever=MagicMock(), redis=MagicMock(), cache_svc=MagicMock())
        edges = graph.get_graph().edges
        entry_targets = [e.target for e in edges if e.source == "__start__"]
        assert "cache_lookup" in entry_targets

    async def test_cache_hit_path_skips_llm_rewrite_and_generate(self):
        from datetime import datetime, timezone
        from app.cache.schemas import CacheEntry, CacheStatus

        cached_entry = CacheEntry(
            query_hash="aabb112233440000",
            original_query="q", normalized_query="q",
            chunks=[], status=CacheStatus.APPROVED,
            created_at=datetime.now(tz=timezone.utc),
            query_embedding=[1.0, 0.0],
            answer="cached answer here",
        )
        llm = MagicMock()
        llm.embed = AsyncMock(return_value=[1.0, 0.0])
        llm.complete = AsyncMock()        # query_rewrite — must NOT be called
        llm.stream_complete = AsyncMock() # generate — must NOT be called

        redis = MagicMock()
        redis.cache_key = MagicMock(return_value="key")
        inner = MagicMock()
        inner.setex = AsyncMock()
        inner.rpush = AsyncMock()
        redis.client = inner

        cache_svc = MagicMock()
        cache_svc.lookup_by_embedding = AsyncMock(return_value=cached_entry)

        graph = build_graph(llm=llm, retriever=MagicMock(), redis=redis, cache_svc=cache_svc)
        result = await graph.ainvoke({
            "session_id": "s1", "job_id": "j1", "query": "q",
            "top_k": 5, "rewritten_query": "", "chunks": [],
            "reranked_chunks": [], "answer": "", "cache_hit": False,
            "cached_answer": "", "query_embedding": None,
            "rag_cache_hash": None, "error": None,
            "user_id": "", "memory_context": None,
            "chunk_cache_hit": False, "cached_answer": "", "query_embedding": None, "error": None,
        })

        assert result["answer"] == "cached answer here"
        llm.complete.assert_not_awaited()
        llm.stream_complete.assert_not_called()

    async def test_cache_miss_path_runs_full_pipeline(self):
        async def _stream(prompt, **kw):
            yield "generated answer"

        llm = MagicMock()
        llm.embed = AsyncMock(return_value=[0.1] * 5)
        llm.complete = AsyncMock(return_value="rewritten query")
        llm.stream_complete = _stream

        redis = MagicMock()
        redis.cache_key = MagicMock(return_value="key")
        inner = MagicMock()
        inner.setex = AsyncMock()
        inner.rpush = AsyncMock()
        redis.client = inner

        chunk = _chunk()
        cache_svc = MagicMock()
        cache_svc.lookup_by_embedding = AsyncMock(return_value=None)
        cache_svc.get_or_retrieve = AsyncMock(return_value=([chunk], False, "aabb112233440000"))
        cache_svc.save_answer = AsyncMock()

        graph = build_graph(llm=llm, retriever=MagicMock(), redis=redis, cache_svc=cache_svc)
        result = await graph.ainvoke({
            "session_id": "s1", "job_id": "j1", "query": "q",
            "top_k": 5, "rewritten_query": "", "chunks": [],
            "reranked_chunks": [], "answer": "", "cache_hit": False,
            "cached_answer": "", "query_embedding": None,
            "rag_cache_hash": None, "error": None,
            "user_id": "", "memory_context": None,
            "chunk_cache_hit": False, "cached_answer": "", "query_embedding": None, "error": None,
        })

        assert result["answer"] == "generated answer"
        llm.complete.assert_awaited()  # query_rewrite and rerank both called complete

    async def test_full_pipeline_with_memory_context_injection(self):
        """When memory_svc is provided and user_id set, retrieve_memory runs
        and the memory context flows into generate's prompt."""
        from app.memory.schemas import ConversationTurn, MemoryContext, MemorySummary, StaticFact

        async def _stream(prompt, **kw):
            yield "memory-aware answer"

# ---------------------------------------------------------------------------
# Bug 1 fix: Layer 2 chunk hit skips entity_extraction and rerank
# ---------------------------------------------------------------------------

class TestLayer2ChunkHitSkipsRerank:
    async def test_chunk_cache_hit_skips_entity_extraction_and_rerank(self):
        """When Layer 2 chunk cache hits, entity_extraction and rerank must be bypassed."""
        async def _stream(prompt, **kw):
            yield "chunk-cached answer"

        llm = MagicMock()
        llm.embed = AsyncMock(return_value=[0.1] * 5)
        llm.complete = AsyncMock(return_value="rewritten query")
        llm.stream_complete = _stream

        redis = MagicMock()
        redis.cache_key = MagicMock(return_value="key")
        inner = MagicMock()
        inner.setex = AsyncMock()
        inner.rpush = AsyncMock()
        redis.client = inner

        chunk = _chunk()
        cache_svc = MagicMock()
        cache_svc.lookup_by_embedding = AsyncMock(return_value=None)
        cache_svc.get_or_retrieve = AsyncMock(return_value=([chunk], False, "aabb112233440000"))
        cache_svc.save_answer = AsyncMock()

        memory_ctx = MemoryContext(
            turns=[
                ConversationTurn(session_id="s1", role="user", content="I like conciseness.", ts=1.0),
            ],
            summary=None,
            static_facts=[],
        )
        memory_svc = MagicMock()
        memory_svc.retrieve_context = AsyncMock(return_value=memory_ctx)

        graph = build_graph(
            llm=llm, retriever=MagicMock(), redis=redis,
            cache_svc=cache_svc, memory_svc=memory_svc,
        )
        cache_svc.lookup_by_embedding = AsyncMock(return_value=None)  # Layer 1 miss
        # Layer 2 hit: returns chunks, cache_hit=True
        cache_svc.get_or_retrieve = AsyncMock(return_value=([chunk], True, "aabb112233440000"))
        cache_svc.save_answer = AsyncMock()

        graph = build_graph(llm=llm, retriever=MagicMock(), redis=redis, cache_svc=cache_svc)
        result = await graph.ainvoke({
            "session_id": "s1", "job_id": "j1", "query": "q",
            "top_k": 5, "rewritten_query": "", "chunks": [],
            "reranked_chunks": [], "answer": "", "cache_hit": False,
            "cached_answer": "", "query_embedding": None,
            "rag_cache_hash": None, "error": None,
            "user_id": "u1", "memory_context": None,
        })

        assert result["answer"] == "memory-aware answer"
        # Memory service should have been called
        memory_svc.retrieve_context.assert_awaited_once_with(
            "s1", "u1", query_embedding=[0.1] * 5,
        )
            "cached_answer": "", "query_embedding": None, "rag_cache_hash": None,
            "chunk_cache_hit": False, "error": None,
        })

        assert result["answer"] == "chunk-cached answer"
        assert llm.complete.await_count <= 1, (
            f"rerank must be skipped on chunk cache hit; llm.complete called {llm.complete.await_count} times"
        )
        cache_svc.save_answer.assert_not_awaited()
