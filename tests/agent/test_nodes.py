"""Unit tests for M4 LangGraph agent nodes.

Each node receives an AgentState dict + keyword deps (llm, retriever, redis)
and returns a partial state dict with only the fields it updates.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.agent.nodes import (
    cache_write,
    entity_extraction,
    generate,
    query_rewrite,
    rerank,
    retrieval,
)
from app.models.chunk import ChunkSchema


def _chunk(**overrides) -> ChunkSchema:
    defaults = dict(
        doc_id="d1",
        section_id="s1",
        chunk_index=0,
        content_hash="abc123",
        version="v1",
        content="FastAPI is a modern web framework.",
    )
    return ChunkSchema(**(defaults | overrides))


def _state(**overrides) -> dict:
    base = dict(
        session_id="sess-1",
        job_id="job-1",
        query="what is fastapi?",
        top_k=5,
        rewritten_query="",
        chunks=[],
        reranked_chunks=[],
        answer="",
        error=None,
        retry_count=0,
    )
    return {**base, **overrides}


# ---------------------------------------------------------------------------
# query_rewrite
# ---------------------------------------------------------------------------

class TestQueryRewrite:
    async def test_sets_rewritten_query_from_llm(self):
        llm = MagicMock()
        llm.complete = AsyncMock(return_value="optimized fastapi introduction")
        state = _state()

        result = await query_rewrite(state, llm=llm, retriever=None, redis=None)

        assert result["rewritten_query"] == "optimized fastapi introduction"
        llm.complete.assert_awaited_once()

    async def test_falls_back_to_original_query_when_llm_returns_empty(self):
        llm = MagicMock()
        llm.complete = AsyncMock(return_value="")
        state = _state(query="what is python?")

        result = await query_rewrite(state, llm=llm, retriever=None, redis=None)

        assert result["rewritten_query"] == "what is python?"

    async def test_falls_back_to_original_query_when_llm_returns_whitespace(self):
        llm = MagicMock()
        llm.complete = AsyncMock(return_value="   ")
        state = _state(query="explain async")

        result = await query_rewrite(state, llm=llm, retriever=None, redis=None)

        assert result["rewritten_query"] == "explain async"


# ---------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------

class TestRetrieval:
    async def test_fills_chunks_from_retriever(self):
        chunk = _chunk()
        retriever = MagicMock()
        retriever.retrieve = AsyncMock(return_value=[chunk])
        state = _state(rewritten_query="fastapi intro", top_k=3)

        result = await retrieval(state, llm=None, retriever=retriever, redis=None)

        assert result["chunks"] == [chunk]
        retriever.retrieve.assert_awaited_once_with("fastapi intro", top_k=3)

    async def test_returns_empty_list_when_retriever_finds_nothing(self):
        retriever = MagicMock()
        retriever.retrieve = AsyncMock(return_value=[])
        state = _state(rewritten_query="obscure topic")

        result = await retrieval(state, llm=None, retriever=retriever, redis=None)

        assert result["chunks"] == []

    async def test_uses_rewritten_query_not_original(self):
        retriever = MagicMock()
        retriever.retrieve = AsyncMock(return_value=[])
        state = _state(query="original query", rewritten_query="better query")

        await retrieval(state, llm=None, retriever=retriever, redis=None)

        retriever.retrieve.assert_awaited_once_with("better query", top_k=5)


# ---------------------------------------------------------------------------
# entity_extraction  (pass-through placeholder)
# ---------------------------------------------------------------------------

class TestEntityExtraction:
    async def test_copies_chunks_to_reranked_chunks(self):
        chunk = _chunk()
        state = _state(chunks=[chunk])

        result = await entity_extraction(state, llm=None, retriever=None, redis=None)

        assert result["reranked_chunks"] == [chunk]

    async def test_empty_chunks_yields_empty_reranked(self):
        state = _state(chunks=[])

        result = await entity_extraction(state, llm=None, retriever=None, redis=None)

        assert result["reranked_chunks"] == []


# ---------------------------------------------------------------------------
# rerank
# ---------------------------------------------------------------------------

class TestRerank:
    async def test_returns_empty_list_without_calling_llm_on_no_chunks(self):
        llm = MagicMock()
        llm.complete = AsyncMock()
        state = _state(reranked_chunks=[])

        result = await rerank(state, llm=llm, retriever=None, redis=None)

        assert result["reranked_chunks"] == []
        llm.complete.assert_not_awaited()

    async def test_reorders_chunks_based_on_llm_ranking(self):
        chunk1 = _chunk(chunk_index=0, content="less relevant")
        chunk2 = _chunk(chunk_index=1, content="highly relevant fastapi")
        llm = MagicMock()
        llm.complete = AsyncMock(return_value="2, 1")
        state = _state(reranked_chunks=[chunk1, chunk2], query="fastapi", rewritten_query="fastapi intro")

        result = await rerank(state, llm=llm, retriever=None, redis=None)

        assert result["reranked_chunks"][0] == chunk2
        assert result["reranked_chunks"][1] == chunk1

    async def test_falls_back_to_original_order_on_parse_error(self):
        chunk1 = _chunk(chunk_index=0)
        chunk2 = _chunk(chunk_index=1, content="second chunk")
        llm = MagicMock()
        llm.complete = AsyncMock(return_value="not, numbers, at, all")
        state = _state(reranked_chunks=[chunk1, chunk2], query="fastapi", rewritten_query="fastapi intro")

        result = await rerank(state, llm=llm, retriever=None, redis=None)

        assert result["reranked_chunks"] == [chunk1, chunk2]

    async def test_propagates_llm_communication_errors(self):
        chunk = _chunk()
        llm = MagicMock()
        llm.complete = AsyncMock(side_effect=RuntimeError("LLM timeout"))
        state = _state(reranked_chunks=[chunk], query="q", rewritten_query="q rewritten")

        with pytest.raises(RuntimeError, match="LLM timeout"):
            await rerank(state, llm=llm, retriever=None, redis=None)

    async def test_calls_llm_with_rewritten_query_not_original(self):
        chunk = _chunk()
        llm = MagicMock()
        llm.complete = AsyncMock(return_value="1")
        state = _state(
            reranked_chunks=[chunk],
            query="original query",
            rewritten_query="optimized rewritten query",
        )

        await rerank(state, llm=llm, retriever=None, redis=None)

        prompt_arg = llm.complete.call_args[0][0]
        assert "optimized rewritten query" in prompt_arg
        assert "original query" not in prompt_arg


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

class TestGenerate:
    async def test_fills_answer_using_llm(self):
        chunk = _chunk(content="FastAPI is a web framework for Python.")
        llm = MagicMock()
        llm.complete = AsyncMock(return_value="FastAPI is a fast Python web framework.")
        state = _state(reranked_chunks=[chunk], query="what is fastapi?")

        result = await generate(state, llm=llm, retriever=None, redis=None)

        assert result["answer"] == "FastAPI is a fast Python web framework."
        llm.complete.assert_awaited_once()

    async def test_includes_chunk_content_in_prompt(self):
        chunk = _chunk(content="unique content marker xyz")
        llm = MagicMock()
        llm.complete = AsyncMock(return_value="answer")
        state = _state(reranked_chunks=[chunk], query="q")

        await generate(state, llm=llm, retriever=None, redis=None)

        prompt_arg = llm.complete.call_args[0][0]
        assert "unique content marker xyz" in prompt_arg

    async def test_still_calls_llm_on_empty_chunks(self):
        llm = MagicMock()
        llm.complete = AsyncMock(return_value="I don't have enough context to answer.")
        state = _state(reranked_chunks=[], query="what is fastapi?")

        result = await generate(state, llm=llm, retriever=None, redis=None)

        llm.complete.assert_awaited_once()
        assert result["answer"] == "I don't have enough context to answer."


# ---------------------------------------------------------------------------
# cache_write
# ---------------------------------------------------------------------------

class TestCacheWrite:
    async def test_writes_answer_to_redis_with_ttl(self):
        redis = MagicMock()
        redis.cache_key = MagicMock(return_value="v1:rag:sess-1:abc123")
        inner = MagicMock()
        inner.setex = AsyncMock()
        redis.client = inner
        state = _state(session_id="sess-1", query="what is fastapi?", answer="It is fast.")

        result = await cache_write(state, llm=None, retriever=None, redis=redis)

        redis.cache_key.assert_called_once()
        inner.setex.assert_awaited_once()
        assert result == {}

    async def test_cache_key_includes_session_and_query_hash(self):
        redis = MagicMock()
        redis.cache_key = MagicMock(return_value="v1:rag:sess-abc:hash")
        inner = MagicMock()
        inner.setex = AsyncMock()
        redis.client = inner
        state = _state(session_id="sess-abc", query="specific query", answer="ans")

        await cache_write(state, llm=None, retriever=None, redis=redis)

        call_args = redis.cache_key.call_args
        assert "sess-abc" in str(call_args)
