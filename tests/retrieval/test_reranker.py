"""Unit tests for LLMReranker — LLM-based cross-encoder reranking."""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.chunk import ChunkSchema
from app.retrieval.reranker import LLMReranker


def _chunk(hash_: str, content: str = "sample text") -> ChunkSchema:
    return ChunkSchema(
        doc_id="d1",
        section_id="s0",
        chunk_index=0,
        content_hash=hash_,
        version="v1",
        content=content,
    )


def _make_llm(response: str | None = None, side_effect=None):
    llm = MagicMock()
    if side_effect is not None:
        llm.complete = AsyncMock(side_effect=side_effect)
    else:
        llm.complete = AsyncMock(
            return_value=response or json.dumps({"ranked_indices": [0]})
        )
    return llm


class TestLLMRerankerRerank:
    async def test_empty_chunks_returns_empty(self):
        llm = _make_llm()
        result = await LLMReranker(llm=llm).rerank("query", [], top_n=5)
        assert result == []

    async def test_empty_chunks_skips_llm(self):
        llm = _make_llm()
        await LLMReranker(llm=llm).rerank("query", [], top_n=5)
        llm.complete.assert_not_awaited()

    async def test_single_chunk_returns_without_llm(self):
        llm = _make_llm()
        chunks = [_chunk("a")]
        result = await LLMReranker(llm=llm).rerank("query", chunks, top_n=1)
        llm.complete.assert_not_awaited()
        assert result[0].content_hash == "a"

    async def test_calls_llm_complete_for_multiple_chunks(self):
        llm = _make_llm(response=json.dumps({"ranked_indices": [0, 1]}))
        chunks = [_chunk("a"), _chunk("b")]
        await LLMReranker(llm=llm).rerank("query", chunks, top_n=2)
        llm.complete.assert_awaited_once()

    async def test_prompt_contains_query(self):
        llm = _make_llm(response=json.dumps({"ranked_indices": [0, 1]}))
        chunks = [_chunk("a"), _chunk("b")]
        await LLMReranker(llm=llm).rerank("my search query", chunks, top_n=2)
        prompt = llm.complete.call_args[0][0]
        assert "my search query" in prompt

    async def test_prompt_contains_chunk_content(self):
        llm = _make_llm(response=json.dumps({"ranked_indices": [0, 1]}))
        chunks = [_chunk("a", content="unique content abc"), _chunk("b")]
        await LLMReranker(llm=llm).rerank("query", chunks, top_n=2)
        prompt = llm.complete.call_args[0][0]
        assert "unique content abc" in prompt

    async def test_query_braces_do_not_break_prompt(self):
        llm = _make_llm(response=json.dumps({"ranked_indices": [0, 1]}))
        chunks = [_chunk("a"), _chunk("b")]
        await LLMReranker(llm=llm).rerank("{malicious} query}", chunks, top_n=2)
        prompt = llm.complete.call_args[0][0]
        assert "{malicious}" in prompt

    async def test_reorders_by_llm_response(self):
        response = json.dumps({"ranked_indices": [2, 0, 1]})
        llm = _make_llm(response=response)
        chunks = [_chunk("a"), _chunk("b"), _chunk("c")]
        result = await LLMReranker(llm=llm).rerank("query", chunks, top_n=3)
        assert result[0].content_hash == "c"
        assert result[1].content_hash == "a"
        assert result[2].content_hash == "b"

    async def test_fallback_on_invalid_json(self):
        llm = _make_llm(response="not valid json at all")
        chunks = [_chunk("a"), _chunk("b")]
        result = await LLMReranker(llm=llm).rerank("query", chunks, top_n=2)
        assert result[0].content_hash == "a"
        assert result[1].content_hash == "b"

    async def test_fallback_on_llm_exception(self):
        llm = _make_llm(side_effect=RuntimeError("LLM service down"))
        chunks = [_chunk("a"), _chunk("b")]
        result = await LLMReranker(llm=llm).rerank("query", chunks, top_n=2)
        assert [c.content_hash for c in result] == ["a", "b"]

    async def test_fallback_on_missing_ranked_indices_key(self):
        llm = _make_llm(response=json.dumps({"result": [0, 1]}))
        chunks = [_chunk("a"), _chunk("b")]
        result = await LLMReranker(llm=llm).rerank("query", chunks, top_n=2)
        assert [c.content_hash for c in result] == ["a", "b"]

    async def test_fallback_when_ranked_indices_is_not_list(self):
        llm = _make_llm(response=json.dumps({"ranked_indices": 42}))
        chunks = [_chunk("a"), _chunk("b")]
        result = await LLMReranker(llm=llm).rerank("query", chunks, top_n=2)
        assert [c.content_hash for c in result] == ["a", "b"]

    async def test_respects_top_n_limit(self):
        response = json.dumps({"ranked_indices": [0, 1, 2]})
        llm = _make_llm(response=response)
        chunks = [_chunk("a"), _chunk("b"), _chunk("c")]
        result = await LLMReranker(llm=llm).rerank("query", chunks, top_n=2)
        assert len(result) == 2

    async def test_out_of_bounds_indices_filtered(self):
        response = json.dumps({"ranked_indices": [0, 99, 1]})
        llm = _make_llm(response=response)
        chunks = [_chunk("a"), _chunk("b")]
        result = await LLMReranker(llm=llm).rerank("query", chunks, top_n=2)
        assert all(c.content_hash in {"a", "b"} for c in result)
