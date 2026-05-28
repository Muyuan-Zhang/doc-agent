"""
Tests for app/cache/query_rewriter.py.

Covers:
- normalize(): lowercasing, whitespace collapse, punctuation removal, CJK preservation
- hash_query(): 16-char hex, stability, uniqueness
- rewrite(): LLM disabled path, LLM enabled path, LLM fallback on error
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.cache.query_rewriter import QueryRewriter
from app.core.config import Settings


def _make_llm(**overrides) -> MagicMock:
    m = MagicMock()
    m.complete = AsyncMock(return_value=overrides.get("complete_return", "rewritten query"))
    return m


def _make_cfg(**overrides) -> Settings:
    return Settings(
        cache_rewrite_enabled=overrides.get("cache_rewrite_enabled", False),
    )


# ---------------------------------------------------------------------------
# normalize()
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_lowercases_query(self):
        rw = QueryRewriter(_make_llm())
        assert rw.normalize("HELLO World") == "hello world"

    def test_strips_leading_trailing_whitespace(self):
        rw = QueryRewriter(_make_llm())
        assert rw.normalize("  hello  ") == "hello"

    def test_collapses_internal_whitespace(self):
        rw = QueryRewriter(_make_llm())
        assert rw.normalize("what  is   the  deadline") == "what is the deadline"

    def test_removes_english_punctuation(self):
        rw = QueryRewriter(_make_llm())
        result = rw.normalize("What is the deadline?!")
        assert "?" not in result
        assert "!" not in result

    def test_removes_common_symbols(self):
        rw = QueryRewriter(_make_llm())
        result = rw.normalize("hello, world; how (are) you?")
        assert "," not in result
        assert ";" not in result
        assert "(" not in result
        assert ")" not in result

    def test_preserves_cjk_characters(self):
        rw = QueryRewriter(_make_llm())
        result = rw.normalize("截止日期是什么时候？")
        assert "截止日期是什么时候" in result

    def test_preserves_alphanumeric(self):
        rw = QueryRewriter(_make_llm())
        result = rw.normalize("Q3 report 2024")
        assert "q3" in result
        assert "report" in result
        assert "2024" in result

    def test_empty_string_returns_empty(self):
        rw = QueryRewriter(_make_llm())
        assert rw.normalize("") == ""

    def test_whitespace_only_returns_empty(self):
        rw = QueryRewriter(_make_llm())
        assert rw.normalize("   ") == ""


# ---------------------------------------------------------------------------
# hash_query()
# ---------------------------------------------------------------------------

class TestHashQuery:
    def test_returns_16_char_hex_string(self):
        rw = QueryRewriter(_make_llm())
        h = rw.hash_query("what is the deadline")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_same_input_produces_same_hash(self):
        rw = QueryRewriter(_make_llm())
        h1 = rw.hash_query("test query")
        h2 = rw.hash_query("test query")
        assert h1 == h2

    def test_different_inputs_produce_different_hashes(self):
        rw = QueryRewriter(_make_llm())
        h1 = rw.hash_query("query one")
        h2 = rw.hash_query("query two")
        assert h1 != h2

    def test_empty_string_produces_consistent_hash(self):
        rw = QueryRewriter(_make_llm())
        h1 = rw.hash_query("")
        h2 = rw.hash_query("")
        assert h1 == h2
        assert len(h1) == 16


# ---------------------------------------------------------------------------
# rewrite()
# ---------------------------------------------------------------------------

class TestRewrite:
    async def test_returns_normalized_and_hash_without_llm(self):
        cfg = _make_cfg(cache_rewrite_enabled=False)
        rw = QueryRewriter(_make_llm(), cfg)
        normalized, query_hash = await rw.rewrite("Hello World!")
        assert normalized == "hello world"
        assert len(query_hash) == 16

    async def test_hash_is_stable_across_repeated_calls(self):
        cfg = _make_cfg(cache_rewrite_enabled=False)
        rw = QueryRewriter(_make_llm(), cfg)
        _, h1 = await rw.rewrite("same query")
        _, h2 = await rw.rewrite("same query")
        assert h1 == h2

    async def test_skips_llm_when_rewrite_disabled(self):
        cfg = _make_cfg(cache_rewrite_enabled=False)
        llm = _make_llm()
        rw = QueryRewriter(llm, cfg)
        await rw.rewrite("test query")
        llm.complete.assert_not_awaited()

    async def test_calls_llm_complete_when_rewrite_enabled(self):
        cfg = _make_cfg(cache_rewrite_enabled=True)
        llm = _make_llm(complete_return="canonical form")
        rw = QueryRewriter(llm, cfg)
        normalized, _ = await rw.rewrite("What is the deadline?")
        llm.complete.assert_awaited_once()
        assert "canonical form" in normalized

    async def test_llm_output_is_also_normalized(self):
        cfg = _make_cfg(cache_rewrite_enabled=True)
        llm = _make_llm(complete_return="  CANONICAL FORM!  ")
        rw = QueryRewriter(llm, cfg)
        normalized, _ = await rw.rewrite("test")
        assert normalized == "canonical form"

    async def test_falls_back_to_normalize_when_llm_raises(self):
        cfg = _make_cfg(cache_rewrite_enabled=True)
        llm = _make_llm()
        llm.complete = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
        rw = QueryRewriter(llm, cfg)
        normalized, query_hash = await rw.rewrite("What is deadline?")
        assert normalized == "what is deadline"
        assert len(query_hash) == 16

    async def test_falls_back_when_llm_returns_empty_string(self):
        cfg = _make_cfg(cache_rewrite_enabled=True)
        llm = _make_llm(complete_return="")
        rw = QueryRewriter(llm, cfg)
        normalized, _ = await rw.rewrite("What is the deadline?")
        assert "deadline" in normalized

    async def test_hash_differs_between_semantically_different_queries(self):
        cfg = _make_cfg(cache_rewrite_enabled=False)
        rw = QueryRewriter(_make_llm(), cfg)
        _, h1 = await rw.rewrite("query about deadlines")
        _, h2 = await rw.rewrite("query about budgets")
        assert h1 != h2

    async def test_hash_stable_regardless_of_llm_rewrite(self):
        """Hash must equal hash(normalize(raw)) even when LLM rewrites the display form."""
        cfg = _make_cfg(cache_rewrite_enabled=True)
        llm = _make_llm(complete_return="completely different canonical form")
        rw = QueryRewriter(llm, cfg)
        _, query_hash = await rw.rewrite("What is the deadline?")
        expected_hash = rw.hash_query(rw.normalize("What is the deadline?"))
        assert query_hash == expected_hash
