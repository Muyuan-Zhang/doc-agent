"""Unit tests for MemoryService."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.memory.schemas import ConversationTurn, MemoryContext, MemorySummary, StaticFact
from app.memory.service import MemoryService


def _make_clients():
    pg = MagicMock()
    redis = MagicMock()
    milvus = MagicMock()
    llm = MagicMock()
    return pg, redis, milvus, llm


def _make_service(recent=None, summary_store=None, static_store=None):
    pg, redis, milvus, llm = _make_clients()
    svc = MemoryService(pg=pg, redis=redis, milvus=milvus, llm=llm)
    # Always inject mock stores so real stores never touch raw MagicMock clients
    svc._recent = recent if recent is not None else _make_recent()
    svc._summary = summary_store if summary_store is not None else _make_summary_store()
    svc._static = static_store if static_store is not None else _make_static_store()
    return svc


def _make_recent(count: int = 1, turns: list | None = None):
    m = MagicMock()
    m.append_turn = AsyncMock(return_value=count)
    m.get_turns = AsyncMock(return_value=turns or [])
    m.count = AsyncMock(return_value=count)
    m.clear = AsyncMock()
    return m


def _make_summary_store(summary: MemorySummary | None = None, compact_return: MemorySummary | None = None):
    m = MagicMock()
    default_summary = MemorySummary(
        summary_id="s1", user_id="u1", session_id="sess",
        summary_text="default summary", content_hash="a" * 64
    )
    m.get_latest_summary = AsyncMock(return_value=summary)
    m.compact = AsyncMock(return_value=compact_return or default_summary)
    return m


def _make_static_store(facts: list | None = None, fact_return: StaticFact | None = None):
    m = MagicMock()
    default_fact = StaticFact(
        fact_id="f1", user_id="u1", content="fact", content_hash="b" * 64
    )
    m.search_facts = AsyncMock(return_value=facts or [])
    m.add_fact = AsyncMock(return_value=fact_return or default_fact)
    m.delete_fact = AsyncMock()
    return m


def _make_turn(content: str = "hello") -> ConversationTurn:
    return ConversationTurn(session_id="sess", role="user", content=content, ts=1.0)


class TestAppendTurn:
    async def test_appends_to_recent(self):
        recent = _make_recent(count=1)
        svc = _make_service(recent=recent)
        await svc.append_turn("sess", "user-1", "user", "hello")
        recent.append_turn.assert_awaited_once()

    async def test_does_not_compact_below_threshold(self):
        recent = _make_recent(count=5)
        summary = _make_summary_store()
        svc = _make_service(recent=recent, summary_store=summary)
        await svc.append_turn("sess", "user-1", "user", "hello")
        summary.compact.assert_not_awaited()

    async def test_compacts_at_threshold(self):
        from app.core.config import settings
        recent = _make_recent(count=settings.memory_summary_threshold)
        summary = _make_summary_store()
        svc = _make_service(recent=recent, summary_store=summary)
        await svc.append_turn("sess", "user-1", "user", "hello")
        summary.compact.assert_awaited_once()

    async def test_clears_recent_after_compact(self):
        from app.core.config import settings
        recent = _make_recent(count=settings.memory_summary_threshold)
        svc = _make_service(recent=recent)
        await svc.append_turn("sess", "user-1", "user", "hello")
        recent.clear.assert_awaited_once_with(svc._redis, "sess")

    async def test_no_clear_below_threshold(self):
        recent = _make_recent(count=3)
        svc = _make_service(recent=recent)
        await svc.append_turn("sess", "user-1", "user", "hello")
        recent.clear.assert_not_awaited()


class TestRetrieveContext:
    async def test_returns_memory_context(self):
        svc = _make_service()
        result = await svc.retrieve_context("sess", "user-1")
        assert isinstance(result, MemoryContext)

    async def test_context_includes_recent_turns(self):
        turns = [_make_turn("msg1"), _make_turn("msg2")]
        recent = _make_recent(turns=turns)
        svc = _make_service(recent=recent)
        result = await svc.retrieve_context("sess", "user-1")
        assert len(result.turns) == 2
        assert result.turns[0].content == "msg1"

    async def test_context_includes_summary(self):
        summ = MemorySummary(
            summary_id="s1", user_id="user-1", session_id="sess",
            summary_text="past events", content_hash="c" * 64
        )
        summary_store = _make_summary_store(summary=summ)
        svc = _make_service(summary_store=summary_store)
        result = await svc.retrieve_context("sess", "user-1")
        assert result.summary is not None
        assert result.summary.summary_text == "past events"

    async def test_no_static_facts_without_embedding(self):
        static = _make_static_store()
        svc = _make_service(static_store=static)
        result = await svc.retrieve_context("sess", "user-1", query_embedding=None)
        static.search_facts.assert_not_awaited()
        assert result.static_facts == []

    async def test_searches_static_facts_with_embedding(self):
        facts = [StaticFact(fact_id="f1", user_id="u", content="c", content_hash="d" * 64)]
        static = _make_static_store(facts=facts)
        svc = _make_service(static_store=static)
        result = await svc.retrieve_context("sess", "user-1", query_embedding=[0.1, 0.2])
        assert len(result.static_facts) == 1
        static.search_facts.assert_awaited_once()

    async def test_context_summary_none_when_no_history(self):
        summary_store = _make_summary_store(summary=None)
        svc = _make_service(summary_store=summary_store)
        result = await svc.retrieve_context("sess", "user-1")
        assert result.summary is None


class TestSummarizeSession:
    async def test_returns_memory_summary(self):
        svc = _make_service()
        result = await svc.summarize_session("sess", "user-1")
        assert isinstance(result, MemorySummary)

    async def test_clears_recent_after_summarize(self):
        recent = _make_recent(turns=[_make_turn()])
        svc = _make_service(recent=recent)
        await svc.summarize_session("sess", "user-1")
        recent.clear.assert_awaited_once()

    async def test_compact_called_with_turns(self):
        turns = [_make_turn("t1"), _make_turn("t2")]
        recent = _make_recent(turns=turns)
        summary_store = _make_summary_store()
        svc = _make_service(recent=recent, summary_store=summary_store)
        await svc.summarize_session("sess", "user-1")
        summary_store.compact.assert_awaited_once()
        _, _, uid, sid, passed_turns = summary_store.compact.call_args[0]
        assert uid == "user-1"
        assert sid == "sess"
        assert len(passed_turns) == 2

    async def test_passes_previous_summary_for_incremental(self):
        prev = MemorySummary(
            summary_id="prev", user_id="user-1", session_id="sess",
            summary_text="old context", content_hash="c" * 64,
        )
        summary_store = _make_summary_store(summary=prev)
        svc = _make_service(summary_store=summary_store)
        await svc.summarize_session("sess", "user-1")
        kwargs = summary_store.compact.call_args[1]
        assert kwargs.get("previous_summary") is prev


class TestClearRetry:
    async def test_clear_retried_on_failure(self):
        from unittest.mock import AsyncMock
        from app.memory.service import _clear_with_retry
        from app.memory.recent import RecentMemoryStore

        recent = MagicMock()
        redis = MagicMock()
        fail_once = AsyncMock(side_effect=[Exception("timeout"), None])
        recent.clear = fail_once
        await _clear_with_retry(recent, redis, "sess")
        assert fail_once.await_count == 2

    async def test_clear_gives_up_after_max_retries(self):
        from unittest.mock import AsyncMock
        from app.memory.service import _clear_with_retry, _CLEAR_RETRIES

        recent = MagicMock()
        redis = MagicMock()
        recent.clear = AsyncMock(side_effect=Exception("always fails"))
        await _clear_with_retry(recent, redis, "sess")  # must not raise
        assert recent.clear.await_count == _CLEAR_RETRIES


class TestAddStaticFact:
    async def test_delegates_to_static_store(self):
        static = _make_static_store()
        svc = _make_service(static_store=static)
        result = await svc.add_static_fact("user-1", "some knowledge")
        static.add_fact.assert_awaited_once()
        assert isinstance(result, StaticFact)

    async def test_passes_user_id_and_content(self):
        static = _make_static_store()
        svc = _make_service(static_store=static)
        await svc.add_static_fact("user-abc", "my content")
        args = static.add_fact.call_args[0]
        assert "user-abc" in args
        assert "my content" in args


class TestDeleteStaticFact:
    async def test_delegates_to_static_store(self):
        static = _make_static_store()
        svc = _make_service(static_store=static)
        await svc.delete_static_fact("fact-1", "user-1")
        static.delete_fact.assert_awaited_once()

    async def test_passes_fact_id_and_user_id(self):
        static = _make_static_store()
        svc = _make_service(static_store=static)
        await svc.delete_static_fact("fact-xyz", "user-xyz")
        args = static.delete_fact.call_args[0]
        assert "fact-xyz" in args
        assert "user-xyz" in args
