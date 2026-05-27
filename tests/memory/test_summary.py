"""Unit tests for SummaryMemoryStore."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.memory.schemas import ConversationTurn, MemorySummary
from app.memory.summary import SummaryMemoryStore


def _make_pg(fetchone_return=None):
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(
        return_value=MagicMock(fetchone=MagicMock(return_value=fetchone_return))
    )
    ctx_begin = AsyncMock()
    ctx_begin.__aenter__ = AsyncMock(return_value=mock_conn)
    ctx_begin.__aexit__ = AsyncMock(return_value=None)
    ctx_connect = AsyncMock()
    ctx_connect.__aenter__ = AsyncMock(return_value=mock_conn)
    ctx_connect.__aexit__ = AsyncMock(return_value=None)
    engine = MagicMock()
    engine.begin = MagicMock(return_value=ctx_begin)
    engine.connect = MagicMock(return_value=ctx_connect)
    pg = MagicMock()
    pg.engine = engine
    return pg, mock_conn


def _make_llm(complete_return: str = "Summary text.") -> MagicMock:
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=complete_return)
    return llm


def _make_summary() -> MemorySummary:
    return MemorySummary(
        summary_id="sum-1",
        user_id="user-1",
        session_id="sess-1",
        summary_text="A brief summary.",
        content_hash="a" * 64,
    )


def _make_turns(n: int = 3) -> list[ConversationTurn]:
    roles = ["user", "assistant"]
    return [
        ConversationTurn(session_id="sess-1", role=roles[i % 2], content=f"msg{i}", ts=float(i))
        for i in range(n)
    ]


class TestGetLatestSummary:
    async def test_returns_none_when_no_rows(self):
        pg, _ = _make_pg(fetchone_return=None)
        store = SummaryMemoryStore()
        result = await store.get_latest_summary(pg, "user-1")
        assert result is None

    async def test_returns_summary_from_row(self):
        row = ("sum-1", "user-1", "sess-1", "A summary.", "hash" * 16)
        pg, _ = _make_pg(fetchone_return=row)
        store = SummaryMemoryStore()
        result = await store.get_latest_summary(pg, "user-1")
        assert result is not None
        assert result.summary_id == "sum-1"
        assert result.user_id == "user-1"
        assert result.summary_text == "A summary."

    async def test_queries_by_user_id(self):
        pg, conn = _make_pg(fetchone_return=None)
        store = SummaryMemoryStore()
        await store.get_latest_summary(pg, "user-xyz")
        conn.execute.assert_awaited_once()
        params = conn.execute.call_args[0][1]
        assert params["user_id"] == "user-xyz"


class TestSaveSummary:
    async def test_executes_upsert(self):
        pg, conn = _make_pg()
        store = SummaryMemoryStore()
        summary = _make_summary()
        await store.save_summary(pg, summary)
        conn.execute.assert_awaited_once()
        params = conn.execute.call_args[0][1]
        assert params["summary_id"] == "sum-1"
        assert params["user_id"] == "user-1"
        assert params["session_id"] == "sess-1"
        assert params["summary_text"] == "A brief summary."

    async def test_uses_begin_transaction(self):
        pg, _ = _make_pg()
        store = SummaryMemoryStore()
        await store.save_summary(pg, _make_summary())
        pg.engine.begin.assert_called_once()


class TestCompact:
    async def test_calls_llm_complete(self):
        pg, _ = _make_pg()
        llm = _make_llm("Compact summary.")
        store = SummaryMemoryStore()
        turns = _make_turns(3)
        await store.compact(pg, llm, "user-1", "sess-1", turns)
        llm.complete.assert_awaited_once()
        prompt = llm.complete.call_args[0][0]
        assert "msg0" in prompt
        assert "msg1" in prompt

    async def test_returns_memory_summary(self):
        pg, _ = _make_pg()
        llm = _make_llm("The summary.")
        store = SummaryMemoryStore()
        result = await store.compact(pg, llm, "user-1", "sess-1", _make_turns(2))
        assert isinstance(result, MemorySummary)
        assert result.summary_text == "The summary."
        assert result.user_id == "user-1"
        assert result.session_id == "sess-1"

    async def test_summary_id_is_uuid(self):
        pg, _ = _make_pg()
        store = SummaryMemoryStore()
        result = await store.compact(pg, _make_llm(), "u", "s", _make_turns(1))
        import re
        assert re.match(r"^[0-9a-f-]{36}$", result.summary_id)

    async def test_content_hash_is_64_chars(self):
        pg, _ = _make_pg()
        store = SummaryMemoryStore()
        result = await store.compact(pg, _make_llm("text"), "u", "s", _make_turns(1))
        assert len(result.content_hash) == 64

    async def test_saves_summary_to_pg(self):
        pg, conn = _make_pg()
        store = SummaryMemoryStore()
        await store.compact(pg, _make_llm(), "u", "s", _make_turns(1))
        conn.execute.assert_awaited_once()
