"""Unit tests for SummaryMemoryStore."""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.memory.schemas import ConversationTurn, MemorySummary
from app.memory.summary import SummaryMemoryStore, _MAX_INPUT_TURNS


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

# Row now has 7 columns: id, user_id, session_id, text, hash, importance_score, structured_facts
_FULL_ROW = ("sum-1", "user-1", "sess-1", "A summary.", "hash" * 16, 1.0, {})


class TestGetLatestSummary:
    async def test_returns_none_when_no_rows(self):
        pg, _ = _make_pg(fetchone_return=None)
        store = SummaryMemoryStore()
        result = await store.get_latest_summary(pg, "user-1", "sess-1")
        assert result is None

    async def test_returns_summary_from_row(self):
        pg, _ = _make_pg(fetchone_return=_FULL_ROW)
        store = SummaryMemoryStore()
        result = await store.get_latest_summary(pg, "user-1", "sess-1")
        assert result is not None
        assert result.summary_id == "sum-1"
        assert result.user_id == "user-1"
        assert result.summary_text == "A summary."

    async def test_queries_by_user_id_and_session_id(self):
        pg, conn = _make_pg(fetchone_return=None)
        store = SummaryMemoryStore()
        await store.get_latest_summary(pg, "user-xyz", "sess-abc")
        conn.execute.assert_awaited_once()
        params = conn.execute.call_args[0][1]
        assert params["user_id"] == "user-xyz"
        assert params["session_id"] == "sess-abc"

    async def test_returns_importance_score(self):
        row = (*_FULL_ROW[:5], 0.8, {})
        pg, _ = _make_pg(fetchone_return=row)
        store = SummaryMemoryStore()
        result = await store.get_latest_summary(pg, "user-1", "sess-1")
        assert result.importance_score == 0.8

    async def test_returns_structured_facts(self):
        row = (*_FULL_ROW[:5], 1.0, {"key_topics": ["python", "testing"]})
        pg, _ = _make_pg(fetchone_return=row)
        store = SummaryMemoryStore()
        result = await store.get_latest_summary(pg, "user-1", "sess-1")
        assert result.structured_facts == {"key_topics": ["python", "testing"]}


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

    async def test_saves_structured_facts_as_json(self):
        pg, conn = _make_pg()
        store = SummaryMemoryStore()
        summary = MemorySummary(
            summary_id="s1", user_id="u", session_id="s",
            summary_text="t", content_hash="a" * 64,
            structured_facts={"key_topics": ["a", "b"]},
        )
        await store.save_summary(pg, summary)
        params = conn.execute.call_args[0][1]
        parsed = json.loads(params["structured_facts"])
        assert parsed["key_topics"] == ["a", "b"]


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

    async def test_parses_json_from_llm(self):
        llm_json = json.dumps({
            "summary_text": "Parsed summary.",
            "key_topics": ["topic-a", "topic-b"],
        })
        pg, _ = _make_pg()
        store = SummaryMemoryStore()
        result = await store.compact(pg, _make_llm(llm_json), "u", "s", _make_turns(2))
        assert result.summary_text == "Parsed summary."
        assert result.structured_facts == {"key_topics": ["topic-a", "topic-b"]}

    async def test_falls_back_when_llm_returns_plain_text(self):
        pg, _ = _make_pg()
        store = SummaryMemoryStore()
        result = await store.compact(pg, _make_llm("plain text"), "u", "s", _make_turns(1))
        assert result.summary_text == "plain text"
        assert result.structured_facts == {}

    async def test_includes_previous_summary_in_prompt(self):
        previous = MemorySummary(
            summary_id="prev", user_id="u", session_id="s",
            summary_text="Earlier context.", content_hash="b" * 64,
        )
        pg, _ = _make_pg()
        llm = _make_llm("new summary")
        store = SummaryMemoryStore()
        await store.compact(pg, llm, "u", "s", _make_turns(2), previous_summary=previous)
        prompt = llm.complete.call_args[0][0]
        assert "Earlier context." in prompt

    async def test_caps_input_at_max_turns(self):
        pg, _ = _make_pg()
        llm = _make_llm("summary")
        store = SummaryMemoryStore()
        many_turns = _make_turns(_MAX_INPUT_TURNS + 5)
        await store.compact(pg, llm, "u", "s", many_turns)
        prompt = llm.complete.call_args[0][0]
        # The first 5 turns should be excluded (windowed to last _MAX_INPUT_TURNS)
        assert f"msg{_MAX_INPUT_TURNS - 1}" in prompt
        assert "msg0" not in prompt

    async def test_returns_previous_summary_when_turns_empty(self):
        prev = MemorySummary(
            summary_id="prev", user_id="u", session_id="s",
            summary_text="existing summary", content_hash="a" * 64,
        )
        pg, _ = _make_pg()
        llm = _make_llm()
        store = SummaryMemoryStore()
        result = await store.compact(pg, llm, "u", "s", [], previous_summary=prev)
        assert result is prev
        llm.complete.assert_not_awaited()

    async def test_raises_when_turns_empty_and_no_previous(self):
        pg, _ = _make_pg()
        store = SummaryMemoryStore()
        with pytest.raises(ValueError, match="no turns"):
            await store.compact(pg, _make_llm(), "u", "sess-x", [])

    async def test_strips_markdown_fences_from_llm_response(self):
        fenced = '```json\n{"summary_text": "Fenced summary.", "key_topics": ["a"]}\n```'
        pg, _ = _make_pg()
        store = SummaryMemoryStore()
        result = await store.compact(pg, _make_llm(fenced), "u", "s", _make_turns(1))
        assert result.summary_text == "Fenced summary."
        assert result.structured_facts == {"key_topics": ["a"]}
