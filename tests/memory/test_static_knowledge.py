"""Unit tests for StaticKnowledgeStore."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.exceptions import NotFoundError
from app.memory.schemas import StaticFact
from app.memory.static_knowledge import StaticKnowledgeStore


def _make_pg(fetchone_return=None, rowcount: int = 1):
    mock_conn = AsyncMock()
    mock_result = MagicMock()
    mock_result.fetchone = MagicMock(return_value=fetchone_return)
    mock_result.rowcount = rowcount
    mock_conn.execute = AsyncMock(return_value=mock_result)
    ctx_begin = AsyncMock()
    ctx_begin.__aenter__ = AsyncMock(return_value=mock_conn)
    ctx_begin.__aexit__ = AsyncMock(return_value=None)
    engine = MagicMock()
    engine.begin = MagicMock(return_value=ctx_begin)
    pg = MagicMock()
    pg.engine = engine
    return pg, mock_conn


def _make_milvus(search_return: list | None = None) -> MagicMock:
    m = MagicMock()
    m.memory_insert = AsyncMock()
    m.memory_search = AsyncMock(return_value=search_return or [])
    m.memory_delete = AsyncMock()
    return m


def _make_llm(embedding: list[float] | None = None) -> MagicMock:
    llm = MagicMock()
    llm.embed = AsyncMock(return_value=embedding or [0.1, 0.2, 0.3])
    return llm


class TestAddFact:
    async def test_returns_static_fact(self):
        pg, _ = _make_pg()
        milvus = _make_milvus()
        llm = _make_llm()
        store = StaticKnowledgeStore()
        result = await store.add_fact(pg, milvus, llm, "user-1", "Python is a language.")
        assert isinstance(result, StaticFact)
        assert result.user_id == "user-1"
        assert result.content == "Python is a language."

    async def test_fact_id_is_uuid(self):
        pg, _ = _make_pg()
        store = StaticKnowledgeStore()
        result = await store.add_fact(_make_pg()[0], _make_milvus(), _make_llm(), "u", "content")
        import re
        assert re.match(r"^[0-9a-f-]{36}$", result.fact_id)

    async def test_content_hash_is_64_chars(self):
        pg, _ = _make_pg()
        store = StaticKnowledgeStore()
        result = await store.add_fact(pg, _make_milvus(), _make_llm(), "u", "some content")
        assert len(result.content_hash) == 64

    async def test_embedding_stored_in_fact(self):
        vec = [0.1, 0.9, 0.5]
        pg, _ = _make_pg()
        store = StaticKnowledgeStore()
        result = await store.add_fact(pg, _make_milvus(), _make_llm(vec), "u", "content")
        assert result.embedding == vec

    async def test_calls_llm_embed(self):
        pg, _ = _make_pg()
        llm = _make_llm()
        store = StaticKnowledgeStore()
        await store.add_fact(pg, _make_milvus(), llm, "u", "my fact")
        llm.embed.assert_awaited_once_with("my fact")

    async def test_inserts_into_milvus(self):
        pg, _ = _make_pg()
        milvus = _make_milvus()
        llm = _make_llm([0.5])
        store = StaticKnowledgeStore()
        result = await store.add_fact(pg, milvus, llm, "user-1", "some fact")
        milvus.memory_insert.assert_awaited_once()
        entity = milvus.memory_insert.call_args[0][0][0]
        assert entity["user_id"] == "user-1"
        assert entity["embedding"] == [0.5]
        assert entity["fact_id"] == result.fact_id

    async def test_inserts_into_postgres(self):
        pg, conn = _make_pg()
        store = StaticKnowledgeStore()
        await store.add_fact(pg, _make_milvus(), _make_llm(), "user-1", "fact")
        conn.execute.assert_awaited_once()
        params = conn.execute.call_args[0][1]
        assert params["user_id"] == "user-1"
        assert params["content"] == "fact"

    async def test_embed_error_propagates_without_storing(self):
        llm = MagicMock()
        llm.embed = AsyncMock(side_effect=RuntimeError("embedding service down"))
        pg, conn = _make_pg()
        milvus = _make_milvus()
        store = StaticKnowledgeStore()
        with pytest.raises(RuntimeError, match="embedding service down"):
            await store.add_fact(pg, milvus, llm, "u", "content")
        conn.execute.assert_not_awaited()
        milvus.memory_insert.assert_not_awaited()


class TestSearchFacts:
    async def test_empty_when_no_results(self):
        milvus = _make_milvus(search_return=[])
        store = StaticKnowledgeStore()
        result = await store.search_facts(milvus, [0.1, 0.2], "user-1")
        assert result == []

    async def test_returns_static_facts(self):
        hits = [
            {"fact_id": "f1", "user_id": "user-1", "content": "fact content"},
            {"fact_id": "f2", "user_id": "user-1", "content": "another fact"},
        ]
        milvus = _make_milvus(search_return=hits)
        store = StaticKnowledgeStore()
        result = await store.search_facts(milvus, [0.1], "user-1")
        assert len(result) == 2
        assert result[0].fact_id == "f1"
        assert result[0].content == "fact content"

    async def test_passes_user_id_and_top_k(self):
        milvus = _make_milvus()
        store = StaticKnowledgeStore()
        await store.search_facts(milvus, [0.0], "user-xyz", top_k=3)
        milvus.memory_search.assert_awaited_once_with([0.0], "user-xyz", 3)


class TestDeleteFact:
    async def test_deletes_from_pg_and_milvus(self):
        pg, conn = _make_pg(rowcount=1)
        milvus = _make_milvus()
        store = StaticKnowledgeStore()
        await store.delete_fact(pg, milvus, "fact-1", "user-1")
        conn.execute.assert_awaited_once()
        milvus.memory_delete.assert_awaited_once_with("fact-1")

    async def test_raises_not_found_when_rowcount_zero(self):
        pg, _ = _make_pg(rowcount=0)
        milvus = _make_milvus()
        store = StaticKnowledgeStore()
        with pytest.raises(NotFoundError):
            await store.delete_fact(pg, milvus, "missing-fact", "user-1")

    async def test_milvus_not_called_on_pg_miss(self):
        pg, _ = _make_pg(rowcount=0)
        milvus = _make_milvus()
        store = StaticKnowledgeStore()
        with pytest.raises(NotFoundError):
            await store.delete_fact(pg, milvus, "f", "u")
        milvus.memory_delete.assert_not_awaited()
