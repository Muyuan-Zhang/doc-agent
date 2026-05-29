"""Unit tests for BM25Strategy — PostgreSQL full-text retrieval."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.chunk import ChunkSchema
from app.retrieval.bm25 import BM25Strategy


def _make_pg(rows=None):
    mock_conn = AsyncMock()
    mock_result = MagicMock()
    mock_result.fetchall = MagicMock(return_value=rows or [])
    mock_conn.execute = AsyncMock(return_value=mock_result)
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    ctx.__aexit__ = AsyncMock(return_value=None)
    engine = MagicMock()
    engine.connect = MagicMock(return_value=ctx)
    pg = MagicMock()
    pg.engine = engine
    return pg, mock_conn


def _row(
    chunk_id="d1:s0:000000",
    doc_id="d1",
    section_id="s0",
    chunk_index=0,
    parent_chunk_id=None,
    content_hash="h0",
    version="v1",
    content="hello world",
    score=0.9,
):
    return (chunk_id, doc_id, section_id, chunk_index, parent_chunk_id, content_hash, version, content, score)


class TestBM25StrategyRetrieve:
    async def test_empty_query_returns_empty(self):
        pg, _ = _make_pg()
        result = await BM25Strategy(pg=pg).retrieve("", top_k=5)
        assert result == []

    async def test_whitespace_only_query_returns_empty(self):
        pg, _ = _make_pg()
        result = await BM25Strategy(pg=pg).retrieve("   ", top_k=5)
        assert result == []

    async def test_no_db_call_for_empty_query(self):
        pg, conn = _make_pg()
        await BM25Strategy(pg=pg).retrieve("", top_k=5)
        conn.execute.assert_not_awaited()

    async def test_executes_sql_for_valid_query(self):
        pg, conn = _make_pg()
        await BM25Strategy(pg=pg).retrieve("python", top_k=5)
        conn.execute.assert_awaited_once()

    async def test_passes_query_in_params(self):
        pg, conn = _make_pg()
        await BM25Strategy(pg=pg).retrieve("fastapi", top_k=3)
        params = conn.execute.call_args[0][1]
        assert params["q"] == "fastapi"

    async def test_passes_top_k_in_params(self):
        pg, conn = _make_pg()
        await BM25Strategy(pg=pg).retrieve("test", top_k=7)
        params = conn.execute.call_args[0][1]
        assert params["top_k"] == 7

    async def test_sql_contains_ts_rank(self):
        pg, conn = _make_pg()
        await BM25Strategy(pg=pg).retrieve("query", top_k=5)
        sql_text = str(conn.execute.call_args[0][0])
        assert "ts_rank" in sql_text.lower()

    async def test_sql_contains_to_tsvector(self):
        pg, conn = _make_pg()
        await BM25Strategy(pg=pg).retrieve("query", top_k=5)
        sql_text = str(conn.execute.call_args[0][0])
        assert "tsvector" in sql_text.lower()

    async def test_empty_db_result_returns_empty(self):
        pg, _ = _make_pg(rows=[])
        result = await BM25Strategy(pg=pg).retrieve("nothing", top_k=5)
        assert result == []

    async def test_returns_chunk_schemas(self):
        pg, _ = _make_pg(rows=[_row()])
        result = await BM25Strategy(pg=pg).retrieve("hello", top_k=5)
        assert len(result) == 1
        assert isinstance(result[0], ChunkSchema)

    async def test_doc_id_populated_from_row(self):
        pg, _ = _make_pg(rows=[_row(doc_id="doc-x")])
        result = await BM25Strategy(pg=pg).retrieve("hi", top_k=5)
        assert result[0].doc_id == "doc-x"

    async def test_content_populated_from_row(self):
        pg, _ = _make_pg(rows=[_row(content="specific content")])
        result = await BM25Strategy(pg=pg).retrieve("specific", top_k=5)
        assert result[0].content == "specific content"

    async def test_content_hash_populated_from_row(self):
        pg, _ = _make_pg(rows=[_row(content_hash="sha256abc")])
        result = await BM25Strategy(pg=pg).retrieve("q", top_k=5)
        assert result[0].content_hash == "sha256abc"

    async def test_no_embedding_in_result(self):
        pg, _ = _make_pg(rows=[_row()])
        result = await BM25Strategy(pg=pg).retrieve("hello", top_k=5)
        assert result[0].embedding is None

    async def test_multiple_rows_all_returned(self):
        rows = [_row(content_hash=f"h{i}", content=f"text{i}") for i in range(3)]
        pg, _ = _make_pg(rows=rows)
        result = await BM25Strategy(pg=pg).retrieve("text", top_k=10)
        assert len(result) == 3

    async def test_parent_chunk_id_populated(self):
        pg, _ = _make_pg(rows=[_row(parent_chunk_id="parent-hash")])
        result = await BM25Strategy(pg=pg).retrieve("q", top_k=5)
        assert result[0].parent_chunk_id == "parent-hash"

    async def test_version_populated_from_row(self):
        pg, _ = _make_pg(rows=[_row(version="v2")])
        result = await BM25Strategy(pg=pg).retrieve("q", top_k=5)
        assert result[0].version == "v2"

    async def test_section_id_populated_from_row(self):
        pg, _ = _make_pg(rows=[_row(section_id="s7")])
        result = await BM25Strategy(pg=pg).retrieve("q", top_k=5)
        assert result[0].section_id == "s7"

    async def test_chunk_index_populated_from_row(self):
        pg, _ = _make_pg(rows=[_row(chunk_index=42)])
        result = await BM25Strategy(pg=pg).retrieve("q", top_k=5)
        assert result[0].chunk_index == 42

    async def test_section_id_and_chunk_index_not_transposed(self):
        pg, _ = _make_pg(rows=[_row(section_id="section-99", chunk_index=7)])
        result = await BM25Strategy(pg=pg).retrieve("q", top_k=5)
        assert result[0].section_id == "section-99"
        assert result[0].chunk_index == 7

    async def test_db_exception_propagates(self):
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(side_effect=Exception("DB connection failed"))
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_conn)
        ctx.__aexit__ = AsyncMock(return_value=None)
        engine = MagicMock()
        engine.connect = MagicMock(return_value=ctx)
        pg = MagicMock()
        pg.engine = engine
        with pytest.raises(Exception, match="DB connection failed"):
            await BM25Strategy(pg=pg).retrieve("query", top_k=5)
