"""Unit tests for ContentDeduplicator."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.knowledge_base.dedup import ContentDeduplicator
from app.models.chunk import ChunkSchema


def _chunk(content_hash: str) -> ChunkSchema:
    return ChunkSchema(
        doc_id="d1",
        section_id="s0000",
        chunk_index=0,
        content_hash=content_hash,
        version="v1",
        content="text",
    )


def _make_pg(existing_hashes: list[str]):
    """Build a mock PostgreSQLClient whose engine.connect returns known hashes."""
    rows = [(h,) for h in existing_hashes]
    mock_result = MagicMock()
    mock_result.__iter__ = MagicMock(return_value=iter(rows))

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=mock_result)

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_ctx.__aexit__ = AsyncMock(return_value=None)

    mock_engine = MagicMock()
    mock_engine.connect = MagicMock(return_value=mock_ctx)

    pg = MagicMock()
    pg.engine = mock_engine
    return pg


class TestContentDeduplicatorFilterNew:
    async def test_empty_chunks_returns_empty(self):
        pg = _make_pg([])
        result = await ContentDeduplicator().filter_new([], pg)
        assert result == []

    async def test_all_new_chunks_returned(self):
        chunks = [_chunk("aaa"), _chunk("bbb")]
        pg = _make_pg([])
        result = await ContentDeduplicator().filter_new(chunks, pg)
        assert len(result) == 2

    async def test_all_existing_returns_empty(self):
        chunks = [_chunk("aaa"), _chunk("bbb")]
        pg = _make_pg(["aaa", "bbb"])
        result = await ContentDeduplicator().filter_new(chunks, pg)
        assert result == []

    async def test_partial_dedup_returns_only_new(self):
        chunks = [_chunk("aaa"), _chunk("bbb"), _chunk("ccc")]
        pg = _make_pg(["bbb"])
        result = await ContentDeduplicator().filter_new(chunks, pg)
        hashes = {c.content_hash for c in result}
        assert hashes == {"aaa", "ccc"}

    async def test_queries_all_hashes_in_one_call(self):
        chunks = [_chunk("aaa"), _chunk("bbb"), _chunk("ccc")]
        pg = _make_pg([])
        await ContentDeduplicator().filter_new(chunks, pg)
        conn = pg.engine.connect.return_value.__aenter__.return_value
        assert conn.execute.await_count == 1

    async def test_passes_hashes_as_list_parameter(self):
        chunks = [_chunk("aaa"), _chunk("bbb")]
        pg = _make_pg([])
        await ContentDeduplicator().filter_new(chunks, pg)
        conn = pg.engine.connect.return_value.__aenter__.return_value
        _, kwargs = conn.execute.call_args
        params = conn.execute.call_args[0][1]
        assert sorted(params["hashes"]) == ["aaa", "bbb"]
