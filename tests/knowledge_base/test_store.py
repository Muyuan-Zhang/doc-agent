"""Unit tests for KnowledgeBaseStore."""
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from app.knowledge_base.parser import ParsedDocument, Section
from app.knowledge_base.store import KnowledgeBaseStore, _chunk_id
from app.models.chunk import ChunkSchema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pg():
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=MagicMock(fetchone=MagicMock(return_value=None)))
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


def _make_milvus():
    m = MagicMock()
    m.ensure_kb_collection = AsyncMock()
    m.insert = AsyncMock(return_value=[])
    m.delete_by_doc_id = AsyncMock()
    return m


def _make_doc() -> ParsedDocument:
    return ParsedDocument(
        doc_id="doc-1",
        filename="test.pdf",
        file_type="pdf",
        sections=(Section(section_id="p0000", heading=None, content="content"),),
        content_hash="abc123",
    )


def _make_chunk(idx: int = 0) -> ChunkSchema:
    return ChunkSchema(
        doc_id="doc-1",
        section_id="s0000",
        chunk_index=idx,
        content_hash=f"hash{idx}",
        version="v1",
        content="text",
        embedding=[0.1, 0.2],
    )


# ---------------------------------------------------------------------------
# chunk_id helper
# ---------------------------------------------------------------------------

class TestChunkIdHelper:
    def test_format_is_doc_section_index(self):
        c = _make_chunk(idx=3)
        assert _chunk_id(c) == "doc-1:s0000:000003"

    def test_index_is_zero_padded_to_6(self):
        c = _make_chunk(idx=1)
        assert _chunk_id(c).endswith(":000001")


# ---------------------------------------------------------------------------
# ensure_schema
# ---------------------------------------------------------------------------

class TestEnsureSchema:
    async def test_calls_pg_execute(self):
        pg, conn = _make_pg()
        milvus = _make_milvus()
        await KnowledgeBaseStore().ensure_schema(pg, milvus)
        conn.execute.assert_awaited()

    async def test_calls_milvus_ensure_kb_collection(self):
        pg, _ = _make_pg()
        milvus = _make_milvus()
        await KnowledgeBaseStore().ensure_schema(pg, milvus)
        milvus.ensure_kb_collection.assert_awaited_once()


# ---------------------------------------------------------------------------
# create_pending_document
# ---------------------------------------------------------------------------

class TestCreatePendingDocument:
    async def test_executes_insert_sql(self):
        pg, conn = _make_pg()
        await KnowledgeBaseStore().create_pending_document(
            "d1", "file.txt", "txt", "v1", pg
        )
        conn.execute.assert_awaited_once()

    async def test_passes_correct_params(self):
        pg, conn = _make_pg()
        await KnowledgeBaseStore().create_pending_document(
            "d1", "file.txt", "txt", "v1", pg
        )
        params = conn.execute.call_args[0][1]
        assert params["doc_id"] == "d1"
        assert params["filename"] == "file.txt"
        assert params["file_type"] == "txt"
        assert params["version"] == "v1"


# ---------------------------------------------------------------------------
# upsert_document
# ---------------------------------------------------------------------------

class TestUpsertDocument:
    async def test_executes_upsert_sql(self):
        pg, conn = _make_pg()
        await KnowledgeBaseStore().upsert_document(_make_doc(), "v1", pg)
        conn.execute.assert_awaited_once()

    async def test_passes_doc_id_and_content_hash(self):
        pg, conn = _make_pg()
        doc = _make_doc()
        await KnowledgeBaseStore().upsert_document(doc, "v1", pg)
        params = conn.execute.call_args[0][1]
        assert params["doc_id"] == "doc-1"
        assert params["content_hash"] == "abc123"


# ---------------------------------------------------------------------------
# save_chunks_meta
# ---------------------------------------------------------------------------

class TestSaveChunksMeta:
    async def test_empty_list_does_not_execute(self):
        pg, conn = _make_pg()
        await KnowledgeBaseStore().save_chunks_meta([], pg)
        conn.execute.assert_not_awaited()

    async def test_executes_insert_for_nonempty(self):
        pg, conn = _make_pg()
        await KnowledgeBaseStore().save_chunks_meta([_make_chunk()], pg)
        conn.execute.assert_awaited_once()

    async def test_passes_list_of_rows(self):
        pg, conn = _make_pg()
        chunks = [_make_chunk(0), _make_chunk(1)]
        await KnowledgeBaseStore().save_chunks_meta(chunks, pg)
        rows = conn.execute.call_args[0][1]
        assert len(rows) == 2

    async def test_row_has_chunk_id(self):
        pg, conn = _make_pg()
        chunk = _make_chunk(2)
        await KnowledgeBaseStore().save_chunks_meta([chunk], pg)
        row = conn.execute.call_args[0][1][0]
        assert row["chunk_id"] == "doc-1:s0000:000002"

    async def test_row_has_content(self):
        pg, conn = _make_pg()
        chunk = _make_chunk(0)
        await KnowledgeBaseStore().save_chunks_meta([chunk], pg)
        row = conn.execute.call_args[0][1][0]
        assert row["content"] == "text"


class TestDDLSchema:
    def test_ddl_includes_content_column(self):
        from app.knowledge_base.store import _DDL
        assert "content" in _DDL

    def test_ddl_includes_gin_index(self):
        from app.knowledge_base.store import _DDL
        assert "gin" in _DDL.lower()


# ---------------------------------------------------------------------------
# save_vectors
# ---------------------------------------------------------------------------

class TestSaveVectors:
    async def test_empty_list_does_not_call_milvus(self):
        milvus = _make_milvus()
        pg = MagicMock()
        await KnowledgeBaseStore().save_vectors([], milvus)
        milvus.insert.assert_not_awaited()

    async def test_chunk_without_embedding_excluded(self):
        milvus = _make_milvus()
        chunk = ChunkSchema(
            doc_id="d1", section_id="s0000", chunk_index=0,
            content_hash="h", version="v1", content="text", embedding=None,
        )
        await KnowledgeBaseStore().save_vectors([chunk], milvus)
        milvus.insert.assert_not_awaited()

    async def test_calls_milvus_insert_with_entities(self):
        milvus = _make_milvus()
        await KnowledgeBaseStore().save_vectors([_make_chunk()], milvus)
        milvus.insert.assert_awaited_once()

    async def test_entity_contains_required_fields(self):
        milvus = _make_milvus()
        await KnowledgeBaseStore().save_vectors([_make_chunk(0)], milvus)
        entity = milvus.insert.call_args[0][0][0]
        for field in ("chunk_id", "doc_id", "section_id", "chunk_index", "version", "content", "embedding"):
            assert field in entity

    async def test_content_truncated_to_4096_chars(self):
        milvus = _make_milvus()
        long_chunk = ChunkSchema(
            doc_id="d1", section_id="s0000", chunk_index=0,
            content_hash="h", version="v1",
            content="x" * 5000,
            embedding=[0.1],
        )
        await KnowledgeBaseStore().save_vectors([long_chunk], milvus)
        entity = milvus.insert.call_args[0][0][0]
        assert len(entity["content"]) == 4096


# ---------------------------------------------------------------------------
# get_document_status
# ---------------------------------------------------------------------------

class TestGetDocumentStatus:
    async def test_returns_none_when_not_found(self):
        pg, conn = _make_pg()
        conn.execute.return_value.fetchone = MagicMock(return_value=None)
        result = await KnowledgeBaseStore().get_document_status("missing", pg)
        assert result is None

    async def test_returns_dict_when_found(self):
        pg, conn = _make_pg()
        conn.execute.return_value.fetchone = MagicMock(
            return_value=("d1", "file.pdf", "indexed", 10, "v1")
        )
        result = await KnowledgeBaseStore().get_document_status("d1", pg)
        assert result == {
            "doc_id": "d1",
            "filename": "file.pdf",
            "status": "indexed",
            "chunk_count": 10,
            "version": "v1",
        }


# ---------------------------------------------------------------------------
# delete_document
# ---------------------------------------------------------------------------

class TestDeleteDocument:
    async def test_calls_milvus_delete_by_doc_id(self):
        pg, conn = _make_pg()
        milvus = _make_milvus()
        await KnowledgeBaseStore().delete_document("doc-1", pg, milvus)
        milvus.delete_by_doc_id.assert_awaited_once_with("doc-1")

    async def test_executes_pg_delete(self):
        pg, conn = _make_pg()
        milvus = _make_milvus()
        await KnowledgeBaseStore().delete_document("doc-1", pg, milvus)
        conn.execute.assert_awaited_once()
