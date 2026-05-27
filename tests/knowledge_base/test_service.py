"""Unit tests for KnowledgeBaseService."""
import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import UploadFile

from app.core.exceptions import NotFoundError, ValidationError
from app.knowledge_base.service import KnowledgeBaseService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_upload(filename: str, content: bytes = b"data") -> UploadFile:
    file = UploadFile(filename=filename, file=io.BytesIO(content))
    return file


def _make_store(*, status_row=None) -> MagicMock:
    store = MagicMock()
    store.create_pending_document = AsyncMock()
    store.get_document_status = AsyncMock(return_value=status_row)
    store.delete_document = AsyncMock()
    return store


def _make_service(store: MagicMock | None = None) -> KnowledgeBaseService:
    pg = MagicMock()
    redis = MagicMock()
    milvus = MagicMock()
    mq = MagicMock()
    embedder = MagicMock()
    svc = KnowledgeBaseService(pg=pg, redis=redis, milvus=milvus, mq=mq, embedder=embedder)
    if store is not None:
        svc._store = store
    return svc


# ---------------------------------------------------------------------------
# prepare_upload
# ---------------------------------------------------------------------------

class TestPrepareUpload:
    async def test_returns_doc_id_and_path(self):
        svc = _make_service(_make_store())
        doc_id, tmp_path = await svc.prepare_upload(_make_upload("test.txt"))
        assert isinstance(doc_id, str) and len(doc_id) == 36
        assert isinstance(tmp_path, Path)

    async def test_rejects_unsupported_extension(self):
        svc = _make_service(_make_store())
        with pytest.raises(ValidationError):
            await svc.prepare_upload(_make_upload("file.docx"))

    async def test_rejects_no_extension(self):
        svc = _make_service(_make_store())
        with pytest.raises(ValidationError):
            await svc.prepare_upload(_make_upload("file"))

    async def test_accepts_pdf(self):
        svc = _make_service(_make_store())
        doc_id, _ = await svc.prepare_upload(_make_upload("doc.pdf", b"%PDF-1.4"))
        assert doc_id

    async def test_calls_create_pending_document(self):
        store = _make_store()
        svc = _make_service(store)
        await svc.prepare_upload(_make_upload("test.txt"))
        store.create_pending_document.assert_awaited_once()

    async def test_pending_doc_id_matches_returned_id(self):
        store = _make_store()
        svc = _make_service(store)
        doc_id, _ = await svc.prepare_upload(_make_upload("test.txt"))
        call_doc_id = store.create_pending_document.call_args.kwargs["doc_id"]
        assert call_doc_id == doc_id

    async def test_temp_file_created(self):
        svc = _make_service(_make_store())
        _, tmp_path = await svc.prepare_upload(_make_upload("test.txt", b"hello"))
        assert tmp_path.exists()
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# run_ingest
# ---------------------------------------------------------------------------

class TestRunIngest:
    async def test_calls_coordinator_ingest(self):
        svc = _make_service(_make_store())
        coord = MagicMock()
        coord.ingest = AsyncMock()
        svc._coordinator = coord
        tmp = Path("/tmp/fake.txt")
        await svc.run_ingest("doc-1", tmp)
        coord.ingest.assert_awaited_once()

    async def test_deletes_temp_file_after_success(self, tmp_path):
        svc = _make_service(_make_store())
        coord = MagicMock()
        coord.ingest = AsyncMock()
        svc._coordinator = coord
        tmp = tmp_path / "doc.txt"
        tmp.write_text("content")
        await svc.run_ingest("doc-1", tmp)
        assert not tmp.exists()

    async def test_deletes_temp_file_even_on_error(self, tmp_path):
        svc = _make_service(_make_store())
        coord = MagicMock()
        coord.ingest = AsyncMock(side_effect=RuntimeError("fail"))
        svc._coordinator = coord
        tmp = tmp_path / "doc.txt"
        tmp.write_text("content")
        with pytest.raises(RuntimeError):
            await svc.run_ingest("doc-1", tmp)
        assert not tmp.exists()


# ---------------------------------------------------------------------------
# get_document_status
# ---------------------------------------------------------------------------

class TestGetDocumentStatus:
    async def test_raises_not_found_when_missing(self):
        svc = _make_service(_make_store(status_row=None))
        with pytest.raises(NotFoundError):
            await svc.get_document_status("unknown-id")

    async def test_returns_status_dict_when_found(self):
        row = {"doc_id": "d1", "filename": "f.pdf", "status": "indexed",
               "chunk_count": 5, "version": "v1"}
        svc = _make_service(_make_store(status_row=row))
        result = await svc.get_document_status("d1")
        assert result == row


# ---------------------------------------------------------------------------
# delete_document
# ---------------------------------------------------------------------------

class TestDeleteDocument:
    async def test_raises_not_found_when_missing(self):
        svc = _make_service(_make_store(status_row=None))
        with pytest.raises(NotFoundError):
            await svc.delete_document("unknown-id")

    async def test_calls_store_delete_when_found(self):
        row = {"doc_id": "d1", "filename": "f.txt", "status": "indexed",
               "chunk_count": 2, "version": "v1"}
        store = _make_store(status_row=row)
        svc = _make_service(store)
        await svc.delete_document("d1")
        store.delete_document.assert_awaited_once()
