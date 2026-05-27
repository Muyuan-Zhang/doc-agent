"""
E2E tests for the knowledge-base API pipeline.

All network clients (postgres, redis, milvus, mq, llm) are mocked via app.state.
The KnowledgeBaseService and UpdateCoordinator are also mocked at app.state
so we test the router → service boundary only.
BackgroundTasks are executed synchronously by patching them.
"""
import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


def _make_client(ping_ok: bool = True) -> MagicMock:
    m = MagicMock()
    m.ping = AsyncMock(return_value=ping_ok)
    return m


def _app_with_state():
    from app import create_app
    app = create_app()
    app.state.postgres = _make_client()
    app.state.redis = _make_client()
    app.state.milvus = _make_client()
    app.state.mq = _make_client()
    app.state.llm = _make_client()
    return app


class TestUploadDocumentEndpoint:
    async def test_upload_returns_202(self):
        app = _app_with_state()
        with patch("app.routers.knowledge_base.KnowledgeBaseService") as mock_svc_cls:
            svc = AsyncMock()
            svc.prepare_upload = AsyncMock(return_value=("doc-uuid-1", "/tmp/f.txt"))
            svc.run_ingest = AsyncMock()
            mock_svc_cls.return_value = svc

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                r = await c.post(
                    "/knowledge-base/documents",
                    files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
                )

        assert r.status_code == 202

    async def test_upload_returns_doc_id(self):
        app = _app_with_state()
        with patch("app.routers.knowledge_base.KnowledgeBaseService") as mock_svc_cls:
            svc = AsyncMock()
            svc.prepare_upload = AsyncMock(return_value=("doc-uuid-42", "/tmp/f.txt"))
            svc.run_ingest = AsyncMock()
            mock_svc_cls.return_value = svc

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                r = await c.post(
                    "/knowledge-base/documents",
                    files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
                )

        assert r.json()["doc_id"] == "doc-uuid-42"

    async def test_upload_invalid_type_returns_422(self):
        app = _app_with_state()
        with patch("app.routers.knowledge_base.KnowledgeBaseService") as mock_svc_cls:
            from app.core.exceptions import ValidationError
            svc = AsyncMock()
            svc.prepare_upload = AsyncMock(side_effect=ValidationError("Unsupported file type"))
            mock_svc_cls.return_value = svc

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                r = await c.post(
                    "/knowledge-base/documents",
                    files={"file": ("test.docx", io.BytesIO(b"data"), "application/octet-stream")},
                )

        assert r.status_code == 422


_DOC_UUID = "a0000000-0000-4000-8000-000000000001"
_MISSING_UUID = "a0000000-0000-4000-8000-000000000002"


class TestGetDocumentStatusEndpoint:
    async def test_status_returns_200_when_found(self):
        app = _app_with_state()
        row = {"doc_id": _DOC_UUID, "filename": "f.txt", "status": "indexed",
               "chunk_count": 5, "version": "v1"}
        with patch("app.routers.knowledge_base.KnowledgeBaseService") as mock_svc_cls:
            svc = AsyncMock()
            svc.get_document_status = AsyncMock(return_value=row)
            mock_svc_cls.return_value = svc

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                r = await c.get(f"/knowledge-base/documents/{_DOC_UUID}/status")

        assert r.status_code == 200
        assert r.json()["status"] == "indexed"

    async def test_status_returns_404_when_not_found(self):
        app = _app_with_state()
        with patch("app.routers.knowledge_base.KnowledgeBaseService") as mock_svc_cls:
            from app.core.exceptions import NotFoundError
            svc = AsyncMock()
            svc.get_document_status = AsyncMock(side_effect=NotFoundError("not found"))
            mock_svc_cls.return_value = svc

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                r = await c.get(f"/knowledge-base/documents/{_MISSING_UUID}/status")

        assert r.status_code == 404

    async def test_status_returns_422_for_non_uuid(self):
        app = _app_with_state()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/knowledge-base/documents/not-a-uuid/status")
        assert r.status_code == 422


class TestDeleteDocumentEndpoint:
    async def test_delete_returns_204(self):
        app = _app_with_state()
        with patch("app.routers.knowledge_base.KnowledgeBaseService") as mock_svc_cls:
            svc = AsyncMock()
            svc.delete_document = AsyncMock()
            mock_svc_cls.return_value = svc

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                r = await c.delete(f"/knowledge-base/documents/{_DOC_UUID}")

        assert r.status_code == 204

    async def test_delete_returns_404_when_not_found(self):
        app = _app_with_state()
        with patch("app.routers.knowledge_base.KnowledgeBaseService") as mock_svc_cls:
            from app.core.exceptions import NotFoundError
            svc = AsyncMock()
            svc.delete_document = AsyncMock(side_effect=NotFoundError("not found"))
            mock_svc_cls.return_value = svc

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                r = await c.delete(f"/knowledge-base/documents/{_MISSING_UUID}")

        assert r.status_code == 404

    async def test_delete_returns_422_for_non_uuid(self):
        app = _app_with_state()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.delete("/knowledge-base/documents/not-a-uuid")
        assert r.status_code == 422
