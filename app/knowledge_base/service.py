import logging
import os
import tempfile
import uuid
from pathlib import Path

from fastapi import UploadFile

from app.clients.milvus import MilvusClient
from app.clients.mq import RedisStreamsMQClient
from app.clients.postgresql import PostgreSQLClient
from app.clients.redis import RedisClient
from app.core.config import settings
from app.core.exceptions import NotFoundError, ValidationError
from app.knowledge_base.chunker import DocumentChunker
from app.knowledge_base.cleaner import TextCleaner
from app.knowledge_base.dedup import ContentDeduplicator
from app.knowledge_base.embedder import ChunkEmbedder
from app.knowledge_base.parser import DocumentParser
from app.knowledge_base.store import KnowledgeBaseStore
from app.knowledge_base.update import UpdateCoordinator

logger = logging.getLogger(__name__)

_ALLOWED = {"pdf", "txt"}
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
_MAGIC_BYTES: dict[str, bytes] = {
    "pdf": b"%PDF-",
}


def _check_magic(ext: str, data: bytes) -> bool:
    magic = _MAGIC_BYTES.get(ext)
    return data[: len(magic)] == magic if magic else True


class KnowledgeBaseService:
    def __init__(
        self,
        pg: PostgreSQLClient,
        redis: RedisClient,
        milvus: MilvusClient,
        mq: RedisStreamsMQClient,
        embedder: ChunkEmbedder,
    ) -> None:
        self._pg = pg
        self._redis = redis
        self._milvus = milvus
        self._mq = mq
        self._store = KnowledgeBaseStore()
        self._coordinator = UpdateCoordinator(
            parser=DocumentParser(),
            cleaner=TextCleaner(),
            chunker=DocumentChunker(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            ),
            dedup=ContentDeduplicator(),
            embedder=embedder,
            store=self._store,
        )

    async def prepare_upload(self, file: UploadFile) -> tuple[str, Path]:
        raw_name = file.filename or "upload"
        filename = Path(raw_name).name[:256] or "upload"
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in _ALLOWED:
            raise ValidationError(f"Unsupported file type: {ext!r}. Allowed: {', '.join(sorted(_ALLOWED))}")

        content = await file.read()
        if len(content) > _MAX_UPLOAD_BYTES:
            raise ValidationError(
                f"File exceeds maximum allowed size of {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB"
            )
        if not _check_magic(ext, content):
            raise ValidationError(f"File content does not match declared type: {ext!r}")

        doc_id = str(uuid.uuid4())
        fd, tmp_name = tempfile.mkstemp(suffix=f".{ext}")
        tmp_path = Path(tmp_name)
        try:
            os.write(fd, content)
            os.close(fd)
            await self._store.create_pending_document(
                doc_id=doc_id,
                filename=filename,
                file_type=ext,
                version=settings.knowledge_base_version,
                pg=self._pg,
            )
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            tmp_path.unlink(missing_ok=True)
            raise
        return doc_id, tmp_path

    async def run_ingest(self, doc_id: str, file_path: Path) -> None:
        try:
            await self._coordinator.ingest(
                file_path, doc_id,
                self._pg, self._redis, self._milvus, self._mq,
            )
        finally:
            try:
                file_path.unlink(missing_ok=True)
            except Exception:
                logger.warning("Could not delete temp file: %s", file_path)

    async def get_document_status(self, doc_id: str) -> dict:
        status = await self._store.get_document_status(doc_id, self._pg)
        if status is None:
            raise NotFoundError(f"Document {doc_id} not found")
        return status

    async def delete_document(self, doc_id: str) -> None:
        status = await self._store.get_document_status(doc_id, self._pg)
        if status is None:
            raise NotFoundError(f"Document {doc_id} not found")
        await self._store.delete_document(doc_id, self._pg, self._milvus)
