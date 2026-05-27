import logging

from sqlalchemy import text

from app.clients.milvus import MilvusClient
from app.clients.postgresql import PostgreSQLClient
from app.knowledge_base.parser import ParsedDocument
from app.models.chunk import ChunkSchema

logger = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id       VARCHAR(36)  PRIMARY KEY,
    filename     VARCHAR(512) NOT NULL,
    file_type    VARCHAR(10)  NOT NULL CHECK (file_type IN ('pdf','txt')),
    status       VARCHAR(50)  NOT NULL DEFAULT 'pending',
    chunk_count  INT          NOT NULL DEFAULT 0,
    version      VARCHAR(100) NOT NULL,
    content_hash VARCHAR(64),
    created_at   TIMESTAMPTZ  DEFAULT NOW(),
    updated_at   TIMESTAMPTZ  DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS chunks_metadata (
    chunk_id        VARCHAR(255) PRIMARY KEY,
    doc_id          VARCHAR(36)  NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    section_id      VARCHAR(255),
    chunk_index     INT          NOT NULL,
    parent_chunk_id VARCHAR(255),
    content_hash    VARCHAR(64)  NOT NULL UNIQUE,
    version         VARCHAR(100) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks_metadata(doc_id);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
"""


def _chunk_id(chunk: ChunkSchema) -> str:
    return f"{chunk.doc_id}:{chunk.section_id}:{chunk.chunk_index:06d}"


class KnowledgeBaseStore:
    async def ensure_schema(self, pg: PostgreSQLClient, milvus: MilvusClient) -> None:
        async with pg.engine.begin() as conn:
            await conn.execute(text(_DDL))
        await milvus.ensure_kb_collection()
        logger.info("Knowledge base schema ensured")

    async def create_pending_document(
        self,
        doc_id: str,
        filename: str,
        file_type: str,
        version: str,
        pg: PostgreSQLClient,
    ) -> None:
        async with pg.engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO documents (doc_id, filename, file_type, status, version)
                    VALUES (:doc_id, :filename, :file_type, 'pending', :version)
                    ON CONFLICT (doc_id) DO NOTHING
                """),
                {"doc_id": doc_id, "filename": filename, "file_type": file_type, "version": version},
            )

    async def upsert_document(
        self, doc: ParsedDocument, version: str, pg: PostgreSQLClient
    ) -> None:
        async with pg.engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO documents (doc_id, filename, file_type, status, version, content_hash)
                    VALUES (:doc_id, :filename, :file_type, 'pending', :version, :content_hash)
                    ON CONFLICT (doc_id) DO UPDATE SET
                        version      = EXCLUDED.version,
                        content_hash = EXCLUDED.content_hash,
                        updated_at   = NOW()
                """),
                {
                    "doc_id": doc.doc_id,
                    "filename": doc.filename,
                    "file_type": doc.file_type,
                    "version": version,
                    "content_hash": doc.content_hash,
                },
            )

    async def update_document_status(
        self,
        doc_id: str,
        status: str,
        chunk_count: int,
        pg: PostgreSQLClient,
    ) -> None:
        async with pg.engine.begin() as conn:
            await conn.execute(
                text("""
                    UPDATE documents
                    SET status = :status, chunk_count = :chunk_count, updated_at = NOW()
                    WHERE doc_id = :doc_id
                """),
                {"doc_id": doc_id, "status": status, "chunk_count": chunk_count},
            )

    async def save_chunks_meta(
        self, chunks: list[ChunkSchema], pg: PostgreSQLClient
    ) -> None:
        if not chunks:
            return
        rows = [
            {
                "chunk_id": _chunk_id(c),
                "doc_id": c.doc_id,
                "section_id": c.section_id,
                "chunk_index": c.chunk_index,
                "parent_chunk_id": c.parent_chunk_id,
                "content_hash": c.content_hash,
                "version": c.version,
            }
            for c in chunks
        ]
        async with pg.engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO chunks_metadata
                        (chunk_id, doc_id, section_id, chunk_index, parent_chunk_id, content_hash, version)
                    VALUES
                        (:chunk_id, :doc_id, :section_id, :chunk_index, :parent_chunk_id, :content_hash, :version)
                    ON CONFLICT (content_hash) DO NOTHING
                """),
                rows,
            )

    async def save_vectors(
        self, chunks: list[ChunkSchema], milvus: MilvusClient
    ) -> None:
        entities = [
            {
                "chunk_id": _chunk_id(c),
                "doc_id": c.doc_id,
                "section_id": c.section_id,
                "chunk_index": c.chunk_index,
                "version": c.version,
                "content": c.content[:4096],
                "embedding": c.embedding,
            }
            for c in chunks
            if c.embedding is not None
        ]
        if entities:
            await milvus.insert(entities)

    async def get_document_status(
        self, doc_id: str, pg: PostgreSQLClient
    ) -> dict | None:
        async with pg.engine.connect() as conn:
            result = await conn.execute(
                text("""
                    SELECT doc_id, filename, status, chunk_count, version
                    FROM documents WHERE doc_id = :doc_id
                """),
                {"doc_id": doc_id},
            )
            row = result.fetchone()
        if row is None:
            return None
        return {
            "doc_id": row[0],
            "filename": row[1],
            "status": row[2],
            "chunk_count": row[3],
            "version": row[4],
        }

    async def delete_document(
        self, doc_id: str, pg: PostgreSQLClient, milvus: MilvusClient
    ) -> None:
        await milvus.delete_by_doc_id(doc_id)
        async with pg.engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM documents WHERE doc_id = :doc_id"),
                {"doc_id": doc_id},
            )
        logger.info("Document deleted: doc_id=%s", doc_id)
