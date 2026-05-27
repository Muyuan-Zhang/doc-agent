import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path

from app.clients.milvus import MilvusClient
from app.clients.mq import RedisStreamsMQClient
from app.clients.postgresql import PostgreSQLClient
from app.clients.redis import RedisClient
from app.core.config import settings
from app.core.exceptions import ServiceUnavailableError
from app.knowledge_base.chunker import DocumentChunker
from app.knowledge_base.cleaner import TextCleaner
from app.knowledge_base.dedup import ContentDeduplicator
from app.knowledge_base.embedder import ChunkEmbedder
from app.knowledge_base.parser import DocumentParser
from app.knowledge_base.store import KnowledgeBaseStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestResult:
    doc_id: str
    chunks_total: int
    chunks_new: int
    version: str


class UpdateCoordinator:
    def __init__(
        self,
        parser: DocumentParser,
        cleaner: TextCleaner,
        chunker: DocumentChunker,
        dedup: ContentDeduplicator,
        embedder: ChunkEmbedder,
        store: KnowledgeBaseStore,
    ) -> None:
        self._parser = parser
        self._cleaner = cleaner
        self._chunker = chunker
        self._dedup = dedup
        self._embedder = embedder
        self._store = store

    async def ingest(
        self,
        file_path: Path,
        doc_id: str,
        pg: PostgreSQLClient,
        redis: RedisClient,
        milvus: MilvusClient,
        mq: RedisStreamsMQClient,
    ) -> IngestResult:
        lock_key = f"{{kb:ingest}}:{doc_id}"
        acquired, token = await redis.acquire_lock(lock_key, ttl_seconds=300)
        if not acquired:
            raise ServiceUnavailableError(f"Document {doc_id} is already being ingested")

        try:
            parsed = self._parser.parse(file_path, doc_id=doc_id)
            await self._store.upsert_document(parsed, settings.knowledge_base_version, pg)
            await self._store.update_document_status(doc_id, "processing", 0, pg)

            cleaned_sections = tuple(
                self._cleaner.clean_section(s) for s in parsed.sections
            )
            cleaned_doc = dataclasses.replace(parsed, sections=cleaned_sections)

            all_chunks = self._chunker.chunk(cleaned_doc, settings.knowledge_base_version)
            new_chunks = await self._dedup.filter_new(all_chunks, pg)
            embedded = await self._embedder.embed(new_chunks)

            await self._store.save_chunks_meta(embedded, pg)
            await self._store.save_vectors(embedded, milvus)
            await self._store.update_document_status(doc_id, "indexed", len(all_chunks), pg)

            await mq.publish({
                "event": "kb_updated",
                "doc_id": doc_id,
                "version": settings.knowledge_base_version,
            })

            logger.info(
                "Ingest complete doc_id=%s total=%d new=%d",
                doc_id, len(all_chunks), len(embedded),
            )
            return IngestResult(
                doc_id=doc_id,
                chunks_total=len(all_chunks),
                chunks_new=len(embedded),
                version=settings.knowledge_base_version,
            )

        except Exception as exc:
            try:
                await self._store.update_document_status(doc_id, "failed", 0, pg)
            except Exception:
                logger.warning("Failed to update status to failed for doc_id=%s", doc_id)
            logger.error("Ingest failed doc_id=%s: %s", doc_id, exc)
            raise

        finally:
            await redis.release_lock(lock_key, token)
