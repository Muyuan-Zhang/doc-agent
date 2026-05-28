import asyncio
import logging
import re
from typing import Any, Callable, TypeVar

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from app.clients.base import AbstractClient
from app.core.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _assert_uuid(value: str) -> None:
    if not _UUID_RE.match(value):
        raise ValueError(f"doc_id must be a UUID v4, got {value!r}")


_SAFE_FILTER_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _safe_filter_str(value: str, field: str = "value") -> None:
    if not _SAFE_FILTER_RE.match(value):
        raise ValueError(
            f"{field} must be 1-64 alphanumeric characters, underscores, or hyphens"
        )


class MilvusClient(AbstractClient):
    """
    Milvus 客户端。所有 collection 操作通过 alias 路由，
    调用方不得绕过此类直接传入 collection_name。
    """

    def __init__(self) -> None:
        self._alias = settings.milvus_alias

    async def _run_sync(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """所有 pymilvus 同步调用的唯一入口。禁止在此类其他方法中直接调用 asyncio.to_thread。"""
        return await asyncio.to_thread(fn, *args, **kwargs)

    async def connect(self) -> None:
        await self._run_sync(
            connections.connect,
            alias=self._alias,
            host=settings.milvus_host,
            port=settings.milvus_port,
        )
        logger.info("Milvus connected via alias=%s", self._alias)

    async def disconnect(self) -> None:
        await self._run_sync(connections.disconnect, alias=self._alias)
        logger.info("Milvus disconnected alias=%s", self._alias)

    async def ping(self) -> bool:
        try:
            await self._run_sync(utility.get_server_version, using=self._alias)
            return True
        except Exception as exc:
            logger.warning("Milvus ping failed: %s", exc)
            return False

    async def ensure_kb_collection(self) -> None:
        """Idempotently create the knowledge-base collection with HNSW index."""
        def _ensure() -> None:
            col_name = settings.milvus_kb_collection
            if utility.has_collection(col_name, using=self._alias):
                return
            fields = [
                FieldSchema("chunk_id", DataType.VARCHAR, max_length=255, is_primary=True, auto_id=False),
                FieldSchema("doc_id", DataType.VARCHAR, max_length=36),
                FieldSchema("section_id", DataType.VARCHAR, max_length=255),
                FieldSchema("chunk_index", DataType.INT64),
                FieldSchema("version", DataType.VARCHAR, max_length=100),
                FieldSchema("content", DataType.VARCHAR, max_length=4096),
                FieldSchema("content_hash", DataType.VARCHAR, max_length=64),
                FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=settings.embedding_dim),
            ]
            schema = CollectionSchema(fields, description="Knowledge base chunks")
            col = Collection(name=col_name, schema=schema, using=self._alias)
            col.create_index(
                field_name="embedding",
                index_params={
                    "metric_type": "COSINE",
                    "index_type": "HNSW",
                    "params": {"M": 16, "efConstruction": 200},
                },
            )
            col.load()
            logger.info("Milvus collection created: %s", col_name)

        await self._run_sync(_ensure)

    async def insert(self, entities: list[dict]) -> list[str]:
        """Insert entities into the knowledge-base collection, return primary keys."""
        def _insert() -> list[str]:
            col = Collection(settings.milvus_kb_collection, using=self._alias)
            result = col.insert(entities)
            return [str(pk) for pk in result.primary_keys]

        return await self._run_sync(_insert)

    async def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete all vectors belonging to a document (two-step: query then delete by PK)."""
        _assert_uuid(doc_id)

        def _delete() -> None:
            col = Collection(settings.milvus_kb_collection, using=self._alias)
            results = col.query(
                expr=f'doc_id == "{doc_id}"',
                output_fields=["chunk_id"],
            )
            if not results:
                return
            ids_expr = ", ".join(f'"{r["chunk_id"]}"' for r in results)
            col.delete(expr=f"chunk_id in [{ids_expr}]")

        await self._run_sync(_delete)

    async def search(
        self,
        *,
        embedding: list[float],
        top_k: int,
        output_fields: list[str] | None = None,
    ) -> list[dict]:
        """HNSW cosine similarity search; returns list of field-value dicts per hit."""
        if output_fields is None:
            output_fields = [
                "chunk_id", "doc_id", "section_id", "chunk_index",
                "version", "content", "content_hash",
            ]

        def _search() -> list[dict]:
            col = Collection(settings.milvus_kb_collection, using=self._alias)
            results = col.search(
                data=[embedding],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"ef": 64}},
                limit=top_k,
                output_fields=output_fields,
            )
            hits = []
            for hit in results[0]:
                entity = {f: hit.entity.get(f) for f in output_fields}
                entity["score"] = hit.score
                hits.append(entity)
    async def ensure_memory_collection(self) -> None:
        """Idempotently create the memory-vectors collection with HNSW index."""
        def _ensure() -> None:
            col_name = settings.memory_milvus_collection
            if utility.has_collection(col_name, using=self._alias):
                return
            fields = [
                FieldSchema("fact_id", DataType.VARCHAR, max_length=36, is_primary=True, auto_id=False),
                FieldSchema("user_id", DataType.VARCHAR, max_length=36),
                FieldSchema("content", DataType.VARCHAR, max_length=4096),
                FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=settings.embedding_dim),
            ]
            schema = CollectionSchema(fields, description="Static memory facts")
            col = Collection(name=col_name, schema=schema, using=self._alias)
            col.create_index(
                field_name="embedding",
                index_params={
                    "metric_type": "COSINE",
                    "index_type": "HNSW",
                    "params": {"M": 16, "efConstruction": 200},
                },
            )
            col.load()
            logger.info("Milvus memory collection created: %s", col_name)

        await self._run_sync(_ensure)

    async def memory_insert(self, entities: list[dict]) -> list[str]:
        """Insert static-fact entities into the memory collection."""
        def _insert() -> list[str]:
            col = Collection(settings.memory_milvus_collection, using=self._alias)
            result = col.insert(entities)
            return [str(pk) for pk in result.primary_keys]

        return await self._run_sync(_insert)

    async def memory_search(
        self, query_embedding: list[float], user_id: str, top_k: int
    ) -> list[dict]:
        """ANN search filtered by user_id; returns list of hit dicts."""
        _safe_filter_str(user_id, "user_id")

        def _search() -> list[dict]:
            col = Collection(settings.memory_milvus_collection, using=self._alias)
            results = col.search(
                data=[query_embedding],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"ef": 64}},
                limit=top_k,
                expr=f'user_id == "{user_id}"',
                output_fields=["fact_id", "user_id", "content"],
            )
            hits = []
            for hit in results[0]:
                hits.append({
                    "fact_id": hit.entity.get("fact_id"),
                    "user_id": hit.entity.get("user_id"),
                    "content": hit.entity.get("content"),
                })
            return hits

        return await self._run_sync(_search)

    async def memory_delete(self, fact_id: str) -> None:
        """Delete a memory vector by fact_id."""
        _assert_uuid(fact_id)

        def _delete() -> None:
            col = Collection(settings.memory_milvus_collection, using=self._alias)
            col.delete(expr=f'fact_id == "{fact_id}"')

        await self._run_sync(_delete)

    async def query_ids_by_doc_id(self, doc_id: str) -> list[str]:
        """Return all chunk_ids stored for a given document."""
        _assert_uuid(doc_id)

        def _query() -> list[str]:
            col = Collection(settings.milvus_kb_collection, using=self._alias)
            results = col.query(
                expr=f'doc_id == "{doc_id}"',
                output_fields=["chunk_id"],
            )
            return [r["chunk_id"] for r in results]

        return await self._run_sync(_query)
