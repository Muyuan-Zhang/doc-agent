import asyncio
import logging
from typing import Any, Callable, TypeVar

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from app.clients.base import AbstractClient
from app.core.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


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

    async def query_ids_by_doc_id(self, doc_id: str) -> list[str]:
        """Return all chunk_ids stored for a given document."""
        def _query() -> list[str]:
            col = Collection(settings.milvus_kb_collection, using=self._alias)
            results = col.query(
                expr=f'doc_id == "{doc_id}"',
                output_fields=["chunk_id"],
            )
            return [r["chunk_id"] for r in results]

        return await self._run_sync(_query)
