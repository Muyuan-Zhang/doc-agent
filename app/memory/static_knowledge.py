import hashlib
import logging
import uuid

from sqlalchemy import text

from app.clients.llm import AbstractLLMClient
from app.clients.milvus import MilvusClient
from app.clients.postgresql import PostgreSQLClient
from app.core.exceptions import NotFoundError
from app.memory.schemas import StaticFact

logger = logging.getLogger(__name__)


class StaticKnowledgeStore:
    async def add_fact(
        self,
        pg: PostgreSQLClient,
        milvus: MilvusClient,
        llm: AbstractLLMClient,
        user_id: str,
        content: str,
    ) -> StaticFact:
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        fact_id = str(uuid.uuid4())

        # Embed before writing to PG — if this fails, the caller can retry cleanly.
        # Storing in PG without a vector would create an un-retrievable orphan record.
        embedding = await llm.embed(content)

        async with pg.engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO memory_static_facts (fact_id, user_id, content, content_hash)
                    VALUES (:fact_id, :user_id, :content, :content_hash)
                    ON CONFLICT (content_hash) DO NOTHING
                """),
                {
                    "fact_id": fact_id,
                    "user_id": user_id,
                    "content": content,
                    "content_hash": content_hash,
                },
            )

        await milvus.memory_insert([{
            "fact_id": fact_id,
            "user_id": user_id,
            "content": content[:4096],
            "embedding": embedding,
        }])

        logger.info("Added static fact user=%s fact_id=%s", user_id, fact_id)
        return StaticFact(
            fact_id=fact_id,
            user_id=user_id,
            content=content,
            content_hash=content_hash,
            embedding=embedding,
        )

    async def search_facts(
        self,
        milvus: MilvusClient,
        query_embedding: list[float],
        user_id: str,
        top_k: int = 5,
    ) -> list[StaticFact]:
        hits = await milvus.memory_search(query_embedding, user_id, top_k)
        return [
            StaticFact(
                fact_id=h["fact_id"],
                user_id=h["user_id"],
                content=h["content"],
                content_hash=hashlib.sha256(h["content"].encode()).hexdigest(),
            )
            for h in hits
        ]

    async def list_facts(
        self,
        pg: PostgreSQLClient,
        user_id: str,
    ) -> list[StaticFact]:
        """Return all static facts for a user (no embedding needed — for management UI)."""
        async with pg.engine.begin() as conn:
            rows = await conn.execute(
                text("""
                    SELECT fact_id, user_id, content, content_hash
                    FROM memory_static_facts
                    WHERE user_id = :user_id
                    ORDER BY created_at DESC
                """),
                {"user_id": user_id},
            )
            return [
                StaticFact(
                    fact_id=str(row.fact_id),
                    user_id=row.user_id,
                    content=row.content,
                    content_hash=row.content_hash,
                )
                for row in rows.fetchall()
            ]

    async def delete_fact(
        self,
        pg: PostgreSQLClient,
        milvus: MilvusClient,
        fact_id: str,
        user_id: str,
    ) -> None:
        async with pg.engine.begin() as conn:
            result = await conn.execute(
                text("""
                    DELETE FROM memory_static_facts
                    WHERE fact_id = :fact_id AND user_id = :user_id
                """),
                {"fact_id": fact_id, "user_id": user_id},
            )
            if result.rowcount == 0:
                raise NotFoundError(f"Static fact {fact_id!r} not found for user {user_id!r}")
        await milvus.memory_delete(fact_id)
        logger.info("Deleted static fact fact_id=%s user=%s", fact_id, user_id)
