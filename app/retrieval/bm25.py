import logging

from sqlalchemy import text

from app.clients.postgresql import PostgreSQLClient
from app.models.chunk import ChunkSchema

logger = logging.getLogger(__name__)

_BM25_SQL = """
    SELECT
        chunk_id,
        doc_id,
        section_id,
        chunk_index,
        parent_chunk_id,
        content_hash,
        version,
        content,
        ts_rank(to_tsvector('english', content), plainto_tsquery('english', :q)) AS score
    FROM chunks_metadata
    WHERE to_tsvector('english', content) @@ plainto_tsquery('english', :q)
    ORDER BY score DESC
    LIMIT :top_k
"""


class BM25Strategy:
    def __init__(self, pg: PostgreSQLClient) -> None:
        self._pg = pg

    async def retrieve(self, query: str, top_k: int, **kwargs) -> list[ChunkSchema]:
        if not query.strip():
            return []
        async with self._pg.engine.connect() as conn:
            result = await conn.execute(
                text(_BM25_SQL),
                {"q": query, "top_k": top_k},
            )
            rows = result.fetchall()
        return [
            ChunkSchema(
                doc_id=row[1],
                section_id=row[2],
                chunk_index=row[3],
                parent_chunk_id=row[4],
                content_hash=row[5],
                version=row[6],
                content=row[7],
            )
            for row in rows
        ]
