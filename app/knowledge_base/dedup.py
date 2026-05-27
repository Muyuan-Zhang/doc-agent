from sqlalchemy import text

from app.clients.postgresql import PostgreSQLClient
from app.models.chunk import ChunkSchema


class ContentDeduplicator:
    async def filter_new(
        self, chunks: list[ChunkSchema], pg: PostgreSQLClient
    ) -> list[ChunkSchema]:
        if not chunks:
            return []
        hashes = [c.content_hash for c in chunks]
        async with pg.engine.connect() as conn:
            result = await conn.execute(
                text("SELECT content_hash FROM chunks_metadata WHERE content_hash = ANY(:hashes)"),
                {"hashes": hashes},
            )
            existing = {row[0] for row in result}
        return [c for c in chunks if c.content_hash not in existing]
