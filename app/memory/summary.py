import hashlib
import logging
import uuid

from sqlalchemy import text

from app.clients.llm import AbstractLLMClient
from app.clients.postgresql import PostgreSQLClient
from app.memory.schemas import ConversationTurn, MemorySummary

logger = logging.getLogger(__name__)

_SUMMARIZE_PROMPT = (
    "Summarize the following conversation concisely, capturing key facts and decisions:\n\n"
    "{dialogue}\n\nSummary:"
)


class SummaryMemoryStore:
    async def get_latest_summary(
        self, pg: PostgreSQLClient, user_id: str
    ) -> MemorySummary | None:
        async with pg.engine.connect() as conn:
            result = await conn.execute(
                text("""
                    SELECT summary_id, user_id, session_id, summary_text, content_hash
                    FROM memory_summaries
                    WHERE user_id = :user_id
                    ORDER BY updated_at DESC
                    LIMIT 1
                """),
                {"user_id": user_id},
            )
            row = result.fetchone()
        if row is None:
            return None
        return MemorySummary(
            summary_id=row[0],
            user_id=row[1],
            session_id=row[2],
            summary_text=row[3],
            content_hash=row[4],
        )

    async def save_summary(self, pg: PostgreSQLClient, summary: MemorySummary) -> None:
        async with pg.engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO memory_summaries
                        (summary_id, user_id, session_id, summary_text, content_hash)
                    VALUES (:summary_id, :user_id, :session_id, :summary_text, :content_hash)
                    ON CONFLICT (user_id, session_id) DO UPDATE SET
                        summary_text = EXCLUDED.summary_text,
                        content_hash = EXCLUDED.content_hash,
                        updated_at   = NOW()
                """),
                {
                    "summary_id": summary.summary_id,
                    "user_id": summary.user_id,
                    "session_id": summary.session_id,
                    "summary_text": summary.summary_text,
                    "content_hash": summary.content_hash,
                },
            )

    async def compact(
        self,
        pg: PostgreSQLClient,
        llm: AbstractLLMClient,
        user_id: str,
        session_id: str,
        turns: list[ConversationTurn],
    ) -> MemorySummary:
        dialogue = "\n".join(f"{t.role}: {t.content}" for t in turns)
        prompt = _SUMMARIZE_PROMPT.format(dialogue=dialogue)
        summary_text = await llm.complete(prompt, max_tokens=512)
        content_hash = hashlib.sha256(summary_text.encode()).hexdigest()
        summary = MemorySummary(
            summary_id=str(uuid.uuid4()),
            user_id=user_id,
            session_id=session_id,
            summary_text=summary_text,
            content_hash=content_hash,
        )
        await self.save_summary(pg, summary)
        logger.info("Compacted session=%s user=%s", session_id, user_id)
        return summary
