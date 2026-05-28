import hashlib
import json
import logging
import uuid
from typing import Any, Optional

from sqlalchemy import text

from app.clients.llm import AbstractLLMClient
from app.clients.postgresql import PostgreSQLClient
from app.memory.schemas import ConversationTurn, MemorySummary

logger = logging.getLogger(__name__)

# Only feed the most recent N turns to the LLM to bound token usage.
_MAX_INPUT_TURNS = 20

_SUMMARIZE_PROMPT = """\
You are a memory compaction assistant. Produce a JSON object with exactly two keys:
- "summary_text": a concise paragraph capturing key facts, decisions and context.
- "key_topics": a list of up to 5 short topic strings.

{context_block}\
Recent conversation:
{dialogue}

Respond with only the JSON object, no markdown fences."""

_CONTEXT_BLOCK = "Previous summary to build upon:\n{previous_text}\n\n"


class SummaryMemoryStore:
    async def get_latest_summary(
        self, pg: PostgreSQLClient, user_id: str, session_id: str
    ) -> MemorySummary | None:
        async with pg.engine.connect() as conn:
            result = await conn.execute(
                text("""
                    SELECT summary_id, user_id, session_id, summary_text, content_hash,
                           importance_score, structured_facts
                    FROM memory_summaries
                    WHERE user_id = :user_id AND session_id = :session_id
                    ORDER BY updated_at DESC
                    LIMIT 1
                """),
                {"user_id": user_id, "session_id": session_id},
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
            importance_score=row[5] or 1.0,
            structured_facts=row[6] or {},
        )

    async def save_summary(self, pg: PostgreSQLClient, summary: MemorySummary) -> None:
        async with pg.engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO memory_summaries
                        (summary_id, user_id, session_id, summary_text, content_hash,
                         importance_score, structured_facts)
                    VALUES (:summary_id, :user_id, :session_id, :summary_text, :content_hash,
                            :importance_score, :structured_facts)
                    ON CONFLICT (user_id, session_id) DO UPDATE SET
                        summary_text     = EXCLUDED.summary_text,
                        content_hash     = EXCLUDED.content_hash,
                        importance_score = EXCLUDED.importance_score,
                        structured_facts = EXCLUDED.structured_facts,
                        updated_at       = NOW()
                """),
                {
                    "summary_id": summary.summary_id,
                    "user_id": summary.user_id,
                    "session_id": summary.session_id,
                    "summary_text": summary.summary_text,
                    "content_hash": summary.content_hash,
                    "importance_score": summary.importance_score,
                    "structured_facts": json.dumps(summary.structured_facts, default=str),
                },
            )

    async def compact(
        self,
        pg: PostgreSQLClient,
        llm: AbstractLLMClient,
        user_id: str,
        session_id: str,
        turns: list[ConversationTurn],
        previous_summary: Optional[MemorySummary] = None,
    ) -> MemorySummary:
        windowed = turns[-_MAX_INPUT_TURNS:]
        if not windowed:
            if previous_summary is not None:
                return previous_summary
            raise ValueError(f"compact() called with no turns for session={session_id!r}")

        dialogue = "\n".join(f"{t.role}: {t.content}" for t in windowed)
        context_block = (
            _CONTEXT_BLOCK.format(previous_text=previous_summary.summary_text)
            if previous_summary
            else ""
        )
        prompt = _SUMMARIZE_PROMPT.format(context_block=context_block, dialogue=dialogue)
        raw = await llm.complete(prompt, max_tokens=512)

        # Strip markdown fences that some LLMs add despite instructions.
        raw_stripped = raw.strip()
        if raw_stripped.startswith("```"):
            raw_stripped = raw_stripped.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        try:
            parsed = json.loads(raw_stripped)
            summary_text = parsed.get("summary_text", raw_stripped)
            structured_facts: dict[str, Any] = {"key_topics": parsed.get("key_topics", [])}
        except (json.JSONDecodeError, AttributeError):
            summary_text = raw_stripped
            structured_facts = {}

        content_hash = hashlib.sha256(summary_text.encode()).hexdigest()
        summary = MemorySummary(
            summary_id=str(uuid.uuid4()),
            user_id=user_id,
            session_id=session_id,
            summary_text=summary_text,
            content_hash=content_hash,
            structured_facts=structured_facts,
        )
        await self.save_summary(pg, summary)
        logger.info("Compacted session=%s user=%s turns=%d", session_id, user_id, len(windowed))
        return summary
