import logging

from app.clients.llm import AbstractLLMClient
from app.clients.milvus import MilvusClient
from app.clients.postgresql import PostgreSQLClient
from app.clients.redis import RedisClient
from app.core.config import settings
from app.memory.recent import RecentMemoryStore
from app.memory.schemas import ConversationTurn, MemoryContext, MemorySummary, StaticFact
from app.memory.static_knowledge import StaticKnowledgeStore
from app.memory.summary import SummaryMemoryStore

logger = logging.getLogger(__name__)


class MemoryService:
    def __init__(
        self,
        pg: PostgreSQLClient,
        redis: RedisClient,
        milvus: MilvusClient,
        llm: AbstractLLMClient,
    ) -> None:
        self._pg = pg
        self._redis = redis
        self._milvus = milvus
        self._llm = llm
        self._recent = RecentMemoryStore()
        self._summary = SummaryMemoryStore()
        self._static = StaticKnowledgeStore()

    async def append_turn(
        self, session_id: str, user_id: str, role: str, content: str
    ) -> None:
        import time
        turn = ConversationTurn(
            session_id=session_id, role=role, content=content, ts=time.time()
        )
        count = await self._recent.append_turn(self._redis, session_id, turn)
        if count >= settings.memory_summary_threshold:
            turns = await self._recent.get_turns(self._redis, session_id)
            await self._summary.compact(self._pg, self._llm, user_id, session_id, turns)
            await self._recent.clear(self._redis, session_id)
            logger.info("Auto-compacted session=%s after %d turns", session_id, count)

    async def retrieve_context(
        self,
        session_id: str,
        user_id: str,
        query_embedding: list[float] | None = None,
    ) -> MemoryContext:
        turns = await self._recent.get_turns(self._redis, session_id)
        summary = await self._summary.get_latest_summary(self._pg, user_id)
        static_facts: list[StaticFact] = []
        if query_embedding is not None:
            static_facts = await self._static.search_facts(
                self._milvus, query_embedding, user_id
            )
        return MemoryContext(turns=turns, summary=summary, static_facts=static_facts)

    async def summarize_session(self, session_id: str, user_id: str) -> MemorySummary:
        turns = await self._recent.get_turns(self._redis, session_id)
        summary = await self._summary.compact(
            self._pg, self._llm, user_id, session_id, turns
        )
        await self._recent.clear(self._redis, session_id)
        return summary

    async def add_static_fact(self, user_id: str, content: str) -> StaticFact:
        return await self._static.add_fact(
            self._pg, self._milvus, self._llm, user_id, content
        )

    async def delete_static_fact(self, fact_id: str, user_id: str) -> None:
        await self._static.delete_fact(self._pg, self._milvus, fact_id, user_id)
