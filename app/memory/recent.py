import logging

from app.clients.redis import RedisClient
from app.core.config import settings
from app.memory.schemas import ConversationTurn

logger = logging.getLogger(__name__)


class RecentMemoryStore:
    async def append_turn(
        self, redis: RedisClient, session_id: str, turn: ConversationTurn
    ) -> int:
        key = redis.cache_key("memory:recent", session_id)
        serialized = turn.model_dump_json()
        await redis.client.rpush(key, serialized)
        await redis.client.ltrim(key, -settings.memory_recent_max_turns, -1)
        count = await redis.client.llen(key)
        await redis.client.expire(key, settings.memory_recent_ttl_seconds)
        logger.debug("Appended turn session=%s count=%d", session_id, count)
        return count

    async def get_turns(
        self, redis: RedisClient, session_id: str
    ) -> list[ConversationTurn]:
        key = redis.cache_key("memory:recent", session_id)
        raw = await redis.client.lrange(key, 0, -1)
        return [ConversationTurn.model_validate_json(r) for r in raw]

    async def count(self, redis: RedisClient, session_id: str) -> int:
        key = redis.cache_key("memory:recent", session_id)
        return await redis.client.llen(key)

    async def clear(self, redis: RedisClient, session_id: str) -> None:
        key = redis.cache_key("memory:recent", session_id)
        await redis.client.delete(key)
        logger.debug("Cleared recent memory session=%s", session_id)
