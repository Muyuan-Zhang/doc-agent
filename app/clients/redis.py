"""
Redis 异步客户端。

Key 命名规范（Cluster 模式预留）：
- 单 key 操作无限制，命名遵循 {kb_version}:{namespace}:{id} 惯例。
- 需要原子多 key 操作（Lua script、MSET、pipeline）时，
  将决定分片的字段用 {} 括起：
    正确：  {user:u123}:session  /  {doc:d456}:lock
    错误：   user:u123:session   （跨 slot 时 Lua script 会抛 CROSSSLOT 错误）
- increment_with_ttl、acquire_lock、release_lock 已遵循此规范。
- 新增方法时，若涉及多 key，请在 PR review 中确认 hash tag 使用。
"""
import logging
import uuid

import redis.asyncio as aioredis

from app.clients.base import AbstractClient
from app.core.config import settings

logger = logging.getLogger(__name__)

_INCR_WITH_TTL_SCRIPT = """
local current = redis.call('INCRBY', KEYS[1], ARGV[1])
if current == tonumber(ARGV[1]) then
    redis.call('EXPIRE', KEYS[1], ARGV[2])
end
return current
"""

_RELEASE_LOCK_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
else
    return 0
end
"""


class RedisClient(AbstractClient):
    def __init__(self) -> None:
        self._client: aioredis.Redis | None = None

    @property
    def client(self) -> aioredis.Redis:
        if self._client is None:
            raise RuntimeError("RedisClient not connected")
        return self._client

    def cache_key(self, namespace: str, *parts: str) -> str:
        return f"{settings.knowledge_base_version}:{namespace}:{':'.join(parts)}"

    async def connect(self) -> None:
        self._client = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=50,
        )
        logger.info("Redis client created")

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("Redis client closed")

    async def ping(self) -> bool:
        try:
            await self.client.ping()
            return True
        except Exception as exc:
            logger.warning("Redis ping failed: %s", exc)
            return False

    async def increment_with_ttl(
        self, key: str, ttl_seconds: int, amount: int = 1
    ) -> int:
        """
        原子 INCR + EXPIRE（Lua script）。
        key 应使用 hash tag，例如：{user:u123}:rate_limit
        """
        result = await self.client.eval(
            _INCR_WITH_TTL_SCRIPT, 1, key, amount, ttl_seconds
        )
        return int(result)

    async def acquire_lock(
        self, key: str, ttl_seconds: int, token: str | None = None
    ) -> tuple[bool, str]:
        """
        SET key token NX EX ttl 分布式锁。
        key 应使用 hash tag，例如：{doc:d456}:update_lock
        返回 (acquired, token)，token 用于 release_lock。
        """
        token = token or str(uuid.uuid4())
        acquired = await self.client.set(key, token, nx=True, ex=ttl_seconds)
        return bool(acquired), token

    async def release_lock(self, key: str, token: str) -> bool:
        """
        Lua script 验证 token 后 DEL，防止误释放他人持有的锁。
        key 需与 acquire_lock 使用相同 hash tag。
        """
        result = await self.client.eval(_RELEASE_LOCK_SCRIPT, 1, key, token)
        return bool(result)
