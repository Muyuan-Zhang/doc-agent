import asyncio
import logging
from abc import abstractmethod

from openai import AsyncOpenAI

from app.clients.base import AbstractClient
from app.core.config import settings

logger = logging.getLogger(__name__)


class AbstractLLMClient(AbstractClient):
    """
    LLM 客户端抽象接口。
    ping() 调用轻量 embedding 验证连通性（熔断器包裹点）。
    """

    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> str:
        """文本补全，返回模型输出字符串。"""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """单文本向量化，返回 embedding 向量。"""

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量向量化，默认并发调用 embed()。子类可覆盖以使用原生批量 API。"""
        return list(await asyncio.gather(*(self.embed(t) for t in texts)))


class OpenAILLMClient(AbstractLLMClient):
    """OpenAI Embeddings API 客户端，使用 background semaphore 限流。"""

    def __init__(self) -> None:
        self._client = None
        self._background_sem = asyncio.Semaphore(settings.llm_semaphore_limits.background)

    async def connect(self) -> None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required but not set")
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        logger.info("OpenAI LLM client created model=%s", settings.openai_embedding_model)

    async def disconnect(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("OpenAI LLM client closed")

    async def ping(self) -> bool:
        try:
            await self.embed("ping")
            return True
        except Exception as exc:
            logger.warning("LLM ping failed: %s", exc)
            return False

    async def complete(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError("complete() implemented in M4")

    async def embed(self, text: str) -> list[float]:
        async with self._background_sem:
            resp = await self._client.embeddings.create(
                input=[text],
                model=settings.openai_embedding_model,
            )
            return resp.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        async with self._background_sem:
            resp = await self._client.embeddings.create(
                input=texts,
                model=settings.openai_embedding_model,
            )
            return [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]
