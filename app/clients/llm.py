import asyncio
import logging
from abc import abstractmethod
from typing import AsyncIterator

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
    async def stream_complete(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """流式文本补全，逐 token yield 输出。"""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """单文本向量化，返回 embedding 向量。"""

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量向量化，默认并发调用 embed()。子类可覆盖以使用原生批量 API。"""
        return list(await asyncio.gather(*(self.embed(t) for t in texts)))


class OpenAILLMClient(AbstractLLMClient):
    """OpenAI Embeddings + Chat API 客户端，按任务类型使用独立 semaphore 限流。"""

    def __init__(self) -> None:
        self._client = None
        self._interactive_sem = asyncio.Semaphore(settings.llm_semaphore_limits.interactive)
        self._background_sem = asyncio.Semaphore(settings.llm_semaphore_limits.background)
        self._audit_sem = asyncio.Semaphore(settings.llm_semaphore_limits.audit)

    async def connect(self) -> None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required but not set")
        kwargs = {"api_key": settings.openai_api_key}
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        self._client = AsyncOpenAI(**kwargs)
        logger.info(
            "OpenAI LLM client created model=%s base_url=%s",
            settings.openai_embedding_model,
            settings.openai_base_url or "default",
        )

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
        max_tokens = kwargs.get("max_tokens", 512)
        async with self._interactive_sem:
            resp = await self._client.chat.completions.create(
                model=settings.openai_chat_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""

    async def stream_complete(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        max_tokens = kwargs.get("max_tokens", 512)
        async with self._interactive_sem:
            async with await self._client.chat.completions.create(
                model=settings.openai_chat_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                stream=True,
            ) as stream:
                async for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content

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
