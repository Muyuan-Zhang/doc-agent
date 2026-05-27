from abc import abstractmethod

from app.clients.base import AbstractClient


class AbstractLLMClient(AbstractClient):
    """
    LLM 客户端抽象接口。
    具体实现（OpenAI / Ollama / vLLM）在 M4 接入。
    ping() 应调用轻量 embedding 或 echo 请求验证连通性。
    此处为熔断器（circuit breaker）包裹点预留。
    """

    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> str:
        """文本补全，返回模型输出字符串。"""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """文本向量化，返回 embedding 向量。"""
