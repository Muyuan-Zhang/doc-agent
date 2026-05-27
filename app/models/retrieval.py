from typing import Protocol, runtime_checkable

from app.models.chunk import ChunkSchema


@runtime_checkable
class RetrievalStrategy(Protocol):
    """M2 混合检索策略接口。实现方不得写死 BM25 或向量检索。"""

    async def retrieve(
        self, query: str, top_k: int, **kwargs
    ) -> list[ChunkSchema]: ...


class HybridRetriever:
    """
    组合多个 RetrievalStrategy，执行融合排序（RRF 等）。
    strategies 列表在 M2 中填充；此处仅声明接口。
    """

    def __init__(self, strategies: list[RetrievalStrategy]) -> None:
        self._strategies = strategies

    async def retrieve(self, query: str, top_k: int, **kwargs) -> list[ChunkSchema]:
        raise NotImplementedError("HybridRetriever 将在 M2 实现")
