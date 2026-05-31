from typing import Optional

from typing_extensions import TypedDict

from app.memory.schemas import MemoryContext
from app.models.chunk import ChunkSchema


class AgentState(TypedDict):
    session_id: str
    job_id: str
    query: str
    top_k: int
    rewritten_query: str
    chunks: list[ChunkSchema]
    reranked_chunks: list[ChunkSchema]
    answer: str
    cache_hit: bool
    cached_answer: str
    query_embedding: Optional[list[float]]
    rag_cache_hash: Optional[str]
    error: Optional[str]
    user_id: str
    memory_context: Optional[MemoryContext]
