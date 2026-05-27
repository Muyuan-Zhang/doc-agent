from typing import Optional

from pydantic import BaseModel, ConfigDict


class ChunkSchema(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    section_id: str
    chunk_index: int
    parent_chunk_id: Optional[str] = None   # 层级检索预留（Graph RAG）
    content_hash: str                        # 去重 & 版本追踪
    version: str                             # 对应 knowledge_base_version
    content: str
    embedding: Optional[list[float]] = None
