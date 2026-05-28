from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from app.models.chunk import ChunkSchema


class CacheStatus(str, Enum):
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"


class CacheEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_hash: str = Field(pattern=r"^[0-9a-f]{16}$")
    original_query: str
    normalized_query: str
    chunks: list[ChunkSchema]
    status: CacheStatus = CacheStatus.PENDING_REVIEW
    approval_count: int = 0
    created_at: datetime
    approved_by: list[str] = Field(default_factory=list)


class ReviewEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_hash: str
    original_query: str
    normalized_query: str
    chunk_count: int
    status: CacheStatus
    approval_count: int
    created_at: datetime
    approved_by: list[str] = Field(default_factory=list)
