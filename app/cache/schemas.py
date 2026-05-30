from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from app.core.exceptions import ValidationError
from app.models.chunk import ChunkSchema


class CacheStatus(str, Enum):
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"


# Allowed status transitions.  APPROVED is terminal — once approved a cache
# entry cannot be rejected (prevents silent cache poisoning via race).
_VALID_TRANSITIONS: dict["CacheStatus", set["CacheStatus"]] = {
    CacheStatus.PENDING_REVIEW: {CacheStatus.APPROVED, CacheStatus.REJECTED},
    CacheStatus.APPROVED: set(),
    CacheStatus.REJECTED: set(),
}


def validate_transition(current: "CacheStatus", new: "CacheStatus") -> None:
    """Raise ValidationError if the transition current → new is not allowed.

    Same-state transitions are always allowed (no-op update).
    """
    if current == new:
        return
    if new not in _VALID_TRANSITIONS[current]:
        raise ValidationError(
            f"Invalid status transition: {current.value!r} → {new.value!r}"
        )


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


class StatsResponse(BaseModel):
    hits: int
    misses: int
    pending: int


class ReviewSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_hash: str
    original_query: str
    normalized_query: str
    chunk_count: int
    status: CacheStatus
    approval_count: int
    created_at: datetime


class ReviewListResponse(BaseModel):
    pending: list[ReviewSummary]
    total: int


class ApproveResponse(BaseModel):
    query_hash: str
    status: CacheStatus
