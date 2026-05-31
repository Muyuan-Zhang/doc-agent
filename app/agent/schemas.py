from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

_SAFE_ID_PATTERN = r"^[a-zA-Z0-9_-]{1,64}$"


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, max_length=64, pattern=_SAFE_ID_PATTERN)
    query: str = Field(min_length=1, max_length=4096)
    top_k: int = Field(default=5, ge=1, le=100)
    user_id: Optional[str] = Field(default=None, min_length=0, max_length=64, pattern=_SAFE_ID_PATTERN)


class QueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: Literal["queued"]


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "error"]
    answer: Optional[str] = None
    error: Optional[str] = None
