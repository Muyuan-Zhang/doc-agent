from typing import Literal, Optional

from pydantic import BaseModel


class QueryRequest(BaseModel):
    session_id: str
    query: str
    top_k: int = 5


class QueryResponse(BaseModel):
    job_id: str
    status: Literal["queued"]


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "error"]
    answer: Optional[str] = None
    error: Optional[str] = None
