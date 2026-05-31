import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Path as PathParam, Query, Request, status
from pydantic import BaseModel, Field

from app.core.rate_limit import rate_limiter
from app.memory.schemas import MemoryContext, MemorySummary, StaticFact
from app.memory.service import MemoryService

router = APIRouter(prefix="/memory", tags=["memory"])
logger = logging.getLogger(__name__)

_UUID_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
_SAFE_ID_PATTERN = r"^[a-zA-Z0-9_-]{1,64}$"

_turns_rate_limit = rate_limiter("memory:turns", limit=600, window_seconds=60)
_summarize_rate_limit = rate_limiter("memory:summarize", limit=50, window_seconds=60)


class AppendTurnRequest(BaseModel):
    session_id: str = Field(..., pattern=_SAFE_ID_PATTERN)
    user_id: str = Field(..., pattern=_SAFE_ID_PATTERN)
    role: Literal["user", "assistant", "system"]
    content: str = Field(..., max_length=32768)


class AddFactRequest(BaseModel):
    user_id: str = Field(..., pattern=_SAFE_ID_PATTERN)
    content: str = Field(..., max_length=32768)


def _svc(request: Request) -> MemoryService:
    try:
        return MemoryService(
            pg=request.app.state.postgres,
            redis=request.app.state.redis,
            milvus=request.app.state.milvus,
            llm=request.app.state.llm,
        )
    except AttributeError as exc:
        raise HTTPException(status_code=503, detail="Service dependencies not ready") from exc


@router.post("/turns", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(_turns_rate_limit)])
async def append_turn(body: AppendTurnRequest, request: Request) -> None:
    await _svc(request).append_turn(
        body.session_id, body.user_id, body.role, body.content
    )


@router.get("/context/{session_id}", response_model=MemoryContext)
async def get_context(
    request: Request,
    session_id: str = PathParam(..., pattern=_SAFE_ID_PATTERN),
    user_id: str = Query(..., pattern=_SAFE_ID_PATTERN),
) -> MemoryContext:
    return await _svc(request).retrieve_context(session_id, user_id)


@router.post("/summarize/{session_id}", response_model=MemorySummary, dependencies=[Depends(_summarize_rate_limit)])
async def summarize_session(
    request: Request,
    session_id: str = PathParam(..., pattern=_SAFE_ID_PATTERN),
    user_id: str = Query(..., pattern=_SAFE_ID_PATTERN),
) -> MemorySummary:
    try:
        return await _svc(request).summarize_session(session_id, user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/static", response_model=StaticFact, status_code=status.HTTP_201_CREATED)
async def add_static_fact(body: AddFactRequest, request: Request) -> StaticFact:
    try:
        return await _svc(request).add_static_fact(body.user_id, body.content)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("add_static_fact failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to add static fact") from exc


@router.delete("/static/{fact_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_static_fact(
    request: Request,
    fact_id: str = PathParam(..., pattern=_UUID_PATTERN),
    user_id: str = Query(..., pattern=_SAFE_ID_PATTERN),
) -> None:
    await _svc(request).delete_static_fact(fact_id, user_id)
