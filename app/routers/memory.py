import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel

from app.memory.schemas import MemoryContext, MemorySummary, StaticFact
from app.memory.service import MemoryService

router = APIRouter(prefix="/memory", tags=["memory"])
logger = logging.getLogger(__name__)


class AppendTurnRequest(BaseModel):
    session_id: str
    user_id: str
    role: str
    content: str


class AddFactRequest(BaseModel):
    user_id: str
    content: str


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


@router.post("/turns", status_code=status.HTTP_204_NO_CONTENT)
async def append_turn(body: AppendTurnRequest, request: Request) -> None:
    await _svc(request).append_turn(
        body.session_id, body.user_id, body.role, body.content
    )


@router.get("/context/{session_id}", response_model=MemoryContext)
async def get_context(
    session_id: str,
    user_id: str = Query(...),
    request: Request = None,
) -> MemoryContext:
    return await _svc(request).retrieve_context(session_id, user_id)


@router.post("/summarize/{session_id}", response_model=MemorySummary)
async def summarize_session(
    session_id: str,
    user_id: str = Query(...),
    request: Request = None,
) -> MemorySummary:
    return await _svc(request).summarize_session(session_id, user_id)


@router.post("/static", response_model=StaticFact, status_code=status.HTTP_201_CREATED)
async def add_static_fact(body: AddFactRequest, request: Request) -> StaticFact:
    return await _svc(request).add_static_fact(body.user_id, body.content)


@router.delete("/static/{fact_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_static_fact(
    fact_id: str,
    user_id: str = Query(...),
    request: Request = None,
) -> None:
    await _svc(request).delete_static_fact(fact_id, user_id)
