import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.models.chunk import ChunkSchema

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=100)


class SearchResponse(BaseModel):
    chunks: list[ChunkSchema]
    query: str
    top_k: int


def _get_retriever(request: Request):
    try:
        return request.app.state.retriever
    except AttributeError as exc:
        raise HTTPException(status_code=503, detail="Retrieval service not ready") from exc


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest, request: Request) -> SearchResponse:
    retriever = _get_retriever(request)
    try:
        chunks = await retriever.retrieve(req.query, req.top_k)
    except Exception:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=503, detail="Retrieval service temporarily unavailable")
    return SearchResponse(chunks=chunks, query=req.query, top_k=req.top_k)
