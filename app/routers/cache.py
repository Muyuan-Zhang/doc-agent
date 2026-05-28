import asyncio
import logging

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.cache.schemas import CacheStatus
from app.cache.service import RagCacheService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/cache", tags=["cache"])


def _svc(request: Request) -> RagCacheService:
    # TODO(M4): move service construction to lifespan and store in app.state
    try:
        return RagCacheService(
            redis=request.app.state.redis,
            llm=request.app.state.llm,
        )
    except AttributeError as exc:
        raise HTTPException(status_code=503, detail="Service dependencies not ready") from exc


class ApproveBody(BaseModel):
    reviewer_id: str = Field(min_length=1, max_length=128)


@router.get("/review")
async def list_pending_reviews(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
) -> dict:
    svc = _svc(request)
    hashes = await svc.review.list_pending(limit=limit)
    raw_entries = await asyncio.gather(*(svc.store.get(h) for h in hashes))
    entries = [
        {
            "query_hash": entry.query_hash,
            "original_query": entry.original_query,
            "normalized_query": entry.normalized_query,
            "chunk_count": len(entry.chunks),
            "status": entry.status.value,
            "approval_count": entry.approval_count,
            "created_at": entry.created_at.isoformat(),
        }
        for entry in raw_entries
        if entry is not None
    ]
    return {"pending": entries, "total": len(entries)}


@router.post("/review/{query_hash}/approve")
# TODO(M4): add authentication dependency to restrict to authorised reviewers
async def approve_entry(
    request: Request,
    query_hash: str,
    body: ApproveBody,
) -> dict:
    svc = _svc(request)
    entry = await svc.store.get(query_hash)
    if entry is None:
        raise HTTPException(status_code=404, detail="Cache entry not found")
    new_status = await svc.review.approve(query_hash, body.reviewer_id)
    return {"query_hash": query_hash, "status": new_status.value}


@router.post("/review/{query_hash}/reject", status_code=204)
# TODO(M4): add authentication dependency to restrict to authorised reviewers
async def reject_entry(request: Request, query_hash: str) -> None:
    svc = _svc(request)
    entry = await svc.store.get(query_hash)
    if entry is None:
        raise HTTPException(status_code=404, detail="Cache entry not found")
    await svc.review.reject(query_hash)


@router.delete("/{query_hash}", status_code=204)
# TODO(M4): add authentication dependency to restrict to authorised reviewers
async def delete_entry(request: Request, query_hash: str) -> None:
    svc = _svc(request)
    deleted = await svc.store.delete(query_hash)
    if not deleted:
        raise HTTPException(status_code=404, detail="Cache entry not found")


@router.get("/stats")
async def get_stats(request: Request) -> dict:
    return await _svc(request).store.get_stats()
