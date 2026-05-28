import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.cache.schemas import CacheStatus
from app.cache.service import RagCacheService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/cache", tags=["cache"])


def _svc(request: Request) -> RagCacheService:
    try:
        return RagCacheService(
            redis=request.app.state.redis,
            llm=request.app.state.llm,
        )
    except AttributeError as exc:
        raise HTTPException(status_code=503, detail="Service dependencies not ready") from exc


class ApproveBody(BaseModel):
    reviewer_id: str


@router.get("/review")
async def list_pending_reviews(request: Request, limit: int = 20) -> dict:
    svc = _svc(request)
    hashes = await svc.review.list_pending(limit=limit)
    entries = []
    for h in hashes:
        entry = await svc.store.get(h)
        if entry is not None:
            entries.append({
                "query_hash": entry.query_hash,
                "original_query": entry.original_query,
                "normalized_query": entry.normalized_query,
                "chunk_count": len(entry.chunks),
                "status": entry.status.value,
                "approval_count": entry.approval_count,
                "created_at": entry.created_at.isoformat(),
            })
    return {"pending": entries, "total": len(entries)}


@router.post("/review/{query_hash}/approve")
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
async def reject_entry(request: Request, query_hash: str) -> None:
    svc = _svc(request)
    entry = await svc.store.get(query_hash)
    if entry is None:
        raise HTTPException(status_code=404, detail="Cache entry not found")
    await svc.review.reject(query_hash)


@router.delete("/{query_hash}", status_code=204)
async def delete_entry(request: Request, query_hash: str) -> None:
    svc = _svc(request)
    deleted = await svc.store.delete(query_hash)
    if not deleted:
        raise HTTPException(status_code=404, detail="Cache entry not found")


@router.get("/stats")
async def get_stats(request: Request) -> dict:
    return await _svc(request).store.get_stats()
