import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from app.cache.schemas import CacheStatus
from app.cache.service import RagCacheService
from app.core.config import settings
from app.core.exceptions import ValidationError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/cache", tags=["cache"])

QueryHash = Annotated[str, Path(pattern=r"^[0-9a-f]{16}$")]

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _require_api_key(api_key: str | None = Security(_api_key_header)) -> None:
    """Reject requests whose X-API-Key doesn't match the configured secret.

    When cache_api_key is empty (default / dev), auth is disabled entirely.
    """
    if settings.cache_api_key and api_key != settings.cache_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _svc(request: Request) -> RagCacheService:
    try:
        return request.app.state.cache_svc
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
    entries_raw = await svc.store.get_many(hashes)
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
        for entry in entries_raw
        if entry is not None
    ]
    return {"pending": entries, "total": len(entries)}


@router.post("/review/{query_hash}/approve", dependencies=[Depends(_require_api_key)])
async def approve_entry(
    request: Request,
    query_hash: QueryHash,
    body: ApproveBody,
) -> dict:
    svc = _svc(request)
    entry = await svc.store.get(query_hash)
    if entry is None:
        raise HTTPException(status_code=404, detail="Cache entry not found")
    new_status = await svc.review.approve(query_hash, body.reviewer_id)
    return {"query_hash": query_hash, "status": new_status.value}


@router.post("/review/{query_hash}/reject", status_code=204, dependencies=[Depends(_require_api_key)])
async def reject_entry(request: Request, query_hash: QueryHash) -> None:
    svc = _svc(request)
    entry = await svc.store.get(query_hash)
    if entry is None:
        raise HTTPException(status_code=404, detail="Cache entry not found")
    try:
        await svc.review.reject(query_hash)
    except ValidationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.delete("/{query_hash}", status_code=204, dependencies=[Depends(_require_api_key)])
async def delete_entry(request: Request, query_hash: QueryHash) -> None:
    svc = _svc(request)
    deleted = await svc.store.delete(query_hash)
    if not deleted:
        raise HTTPException(status_code=404, detail="Cache entry not found")


@router.get("/stats")
async def get_stats(request: Request) -> dict:
    return await _svc(request).store.get_stats()
