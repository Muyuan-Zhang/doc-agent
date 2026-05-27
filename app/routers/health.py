import asyncio
import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health")
async def liveness() -> dict:
    return {"status": "ok", "service": settings.app_name}


@router.get("/health/ready")
async def readiness(request: Request) -> JSONResponse:
    async def check(name: str, client) -> tuple[str, str]:
        try:
            ok = await client.ping()
            return name, "ok" if ok else f"error: ping returned false"
        except Exception as exc:
            return name, f"error: {exc}"

    state = request.app.state
    results = await asyncio.gather(
        check("mysql",  state.mysql),
        check("redis",  state.redis),
        check("milvus", state.milvus),
        check("mq",     state.mq),
    )

    checks = dict(results)
    degraded = any(v != "ok" for v in checks.values())

    return JSONResponse(
        status_code=503 if degraded else 200,
        content={
            "status": "degraded" if degraded else "ok",
            "kb_version": settings.knowledge_base_version,
            "checks": checks,
        },
    )
