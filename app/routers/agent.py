"""M4 Agent API endpoints.

POST /agent/query           — enqueue a query job; returns job_id immediately (async)
GET  /agent/jobs/{job_id}   — poll job status / answer
GET  /agent/stream/{job_id} — stream the answer as SSE once the job completes
"""
import asyncio
import logging
from typing import AsyncIterator
from uuid import UUID, uuid4

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.agent._keys import job_key
from app.agent.schemas import JobStatus, QueryRequest, QueryResponse
from app.core.config import settings
from app.core.exceptions import NotFoundError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])

_STREAM_POLL_INTERVAL = 0.5


async def _init_job(redis, job_id: str) -> None:
    key = job_key(job_id)
    await redis.client.hset(key, mapping={"status": "queued", "answer": "", "error": ""})
    await redis.client.expire(key, settings.agent_job_ttl_seconds)


async def _load_job(redis, job_id: str) -> JobStatus | None:
    data = await redis.client.hgetall(job_key(job_id))
    if not data:
        return None
    return JobStatus(
        job_id=job_id,
        status=data.get("status", "error"),
        answer=data.get("answer") or None,
        error=data.get("error") or None,
    )


@router.post("/query", status_code=202, response_model=QueryResponse)
async def enqueue_query(body: QueryRequest, request: Request) -> QueryResponse:
    job_id = str(uuid4())
    # Write job record before publishing so the consumer never sees a missing key.
    await _init_job(request.app.state.redis, job_id)
    await request.app.state.mq.publish({
        "job_id": job_id,
        "session_id": body.session_id,
        "query": body.query,
        "top_k": str(body.top_k),
    })
    return QueryResponse(job_id=job_id, status="queued")


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str, request: Request) -> JobStatus:
    status = await _load_job(request.app.state.redis, job_id)
    if status is None:
        raise NotFoundError(f"Job {job_id!r} not found")
    return status


@router.get("/stream/{job_id}")
async def stream_answer(job_id: UUID, request: Request) -> StreamingResponse:
    """Stream job answer as SSE events.

    Polls Redis at a 0.5 s interval; emits SSE keepalive comments at
    settings.stream_heartbeat_interval cadence to satisfy proxy timeouts.
    """
    job_id_str = str(job_id)
    redis = request.app.state.redis

    job = await _load_job(redis, job_id_str)
    if job is None:
        raise NotFoundError(f"Job {job_id_str!r} not found")

    async def _sse_gen() -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 60.0
        last_heartbeat = loop.time()

        while loop.time() < deadline:
            data = await redis.client.hgetall(job_key(job_id_str))
            status = data.get("status", "")
            if status == "done":
                answer = data.get("answer", "")
                for word in answer.split():
                    yield f"data: {word}\n\n"
                yield "data: [DONE]\n\n"
                return
            if status == "error":
                err = data.get("error", "unknown error")
                yield f"data: [ERROR] {err}\n\n"
                return
            now = loop.time()
            if now - last_heartbeat >= settings.stream_heartbeat_interval:
                yield ": heartbeat\n\n"
                last_heartbeat = now
            await asyncio.sleep(_STREAM_POLL_INTERVAL)
        yield "data: [TIMEOUT]\n\n"

    return StreamingResponse(_sse_gen(), media_type="text/event-stream")
