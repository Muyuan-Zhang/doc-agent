"""M4 Agent API endpoints.

POST /agent/query           — enqueue a query job; returns job_id immediately (async)
GET  /agent/jobs/{job_id}   — poll job status / answer
GET  /agent/stream/{job_id} — stream the answer as SSE events (named: token/done/error/timeout)
"""
import asyncio
import logging
from uuid import UUID, uuid4

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from app.agent._keys import job_key, token_stream_key
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
        "user_id": body.user_id or "",
    })
    return QueryResponse(job_id=job_id, status="queued")


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str, request: Request) -> JobStatus:
    status = await _load_job(request.app.state.redis, job_id)
    if status is None:
        raise NotFoundError(f"Job {job_id!r} not found")
    return status


@router.get("/stream/{job_id}")
async def stream_answer(job_id: UUID, request: Request) -> EventSourceResponse:
    """Stream job answer as SSE named events.

    Polls the Redis token List at 0.5 s intervals, forwarding each buffered
    LLM token as an ``event: token`` message.  Terminates with ``event: done``,
    ``event: error``, or ``event: timeout``.  Automatic SSE keepalive pings
    are sent by EventSourceResponse at ``settings.stream_heartbeat_interval``.
    """
    job_id_str = str(job_id)
    redis = request.app.state.redis

    job = await _load_job(redis, job_id_str)
    if job is None:
        raise NotFoundError(f"Job {job_id_str!r} not found")

    stream_key = token_stream_key(job_id_str)

    async def _sse_gen():
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 60.0
        sent_idx = 0

        while loop.time() < deadline:
            new_tokens = await redis.client.lrange(stream_key, sent_idx, -1)
            for token in new_tokens:
                yield {"event": "token", "data": token}
            sent_idx += len(new_tokens)

            data = await redis.client.hgetall(job_key(job_id_str))
            status = data.get("status", "")

            if status in ("done", "error"):
                remaining = await redis.client.lrange(stream_key, sent_idx, -1)
                for token in remaining:
                    yield {"event": "token", "data": token}
                if status == "done":
                    yield {"event": "done", "data": ""}
                else:
                    raw_err = data.get("error", "")
                    logger.error("Job %s failed: %s", job_id_str, raw_err)
                    yield {"event": "error", "data": "job_failed"}
                return

            await asyncio.sleep(_STREAM_POLL_INTERVAL)

        yield {"event": "timeout", "data": ""}

    return EventSourceResponse(_sse_gen(), ping=int(settings.stream_heartbeat_interval))
