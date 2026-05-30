"""M4 MQ consumer — runs as a background asyncio.Task in the app lifespan.

The consumer reads jobs from Redis Streams, executes the LangGraph pipeline,
and writes job status back to Redis. It is cancelled (and exits cleanly) when
the FastAPI lifespan shuts down.
"""
import asyncio
import logging

from app.agent._keys import job_key, token_stream_key
from app.agent.state import AgentState
from app.clients.mq import MQMessage
from app.core.config import settings

logger = logging.getLogger(__name__)

# Re-export so external callers can discover the key format via this module.
JOB_STATUS_KEY = job_key


async def _set_job_status(
    redis,
    job_id: str,
    status: str,
    *,
    answer: str = "",
    error: str = "",
) -> None:
    key = job_key(job_id)
    await redis.client.hset(key, mapping={"status": status, "answer": answer, "error": error})
    await redis.client.expire(key, settings.agent_job_ttl_seconds)
    await redis.client.expire(token_stream_key(job_id), settings.agent_job_ttl_seconds)


async def _process_message(msg: MQMessage, graph, redis, mq) -> None:
    job_id = msg.data.get("job_id", "unknown")
    try:
        await _set_job_status(redis, job_id, "running")
        state: AgentState = {
            "session_id": msg.data.get("session_id", ""),
            "job_id": job_id,
            "query": msg.data.get("query", ""),
            "top_k": int(msg.data.get("top_k", "5")),
            "rewritten_query": "",
            "chunks": [],
            "reranked_chunks": [],
            "answer": "",
            "cache_hit": False,
            "error": None,
        }
        result = await graph.ainvoke(state)
        await _set_job_status(redis, job_id, "done", answer=result.get("answer", ""))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error("Job %s failed: %s", job_id, exc)
        logger.debug("Job %s exception detail:", job_id, exc_info=True)
        error_msg = f"{type(exc).__name__}: job processing failed"
        await _set_job_status(redis, job_id, "error", error=error_msg)
    finally:
        await mq.ack(msg.id)


async def run_consumer(mq, graph, redis) -> None:
    """Long-running consumer loop. Cancelled by the app lifespan on shutdown.

    Note: messages within a single batch are processed sequentially.
    Concurrent processing across batches is deferred to a future iteration
    as it requires changes to the task lifecycle and test harness.
    """
    logger.info("MQ consumer started")
    while True:
        try:
            async for msg in mq.consume():
                await _process_message(msg, graph, redis, mq)
        except asyncio.CancelledError:
            logger.info("MQ consumer cancelled")
            raise
        except Exception as exc:
            logger.error("Consumer loop error (will retry): %s", exc)
            await asyncio.sleep(1)
