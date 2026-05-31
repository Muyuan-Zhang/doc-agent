"""M4 MQ consumer — runs as a background asyncio.Task in the app lifespan.

The consumer reads jobs from Redis Streams, executes the LangGraph pipeline,
and writes job status back to Redis. It is cancelled (and exits cleanly) when
the FastAPI lifespan shuts down.
"""
import asyncio
import logging
import time

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
    logger.info("job_status=update job=%s status=%s answer_len=%d error=%.200s", job_id, status, len(answer), error)


async def _process_message(msg: MQMessage, graph, redis, mq) -> None:
    job_id = msg.data.get("job_id", "unknown")
    t0 = time.perf_counter()
    logger.info("job=start job=%s session=%s query=%.120s", job_id, msg.data.get("session_id", ""), msg.data.get("query", ""))

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
            "cached_answer": "",
            "query_embedding": None,
            "rag_cache_hash": None,
            "error": None,
            "user_id": msg.data.get("user_id", ""),
            "memory_context": None,
        }
        result = await graph.ainvoke(state)
        total = time.perf_counter() - t0
        answer = result.get("answer", "")
        logger.info("job=done job=%s answer_len=%d total_elapsed=%.3fs", job_id, len(answer), total)
        await _set_job_status(redis, job_id, "done", answer=answer)
    except asyncio.CancelledError:
        logger.info("job=cancelled job=%s elapsed=%.3fs", job_id, time.perf_counter() - t0)
        raise
    except Exception as exc:
        total = time.perf_counter() - t0
        logger.error("job=failed job=%s error=%s elapsed=%.3fs", job_id, exc, total)
        logger.debug("job=%s exception detail:", job_id, exc_info=True)
        error_msg = f"{type(exc).__name__}: job processing failed"
        await _set_job_status(redis, job_id, "error", error=error_msg)
    finally:
        await mq.ack(msg.id)
        logger.debug("job=acked job=%s", job_id)


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
