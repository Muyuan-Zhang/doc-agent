"""Unit tests for the M4 MQ consumer loop.

Tests exercise _process_message directly (avoids infinite loop complexity)
and also validate run_consumer stops cleanly on CancelledError.
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, call

from app.agent.consumer import _process_message, run_consumer
from app.clients.mq import MQMessage


def _make_msg(job_id: str = "job-1", query: str = "hello", session_id: str = "s1",
             user_id: str = "") -> MQMessage:
    return MQMessage(
        id=f"{job_id}-0",
        data={"job_id": job_id, "session_id": session_id, "query": query, "top_k": "5",
              "user_id": user_id},
        stream="doc-agent:tasks",
    )


def _make_redis_mock() -> MagicMock:
    redis = MagicMock()
    redis.cache_key = MagicMock(return_value="v1:agent:job:job-1")
    inner = MagicMock()
    inner.hset = AsyncMock()
    inner.expire = AsyncMock()
    redis.client = inner
    return redis


def _make_graph_mock(answer: str = "great answer") -> MagicMock:
    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value={
        "answer": answer,
        "error": None,
    })
    return graph


# ---------------------------------------------------------------------------
# _process_message
# ---------------------------------------------------------------------------

class TestProcessMessage:
    async def test_invokes_graph_with_correct_initial_state(self):
        msg = _make_msg(job_id="j1", query="what is redis?", session_id="sess-x",
                        user_id="u1")
        graph = _make_graph_mock()
        redis = _make_redis_mock()
        mq = MagicMock()
        mq.ack = AsyncMock()

        await _process_message(msg, graph, redis, mq)

        invoked_state = graph.ainvoke.call_args[0][0]
        assert invoked_state["job_id"] == "j1"
        assert invoked_state["query"] == "what is redis?"
        assert invoked_state["session_id"] == "sess-x"
        assert invoked_state["top_k"] == 5
        assert invoked_state["user_id"] == "u1"

    async def test_user_id_defaults_to_empty_string_when_not_in_message(self):
        msg = _make_msg(job_id="j1", session_id="sess-x", user_id="")
        graph = _make_graph_mock()
        redis = _make_redis_mock()
        mq = MagicMock()
        mq.ack = AsyncMock()

        await _process_message(msg, graph, redis, mq)

        invoked_state = graph.ainvoke.call_args[0][0]
        assert invoked_state["user_id"] == ""


    async def test_acks_message_on_success(self):
        msg = _make_msg()
        graph = _make_graph_mock()
        redis = _make_redis_mock()
        mq = MagicMock()
        mq.ack = AsyncMock()

        await _process_message(msg, graph, redis, mq)

        mq.ack.assert_awaited_once_with(msg.id)

    async def test_acks_message_on_graph_error(self):
        msg = _make_msg(job_id="j-err")
        graph = MagicMock()
        graph.ainvoke = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
        redis = _make_redis_mock()
        mq = MagicMock()
        mq.ack = AsyncMock()

        await _process_message(msg, graph, redis, mq)

        mq.ack.assert_awaited_once_with(msg.id)

    async def test_sets_status_running_then_done_on_success(self):
        msg = _make_msg(job_id="j-ok")
        graph = _make_graph_mock(answer="perfect answer")
        redis = _make_redis_mock()
        mq = MagicMock()
        mq.ack = AsyncMock()

        await _process_message(msg, graph, redis, mq)

        hset_calls = redis.client.hset.await_args_list
        statuses = [c.kwargs.get("mapping", {}).get("status") for c in hset_calls]
        assert "running" in statuses
        assert "done" in statuses

    async def test_sets_status_error_on_graph_failure(self):
        msg = _make_msg(job_id="j-fail")
        graph = MagicMock()
        graph.ainvoke = AsyncMock(side_effect=ValueError("bad input"))
        redis = _make_redis_mock()
        mq = MagicMock()
        mq.ack = AsyncMock()

        await _process_message(msg, graph, redis, mq)

        hset_calls = redis.client.hset.await_args_list
        statuses = [c.kwargs.get("mapping", {}).get("status") for c in hset_calls]
        assert "error" in statuses

    async def test_token_stream_key_is_expired_on_success(self):
        from app.agent._keys import token_stream_key

        msg = _make_msg(job_id="j-tok")
        graph = _make_graph_mock()
        redis = _make_redis_mock()
        mq = MagicMock()
        mq.ack = AsyncMock()

        await _process_message(msg, graph, redis, mq)

        expire_keys = [c.args[0] for c in redis.client.expire.await_args_list]
        assert token_stream_key("j-tok") in expire_keys

    async def test_token_stream_key_is_expired_on_error(self):
        from app.agent._keys import token_stream_key

        msg = _make_msg(job_id="j-tok-err")
        graph = MagicMock()
        graph.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
        redis = _make_redis_mock()
        mq = MagicMock()
        mq.ack = AsyncMock()

        await _process_message(msg, graph, redis, mq)

        expire_keys = [c.args[0] for c in redis.client.expire.await_args_list]
        assert token_stream_key("j-tok-err") in expire_keys

    async def test_error_message_stored_in_redis(self):
        msg = _make_msg(job_id="j-fail2")
        graph = MagicMock()
        graph.ainvoke = AsyncMock(side_effect=ValueError("specific error text"))
        redis = _make_redis_mock()
        mq = MagicMock()
        mq.ack = AsyncMock()

        await _process_message(msg, graph, redis, mq)

        all_mappings = [
            c.kwargs.get("mapping", {})
            for c in redis.client.hset.await_args_list
        ]
        error_mapping = next(m for m in all_mappings if m.get("status") == "error")
        assert "ValueError" in error_mapping.get("error", "")


# ---------------------------------------------------------------------------
# run_consumer (loop-level)
# ---------------------------------------------------------------------------

class TestRunConsumer:
    async def test_stops_cleanly_on_cancelled_error(self):
        call_count = 0

        async def mock_consume(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise asyncio.CancelledError()
            # Empty async generator on first call.
            # The `if False` branch makes this an async generator without
            # an unreachable statement after return.
            if False:
                yield

        mq = MagicMock()
        mq.consume = mock_consume
        mq.ack = AsyncMock()

        graph = _make_graph_mock()
        redis = _make_redis_mock()

        with pytest.raises(asyncio.CancelledError):
            await run_consumer(mq, graph, redis)

    async def test_processes_multiple_messages_before_cancellation(self):
        messages = [_make_msg(job_id=f"j{i}") for i in range(3)]
        call_count = 0

        async def mock_consume(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                for m in messages:
                    yield m
            else:
                raise asyncio.CancelledError()

        mq = MagicMock()
        mq.consume = mock_consume
        mq.ack = AsyncMock()

        graph = _make_graph_mock()
        redis = _make_redis_mock()

        with pytest.raises(asyncio.CancelledError):
            await run_consumer(mq, graph, redis)

        assert mq.ack.await_count == 3
