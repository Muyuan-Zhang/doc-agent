"""
E2E tests for the M6 consistency pipeline.

All tests use the same mock-state pattern as the rest of the E2E suite:
- No real Redis, MQ, or DB connections
- Redis .client.scan / .unlink calls are tracked to verify invalidation
- MQ .consume is replaced with an async generator of pre-built messages

Test scope:
1. M1 → M6 event format compatibility (published shape matches consumer expectations)
2. Full pipeline: MQMessage → ConsistencyConsumer → CacheInvalidator → Redis UNLINK
3. ConsistencyService lifecycle: start / stop (timing-free)
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.clients.mq import MQMessage
from app.consistency.consumer import ConsistencyConsumer
from app.consistency.invalidator import CacheInvalidator
from app.consistency.service import ConsistencyService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_redis_with_keys(keys: list[str]) -> MagicMock:
    """Redis mock whose SCAN returns the given keys in a single batch."""
    redis = MagicMock()
    redis.client = AsyncMock()
    redis.client.scan = AsyncMock(return_value=(0, keys))
    redis.client.unlink = AsyncMock()
    return redis


def _make_mq_with_messages(*payloads: dict) -> MagicMock:
    """MQ mock whose .consume yields one MQMessage per payload dict."""
    messages = [
        MQMessage(id=f"{i}-0", data=p, stream="doc-agent:tasks")
        for i, p in enumerate(payloads)
    ]

    async def _gen():
        for msg in messages:
            yield msg

    mq = MagicMock()
    mq.consume = MagicMock(return_value=_gen())
    mq.ack = AsyncMock()
    return mq


async def _run_pipeline(event_payload: dict, cache_keys: list[str]) -> tuple[int, MagicMock]:
    """Wire a ConsistencyConsumer end-to-end and return (count, redis_mock)."""
    redis = _make_redis_with_keys(cache_keys)
    mq = _make_mq_with_messages(event_payload)
    inv = CacheInvalidator(redis, namespace="rag", max_iterations=10)
    consumer = ConsistencyConsumer(mq, inv, version="v1")
    count = await consumer.run_once()
    return count, redis


# ---------------------------------------------------------------------------
# M1 ↔ M6 event format compatibility
# ---------------------------------------------------------------------------

class TestM1EventFormatCompatibility:
    """Verify the exact dict M1 publishes is understood by the M6 consumer."""

    async def test_kb_updated_payload_triggers_invalidation(self):
        # UpdateCoordinator.ingest() publishes exactly this dict
        payload = {"event": "kb_updated", "doc_id": "doc-uuid-1", "version": "v1"}
        count, redis = await _run_pipeline(payload, cache_keys=["v1:rag:abc"])
        assert count == 1
        redis.client.unlink.assert_awaited_once()

    async def test_kb_deleted_payload_triggers_invalidation(self):
        # KnowledgeBaseService.delete_document() publishes exactly this dict
        payload = {"event": "kb_deleted", "doc_id": "doc-uuid-1", "version": "v1"}
        count, redis = await _run_pipeline(payload, cache_keys=["v1:rag:abc"])
        assert count == 1
        redis.client.unlink.assert_awaited_once()

    async def test_consumer_uses_version_from_event_not_constructor(self):
        """Version bump scenario: event carries v2, constructor has v1."""
        redis = _make_redis_with_keys(["v2:rag:xyz"])
        mq = _make_mq_with_messages({"event": "kb_updated", "doc_id": "d1", "version": "v2"})
        inv = CacheInvalidator(redis, namespace="rag", max_iterations=10)
        consumer = ConsistencyConsumer(mq, inv, version="v1")
        await consumer.run_once()
        scan_pattern = redis.client.scan.call_args.kwargs["match"]
        assert scan_pattern.startswith("v2:")

    async def test_no_version_in_event_falls_back_to_constructor(self):
        """If event omits version field, constructor default is used."""
        redis = _make_redis_with_keys([])
        mq = _make_mq_with_messages({"event": "kb_updated", "doc_id": "d1"})
        inv = CacheInvalidator(redis, namespace="rag", max_iterations=10)
        consumer = ConsistencyConsumer(mq, inv, version="v1")
        await consumer.run_once()
        scan_pattern = redis.client.scan.call_args.kwargs["match"]
        assert scan_pattern.startswith("v1:")


# ---------------------------------------------------------------------------
# Full invalidation pipeline
# ---------------------------------------------------------------------------

class TestInvalidationPipeline:
    async def test_matching_keys_are_deleted(self):
        payload = {"event": "kb_updated", "doc_id": "d1", "version": "v1"}
        keys = ["v1:rag:q1", "v1:rag:q2", "v1:rag:q3"]
        count, redis = await _run_pipeline(payload, keys)
        redis.client.unlink.assert_awaited_once_with(*keys)

    async def test_no_keys_means_unlink_not_called(self):
        payload = {"event": "kb_updated", "doc_id": "d1", "version": "v1"}
        count, redis = await _run_pipeline(payload, cache_keys=[])
        redis.client.unlink.assert_not_awaited()

    async def test_multiple_events_each_trigger_invalidation(self):
        redis = _make_redis_with_keys(["v1:rag:k"])
        mq = _make_mq_with_messages(
            {"event": "kb_updated", "doc_id": "d1", "version": "v1"},
            {"event": "kb_deleted", "doc_id": "d2", "version": "v1"},
        )

        async def _gen():
            for p in [
                {"event": "kb_updated", "doc_id": "d1", "version": "v1"},
                {"event": "kb_deleted", "doc_id": "d2", "version": "v1"},
            ]:
                yield MQMessage(id="1-0", data=p, stream="s")

        mq.consume = MagicMock(return_value=_gen())
        inv = CacheInvalidator(redis, namespace="rag", max_iterations=10)
        consumer = ConsistencyConsumer(mq, inv, version="v1")
        count = await consumer.run_once()
        assert count == 2
        assert redis.client.scan.await_count == 2

    async def test_all_messages_are_acked(self):
        redis = _make_redis_with_keys([])
        mq = _make_mq_with_messages(
            {"event": "kb_updated", "doc_id": "d1", "version": "v1"},
            {"event": "unknown_event"},
        )
        inv = CacheInvalidator(redis, namespace="rag")
        consumer = ConsistencyConsumer(mq, inv, version="v1")
        await consumer.run_once()
        assert mq.ack.await_count == 2

    async def test_invalid_version_in_event_does_not_cause_scan(self):
        redis = _make_redis_with_keys(["v1:rag:k"])
        payload = {"event": "kb_updated", "doc_id": "d1", "version": "bad*version"}
        mq = _make_mq_with_messages(payload)
        inv = CacheInvalidator(redis, namespace="rag")
        consumer = ConsistencyConsumer(mq, inv, version="v1")
        count = await consumer.run_once()
        assert count == 0
        redis.client.scan.assert_not_awaited()


# ---------------------------------------------------------------------------
# ConsistencyService lifecycle
# ---------------------------------------------------------------------------

class TestConsistencyServiceLifecycle:
    async def test_service_stop_is_clean(self):
        consumer = MagicMock()
        consumer.run_once = AsyncMock(return_value=0)
        svc = ConsistencyService(consumer)
        await svc.start()
        assert svc._task is not None
        await svc.stop()
        assert svc._task is None

    async def test_loop_processes_event_then_continues(self):
        """_loop calls run_once repeatedly; stop cancels it cleanly."""
        call_count = 0

        async def side_effect():
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise asyncio.CancelledError

        consumer = MagicMock()
        consumer.run_once = AsyncMock(side_effect=side_effect)
        svc = ConsistencyService(consumer)
        with pytest.raises(asyncio.CancelledError):
            await svc._loop()
        assert call_count == 2

    async def test_loop_recovers_from_error_without_sleep(self):
        """Error path: loop catches exception and retries (sleep mocked)."""
        call_count = 0

        async def side_effect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient")
            raise asyncio.CancelledError

        consumer = MagicMock()
        consumer.run_once = AsyncMock(side_effect=side_effect)
        svc = ConsistencyService(consumer)
        with patch("app.consistency.service.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(asyncio.CancelledError):
                await svc._loop()
        assert call_count == 2
