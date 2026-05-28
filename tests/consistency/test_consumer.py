"""Unit tests for ConsistencyConsumer."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.clients.mq import MQMessage
from app.consistency.consumer import ConsistencyConsumer


def _make_mq(*messages: MQMessage) -> MagicMock:
    async def _gen():
        for msg in messages:
            yield msg

    mock = MagicMock()
    mock.consume = MagicMock(return_value=_gen())
    mock.ack = AsyncMock()
    return mock


def _make_invalidator() -> MagicMock:
    inv = MagicMock()
    inv.invalidate = AsyncMock(return_value=5)
    return inv


def _msg(event: str, msg_id: str = "123-0") -> MQMessage:
    return MQMessage(id=msg_id, data={"event": event, "doc_id": "d1", "version": "v1"}, stream="s")


class TestConsistencyConsumerNoMessages:
    async def test_returns_zero_when_no_messages(self):
        mq = _make_mq()
        inv = _make_invalidator()
        consumer = ConsistencyConsumer(mq, inv, "v1")
        result = await consumer.run_once()
        assert result == 0

    async def test_invalidator_not_called_when_no_messages(self):
        mq = _make_mq()
        inv = _make_invalidator()
        consumer = ConsistencyConsumer(mq, inv, "v1")
        await consumer.run_once()
        inv.invalidate.assert_not_awaited()


class TestConsistencyConsumerKbUpdated:
    async def test_processes_kb_updated_event(self):
        mq = _make_mq(_msg("kb_updated"))
        inv = _make_invalidator()
        consumer = ConsistencyConsumer(mq, inv, "v1")
        result = await consumer.run_once()
        assert result == 1

    async def test_calls_invalidator_with_version_on_kb_updated(self):
        mq = _make_mq(_msg("kb_updated"))
        inv = _make_invalidator()
        consumer = ConsistencyConsumer(mq, inv, "v1")
        await consumer.run_once()
        inv.invalidate.assert_awaited_once_with("v1")

    async def test_uses_version_from_message_payload(self):
        msg = MQMessage(id="1-0", data={"event": "kb_updated", "doc_id": "d1", "version": "v9"}, stream="s")
        mq = _make_mq(msg)
        inv = _make_invalidator()
        consumer = ConsistencyConsumer(mq, inv, "v1")
        await consumer.run_once()
        inv.invalidate.assert_awaited_once_with("v9")

    async def test_acks_kb_updated_message(self):
        mq = _make_mq(_msg("kb_updated", "456-0"))
        inv = _make_invalidator()
        consumer = ConsistencyConsumer(mq, inv, "v1")
        await consumer.run_once()
        mq.ack.assert_awaited_once_with("456-0")


class TestConsistencyConsumerKbDeleted:
    async def test_processes_kb_deleted_event(self):
        mq = _make_mq(_msg("kb_deleted"))
        inv = _make_invalidator()
        consumer = ConsistencyConsumer(mq, inv, "v1")
        result = await consumer.run_once()
        assert result == 1

    async def test_calls_invalidator_with_version_on_kb_deleted(self):
        mq = _make_mq(_msg("kb_deleted"))
        inv = _make_invalidator()
        consumer = ConsistencyConsumer(mq, inv, "v1")
        await consumer.run_once()
        inv.invalidate.assert_awaited_once_with("v1")

    async def test_acks_kb_deleted_message(self):
        mq = _make_mq(_msg("kb_deleted", "789-0"))
        inv = _make_invalidator()
        consumer = ConsistencyConsumer(mq, inv, "v1")
        await consumer.run_once()
        mq.ack.assert_awaited_once_with("789-0")


class TestConsistencyConsumerUnknownEvent:
    async def test_ignores_unknown_event_type(self):
        mq = _make_mq(_msg("some_other_event"))
        inv = _make_invalidator()
        consumer = ConsistencyConsumer(mq, inv, "v1")
        result = await consumer.run_once()
        assert result == 0

    async def test_still_acks_unknown_event(self):
        mq = _make_mq(_msg("unknown", "111-0"))
        inv = _make_invalidator()
        consumer = ConsistencyConsumer(mq, inv, "v1")
        await consumer.run_once()
        mq.ack.assert_awaited_once_with("111-0")

    async def test_does_not_call_invalidator_for_unknown_event(self):
        mq = _make_mq(_msg("unknown"))
        inv = _make_invalidator()
        consumer = ConsistencyConsumer(mq, inv, "v1")
        await consumer.run_once()
        inv.invalidate.assert_not_awaited()


class TestConsistencyConsumerMultipleMessages:
    async def test_counts_all_relevant_events(self):
        mq = _make_mq(_msg("kb_updated", "1-0"), _msg("kb_deleted", "2-0"), _msg("other", "3-0"))
        inv = _make_invalidator()
        consumer = ConsistencyConsumer(mq, inv, "v1")
        result = await consumer.run_once()
        assert result == 2

    async def test_acks_all_messages_regardless_of_type(self):
        mq = _make_mq(_msg("kb_updated", "1-0"), _msg("unknown", "2-0"))
        inv = _make_invalidator()
        consumer = ConsistencyConsumer(mq, inv, "v1")
        await consumer.run_once()
        assert mq.ack.await_count == 2

    async def test_invalidator_called_once_per_relevant_event(self):
        mq = _make_mq(_msg("kb_updated", "1-0"), _msg("kb_deleted", "2-0"))
        inv = _make_invalidator()
        consumer = ConsistencyConsumer(mq, inv, "v1")
        await consumer.run_once()
        assert inv.invalidate.await_count == 2
