"""Unit tests for ConsistencyService."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.consistency.service import ConsistencyService


def _make_consumer() -> MagicMock:
    c = MagicMock()
    c.run_once = AsyncMock(return_value=0)
    return c


class TestConsistencyServiceStart:
    async def test_start_creates_background_task(self):
        consumer = _make_consumer()
        svc = ConsistencyService(consumer)
        await svc.start()
        assert svc._task is not None
        await svc.stop()

    async def test_task_is_running_after_start(self):
        consumer = _make_consumer()
        svc = ConsistencyService(consumer)
        await svc.start()
        assert not svc._task.done()
        await svc.stop()


class TestConsistencyServiceStop:
    async def test_stop_cancels_task(self):
        consumer = _make_consumer()
        svc = ConsistencyService(consumer)
        await svc.start()
        await svc.stop()
        assert svc._task is None

    async def test_stop_is_idempotent_when_not_started(self):
        consumer = _make_consumer()
        svc = ConsistencyService(consumer)
        await svc.stop()

    async def test_stop_clears_task_reference(self):
        consumer = _make_consumer()
        svc = ConsistencyService(consumer)
        await svc.start()
        await svc.stop()
        assert svc._task is None


class TestConsistencyServiceLoop:
    async def test_loop_reraises_cancelled_error(self):
        consumer = _make_consumer()
        consumer.run_once = AsyncMock(side_effect=asyncio.CancelledError)
        svc = ConsistencyService(consumer)
        with pytest.raises(asyncio.CancelledError):
            await svc._loop()

    async def test_loop_continues_after_runtime_error(self):
        consumer = _make_consumer()
        call_count = 0

        async def side_effect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient error")
            raise asyncio.CancelledError

        consumer.run_once = AsyncMock(side_effect=side_effect)
        svc = ConsistencyService(consumer)
        with pytest.raises(asyncio.CancelledError):
            await svc._loop()
        assert call_count == 2

    async def test_loop_calls_run_once(self):
        consumer = _make_consumer()
        call_count = 0

        async def side_effect():
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                raise asyncio.CancelledError
            return 0

        consumer.run_once = AsyncMock(side_effect=side_effect)
        svc = ConsistencyService(consumer)
        with pytest.raises(asyncio.CancelledError):
            await svc._loop()
        assert call_count >= 1

    async def test_loop_backs_off_after_error(self):
        consumer = _make_consumer()
        call_count = 0

        async def side_effect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("redis down")
            raise asyncio.CancelledError

        consumer.run_once = AsyncMock(side_effect=side_effect)
        svc = ConsistencyService(consumer)
        with patch("app.consistency.service.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(asyncio.CancelledError):
                await svc._loop()
        mock_sleep.assert_awaited_once()

    async def test_loop_resets_backoff_on_success(self):
        from app.consistency.service import _BASE_BACKOFF
        consumer = _make_consumer()
        call_count = 0

        async def side_effect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient")
            if call_count == 2:
                return 0  # success — backoff should reset
            raise asyncio.CancelledError

        consumer.run_once = AsyncMock(side_effect=side_effect)
        svc = ConsistencyService(consumer)
        sleep_calls = []
        async def mock_sleep(n):
            sleep_calls.append(n)
        with patch("app.consistency.service.asyncio.sleep", side_effect=mock_sleep):
            with pytest.raises(asyncio.CancelledError):
                await svc._loop()
        assert sleep_calls == [_BASE_BACKOFF]
