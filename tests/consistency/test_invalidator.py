"""Unit tests for CacheInvalidator."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.consistency.invalidator import CacheInvalidator


def _make_redis(scan_results: list[tuple[int, list[str]]]) -> MagicMock:
    """Build a RedisClient mock whose .client.scan returns the given (cursor, keys) pairs."""
    mock_redis = MagicMock()
    mock_client = AsyncMock()
    mock_redis.client = mock_client
    mock_client.scan = AsyncMock(side_effect=scan_results)
    mock_client.unlink = AsyncMock()
    return mock_redis


class TestCacheInvalidatorNoKeys:
    async def test_returns_zero_when_no_keys_match(self):
        redis = _make_redis([(0, [])])
        inv = CacheInvalidator(redis, namespace="rag")
        result = await inv.invalidate("v1")
        assert result == 0

    async def test_unlink_not_called_when_no_keys(self):
        redis = _make_redis([(0, [])])
        inv = CacheInvalidator(redis, namespace="rag")
        await inv.invalidate("v1")
        redis.client.unlink.assert_not_awaited()


class TestCacheInvalidatorWithKeys:
    async def test_returns_count_of_deleted_keys(self):
        redis = _make_redis([(0, ["v1:rag:abc", "v1:rag:def", "v1:rag:ghi"])])
        inv = CacheInvalidator(redis, namespace="rag")
        result = await inv.invalidate("v1")
        assert result == 3

    async def test_calls_unlink_with_all_keys(self):
        keys = ["v1:rag:abc", "v1:rag:def"]
        redis = _make_redis([(0, keys)])
        inv = CacheInvalidator(redis, namespace="rag")
        await inv.invalidate("v1")
        redis.client.unlink.assert_awaited_once_with(*keys)

    async def test_uses_version_in_scan_pattern(self):
        redis = _make_redis([(0, [])])
        inv = CacheInvalidator(redis, namespace="rag")
        await inv.invalidate("v2")
        assert redis.client.scan.call_args.kwargs["match"] == "v2:rag:*"

    async def test_uses_namespace_in_scan_pattern(self):
        redis = _make_redis([(0, [])])
        inv = CacheInvalidator(redis, namespace="summary")
        await inv.invalidate("v1")
        assert redis.client.scan.call_args.kwargs["match"] == "v1:summary:*"


class TestCacheInvalidatorMaxIterations:
    async def test_stops_at_max_iterations(self):
        redis = MagicMock()
        redis.client = AsyncMock()
        redis.client.scan = AsyncMock(return_value=(5, ["v1:rag:k"]))
        redis.client.unlink = AsyncMock()
        inv = CacheInvalidator(redis, namespace="rag", max_iterations=3)
        result = await inv.invalidate("v1")
        assert redis.client.scan.await_count == 3
        assert result == 3

    async def test_yields_event_loop_between_batches(self):
        redis = _make_redis([(5, ["v1:rag:k1"]), (0, ["v1:rag:k2"])])
        inv = CacheInvalidator(redis, namespace="rag")
        with patch("app.consistency.invalidator.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await inv.invalidate("v1")
        mock_sleep.assert_awaited_once_with(0)

    async def test_no_yield_on_single_batch(self):
        redis = _make_redis([(0, ["v1:rag:k1"])])
        inv = CacheInvalidator(redis, namespace="rag")
        with patch("app.consistency.invalidator.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await inv.invalidate("v1")
        mock_sleep.assert_not_awaited()


class TestCacheInvalidatorPagination:
    async def test_paginates_when_scan_has_multiple_batches(self):
        redis = _make_redis([
            (10, ["v1:rag:k1", "v1:rag:k2"]),
            (0, ["v1:rag:k3"]),
        ])
        inv = CacheInvalidator(redis, namespace="rag")
        result = await inv.invalidate("v1")
        assert result == 3

    async def test_stops_iteration_when_cursor_returns_to_zero(self):
        redis = _make_redis([
            (5, ["v1:rag:k1"]),
            (0, []),
        ])
        inv = CacheInvalidator(redis, namespace="rag")
        result = await inv.invalidate("v1")
        assert result == 1
        assert redis.client.scan.await_count == 2

    async def test_unlink_called_once_per_non_empty_batch(self):
        redis = _make_redis([
            (5, ["v1:rag:k1"]),
            (0, ["v1:rag:k2"]),
        ])
        inv = CacheInvalidator(redis, namespace="rag")
        await inv.invalidate("v1")
        assert redis.client.unlink.await_count == 2
