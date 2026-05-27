"""Unit tests for MilvusClient — P0."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pymilvus import connections, utility

from app.clients.milvus import MilvusClient
from app.core.config import settings


class TestMilvusClientRunSync:
    async def test_run_sync_delegates_to_asyncio_to_thread(self):
        import asyncio

        client = MilvusClient()
        fn = MagicMock(return_value=99)
        with patch.object(asyncio, "to_thread", new=AsyncMock(return_value=99)) as mock_thread:
            result = await client._run_sync(fn, "a", key="b")
        mock_thread.assert_awaited_once_with(fn, "a", key="b")
        assert result == 99


class TestMilvusClientConnect:
    async def test_connect_calls_connections_connect(self):
        client = MilvusClient()
        with patch("app.clients.milvus.connections.connect") as mock_connect:
            with patch("asyncio.to_thread", new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw))):
                await client.connect()
        mock_connect.assert_called_once()

    async def test_connect_passes_host_and_port(self):
        client = MilvusClient()
        calls: list = []

        async def fake_thread(fn, *args, **kwargs):
            calls.append((fn, kwargs))

        with patch("asyncio.to_thread", side_effect=fake_thread):
            await client.connect()

        _, kwargs = calls[0]
        assert kwargs["host"] == settings.milvus_host
        assert kwargs["port"] == settings.milvus_port

    async def test_connect_uses_alias_from_settings(self):
        client = MilvusClient()
        calls: list = []

        async def fake_thread(fn, *args, **kwargs):
            calls.append((fn, kwargs))

        with patch("asyncio.to_thread", side_effect=fake_thread):
            await client.connect()

        _, kwargs = calls[0]
        assert kwargs["alias"] == settings.milvus_alias


class TestMilvusClientDisconnect:
    async def test_disconnect_calls_connections_disconnect(self):
        client = MilvusClient()
        with patch("app.clients.milvus.connections.disconnect") as mock_disconnect:
            with patch("asyncio.to_thread", new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw))):
                await client.disconnect()
        mock_disconnect.assert_called_once()

    async def test_disconnect_passes_alias(self):
        client = MilvusClient()
        calls: list = []

        async def fake_thread(fn, *args, **kwargs):
            calls.append((fn, kwargs))

        with patch("asyncio.to_thread", side_effect=fake_thread):
            await client.disconnect()

        _, kwargs = calls[0]
        assert kwargs["alias"] == settings.milvus_alias


class TestMilvusClientPing:
    async def test_ping_returns_true_when_version_obtained(self):
        client = MilvusClient()

        async def fake_thread(fn, *args, **kwargs):
            return "2.4.0"

        with patch("asyncio.to_thread", side_effect=fake_thread):
            result = await client.ping()
        assert result is True

    async def test_ping_calls_get_server_version(self):
        client = MilvusClient()
        calls: list = []

        async def fake_thread(fn, *args, **kwargs):
            calls.append(fn)
            return "2.4.0"

        with patch("asyncio.to_thread", side_effect=fake_thread):
            await client.ping()

        assert calls[0] is utility.get_server_version

    async def test_ping_returns_false_on_exception(self):
        client = MilvusClient()

        async def fake_thread(fn, *args, **kwargs):
            raise ConnectionError("Milvus unreachable")

        with patch("asyncio.to_thread", side_effect=fake_thread):
            result = await client.ping()
        assert result is False

    async def test_ping_passes_alias_to_get_server_version(self):
        client = MilvusClient()
        calls: list = []

        async def fake_thread(fn, *args, **kwargs):
            calls.append((fn, kwargs))
            return "2.4.0"

        with patch("asyncio.to_thread", side_effect=fake_thread):
            await client.ping()

        _, kwargs = calls[0]
        assert kwargs["using"] == settings.milvus_alias


class TestMilvusClientEnsureKbCollection:
    async def test_does_not_create_if_collection_exists(self):
        client = MilvusClient()
        inner_fn = None

        async def capture_thread(fn, *args, **kwargs):
            nonlocal inner_fn
            inner_fn = fn

        with patch("asyncio.to_thread", side_effect=capture_thread):
            await client.ensure_kb_collection()

        with (
            patch("app.clients.milvus.utility.has_collection", return_value=True) as mock_has,
            patch("app.clients.milvus.Collection") as mock_col,
        ):
            inner_fn()
            mock_has.assert_called_once()
            mock_col.assert_not_called()

    async def test_creates_collection_when_absent(self):
        client = MilvusClient()
        inner_fn = None

        async def capture_thread(fn, *args, **kwargs):
            nonlocal inner_fn
            inner_fn = fn

        with patch("asyncio.to_thread", side_effect=capture_thread):
            await client.ensure_kb_collection()

        mock_instance = MagicMock()
        with (
            patch("app.clients.milvus.utility.has_collection", return_value=False),
            patch("app.clients.milvus.Collection", return_value=mock_instance) as mock_col,
        ):
            inner_fn()
            mock_col.assert_called_once()
            mock_instance.create_index.assert_called_once()
            mock_instance.load.assert_called_once()


class TestMilvusClientInsert:
    async def test_insert_returns_primary_keys_as_strings(self):
        client = MilvusClient()
        mock_col = MagicMock()
        mock_col.insert.return_value = MagicMock(primary_keys=["pk1", "pk2"])

        async def run_fn(fn, *a, **kw):
            with patch("app.clients.milvus.Collection", return_value=mock_col):
                return fn()

        with patch("asyncio.to_thread", side_effect=run_fn):
            result = await client.insert([{"chunk_id": "pk1"}, {"chunk_id": "pk2"}])

        assert result == ["pk1", "pk2"]

    async def test_insert_routes_through_run_sync(self):
        client = MilvusClient()
        calls: list = []

        async def fake_thread(fn, *args, **kwargs):
            calls.append(fn)
            mock_col = MagicMock()
            mock_col.insert.return_value = MagicMock(primary_keys=[])
            with patch("app.clients.milvus.Collection", return_value=mock_col):
                return fn()

        with patch("asyncio.to_thread", side_effect=fake_thread):
            await client.insert([])

        assert len(calls) == 1


_VALID_DOC_UUID = "a0000000-0000-4000-8000-000000000001"


class TestMilvusClientDeleteByDocId:
    async def test_delete_routes_through_run_sync(self):
        client = MilvusClient()
        calls: list = []

        async def fake_thread(fn, *args, **kwargs):
            calls.append(fn)
            mock_col = MagicMock()
            mock_col.query.return_value = []
            with patch("app.clients.milvus.Collection", return_value=mock_col):
                fn()

        with patch("asyncio.to_thread", side_effect=fake_thread):
            await client.delete_by_doc_id(_VALID_DOC_UUID)

        assert len(calls) == 1

    async def test_delete_skips_col_delete_when_no_results(self):
        client = MilvusClient()
        mock_col = MagicMock()
        mock_col.query.return_value = []

        async def run_fn(fn, *a, **kw):
            with patch("app.clients.milvus.Collection", return_value=mock_col):
                fn()

        with patch("asyncio.to_thread", side_effect=run_fn):
            await client.delete_by_doc_id(_VALID_DOC_UUID)

        mock_col.delete.assert_not_called()

    async def test_delete_calls_col_delete_when_results_exist(self):
        client = MilvusClient()
        mock_col = MagicMock()
        mock_col.query.return_value = [{"chunk_id": "c1"}, {"chunk_id": "c2"}]

        async def run_fn(fn, *a, **kw):
            with patch("app.clients.milvus.Collection", return_value=mock_col):
                fn()

        with patch("asyncio.to_thread", side_effect=run_fn):
            await client.delete_by_doc_id(_VALID_DOC_UUID)

        mock_col.delete.assert_called_once()

    async def test_rejects_non_uuid_doc_id(self):
        client = MilvusClient()
        with pytest.raises(ValueError, match="UUID"):
            await client.delete_by_doc_id("doc-1")


class TestMilvusClientQueryIdsByDocId:
    async def test_returns_list_of_chunk_ids(self):
        client = MilvusClient()
        mock_col = MagicMock()
        mock_col.query.return_value = [{"chunk_id": "id1"}, {"chunk_id": "id2"}]

        async def run_fn(fn, *a, **kw):
            with patch("app.clients.milvus.Collection", return_value=mock_col):
                return fn()

        with patch("asyncio.to_thread", side_effect=run_fn):
            result = await client.query_ids_by_doc_id(_VALID_DOC_UUID)

        assert result == ["id1", "id2"]

    async def test_returns_empty_list_when_no_results(self):
        client = MilvusClient()
        mock_col = MagicMock()
        mock_col.query.return_value = []

        async def run_fn(fn, *a, **kw):
            with patch("app.clients.milvus.Collection", return_value=mock_col):
                return fn()

        with patch("asyncio.to_thread", side_effect=run_fn):
            result = await client.query_ids_by_doc_id(_VALID_DOC_UUID)

        assert result == []

    async def test_rejects_non_uuid_doc_id(self):
        client = MilvusClient()
        with pytest.raises(ValueError, match="UUID"):
            await client.query_ids_by_doc_id("doc-1")
