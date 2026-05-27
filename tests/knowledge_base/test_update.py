"""Unit tests for UpdateCoordinator."""
import dataclasses
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.exceptions import ServiceUnavailableError
from app.knowledge_base.update import IngestResult, UpdateCoordinator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_coordinator(
    *,
    parsed_doc=None,
    chunks=None,
    new_chunks=None,
    embedded=None,
) -> tuple[UpdateCoordinator, dict]:
    from app.knowledge_base.parser import ParsedDocument, Section

    if parsed_doc is None:
        parsed_doc = ParsedDocument(
            doc_id="doc-1",
            filename="test.txt",
            file_type="txt",
            sections=(Section(section_id="s0000", heading=None, content="text"),),
            content_hash="hash",
        )

    from app.models.chunk import ChunkSchema
    if chunks is None:
        chunks = [ChunkSchema(
            doc_id="doc-1", section_id="s0000", chunk_index=0,
            content_hash="h1", version="v1", content="text",
        )]
    if new_chunks is None:
        new_chunks = chunks
    if embedded is None:
        embedded = [c.model_copy(update={"embedding": [0.1]}) for c in new_chunks]

    parser = MagicMock()
    parser.parse = MagicMock(return_value=parsed_doc)

    cleaner = MagicMock()
    cleaner.clean_section = MagicMock(side_effect=lambda s: s)

    chunker = MagicMock()
    chunker.chunk = MagicMock(return_value=chunks)

    dedup = MagicMock()
    dedup.filter_new = AsyncMock(return_value=new_chunks)

    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=embedded)

    store = MagicMock()
    store.upsert_document = AsyncMock()
    store.update_document_status = AsyncMock()
    store.save_chunks_meta = AsyncMock()
    store.save_vectors = AsyncMock()

    mocks = {
        "parser": parser, "cleaner": cleaner, "chunker": chunker,
        "dedup": dedup, "embedder": embedder, "store": store,
    }
    coordinator = UpdateCoordinator(
        parser=parser, cleaner=cleaner, chunker=chunker,
        dedup=dedup, embedder=embedder, store=store,
    )
    return coordinator, mocks


def _make_redis(acquired: bool = True) -> MagicMock:
    r = MagicMock()
    r.acquire_lock = AsyncMock(return_value=(acquired, "token-abc"))
    r.release_lock = AsyncMock(return_value=True)
    return r


def _make_mq() -> MagicMock:
    m = MagicMock()
    m.publish = AsyncMock(return_value="msg-id")
    return m


def _make_pg() -> MagicMock:
    return MagicMock()


def _make_milvus() -> MagicMock:
    return MagicMock()


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------

class TestUpdateCoordinatorSuccess:
    async def test_returns_ingest_result(self):
        coord, _ = _make_coordinator()
        result = await coord.ingest(
            Path("f.txt"), "doc-1", _make_pg(), _make_redis(), _make_milvus(), _make_mq()
        )
        assert isinstance(result, IngestResult)

    async def test_result_doc_id_matches(self):
        coord, _ = _make_coordinator()
        result = await coord.ingest(
            Path("f.txt"), "doc-1", _make_pg(), _make_redis(), _make_milvus(), _make_mq()
        )
        assert result.doc_id == "doc-1"

    async def test_result_chunks_total_equals_all_chunks(self):
        from app.models.chunk import ChunkSchema
        chunks = [
            ChunkSchema(doc_id="d1", section_id="s0000", chunk_index=i,
                        content_hash=f"h{i}", version="v1", content="t")
            for i in range(4)
        ]
        coord, _ = _make_coordinator(chunks=chunks, new_chunks=chunks[:2],
                                     embedded=[c.model_copy(update={"embedding": [0.1]}) for c in chunks[:2]])
        result = await coord.ingest(
            Path("f.txt"), "d1", _make_pg(), _make_redis(), _make_milvus(), _make_mq()
        )
        assert result.chunks_total == 4
        assert result.chunks_new == 2

    async def test_upsert_document_called(self):
        coord, mocks = _make_coordinator()
        await coord.ingest(
            Path("f.txt"), "doc-1", _make_pg(), _make_redis(), _make_milvus(), _make_mq()
        )
        mocks["store"].upsert_document.assert_awaited_once()

    async def test_status_set_to_processing_then_indexed(self):
        coord, mocks = _make_coordinator()
        await coord.ingest(
            Path("f.txt"), "doc-1", _make_pg(), _make_redis(), _make_milvus(), _make_mq()
        )
        calls = [c.args[1] for c in mocks["store"].update_document_status.await_args_list]
        assert "processing" in calls
        assert "indexed" in calls

    async def test_mq_publish_called_with_event(self):
        coord, _ = _make_coordinator()
        mq = _make_mq()
        await coord.ingest(
            Path("f.txt"), "doc-1", _make_pg(), _make_redis(), _make_milvus(), mq
        )
        mq.publish.assert_awaited_once()
        payload = mq.publish.call_args[0][0]
        assert payload["event"] == "kb_updated"
        assert payload["doc_id"] == "doc-1"

    async def test_lock_released_on_success(self):
        coord, _ = _make_coordinator()
        redis = _make_redis()
        await coord.ingest(
            Path("f.txt"), "doc-1", _make_pg(), redis, _make_milvus(), _make_mq()
        )
        redis.release_lock.assert_awaited_once()

    async def test_parser_receives_doc_id(self):
        coord, mocks = _make_coordinator()
        await coord.ingest(
            Path("f.txt"), "doc-1", _make_pg(), _make_redis(), _make_milvus(), _make_mq()
        )
        mocks["parser"].parse.assert_called_once_with(Path("f.txt"), doc_id="doc-1")


# ---------------------------------------------------------------------------
# Lock failure
# ---------------------------------------------------------------------------

class TestUpdateCoordinatorLockFail:
    async def test_raises_service_unavailable_when_lock_not_acquired(self):
        coord, _ = _make_coordinator()
        redis = _make_redis(acquired=False)
        with pytest.raises(ServiceUnavailableError):
            await coord.ingest(
                Path("f.txt"), "doc-1", _make_pg(), redis, _make_milvus(), _make_mq()
            )

    async def test_does_not_call_store_when_lock_not_acquired(self):
        coord, mocks = _make_coordinator()
        redis = _make_redis(acquired=False)
        with pytest.raises(ServiceUnavailableError):
            await coord.ingest(
                Path("f.txt"), "doc-1", _make_pg(), redis, _make_milvus(), _make_mq()
            )
        mocks["store"].upsert_document.assert_not_awaited()


# ---------------------------------------------------------------------------
# Error / rollback
# ---------------------------------------------------------------------------

class TestUpdateCoordinatorRollback:
    async def test_status_set_to_failed_on_error(self):
        coord, mocks = _make_coordinator()
        mocks["embedder"].embed = AsyncMock(side_effect=RuntimeError("embed failed"))

        with pytest.raises(RuntimeError, match="embed failed"):
            await coord.ingest(
                Path("f.txt"), "doc-1", _make_pg(), _make_redis(), _make_milvus(), _make_mq()
            )

        calls = [c.args[1] for c in mocks["store"].update_document_status.await_args_list]
        assert "failed" in calls

    async def test_lock_released_on_error(self):
        coord, mocks = _make_coordinator()
        mocks["embedder"].embed = AsyncMock(side_effect=RuntimeError("embed failed"))
        redis = _make_redis()

        with pytest.raises(RuntimeError):
            await coord.ingest(
                Path("f.txt"), "doc-1", _make_pg(), redis, _make_milvus(), _make_mq()
            )

        redis.release_lock.assert_awaited_once()

    async def test_original_error_propagates(self):
        coord, mocks = _make_coordinator()
        mocks["dedup"].filter_new = AsyncMock(side_effect=ValueError("dedup error"))

        with pytest.raises(ValueError, match="dedup error"):
            await coord.ingest(
                Path("f.txt"), "doc-1", _make_pg(), _make_redis(), _make_milvus(), _make_mq()
            )
