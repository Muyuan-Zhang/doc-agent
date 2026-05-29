"""
Tests for app/cache/schemas.py.

Covers:
- CacheStatus enum values and str inheritance
- CacheEntry frozen invariant, field defaults, JSON round-trip
- ReviewEntry field requirements
"""
import pytest
from datetime import datetime, timezone

from pydantic import ValidationError as PydanticValidationError

from app.cache.schemas import CacheEntry, CacheStatus, ReviewEntry
from app.models.chunk import ChunkSchema


def _make_chunk(**overrides) -> ChunkSchema:
    defaults = dict(
        doc_id="doc-1",
        section_id="sec-1",
        chunk_index=0,
        content_hash="abc123",
        version="v1",
        content="test content",
    )
    return ChunkSchema(**(defaults | overrides))


def _make_entry(**overrides) -> CacheEntry:
    defaults = dict(
        query_hash="deadbeefcafe0000",
        original_query="What is the deadline?",
        normalized_query="what is the deadline",
        chunks=[_make_chunk()],
        created_at=datetime.now(tz=timezone.utc),
    )
    return CacheEntry(**(defaults | overrides))


# ---------------------------------------------------------------------------
# CacheStatus
# ---------------------------------------------------------------------------

class TestCacheStatus:
    def test_pending_review_value(self):
        assert CacheStatus.PENDING_REVIEW == "pending_review"

    def test_approved_value(self):
        assert CacheStatus.APPROVED == "approved"

    def test_rejected_value(self):
        assert CacheStatus.REJECTED == "rejected"

    def test_is_str_subclass(self):
        assert isinstance(CacheStatus.APPROVED, str)

    def test_all_three_members_exist(self):
        assert len(CacheStatus) == 3


# ---------------------------------------------------------------------------
# CacheEntry
# ---------------------------------------------------------------------------

class TestCacheEntry:
    def test_default_status_is_pending_review(self):
        entry = _make_entry()
        assert entry.status == CacheStatus.PENDING_REVIEW

    def test_default_approval_count_is_zero(self):
        entry = _make_entry()
        assert entry.approval_count == 0

    def test_default_approved_by_is_empty_list(self):
        entry = _make_entry()
        assert entry.approved_by == []

    def test_frozen_prevents_field_reassignment(self):
        entry = _make_entry()
        with pytest.raises((TypeError, PydanticValidationError)):
            entry.status = CacheStatus.APPROVED  # type: ignore[misc]

    def test_accepts_approved_status_on_construction(self):
        entry = _make_entry(status=CacheStatus.APPROVED)
        assert entry.status == CacheStatus.APPROVED

    def test_accepts_rejected_status_on_construction(self):
        entry = _make_entry(status=CacheStatus.REJECTED)
        assert entry.status == CacheStatus.REJECTED

    def test_stores_chunks_list(self):
        chunk = _make_chunk(content="hello")
        entry = _make_entry(chunks=[chunk])
        assert len(entry.chunks) == 1
        assert entry.chunks[0].content == "hello"

    def test_stores_approved_by_list(self):
        entry = _make_entry(approved_by=["alice", "bob"])
        assert entry.approved_by == ["alice", "bob"]

    def test_json_round_trip_preserves_all_fields(self):
        entry = _make_entry(
            status=CacheStatus.APPROVED,
            approval_count=2,
            approved_by=["alice"],
        )
        raw = entry.model_dump_json()
        restored = CacheEntry.model_validate_json(raw)
        assert restored.query_hash == entry.query_hash
        assert restored.original_query == entry.original_query
        assert restored.normalized_query == entry.normalized_query
        assert restored.status == entry.status
        assert restored.approval_count == entry.approval_count
        assert restored.approved_by == entry.approved_by
        assert len(restored.chunks) == len(entry.chunks)

    def test_json_round_trip_preserves_chunk_content(self):
        chunk = _make_chunk(content="important text")
        entry = _make_entry(chunks=[chunk])
        raw = entry.model_dump_json()
        restored = CacheEntry.model_validate_json(raw)
        assert restored.chunks[0].content == "important text"

    def test_model_dump_returns_dict(self):
        entry = _make_entry()
        data = entry.model_dump()
        assert isinstance(data, dict)
        assert data["query_hash"] == entry.query_hash


# ---------------------------------------------------------------------------
# ReviewEntry
# ---------------------------------------------------------------------------

class TestReviewEntry:
    def test_stores_all_required_fields(self):
        entry = ReviewEntry(
            query_hash="abc123",
            original_query="test?",
            normalized_query="test",
            chunk_count=2,
            status=CacheStatus.PENDING_REVIEW,
            approval_count=0,
            created_at=datetime.now(tz=timezone.utc),
        )
        assert entry.query_hash == "abc123"
        assert entry.chunk_count == 2
        assert entry.status == CacheStatus.PENDING_REVIEW

    def test_frozen_prevents_field_reassignment(self):
        entry = ReviewEntry(
            query_hash="abc123",
            original_query="q",
            normalized_query="q",
            chunk_count=1,
            status=CacheStatus.PENDING_REVIEW,
            approval_count=0,
            created_at=datetime.now(tz=timezone.utc),
        )
        with pytest.raises((TypeError, PydanticValidationError)):
            entry.chunk_count = 5  # type: ignore[misc]
