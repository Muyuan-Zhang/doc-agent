"""
Tests for app/cache/quality.py — embedding similarity quality scorer.

Covers:
- Identical embeddings → 1.0
- Orthogonal embeddings → 0.0
- Top-3 averaging
- No embeddings in chunks → 0.0
- Mixed None embeddings
- Fewer than 3 chunks
- Realistic partial-match embeddings
"""
import math
import pytest

from app.models.chunk import ChunkSchema


# Import the module under test — will fail (RED) until quality.py is created
try:
    from app.cache.quality import compute_quality, cosine_similarity
except ImportError:
    compute_quality = None  # type: ignore[assignment]
    cosine_similarity = None  # type: ignore[assignment]


def _make_chunk(content: str = "test", embedding: list[float] | None = None) -> ChunkSchema:
    return ChunkSchema(
        doc_id="d1", section_id="s1", chunk_index=0,
        content_hash="abc", version="v1", content=content,
        embedding=embedding,
    )


# ---------------------------------------------------------------------------
# cosine_similarity unit tests
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors_return_one(self):
        v = [1.0, 2.0, 3.0]
        assert math.isclose(cosine_similarity(v, v), 1.0, rel_tol=1e-9)

    def test_orthogonal_vectors_return_zero(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert math.isclose(cosine_similarity(a, b), 0.0, abs_tol=1e-9)

    def test_opposite_vectors_return_negative_one(self):
        a = [1.0, 2.0]
        b = [-1.0, -2.0]
        assert math.isclose(cosine_similarity(a, b), -1.0, rel_tol=1e-9)

    def test_partial_match(self):
        a = [1.0, 0.0]
        b = [0.7071, 0.7071]  # 45 degrees
        result = cosine_similarity(a, b)
        assert 0.70 < result < 0.72

    def test_zero_vector_returns_zero(self):
        a = [0.0, 0.0]
        b = [1.0, 2.0]
        assert cosine_similarity(a, b) == 0.0

    def test_both_zero_vectors_return_zero(self):
        assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0

    def test_different_dimensions_raises_value_error(self):
        with pytest.raises(ValueError, match="dimension"):
            cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_empty_vectors_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            cosine_similarity([], [])


# ---------------------------------------------------------------------------
# compute_quality
# ---------------------------------------------------------------------------

class TestComputeQuality:
    def test_returns_one_for_perfect_match(self):
        emb = [0.1] * 10
        q_emb = [0.1] * 10
        chunks = [_make_chunk(embedding=emb)]
        result = compute_quality(q_emb, chunks)
        assert math.isclose(result, 1.0, rel_tol=1e-9)

    def test_uses_top_three_similarities(self):
        q_emb = [1.0, 0.0, 0.0]
        chunks = [
            _make_chunk(embedding=[1.0, 0.0, 0.0]),   # sim 1.0
            _make_chunk(embedding=[0.9, 0.1, 0.0]),   # sim ~0.994
            _make_chunk(embedding=[0.8, 0.2, 0.0]),   # sim ~0.97
            _make_chunk(embedding=[0.0, 1.0, 0.0]),   # sim 0.0 (excluded by top-3)
            _make_chunk(embedding=[0.0, 0.0, 1.0]),   # sim 0.0 (excluded by top-3)
        ]
        result = compute_quality(q_emb, chunks)
        assert 0.95 < result <= 1.0

    def test_returns_zero_when_no_chunks_have_embeddings(self):
        chunks = [
            _make_chunk(embedding=None),
            _make_chunk(embedding=None),
        ]
        assert compute_quality([1.0, 0.0], chunks) == 0.0

    def test_handles_fewer_than_three_chunks(self):
        q_emb = [1.0, 0.0]
        chunks = [
            _make_chunk(embedding=[1.0, 0.0]),   # sim 1.0
            _make_chunk(embedding=[0.5, 0.866]),  # sim 0.5
        ]
        result = compute_quality(q_emb, chunks)
        expected = (1.0 + 0.5) / 2
        assert math.isclose(result, expected, abs_tol=1e-4)

    def test_mixed_some_embeddings_none(self):
        q_emb = [1.0, 0.0]
        chunks = [
            _make_chunk(embedding=None),
            _make_chunk(embedding=[1.0, 0.0]),   # sim 1.0
            _make_chunk(embedding=None),
        ]
        result = compute_quality(q_emb, chunks)
        assert math.isclose(result, 1.0, rel_tol=1e-9)

    def test_single_chunk_with_embedding(self):
        q_emb = [1.0, 2.0, 3.0]
        chunks = [_make_chunk(embedding=[1.0, 2.0, 3.0])]
        assert math.isclose(compute_quality(q_emb, chunks), 1.0, rel_tol=1e-9)

    def test_empty_chunks_list_returns_zero(self):
        assert compute_quality([1.0, 0.0], []) == 0.0
