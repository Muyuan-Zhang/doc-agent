"""Unit tests for rrf_fuse — pure Reciprocal Rank Fusion function."""
from app.models.chunk import ChunkSchema
from app.retrieval.rrf import rrf_fuse


def _chunk(hash_: str, content: str = "text") -> ChunkSchema:
    return ChunkSchema(
        doc_id="d1",
        section_id="s0",
        chunk_index=0,
        content_hash=hash_,
        version="v1",
        content=content,
    )


class TestRrfFuse:
    def test_empty_input_returns_empty(self):
        assert rrf_fuse([]) == []

    def test_single_list_preserves_order(self):
        chunks = [_chunk("a"), _chunk("b"), _chunk("c")]
        result = rrf_fuse([chunks])
        assert [c.content_hash for c in result] == ["a", "b", "c"]

    def test_deduplicates_by_content_hash(self):
        list1 = [_chunk("a"), _chunk("b")]
        list2 = [_chunk("a"), _chunk("c")]
        result = rrf_fuse([list1, list2])
        hashes = [c.content_hash for c in result]
        assert hashes.count("a") == 1

    def test_chunk_in_both_lists_ranks_first(self):
        shared = _chunk("shared")
        unique1 = _chunk("unique1")
        unique2 = _chunk("unique2")
        result = rrf_fuse([[shared, unique1], [shared, unique2]])
        assert result[0].content_hash == "shared"

    def test_higher_rank_yields_higher_score(self):
        chunks = [_chunk("first"), _chunk("second"), _chunk("third")]
        result = rrf_fuse([chunks])
        assert result[0].content_hash == "first"
        assert result[-1].content_hash == "third"

    def test_all_chunks_appear_in_output(self):
        list1 = [_chunk("a"), _chunk("b")]
        list2 = [_chunk("c"), _chunk("d")]
        result = rrf_fuse([list1, list2])
        hashes = {c.content_hash for c in result}
        assert hashes == {"a", "b", "c", "d"}

    def test_empty_sublists_are_handled(self):
        chunks = [_chunk("a")]
        result = rrf_fuse([chunks, []])
        assert len(result) == 1

    def test_custom_k_parameter_accepted(self):
        chunks = [_chunk("a")]
        assert len(rrf_fuse([chunks], k=1)) == 1
        assert len(rrf_fuse([chunks], k=100)) == 1

    def test_score_formula_k60_rank1(self):
        # With k=60, rank=1: score = 1/(60+1) ≈ 0.01639
        # With two lists both rank 1: score ≈ 0.03278 vs 0.01639 for rank-2-only
        top = _chunk("top")
        bottom = _chunk("bottom")
        result = rrf_fuse([[top], [top, bottom]], k=60)
        assert result[0].content_hash == "top"
