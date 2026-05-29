from app.models.chunk import ChunkSchema


def rrf_fuse(
    ranked_lists: list[list[ChunkSchema]],
    k: int = 60,
) -> list[ChunkSchema]:
    """Reciprocal Rank Fusion over multiple ranked result lists.

    Deduplicates by content_hash, preferring instances that carry an embedding
    (vector hits) over instances without one (BM25 hits).
    Scores: Σ 1/(k + rank_i). Returns descending-score order.
    """
    if k <= 0:
        raise ValueError(f"rrf_fuse: k must be positive, got {k}")

    scores: dict[str, float] = {}
    best: dict[str, ChunkSchema] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            key = chunk.content_hash
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in best or (
                chunk.embedding is not None and best[key].embedding is None
            ):
                best[key] = chunk

    sorted_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [best[key] for key in sorted_keys]
