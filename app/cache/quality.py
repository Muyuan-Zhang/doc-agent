"""Embedding-based quality scoring for RAG cache auto-approval.

Computes a 0.0-1.0 quality score by measuring cosine similarity between
a query embedding and retrieved chunk embeddings.
"""
import logging
import math

from app.models.chunk import ChunkSchema

logger = logging.getLogger(__name__)

_TOP_N = 3


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return the cosine similarity between two vectors.

    Raises ValueError if vectors differ in length or are empty.
    Returns 0.0 when either vector has zero magnitude.
    """
    if not a or not b:
        raise ValueError("Vectors must not be empty")
    if len(a) != len(b):
        raise ValueError(
            f"Vector dimension mismatch: {len(a)} vs {len(b)}"
        )

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


def compute_quality(
    query_embedding: list[float],
    chunks: list[ChunkSchema],
) -> float:
    """Compute a 0.0-1.0 quality score based on embedding similarity.

    Computes cosine similarity between the query embedding and each chunk's
    embedding, then averages the top-3 scores. Returns 0.0 when no chunks
    have embeddings.

    Higher scores indicate the retrieved chunks are semantically close to
    the query — a proxy for retrieval quality.
    """
    similarities: list[float] = []
    for chunk in chunks:
        if chunk.embedding is not None:
            try:
                sim = cosine_similarity(query_embedding, chunk.embedding)
                similarities.append(sim)
            except ValueError as exc:
                logger.warning(
                    "quality=skip_chunk doc_id=%s chunk_index=%d error=%s",
                    chunk.doc_id, chunk.chunk_index, exc,
                )

    if not similarities:
        logger.info("quality=no_embeddings chunk_count=%d", len(chunks))
        return 0.0

    similarities.sort(reverse=True)
    top_k = similarities[:_TOP_N]
    score = sum(top_k) / len(top_k)
    logger.debug("quality=scored mean=%.4f top_k=%s", score, top_k)
    return score
