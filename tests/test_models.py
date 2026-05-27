"""Unit tests for Pydantic models and Protocol — P1."""
import pytest
from pydantic import ValidationError as PydanticValidationError

from app.models.chunk import ChunkSchema
from app.models.retrieval import HybridRetriever, RetrievalStrategy

_CHUNK = dict(
    doc_id="doc1",
    section_id="sec1",
    chunk_index=0,
    content_hash="abc123",
    version="v1",
    content="some text",
)


class TestChunkSchema:
    def test_valid_chunk_creates_successfully(self):
        chunk = ChunkSchema(**_CHUNK)
        assert chunk.doc_id == "doc1"
        assert chunk.content == "some text"

    def test_optional_fields_default_to_none(self):
        chunk = ChunkSchema(**_CHUNK)
        assert chunk.parent_chunk_id is None
        assert chunk.embedding is None

    def test_embedding_field_accepted(self):
        chunk = ChunkSchema(**_CHUNK, embedding=[0.1, 0.2, 0.3])
        assert chunk.embedding == [0.1, 0.2, 0.3]

    def test_frozen_raises_on_mutation(self):
        chunk = ChunkSchema(**_CHUNK)
        with pytest.raises(Exception):
            chunk.content = "mutated"  # type: ignore[misc]

    def test_frozen_model_is_hashable(self):
        chunk = ChunkSchema(**_CHUNK)
        assert isinstance(hash(chunk), int)

    def test_missing_required_field_raises(self):
        with pytest.raises(PydanticValidationError):
            ChunkSchema(doc_id="d1", section_id="s1", chunk_index=0)


class TestRetrievalStrategyProtocol:
    def test_class_with_retrieve_method_satisfies_protocol(self):
        class ConcreteStrategy:
            async def retrieve(self, query: str, top_k: int, **kwargs) -> list:
                return []

        assert isinstance(ConcreteStrategy(), RetrievalStrategy)

    def test_class_without_retrieve_does_not_satisfy_protocol(self):
        class BadStrategy:
            pass

        assert not isinstance(BadStrategy(), RetrievalStrategy)


class TestHybridRetriever:
    async def test_retrieve_raises_not_implemented(self):
        retriever = HybridRetriever(strategies=[])
        with pytest.raises(NotImplementedError):
            await retriever.retrieve("query", top_k=5)

    def test_strategies_stored(self):
        class S:
            async def retrieve(self, query: str, top_k: int, **kwargs) -> list:
                return []

        strategies = [S(), S()]
        retriever = HybridRetriever(strategies=strategies)
        assert retriever._strategies is strategies
