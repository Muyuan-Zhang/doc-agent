"""Unit tests for POST /retrieval/search endpoint."""
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.models.chunk import ChunkSchema


def _chunk(hash_: str = "h0") -> ChunkSchema:
    return ChunkSchema(
        doc_id="d1",
        section_id="s0",
        chunk_index=0,
        content_hash=hash_,
        version="v1",
        content="sample text",
    )


def _make_retriever(results: list[ChunkSchema] | None = None) -> MagicMock:
    r = MagicMock()
    r.retrieve = AsyncMock(return_value=results or [_chunk()])
    return r


async def _app_with_retriever(retriever=None):
    from app import create_app
    app = create_app()
    app.state.retriever = retriever or _make_retriever()
    return app


class TestSearchEndpointHappyPath:
    async def test_returns_200(self):
        app = await _app_with_retriever()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post("/retrieval/search", json={"query": "test query"})
        assert response.status_code == 200

    async def test_response_contains_chunks_key(self):
        app = await _app_with_retriever()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post("/retrieval/search", json={"query": "test"})
        assert "chunks" in response.json()

    async def test_response_contains_query(self):
        app = await _app_with_retriever()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post("/retrieval/search", json={"query": "my query"})
        assert response.json()["query"] == "my query"

    async def test_response_contains_top_k(self):
        app = await _app_with_retriever()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post("/retrieval/search", json={"query": "test", "top_k": 3})
        assert response.json()["top_k"] == 3

    async def test_default_top_k_is_5(self):
        app = await _app_with_retriever()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post("/retrieval/search", json={"query": "test"})
        assert response.json()["top_k"] == 5

    async def test_calls_retriever_with_query_and_top_k(self):
        retriever = _make_retriever()
        app = await _app_with_retriever(retriever)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            await c.post("/retrieval/search", json={"query": "hello", "top_k": 4})
        retriever.retrieve.assert_awaited_once_with("hello", 4)

    async def test_chunks_list_length_matches_retriever_results(self):
        chunks = [_chunk(f"h{i}") for i in range(3)]
        retriever = _make_retriever(results=chunks)
        app = await _app_with_retriever(retriever)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post("/retrieval/search", json={"query": "test"})
        assert len(response.json()["chunks"]) == 3

    async def test_chunk_fields_in_response(self):
        retriever = _make_retriever(results=[_chunk("hash-abc")])
        app = await _app_with_retriever(retriever)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post("/retrieval/search", json={"query": "test"})
        chunk = response.json()["chunks"][0]
        assert chunk["content_hash"] == "hash-abc"
        assert "content" in chunk
        assert "doc_id" in chunk


class TestSearchEndpointValidation:
    async def test_missing_query_returns_422(self):
        app = await _app_with_retriever()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post("/retrieval/search", json={"top_k": 3})
        assert response.status_code == 422

    async def test_empty_query_returns_422(self):
        app = await _app_with_retriever()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post("/retrieval/search", json={"query": ""})
        assert response.status_code == 422

    async def test_query_exceeding_max_length_returns_422(self):
        app = await _app_with_retriever()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post("/retrieval/search", json={"query": "a" * 1001})
        assert response.status_code == 422

    async def test_invalid_top_k_type_returns_422(self):
        app = await _app_with_retriever()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post("/retrieval/search", json={"query": "test", "top_k": "bad"})
        assert response.status_code == 422

    async def test_top_k_zero_returns_422(self):
        app = await _app_with_retriever()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post("/retrieval/search", json={"query": "test", "top_k": 0})
        assert response.status_code == 422

    async def test_top_k_exceeding_max_returns_422(self):
        app = await _app_with_retriever()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post("/retrieval/search", json={"query": "test", "top_k": 101})
        assert response.status_code == 422


class TestSearchEndpointErrors:
    async def test_retriever_not_ready_returns_503(self):
        from app import create_app
        app = create_app()
        # Do NOT set app.state.retriever
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post("/retrieval/search", json={"query": "test"})
        assert response.status_code == 503

    async def test_retriever_exception_returns_503(self):
        retriever = MagicMock()
        retriever.retrieve = AsyncMock(side_effect=RuntimeError("all strategies failed"))
        app = await _app_with_retriever(retriever)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post("/retrieval/search", json={"query": "test"})
        assert response.status_code == 503

    async def test_error_response_does_not_leak_exception_detail(self):
        retriever = MagicMock()
        retriever.retrieve = AsyncMock(side_effect=RuntimeError("secret internal state"))
        app = await _app_with_retriever(retriever)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post("/retrieval/search", json={"query": "test"})
        assert "secret internal state" not in response.text
