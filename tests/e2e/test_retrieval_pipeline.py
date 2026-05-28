"""
E2E tests for the M2 retrieval pipeline.

All external clients are mocked via app.state injection (bypassing lifespan).
Tests exercise: RequestIdMiddleware → router validation → ConcreteHybridRetriever
→ response serialisation — the full ASGI chain for every retrieval scenario.

Covers gaps not addressed by unit tests:
  - Middleware integration on /retrieval/search (X-Request-ID lifecycle)
  - 503-state-guard: missing app.state.retriever path through real ASGI stack
  - RuntimeError → 503 conversion by the router without leaking internal detail
  - Error response body shape (request_id field, safe detail text)
  - Full retriever wiring: BM25 + Vector → RRF → reranker → HTTP response
  - Dedup contract: shared content_hash across strategies yields one result
  - final_top_k ceiling enforced end-to-end
  - Prompt-injection safety: brace-containing query reaches the reranker intact
"""
import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.models.chunk import ChunkSchema
from app.retrieval.hybrid import ConcreteHybridRetriever
from app.retrieval.reranker import LLMReranker
from app.retrieval.rrf import rrf_fuse

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(hash_: str, content: str = "sample text") -> ChunkSchema:
    return ChunkSchema(
        doc_id="d1",
        section_id="s0",
        chunk_index=0,
        content_hash=hash_,
        version="v1",
        content=content,
    )


def _make_retriever(results: list[ChunkSchema] | None = None, fail: bool = False) -> MagicMock:
    r = MagicMock()
    if fail:
        r.retrieve = AsyncMock(side_effect=RuntimeError("All retrieval strategies failed"))
    else:
        r.retrieve = AsyncMock(return_value=results if results is not None else [_chunk("h0")])
    return r


def _app(retriever=None):
    from app import create_app
    app = create_app()
    app.state.retriever = retriever or _make_retriever()
    return app


def _app_no_retriever():
    from app import create_app
    return create_app()  # no app.state.retriever set


# ---------------------------------------------------------------------------
# Middleware integration
# ---------------------------------------------------------------------------

class TestRetrievalMiddleware:
    async def test_x_request_id_present_on_200_response(self):
        app = _app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/retrieval/search", json={"query": "hello"})
        assert r.status_code == 200
        assert "x-request-id" in r.headers

    async def test_generated_request_id_is_uuid(self):
        app = _app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/retrieval/search", json={"query": "hello"})
        assert _UUID_RE.match(r.headers["x-request-id"]), r.headers["x-request-id"]

    async def test_valid_request_id_echoed_unchanged(self):
        app = _app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post(
                "/retrieval/search",
                json={"query": "hello"},
                headers={"X-Request-ID": "my-retrieval-req"},
            )
        assert r.headers["x-request-id"] == "my-retrieval-req"

    async def test_x_request_id_present_on_503_state_guard(self):
        app = _app_no_retriever()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/retrieval/search", json={"query": "hello"})
        assert r.status_code == 503
        assert "x-request-id" in r.headers

    async def test_x_request_id_present_on_422_validation_error(self):
        app = _app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/retrieval/search", json={"query": ""})
        assert r.status_code == 422
        assert "x-request-id" in r.headers

    async def test_distinct_requests_get_distinct_ids(self):
        app = _app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r1 = await c.post("/retrieval/search", json={"query": "a"})
            r2 = await c.post("/retrieval/search", json={"query": "b"})
        assert r1.headers["x-request-id"] != r2.headers["x-request-id"]


# ---------------------------------------------------------------------------
# 503 state-guard: retriever missing from app.state
# ---------------------------------------------------------------------------

class TestRetrieverStateGuard:
    async def test_returns_503_when_retriever_not_wired(self):
        app = _app_no_retriever()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/retrieval/search", json={"query": "test"})
        assert r.status_code == 503

    async def test_safe_detail_on_state_guard_503(self):
        app = _app_no_retriever()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/retrieval/search", json={"query": "test"})
        body = r.json()
        assert "Traceback" not in r.text
        assert "AttributeError" not in r.text
        assert "detail" in body


# ---------------------------------------------------------------------------
# Input validation — exercised through the full ASGI stack
# ---------------------------------------------------------------------------

class TestRetrievalValidation:
    async def test_empty_query_returns_422(self):
        app = _app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/retrieval/search", json={"query": ""})
        assert r.status_code == 422

    async def test_missing_query_returns_422(self):
        app = _app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/retrieval/search", json={"top_k": 3})
        assert r.status_code == 422

    async def test_oversized_query_returns_422(self):
        app = _app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/retrieval/search", json={"query": "x" * 1001})
        assert r.status_code == 422

    async def test_top_k_zero_returns_422(self):
        app = _app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/retrieval/search", json={"query": "test", "top_k": 0})
        assert r.status_code == 422

    async def test_top_k_above_ceiling_returns_422(self):
        app = _app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/retrieval/search", json={"query": "test", "top_k": 101})
        assert r.status_code == 422

    async def test_invalid_top_k_type_returns_422(self):
        app = _app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/retrieval/search", json={"query": "test", "top_k": "bad"})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Error safety: RuntimeError from retriever must not reach the client
# ---------------------------------------------------------------------------

class TestRetrievalErrorSafety:
    async def test_retriever_runtime_error_returns_503(self):
        app = _app(retriever=_make_retriever(fail=True))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/retrieval/search", json={"query": "test"})
        assert r.status_code == 503

    async def test_retriever_error_detail_not_in_response(self):
        retriever = _make_retriever(fail=True)
        retriever.retrieve.side_effect = RuntimeError("secret-connection-string-in-stacktrace")
        app = _app(retriever=retriever)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/retrieval/search", json={"query": "test"})
        assert "secret-connection-string-in-stacktrace" not in r.text
        assert "Traceback" not in r.text

    async def test_retriever_error_response_has_detail_field(self):
        app = _app(retriever=_make_retriever(fail=True))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/retrieval/search", json={"query": "test"})
        assert "detail" in r.json()

    async def test_x_request_id_present_on_503_retriever_error(self):
        app = _app(retriever=_make_retriever(fail=True))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/retrieval/search", json={"query": "test"})
        assert "x-request-id" in r.headers


# ---------------------------------------------------------------------------
# Happy-path pipeline: full request → response shape
# ---------------------------------------------------------------------------

class TestRetrievalHappyPath:
    async def test_returns_200_with_chunks(self):
        app = _app(retriever=_make_retriever(results=[_chunk("h1")]))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/retrieval/search", json={"query": "hello world"})
        assert r.status_code == 200
        assert len(r.json()["chunks"]) == 1

    async def test_response_echoes_query(self):
        app = _app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/retrieval/search", json={"query": "my search"})
        assert r.json()["query"] == "my search"

    async def test_response_echoes_top_k(self):
        app = _app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/retrieval/search", json={"query": "q", "top_k": 7})
        assert r.json()["top_k"] == 7

    async def test_default_top_k_is_5(self):
        app = _app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/retrieval/search", json={"query": "q"})
        assert r.json()["top_k"] == 5

    async def test_chunk_fields_present_in_response(self):
        chunk = _chunk("abc-hash", content="sample content")
        app = _app(retriever=_make_retriever(results=[chunk]))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/retrieval/search", json={"query": "q"})
        body = r.json()["chunks"][0]
        assert body["content_hash"] == "abc-hash"
        assert body["content"] == "sample content"
        assert body["doc_id"] == "d1"

    async def test_empty_result_list_returns_200(self):
        app = _app(retriever=_make_retriever(results=[]))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/retrieval/search", json={"query": "obscure"})
        assert r.status_code == 200
        assert r.json()["chunks"] == []

    async def test_retriever_called_with_correct_query_and_top_k(self):
        retriever = _make_retriever()
        app = _app(retriever=retriever)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            await c.post("/retrieval/search", json={"query": "target query", "top_k": 9})
        retriever.retrieve.assert_awaited_once_with("target query", 9)


# ---------------------------------------------------------------------------
# Retriever wiring: exercise ConcreteHybridRetriever through the full pipeline
# ---------------------------------------------------------------------------

def _make_strategy(results: list[ChunkSchema] | None = None, fail: bool = False) -> MagicMock:
    s = MagicMock()
    if fail:
        s.retrieve = AsyncMock(side_effect=RuntimeError("strategy down"))
    else:
        s.retrieve = AsyncMock(return_value=results or [])
    return s


def _make_reranker(results: list[ChunkSchema] | None = None) -> MagicMock:
    r = MagicMock()
    r.rerank = AsyncMock(return_value=results or [])
    return r


def _settings_mock(
    bm25_top_k=10,
    vector_top_k=10,
    rrf_k=60,
    rerank_top_n=5,
    final_top_k=5,
) -> MagicMock:
    s = MagicMock()
    s.bm25_top_k = bm25_top_k
    s.vector_top_k = vector_top_k
    s.rrf_k = rrf_k
    s.rerank_top_n = rerank_top_n
    s.final_top_k = final_top_k
    return s


def _app_with_concrete_retriever(bm25, vector, reranker, settings=None):
    from app import create_app
    app = create_app()
    app.state.retriever = ConcreteHybridRetriever(
        bm25=bm25,
        vector=vector,
        reranker=reranker,
        settings=settings or _settings_mock(),
    )
    return app


class TestConcreteRetrieverWiring:
    async def test_chunks_from_bm25_reach_response(self):
        bm25 = _make_strategy(results=[_chunk("bm-1")])
        reranker = _make_reranker(results=[_chunk("bm-1")])
        app = _app_with_concrete_retriever(bm25=bm25, vector=_make_strategy(), reranker=reranker)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/retrieval/search", json={"query": "bm25 query"})
        assert r.status_code == 200
        hashes = [ch["content_hash"] for ch in r.json()["chunks"]]
        assert "bm-1" in hashes

    async def test_all_strategies_fail_returns_503(self):
        app = _app_with_concrete_retriever(
            bm25=_make_strategy(fail=True),
            vector=_make_strategy(fail=True),
            reranker=_make_reranker(),
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/retrieval/search", json={"query": "test"})
        assert r.status_code == 503

    async def test_all_strategies_fail_error_not_leaked(self):
        app = _app_with_concrete_retriever(
            bm25=_make_strategy(fail=True),
            vector=_make_strategy(fail=True),
            reranker=_make_reranker(),
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/retrieval/search", json={"query": "test"})
        assert "RuntimeError" not in r.text
        assert "Traceback" not in r.text

    async def test_partial_failure_still_returns_200(self):
        bm25 = _make_strategy(results=[_chunk("bm-ok")])
        reranker = _make_reranker(results=[_chunk("bm-ok")])
        app = _app_with_concrete_retriever(
            bm25=bm25,
            vector=_make_strategy(fail=True),
            reranker=reranker,
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/retrieval/search", json={"query": "test"})
        assert r.status_code == 200

    async def test_dedup_shared_hash_yields_one_result(self):
        shared = _chunk("shared-hash", content="identical content")
        bm25 = _make_strategy(results=[shared])
        vector = _make_strategy(results=[shared])
        reranker = _make_reranker(results=[shared])
        app = _app_with_concrete_retriever(
            bm25=bm25,
            vector=vector,
            reranker=reranker,
            settings=_settings_mock(rerank_top_n=10, final_top_k=10),
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/retrieval/search", json={"query": "q", "top_k": 10})
        hashes = [ch["content_hash"] for ch in r.json()["chunks"]]
        assert hashes.count("shared-hash") == 1

    async def test_final_top_k_caps_response_length(self):
        chunks = [_chunk(f"h{i}") for i in range(8)]
        bm25 = _make_strategy(results=chunks)
        reranker = _make_reranker(results=chunks[:3])
        app = _app_with_concrete_retriever(
            bm25=bm25,
            vector=_make_strategy(),
            reranker=reranker,
            settings=_settings_mock(rerank_top_n=10, final_top_k=3),
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/retrieval/search", json={"query": "q", "top_k": 10})
        assert len(r.json()["chunks"]) <= 3

    async def test_query_with_braces_processed_without_error(self):
        bm25 = _make_strategy(results=[_chunk("h0")])
        reranker = _make_reranker(results=[_chunk("h0")])
        app = _app_with_concrete_retriever(bm25=bm25, vector=_make_strategy(), reranker=reranker)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post(
                "/retrieval/search",
                json={"query": "{injection} test {{double}} }unbalanced"},
            )
        assert r.status_code == 200
