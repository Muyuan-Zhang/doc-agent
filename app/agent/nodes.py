"""LangGraph node functions for the M4 agent pipeline.

Each node receives the full AgentState and keyword-only client dependencies,
and returns a partial state dict containing only the fields it updates.

Dependency injection is done via functools.partial in graph.py so LangGraph
can call each node with only the state argument.
"""
import hashlib
import logging

from app.agent.state import AgentState
from app.core.config import settings

logger = logging.getLogger(__name__)


async def query_rewrite(state: AgentState, *, llm, retriever, redis) -> dict:
    prompt = (
        "Rewrite the following search query to improve document retrieval accuracy. "
        "Return only the rewritten query, nothing else.\n\n"
        f"Original query: {state['query']}\n\nRewritten query:"
    )
    rewritten = await llm.complete(prompt)
    return {"rewritten_query": rewritten.strip() or state["query"]}


async def retrieval(state: AgentState, *, llm, retriever, redis) -> dict:
    chunks = await retriever.retrieve(state["rewritten_query"], top_k=state["top_k"])
    return {"chunks": list(chunks)}


async def entity_extraction(state: AgentState, *, llm, retriever, redis) -> dict:
    # Pass-through placeholder for future Graph RAG entity extraction.
    return {"reranked_chunks": list(state["chunks"])}


async def rerank(state: AgentState, *, llm, retriever, redis) -> dict:
    chunks = state["reranked_chunks"]
    if not chunks:
        return {"reranked_chunks": []}

    numbered = "\n".join(f"{i + 1}. {c.content}" for i, c in enumerate(chunks))
    prompt = (
        f"Query: {state['query']}\n\n"
        f"Rank these passages by relevance (most relevant first). "
        f"Return only numbers comma-separated, e.g.: 2, 1, 3\n\n"
        f"Passages:\n{numbered}\n\nRanking:"
    )
    try:
        raw = await llm.complete(prompt)
        indices = [int(n.strip()) - 1 for n in raw.split(",") if n.strip().isdigit()]
        valid = [i for i in indices if 0 <= i < len(chunks)]
        ranked = [chunks[i] for i in valid]
        ranked_set = set(valid)
        remainder = [chunks[i] for i in range(len(chunks)) if i not in ranked_set]
        return {"reranked_chunks": ranked + remainder}
    except Exception as exc:
        logger.warning("Rerank LLM call failed, keeping original order: %s", exc)
        return {"reranked_chunks": chunks}


async def generate(state: AgentState, *, llm, retriever, redis) -> dict:
    context = "\n\n".join(c.content for c in state["reranked_chunks"])
    prompt = (
        "You are a helpful assistant. Answer the question using the provided context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {state['query']}\n\nAnswer:"
    )
    answer = await llm.complete(prompt)
    return {"answer": answer}


async def cache_write(state: AgentState, *, llm, retriever, redis) -> dict:
    query_hash = hashlib.sha256(state["query"].encode()).hexdigest()[:16]
    key = redis.cache_key("rag", state["session_id"], query_hash)
    await redis.client.setex(key, settings.agent_job_ttl_seconds, state["answer"])
    return {}
