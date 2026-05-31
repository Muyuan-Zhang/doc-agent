"""LangGraph StateGraph wiring for the M4 agent pipeline."""
import logging
from functools import partial

from langgraph.graph import END, StateGraph

from app.agent.nodes import (
    cache_lookup,
    cache_write,
    entity_extraction,
    generate,
    query_rewrite,
    rerank,
    retrieval,
    stream_cached,
)
from app.agent.state import AgentState

logger = logging.getLogger(__name__)


def build_graph(llm, retriever, redis, cache_svc, memory_svc=None):
    """Compile the agent graph with bound client dependencies.

    When ``memory_svc`` is provided, a ``retrieve_memory`` node is inserted
    between ``cache_lookup`` and ``query_rewrite`` on the cache-miss path,
    and the resulting ``MemoryContext`` is available to the ``generate`` node.
    """
    def _bind(node_fn):
        return partial(node_fn, llm=llm, retriever=retriever, redis=redis, cache_svc=cache_svc)

    def _bind_memory(node_fn):
        return partial(node_fn, memory_svc=memory_svc)

    g = StateGraph(AgentState)
    g.add_node("cache_lookup",      _bind(cache_lookup))
    g.add_node("stream_cached",     _bind(stream_cached))
    g.add_node("query_rewrite",     _bind(query_rewrite))
    g.add_node("retrieval",         _bind(retrieval))
    g.add_node("entity_extraction", _bind(entity_extraction))
    g.add_node("rerank",            _bind(rerank))
    g.add_node("generate",          _bind(generate))
    g.add_node("cache_write",       _bind(cache_write))

    if memory_svc is not None:
        from app.agent.nodes import retrieve_memory
        g.add_node("retrieve_memory", _bind_memory(retrieve_memory))

    def _after_lookup(state: AgentState) -> str:
        if state["cache_hit"]:
            decision = "stream_cached"
        elif memory_svc is not None and state.get("user_id"):
            decision = "retrieve_memory"
        else:
            decision = "query_rewrite"
        logger.info("graph=route job=%s from=cache_lookup to=%s cache_hit=%s", state.get("job_id"), decision, state["cache_hit"])
        return decision

    def _after_retrieval(state: AgentState) -> str:
        return "generate" if state["chunk_cache_hit"] else "entity_extraction"

    g.set_entry_point("cache_lookup")

    if memory_svc is not None:
        g.add_conditional_edges("cache_lookup", _after_lookup,
                                ["stream_cached", "retrieve_memory", "query_rewrite"])
        g.add_edge("retrieve_memory", "query_rewrite")
    else:
        g.add_conditional_edges("cache_lookup", _after_lookup,
                                ["stream_cached", "query_rewrite"])

    g.add_edge("stream_cached",     "cache_write")
    g.add_edge("query_rewrite",     "retrieval")
    g.add_conditional_edges("retrieval", _after_retrieval, ["generate", "entity_extraction"])
    g.add_edge("entity_extraction", "rerank")
    g.add_edge("rerank",            "generate")
    g.add_edge("generate",          "cache_write")
    g.add_edge("cache_write",       END)

    return g.compile()
