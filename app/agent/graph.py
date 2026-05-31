"""LangGraph StateGraph wiring for the M4 agent pipeline."""
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


def build_graph(llm, retriever, redis, cache_svc):
    """Compile the agent graph with bound client dependencies."""
    def _bind(node_fn):
        return partial(node_fn, llm=llm, retriever=retriever, redis=redis, cache_svc=cache_svc)

    g = StateGraph(AgentState)
    g.add_node("cache_lookup",      _bind(cache_lookup))
    g.add_node("stream_cached",     _bind(stream_cached))
    g.add_node("query_rewrite",     _bind(query_rewrite))
    g.add_node("retrieval",         _bind(retrieval))
    g.add_node("entity_extraction", _bind(entity_extraction))
    g.add_node("rerank",            _bind(rerank))
    g.add_node("generate",          _bind(generate))
    g.add_node("cache_write",       _bind(cache_write))

    def _after_lookup(state: AgentState) -> str:
        return "stream_cached" if state["cache_hit"] else "query_rewrite"

    def _after_retrieval(state: AgentState) -> str:
        return "generate" if state["chunk_cache_hit"] else "entity_extraction"

    g.set_entry_point("cache_lookup")
    g.add_conditional_edges("cache_lookup", _after_lookup, ["stream_cached", "query_rewrite"])
    g.add_edge("stream_cached",     "cache_write")
    g.add_edge("query_rewrite",     "retrieval")
    g.add_conditional_edges("retrieval", _after_retrieval, ["generate", "entity_extraction"])
    g.add_edge("entity_extraction", "rerank")
    g.add_edge("rerank",            "generate")
    g.add_edge("generate",          "cache_write")
    g.add_edge("cache_write",       END)

    return g.compile()
