"""LangGraph StateGraph wiring for the M4 agent pipeline."""
from functools import partial

from langgraph.graph import END, StateGraph

from app.agent.nodes import (
    cache_write,
    entity_extraction,
    generate,
    query_rewrite,
    rerank,
    retrieval,
)
from app.agent.state import AgentState


def build_graph(llm, retriever, redis):
    """Compile the agent graph with bound client dependencies."""
    def _bind(node_fn):
        return partial(node_fn, llm=llm, retriever=retriever, redis=redis)

    g = StateGraph(AgentState)
    g.add_node("query_rewrite",     _bind(query_rewrite))
    g.add_node("retrieval",         _bind(retrieval))
    g.add_node("entity_extraction", _bind(entity_extraction))
    g.add_node("rerank",            _bind(rerank))
    g.add_node("generate",          _bind(generate))
    g.add_node("cache_write",       _bind(cache_write))

    g.set_entry_point("query_rewrite")
    g.add_edge("query_rewrite",     "retrieval")
    g.add_edge("retrieval",         "entity_extraction")
    g.add_edge("entity_extraction", "rerank")
    g.add_edge("rerank",            "generate")
    g.add_edge("generate",          "cache_write")
    g.add_edge("cache_write",       END)

    return g.compile()
