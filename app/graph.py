"""LangGraph state graph for the agent."""

from langgraph.graph import StateGraph, END
from app.state import AgentState


def _decide(state: AgentState) -> str:
    """Decide next node based on state."""
    if state["retrieval_count"] >= 2:
        return "generate"
    return "search_kb"


def _search_kb(state: AgentState) -> AgentState:
    from app.query import _search
    msg = state["messages"][-1]["content"] if state["messages"] else ""
    try:
        ctx = _search(msg)
        state["source"] = "kb"
        state["context"] = ctx
    except ValueError:
        state["context"] = ""
    state["retrieval_count"] += 1
    return state


def _generate(state: AgentState) -> AgentState:
    from app.chain import build_llm
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = PromptTemplate.from_template(
        "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    chain = prompt | build_llm() | StrOutputParser()
    msg = state["messages"][-1]["content"] if state["messages"] else ""
    state["response"] = chain.invoke({"context": state["context"], "question": msg})
    return state


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("search_kb", _search_kb)
    graph.add_node("generate", _generate)

    graph.set_entry_point("search_kb")
    graph.add_conditional_edges("search_kb", _decide, {"search_kb": "search_kb", "generate": "generate"})
    graph.add_edge("generate", END)

    return graph.compile()
