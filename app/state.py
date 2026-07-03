"""State definition for LangGraph agent."""

from typing import List, TypedDict


class AgentState(TypedDict):
    messages: List[dict]
    source: str
    retrieval_count: int
    context: str
    response: str
