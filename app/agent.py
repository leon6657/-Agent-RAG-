"""Simple custom agent: always search KB first, fall back to chat if no relevant results."""

from datetime import date

from app.ingest import build_embeddings
from app.memory import SimpleMemory
from app.config import config
from app import store

_KB_PROMPT = """Current date: {current_date}

Based on the context below, answer the question.

Context:
{context}

Question: {question}

Answer concisely based on the context."""

_CHAT_PROMPT = """Current date: {current_date}

You are a helpful assistant. Answer the question naturally.

Conversation history:
{history}

Question: {question}

Answer:"""


memory = SimpleMemory(window_size=5)
_SEARCH_THRESHOLD = 0.35


def _call_llm(prompt_template: str, variables: dict) -> str:
    from app.chain import build_llm as _build
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | _build() | StrOutputParser()
    return chain.invoke(variables)


def _search_kb(query: str) -> str:
    from app.query import _search
    try:
        return _search(query)
    except ValueError as e:
        return ""


def _has_relevant(query: str) -> bool:
    """Check if KB has relevant content for this query."""
    emb = build_embeddings()
    vec = emb.embed_query(query)
    top_score = store.search_top_score(vec)
    return top_score >= _SEARCH_THRESHOLD


def chat(message: str) -> str:
    today = date.today().isoformat()
    history = memory.get_history()

    # Always search KB first, check relevance
    if _has_relevant(message):
        context = _search_kb(message)
        if context:
            response = _call_llm(_KB_PROMPT, {"context": context, "question": message, "current_date": today})
        else:
            response = _call_llm(_CHAT_PROMPT, {"history": history or "None", "question": message, "current_date": today})
    else:
        response = _call_llm(_CHAT_PROMPT, {"history": history or "None", "question": message, "current_date": today})

    memory.add_user(message)
    memory.add_assistant(response)
    return response
