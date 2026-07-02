"""Simple custom agent with tool selection and memory."""

from datetime import date

from app.ingest import build_embeddings  # noqa: F401 - import order

from app.memory import SimpleMemory
from app.config import config

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

_SELECT_PROMPT = """You have a knowledge base with notes about Python, ML, data structures ONLY.
Decide if you MUST search the knowledge base to answer.

Search ONLY if: the question asks about Python, ML, data structures, or specific technical concepts that are IN your notes.
Do NOT search for: dates, weather, news, mathematics, general knowledge, greetings, creative tasks, code generation, or anything NOT in your notes.

Current date: {current_date}

Question: {question}

Respond with exactly "search" or "chat":"""


memory = SimpleMemory(window_size=5)


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
        return f"Error: {e}"


def _should_search(question: str) -> bool:
    today = date.today().isoformat()
    decision = _call_llm(_SELECT_PROMPT, {"question": question, "current_date": today}).strip().lower()
    return "search" in decision


def chat(message: str) -> str:
    today = date.today().isoformat()
    history = memory.get_history()

    if _should_search(message):
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
