"""Simple custom agent with KB search + DeepSeek web search fallback."""

from datetime import date
import json

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

You are a helpful assistant with expertise in programming, tech, and general knowledge.

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
    except ValueError:
        return ""


def _has_relevant(query: str) -> bool:
    emb = build_embeddings()
    vec = emb.embed_query(query)
    return store.search_top_score(vec) >= _SEARCH_THRESHOLD


def _call_with_search(history: str, question: str, today: str) -> str:
    """Call DeepSeek API with built-in web search enabled."""
    import requests as _req
    key = config.deepseek_api_key
    if not key or key.startswith("sk-your"):
        return _call_llm(_CHAT_PROMPT, {"history": history or "None", "question": question, "current_date": today})

    try:
        resp = _req.post(
            "https://api.deepseek.com/v1/chat/completions",
            json={
                "model": config.deepseek_model,
                "messages": [
                    {"role": "system", "content": f"Current date: {today}. Answer based on web search results."},
                    {"role": "user", "content": f"Conversation history:\n{history or 'None'}\n\nQuestion: {question}"},
                ],
                "enable_search": True,
                "temperature": 0.3,
                "max_tokens": 1024,
            },
            headers={"Authorization": f"Bearer {key}"},
            timeout=30,
        )
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception:
        return _call_llm(_CHAT_PROMPT, {"history": history or "None", "question": question, "current_date": today})


def chat(message: str) -> str:
    today = date.today().isoformat()
    history = memory.get_history()

    if _has_relevant(message):
        context = _search_kb(message)
        if context:
            response = _call_llm(_KB_PROMPT, {"context": context, "question": message, "current_date": today})
        else:
            response = _call_with_search(history, message, today)
    else:
        response = _call_with_search(history, message, today)

    memory.add_user(message)
    memory.add_assistant(response)
    return response
