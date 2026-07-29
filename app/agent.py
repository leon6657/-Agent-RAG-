"""Simple custom agent with KB search + DeepSeek web search fallback."""

from datetime import date
import json

from app.runtime import embed_query_cached
from app.memory import SimpleMemory
from app.config import config
from app import store

_ANSWER_STYLE = (
    "\u8bf7\u7528\u4e2d\u6587\u56de\u7b54\uff0c\u7ed3\u6784\u6e05\u6670\uff0c"
    "\u50cf\u6280\u672f\u8bf4\u660e\u6587\u4e00\u6837\u6613\u8bfb\u3002\n"
    "\u4f18\u5148\u4f7f\u7528\uff1a\n"
    "- \u7b80\u77ed\u5f00\u5934\u8bf4\u660e\u6838\u5fc3\u4f5c\u7528\n"
    "- \u5fc5\u8981\u65f6\u4f7f\u7528 Markdown \u8868\u683c\u505a\u5bf9\u6bd4\n"
    "- \u793a\u4f8b\u4ee3\u7801\u4f7f\u7528 Markdown \u4ee3\u7801\u5757\n"
    "- \u7ed3\u5c3e\u7ed9\u51fa\u6ce8\u610f\u4e8b\u9879\u6216\u4e00\u53e5\u8bdd\u603b\u7ed3\n"
    "\u4e0d\u8981\u5806\u53e0\u8fc7\u591a\u661f\u53f7\u3002"
    "\u4e0d\u8981\u4f7f\u7528 ### \u8fd9\u7c7b Markdown \u6807\u9898\u7b26\u53f7"
    "\u76f4\u63a5\u51fa\u73b0\u5728\u56de\u7b54\u91cc\uff1b\u5982\u679c\u9700\u8981"
    "\u5206\u8282\uff0c\u8bf7\u4f7f\u7528\u7b80\u77ed\u6807\u9898\u6216\u8fde\u7eed"
    "\u7f16\u53f7\uff081. 2. 3.\uff09\uff0c\u4e0d\u8981\u6bcf\u4e2a\u5206\u8282"
    "\u90fd\u4ece 1. \u5f00\u59cb\u3002"
)

_KB_PROMPT = """Current date: {current_date}

Based on the context below, answer the question.
{style}

Context:
{context}

Question: {question}

Answer concisely based on the context."""

_CHAT_PROMPT = """Current date: {current_date}

You are a helpful assistant with expertise in programming, tech, and general knowledge.
{style}

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


def _stream_llm(prompt_template: str, variables: dict):
    from app.chain import build_llm as _build
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | _build() | StrOutputParser()
    yield from chain.stream(variables)


def _search_kb(query: str) -> str:
    from app.query import _search
    try:
        return _search(query)
    except ValueError:
        return ""


def _has_relevant(query: str) -> bool:
    vec = embed_query_cached(query)
    return store.search_top_score(vec) >= _SEARCH_THRESHOLD


def _call_with_search(history: str, question: str, today: str) -> str:
    """Call DeepSeek API with built-in web search enabled."""
    import requests as _req
    key = config.deepseek_api_key
    if not key or key.startswith("sk-your"):
        return _call_llm(_CHAT_PROMPT, {"history": history or "None", "question": question, "current_date": today, "style": _ANSWER_STYLE})

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
        return _call_llm(_CHAT_PROMPT, {"history": history or "None", "question": question, "current_date": today, "style": _ANSWER_STYLE})


def _call_deepseek_direct(question: str, history: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=config.deepseek_api_key, base_url=config.deepseek_api_base)
    r = client.chat.completions.create(
        model=config.deepseek_model,
        messages=[
            {"role": "system", "content": _ANSWER_STYLE},
            {"role": "user", "content": question},
        ],
        temperature=0.3,
        max_tokens=2048,
        extra_body={"enable_search": True},
    )
    return r.choices[0].message.content


def _stream_deepseek_direct(question: str, history: str):
    from openai import OpenAI
    client = OpenAI(api_key=config.deepseek_api_key, base_url=config.deepseek_api_base)
    stream = client.chat.completions.create(
        model=config.deepseek_model,
        messages=[
            {"role": "system", "content": _ANSWER_STYLE},
            {"role": "user", "content": f"Conversation history:\n{history or 'None'}\n\nQuestion: {question}"},
        ],
        temperature=0.3,
        max_tokens=2048,
        extra_body={"enable_search": True},
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        text = getattr(delta, "content", None)
        if text:
            yield text


def chat(message: str) -> str:
    today = date.today().isoformat()
    history = memory.get_history()

    context = _search_kb(message)
    if context:
        if len(message) >= 4 and message[:4] not in context:
            context = ''
    if context:
        response = _call_llm(_KB_PROMPT, {"context": context, "question": message, "current_date": today, "style": _ANSWER_STYLE})
    else:
        response = _call_deepseek_direct(message, history)

    memory.add_user(message)
    memory.add_assistant(response)
    return response


def chat_stream_events(message: str):
    today = date.today().isoformat()
    history = memory.get_history()

    context = _search_kb(message)
    if context:
        if len(message) >= 4 and message[:4] not in context:
            context = ""

    mode = "agent_kb" if context else "agent"
    yield {"event": "meta", "mode": mode, "sources": [], "top_score": None}

    response_parts = []
    try:
        if context:
            chunks = _stream_llm(
                _KB_PROMPT,
                {"context": context, "question": message, "current_date": today, "style": _ANSWER_STYLE},
            )
        else:
            chunks = _stream_deepseek_direct(message, history)

        for chunk in chunks:
            response_parts.append(chunk)
            yield {"event": "token", "text": chunk}
    except Exception:
        response = chat(message)
        response_parts = [response]
        yield {"event": "token", "text": response}
        yield {"event": "done"}
        return

    response = "".join(response_parts)
    memory.add_user(message)
    memory.add_assistant(response)
    yield {"event": "done"}
