"""Simple custom agent with tool selection and memory."""

# Import order: community BEFORE openai
from app.ingest import build_embeddings  # noqa: F401 - forces langchain_community import

from app.memory import SimpleMemory
from app.config import config

_KB_PROMPT = """Based on the context below, answer the question.

Context:
{context}

Question: {question}

Answer concisely based on the context. If the context is not relevant, say so."""

_CHAT_PROMPT = """You are a helpful assistant. Answer the question naturally.

Conversation history:
{history}

Question: {question}

Answer:"""

_SELECT_PROMPT = """You have access to a knowledge base with notes about Python, ML, data structures.
Given the user's question, decide if you need to search the knowledge base.

Question: {question}

Reply with exactly "search" if the question is about Python, ML, data structures, programming concepts, or specific topics from your notes.
Reply with "chat" for greetings, general conversation, code examples, or anything else.

Decision:"""


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
    decision = _call_llm(_SELECT_PROMPT, {"question": question}).strip().lower()
    return "search" in decision


def chat(message: str) -> str:
    history = memory.get_history()

    if _should_search(message):
        context = _search_kb(message)
        if context:
            response = _call_llm(_KB_PROMPT, {"context": context, "question": message})
        else:
            response = _call_llm(_CHAT_PROMPT, {"history": history or "None", "question": message})
    else:
        response = _call_llm(_CHAT_PROMPT, {"history": history or "None", "question": message})

    memory.add_user(message)
    memory.add_assistant(response)
    return response
