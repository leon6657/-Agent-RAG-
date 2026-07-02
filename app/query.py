"""Query interface: ask questions against the RAG knowledge base."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.config import config
from app.ingest import build_embeddings
from app import store


def _search(query: str) -> str:
    if store.count() == 0:
        raise ValueError("Vector store is empty. Run 'python main.py --ingest' first.")
    emb = build_embeddings()
    vector = emb.embed_query(query)
    docs = store.search(vector, k=config.retrieval_top_k)
    parts = []
    for d in docs:
        src = d.metadata.get("source", "?")
        parts.append(f"[{src}]\n{d.page_content}")
    return "\n\n".join(parts)


def ask(question: str) -> str:
    context = _search(question)
    from app.chain import build_llm
    prompt = ChatPromptTemplate.from_template(
        "Based on the context below, answer the question.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    chain = prompt | build_llm() | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


def ask_stream(question: str):
    context = _search(question)
    from app.chain import build_llm
    prompt = ChatPromptTemplate.from_template(
        "Based on the context below, answer the question.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    chain = prompt | build_llm() | StrOutputParser()
    for chunk in chain.stream({"context": context, "question": question}):
        yield chunk
