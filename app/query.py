"""Query interface: ask questions against the RAG knowledge base."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app import store
from app.chain import build_llm
from app.config import config
from app.ingest import build_embeddings


NO_CONTEXT_ANSWER = (
    "\u77e5\u8bc6\u5e93\u4e2d\u6ca1\u6709\u627e\u5230\u8db3\u591f"
    "\u53ef\u9760\u7684\u4f9d\u636e\u6765\u56de\u7b54\u8fd9\u4e2a"
    "\u95ee\u9898\u3002\u5efa\u8bae\u5148\u8865\u5145\u76f8\u5173"
    "\u8d44\u6599\uff0c\u6216\u6362\u4e00\u4e2a\u66f4\u8d34\u8fd1"
    "\u5f53\u524d\u77e5\u8bc6\u5e93\u7684\u95ee\u9898\u3002"
)

ANSWER_STYLE_INSTRUCTIONS = (
    "\u8bf7\u7528\u4e2d\u6587\u56de\u7b54\uff0c\u683c\u5f0f\u5c3d\u91cf"
    "\u6e05\u6670\u3001\u50cf\u4e00\u7bc7\u53ef\u9605\u8bfb\u7684\u6280\u672f"
    "\u8bf4\u660e\u3002\n"
    "\u5efa\u8bae\u7ed3\u6784\uff1a\n"
    "1. \u5148\u7528\u4e00\u5c0f\u6bb5\u8bf4\u660e\u6838\u5fc3\u4f5c\u7528\u3002\n"
    "2. \u5982\u679c\u95ee\u9898\u6d89\u53ca\u5bf9\u6bd4\uff0c\u7528 Markdown "
    "\u8868\u683c\u5c55\u793a\u533a\u522b\u3002\n"
    "3. \u5982\u679c\u9002\u5408\uff0c\u7ed9\u51fa\u7b80\u77ed\u793a\u4f8b\uff0c"
    "\u4ee3\u7801\u7528 Markdown \u4ee3\u7801\u5757\u3002\n"
    "4. \u6700\u540e\u7ed9\u51fa\u6ce8\u610f\u4e8b\u9879\u6216\u4e00\u53e5\u8bdd"
    "\u603b\u7ed3\u3002\n"
    "\u53ef\u4ee5\u4f7f\u7528 Markdown \u6807\u9898\u3001\u52a0\u7c97\u3001"
    "\u5217\u8868\u3001\u8868\u683c\u548c\u4ee3\u7801\u5757\uff0c\u4f46\u4e0d\u8981"
    "\u5806\u53e0\u8fc7\u591a\u661f\u53f7\u3002"
)


def _retrieve_docs(query: str):
    if store.count() == 0:
        raise ValueError("Vector store is empty. Run 'python main.py --ingest' first.")
    emb = build_embeddings()
    vector = emb.embed_query(query)
    return store.search_cached(vector, k=config.retrieval_top_k)


def _format_context(docs) -> str:
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "?")
        parts.append(f"[{source}]\n{doc.page_content}")
    return "\n\n".join(parts)


def _search(query: str) -> str:
    return _format_context(_retrieve_docs(query))


def _source_payload(docs) -> list[dict]:
    sources = []
    for doc in docs:
        text = " ".join(doc.page_content.split())
        sources.append(
            {
                "source": doc.metadata.get("source", doc.metadata.get("filename", "?")),
                "filename": doc.metadata.get("filename", doc.metadata.get("source", "?")),
                "score": doc.metadata.get("score", 0.0),
                "preview": text[:180],
            }
        )
    return sources


def _top_score(docs) -> float:
    if not docs:
        return 0.0
    return float(docs[0].metadata.get("score", 0.0))


def _append_citations(answer: str, sources: list[dict]) -> str:
    if "sources:" in answer.lower():
        return answer
    filenames = []
    for source in sources:
        filename = source.get("filename") or source.get("source")
        if filename and filename not in filenames:
            filenames.append(filename)
    if not filenames:
        return answer
    return f"{answer.rstrip()}\n\nSources: {', '.join(filenames)}"


def ask_with_sources(question: str) -> dict:
    docs = _retrieve_docs(question)
    top_score = _top_score(docs)
    sources = _source_payload(docs)

    if top_score < config.rag_min_score:
        return {
            "answer": NO_CONTEXT_ANSWER,
            "sources": sources,
            "mode": "no_context",
            "top_score": top_score,
        }

    context = _format_context(docs)
    prompt = ChatPromptTemplate.from_template(
        "You are a strict RAG assistant. Answer only from the Context below. "
        "Do not invent facts that are not supported by the Context. "
        "{style}\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    chain = prompt | build_llm() | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question, "style": ANSWER_STYLE_INSTRUCTIONS})
    return {
        "answer": _append_citations(answer, sources),
        "sources": sources,
        "mode": "rag",
        "top_score": top_score,
    }


def ask(question: str) -> str:
    return ask_with_sources(question)["answer"]


def ask_stream(question: str):
    docs = _retrieve_docs(question)
    if _top_score(docs) < config.rag_min_score:
        yield NO_CONTEXT_ANSWER
        return

    context = _format_context(docs)
    prompt = ChatPromptTemplate.from_template(
        "You are a strict RAG assistant. Answer only from the Context below. "
        "Do not invent facts that are not supported by the Context.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    chain = prompt | build_llm() | StrOutputParser()
    for chunk in chain.stream({"context": context, "question": question}):
        yield chunk


def ask_stream_events(question: str):
    docs = _retrieve_docs(question)
    top_score = _top_score(docs)
    sources = _source_payload(docs)

    if top_score < config.rag_min_score:
        yield {
            "event": "meta",
            "sources": sources,
            "mode": "no_context",
            "top_score": top_score,
        }
        yield {"event": "token", "text": NO_CONTEXT_ANSWER}
        yield {"event": "done"}
        return

    yield {
        "event": "meta",
        "sources": sources,
        "mode": "rag",
        "top_score": top_score,
    }

    context = _format_context(docs)
    prompt = ChatPromptTemplate.from_template(
        "You are a strict RAG assistant. Answer only from the Context below. "
        "Do not invent facts that are not supported by the Context. "
        "{style}\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    chain = prompt | build_llm() | StrOutputParser()
    answer_parts = []
    for chunk in chain.stream({"context": context, "question": question, "style": ANSWER_STYLE_INSTRUCTIONS}):
        answer_parts.append(chunk)
        yield {"event": "token", "text": chunk}

    answer = "".join(answer_parts)
    citation_text = _append_citations(answer, sources)
    suffix = citation_text[len(answer):]
    if suffix:
        yield {"event": "token", "text": suffix}
    yield {"event": "done"}
