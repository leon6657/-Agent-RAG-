"""Query interface: ask questions against the RAG knowledge base."""

from typing import List

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from app.config import config
from app.ingest import _get_chroma_collection, build_embeddings
from app.chain import build_chain


def _search_for_retriever(query: str) -> List[Document]:
    """Search ChromaDB for relevant documents."""
    coll = _get_chroma_collection()
    if coll.count() == 0:
        raise ValueError("Vector store is empty. Run 'python main.py --ingest' first.")
    emb = build_embeddings()
    vector = emb.embed_query(query)
    results = coll.query(query_embeddings=[vector], n_results=config.retrieval_top_k)
    docs = []
    doc_ids = results.get("ids", [[]])[0]
    documents_list = results.get("documents", [[]])[0]
    metadatas_list = results.get("metadatas", [[]])[0]
    for i in range(len(doc_ids)):
        meta = metadatas_list[i] if i < len(metadatas_list) and metadatas_list[i] else {}
        docs.append(Document(page_content=documents_list[i], metadata=meta))
    return docs


def get_retriever():
    """Get a retriever for the RAG chain."""
    return RunnableLambda(_search_for_retriever)


def ask(question: str) -> str:
    retriever = get_retriever()
    chain = build_chain(retriever)
    return chain.invoke(question)


def ask_stream(question: str):
    retriever = get_retriever()
    chain = build_chain(retriever)
    for chunk in chain.stream(question):
        yield chunk
