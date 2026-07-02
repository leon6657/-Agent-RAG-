"""Query interface: ask questions against the RAG knowledge base."""

from app.config import config
from app.ingest import _get_chroma_collection, build_embeddings
from app.chain import build_chain


def get_retriever():
    """Get the ChromaDB retriever."""
    collection = _get_chroma_collection()
    count = collection.count()
    if count == 0:
        raise ValueError(
            "Vector store is empty. Run 'python main.py --ingest' first."
        )

    from langchain_community.vectorstores import Chroma as LangChainChroma

    emb = build_embeddings()
    from chromadb import PersistentClient
    client = PersistentClient(path=config.chroma_persist_dir)
    vector_store = LangChainChroma(
        client=client,
        embedding_function=emb,
        collection_name="rag_kb",
    )
    return vector_store.as_retriever(search_kwargs={"k": config.retrieval_top_k})


def ask(question: str) -> str:
    """Ask a question and get an answer with sources."""
    retriever = get_retriever()
    chain = build_chain(retriever)
    return chain.invoke(question)


def ask_stream(question: str):
    """Ask a question and stream the answer token by token."""
    retriever = get_retriever()
    chain = build_chain(retriever)
    for chunk in chain.stream(question):
        yield chunk
