"""Document ingestion pipeline: load -> split -> embed -> store."""

import os
import uuid
from pathlib import Path
from typing import List, Optional

import chromadb
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import config

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class MarkdownLoader(TextLoader):
    """Loads .md files with UTF-8 encoding."""

    def __init__(self, file_path: str):
        super().__init__(file_path, encoding="utf-8")


def load_markdown_files(data_dir: Optional[str] = None) -> List[Document]:
    """Load all .md files from data_dir."""
    if data_dir is None:
        data_dir = config.data_dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        return []

    loader = DirectoryLoader(
        data_dir,
        glob="**/*.md",
        loader_cls=MarkdownLoader,
        show_progress=True,
    )
    return loader.load()


def split_documents(docs: List[Document]) -> List[Document]:
    """Split documents into chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", "\u3002", "\uff01", "\uff1f", " "],
    )
    return text_splitter.split_documents(docs)


def _get_model_path() -> str:
    snapshots_dir = _PROJECT_ROOT / ".hf_cache" / "hub" / "models--BAAI--bge-small-zh-v1.5" / "snapshots"
    if snapshots_dir.exists():
        snapshots = os.listdir(str(snapshots_dir))
        if snapshots:
            return str(snapshots_dir / snapshots[0])
    return "BAAI/bge-small-zh-v1.5"


def build_embeddings():
    """Create the BGE embedding model, using local cache if available."""
    model_path = _get_model_path()
    return HuggingFaceBgeEmbeddings(
        model_name=model_path,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _get_chroma_collection(persist_directory: Optional[str] = None) -> chromadb.Collection:
    """Get or create a ChromaDB collection (uses no default embedding)."""
    if persist_directory is None:
        persist_directory = config.chroma_persist_dir
    client = chromadb.PersistentClient(path=persist_directory)
    return client.get_or_create_collection(name="rag_kb")


def add_documents(collection: chromadb.Collection, docs: List[Document]):
    """Compute embeddings and upsert documents into ChromaDB."""
    emb = build_embeddings()
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]

    print(f"Computing {len(texts)} embedding(s)...", flush=True)
    vectors = emb.embed_documents(texts)
    print(f"Embedding dim: {len(vectors[0])}", flush=True)

    ids = [str(uuid.uuid4()) for _ in texts]
    collection.upsert(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
    print(f"Stored {len(ids)} chunks in ChromaDB", flush=True)


def run_ingest() -> int:
    """Full ingestion pipeline. Returns number of chunks ingested."""
    print(f"Loading documents from {config.data_dir}...")
    docs = load_markdown_files()
    if not docs:
        print("No .md files found. Add some to the data/ directory.")
        return 0
    print(f"Loaded {len(docs)} document(s)")

    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunk(s)")

    print("Building vector store...")
    collection = _get_chroma_collection()
    add_documents(collection, chunks)
    print(f"Vector store saved to {config.chroma_persist_dir}")

    return len(chunks)


def get_retriever():
    """Get a LangChain retriever wrapping the ChromaDB collection."""
    from langchain_community.vectorstores import Chroma as LangChainChroma

    emb = build_embeddings()
    client = chromadb.PersistentClient(path=config.chroma_persist_dir)
    vector_store = LangChainChroma(
        client=client,
        embedding_function=emb,
        collection_name="rag_kb",
    )
    return vector_store.as_retriever(search_kwargs={"k": config.retrieval_top_k})
