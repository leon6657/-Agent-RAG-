"""Document ingestion pipeline: load -> split -> embed -> store."""

import os
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import config
from app import store

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class MarkdownLoader(TextLoader):
    def __init__(self, file_path: str):
        super().__init__(file_path, encoding="utf-8")


def load_markdown_files(data_dir: Optional[str] = None) -> List[Document]:
    if data_dir is None:
        data_dir = config.data_dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        return []
    loader = DirectoryLoader(
        data_dir, glob="**/*.md", loader_cls=MarkdownLoader, show_progress=True
    )
    return loader.load()


def split_documents(docs: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", "\u3002", "\uff01", "\uff1f", " "],
    )
    return text_splitter.split_documents(docs)


def _get_model_path() -> str:
    snapshots_dir = (
        _PROJECT_ROOT
        / ".hf_cache"
        / "hub"
        / "models--BAAI--bge-small-zh-v1.5"
        / "snapshots"
    )
    if snapshots_dir.exists():
        snapshots = os.listdir(str(snapshots_dir))
        if snapshots:
            return str(snapshots_dir / snapshots[0])
    return "BAAI/bge-small-zh-v1.5"


def build_embeddings():
    model_path = _get_model_path()
    return HuggingFaceBgeEmbeddings(
        model_name=model_path,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def run_ingest() -> int:
    print(f"Loading documents from {config.data_dir}...")
    docs = load_markdown_files()
    if not docs:
        print("No .md files found. Add some to the data/ directory.")
        return 0
    print(f"Loaded {len(docs)} document(s)")

    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunk(s)")

    print("Computing embeddings...")
    emb = build_embeddings()
    texts = [c.page_content for c in chunks]
    vectors = emb.embed_documents(texts)
    print(f"Embedding dim: {len(vectors[0])}")

    store.clear()
    store.add_documents(chunks, vectors)
    print(f"Stored {len(chunks)} chunks in vector store")
    return len(chunks)
