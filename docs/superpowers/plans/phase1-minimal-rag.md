# Phase 1: Minimal RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal RAG system that ingests Markdown notes into ChromaDB and answers questions via DeepSeek Chat + BGE embedding.

**Architecture:** A CLI tool with two commands (`--ingest` and `--query`). Ingest loads .md files from `data/`, splits into chunks, embeds via BGE, stores in ChromaDB. Query takes a question, retrieves top-4 chunks, and generates an answer via DeepSeek Chat using an LCEL chain.

**Tech Stack:** Python 3.10+, LangChain, ChromaDB, BGE (sentence-transformers), DeepSeek Chat API

## Global Constraints

- Python >= 3.10
- All dependencies listed in `pyproject.toml`
- DeepSeek API key stored in `.env`
- ChromaDB persisted to `chroma_db/` (in .gitignore)
- BGE model: BAAI/bge-small-zh-v1.5 (auto-download on first run)
- LLM: DeepSeek Chat via OpenAI-compatible API
- Chunk size: 500, overlap: 50
- Retrieval top_k: 4
- All source code in `app/` package
- CLI entry via `main.py`

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.env`
- Create: `app/__init__.py`

**Interfaces:**
- Consumes: nothing
- Produces: project skeleton with all directories in place

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p app data evaluation tests notebooks
```

- [ ] **Step 2: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "rag-kb"
version = "0.1.0"
description = "A minimal RAG knowledge base with Markdown notes"
requires-python = ">=3.10"
dependencies = [
    "langchain-core>=0.3",
    "langchain-community>=0.3",
    "langchain-openai>=0.2",
    "chromadb>=0.5",
    "sentence-transformers>=3.0",
    "python-dotenv>=1.0",
    "pytest>=8.0",
]
```

- [ ] **Step 3: Create .env**

```env
DEEPSEEK_API_KEY=sk-your-deepseek-api-key-here
```

- [ ] **Step 4: Create app/__init__.py**

Empty file.

- [ ] **Step 5: Add a sample Markdown note to test with**

Create `data/hello.md`:

```markdown
# Python Decorator

A decorator is a function that takes another function and extends its behavior without explicitly modifying it.

## Example

```python
def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")
    return wrapper

@my_decorator
def say_whee():
    print("Whee!")
```

## Key Points

- Decorators wrap functions to add behavior
- They use the @ syntax
- The wrapper function can access the original function's arguments
```

- [ ] **Step 6: Verify Python imports work**

```bash
cd rag-project
python -c "from langchain_core.documents import Document; print('langchain ok')"
python -c "import chromadb; print('chromadb ok')"
python -c "from sentence_transformers import SentenceTransformer; print('sentence-transformers ok')"
```

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .env app/ data/
git commit -m "chore: scaffold project structure"
```

---

### Task 2: Config Module

**Files:**
- Create: `app/config.py`

**Interfaces:**
- Consumes: `.env` file
- Produces: `config` object with attributes: `deepseek_api_key`, `deepseek_api_base`, `embedding_model_name`, `chunk_size`, `chunk_overlap`, `chroma_persist_dir`, `data_dir`

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import config

def test_config_has_required_fields():
    assert hasattr(config, "deepseek_api_key")
    assert hasattr(config, "embedding_model_name")
    assert hasattr(config, "chunk_size")
    assert hasattr(config, "chroma_persist_dir")
    assert hasattr(config, "data_dir")

def test_config_defaults():
    assert config.chunk_size == 500
    assert config.chunk_overlap == 50
    assert config.deepseek_api_base == "https://api.deepseek.com"
    assert config.retrieval_top_k == 4
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd rag-project
python -m pytest tests/test_config.py -v
```
Expected: ModuleNotFoundError or ImportError for app.config

- [ ] **Step 3: Write app/config.py**

```python
"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    deepseek_api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    deepseek_api_base: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    embedding_model_name: str = "BAAI/bge-small-zh-v1.5"

    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_top_k: int = 4

    chroma_persist_dir: str = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "chroma_db"
    )
    data_dir: str = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data"
    )

    def __post_init__(self):
        if not self.deepseek_api_key:
            import warnings
            warnings.warn(
                "DEEPSEEK_API_KEY not set. Set it in .env or environment variables."
            )


config = Config()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd rag-project
python -m pytest tests/test_config.py -v
```
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add app/config.py tests/test_config.py
git commit -m "feat: add config module with env loading"
```

---

### Task 3: Ingestion Pipeline

**Files:**
- Create: `app/ingest.py`

**Interfaces:**
- Consumes: `config` from `app/config.py`, .md files from `config.data_dir`
- Produces: `build_vector_store() -> Chroma` that can be used by retriever

- [ ] **Step 1: Write the failing test**

Create `tests/test_ingest.py`:

```python
import os
import sys
import shutil
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.ingest import load_markdown_files, split_documents


def test_load_markdown_files_finds_md():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test .md file
        test_file = os.path.join(tmpdir, "test.md")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("# Hello\n\nWorld")
        # Create a non-md file that should be ignored
        with open(os.path.join(tmpdir, "notes.txt"), "w") as f:
            f.write("ignore me")

        docs = load_markdown_files(tmpdir)
        assert len(docs) == 1
        assert docs[0].metadata["source"].endswith("test.md")


def test_split_documents_creates_chunks():
    from langchain_core.documents import Document

    docs = [Document(page_content="A " * 1000, metadata={"source": "test.md"})]
    chunks = split_documents(docs)
    assert len(chunks) > 1  # should be split into multiple chunks
    assert all("source" in c.metadata for c in chunks)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd rag-project
python -m pytest tests/test_ingest.py -v
```
Expected: ImportError for app.ingest

- [ ] **Step 3: Write app/ingest.py**

```python
"""Document ingestion pipeline: load -> split -> embed -> store."""

import os
from typing import List

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import config


# Custom loader for Markdown files - wraps TextLoader for .md extension
class MarkdownLoader(TextLoader):
    def __init__(self, file_path: str):
        super().__init__(file_path, encoding="utf-8")


def load_markdown_files(data_dir: str = None) -> List[Document]:
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
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", "\u3002", "\uff01", "\uff1f", " "],
    )
    return text_splitter.split_documents(docs)


def build_embeddings():
    """Create BGE embedding model."""
    return HuggingFaceBgeEmbeddings(
        model_name=config.embedding_model_name,
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(
    docs: List[Document] = None,
    persist_directory: str = None,
) -> Chroma:
    """Build or rebuild ChromaDB from documents."""
    if persist_directory is None:
        persist_directory = config.chroma_persist_dir

    embeddings = build_embeddings()

    if docs:
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_directory,
        )
    else:
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )

    return vector_store


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
    build_vector_store(chunks)
    print(f"Vector store saved to {config.chroma_persist_dir}")

    return len(chunks)
```

- [ ] **Step 4: Run unit tests**

```bash
cd rag-project
python -m pytest tests/test_ingest.py -v
```
Expected: 2 passed

- [ ] **Step 5: Run ingest with the sample data**

```bash
cd rag-project
python -c "from app.ingest import run_ingest; run_ingest()"
```
Expected: "Loaded 1 document(s)", "Split into N chunk(s)", "Vector store saved to..."

- [ ] **Step 6: Verify ChromaDB was created**

```bash
ls chroma_db/
```
Expected: chroma.sqlite3 and related files exist

- [ ] **Step 7: Commit**

```bash
git add app/ingest.py tests/test_ingest.py
git commit -m "feat: add document ingestion pipeline with ChromaDB"
```

---

### Task 4: Chain and Query

**Files:**
- Create: `app/chain.py`
- Create: `app/query.py`

**Interfaces:**
- Consumes: `config` from `app/config.py`, `build_vector_store` / `build_embeddings` from `app/ingest.py`
- Produces: `ask(question: str) -> str` query function

- [ ] **Step 1: Write the failing test**

Create `tests/test_query.py`:

```python
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.chain import build_prompt
def test_build_prompt_includes_context():
    prompt = build_prompt()
    assert "{context}" in prompt.template
    assert "{question}" in prompt.template
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd rag-project
python -m pytest tests/test_query.py -v
```
Expected: ImportError for app.chain

- [ ] **Step 3: Write app/chain.py**

```python
"""LCEL chain definition for RAG question answering."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from app.config import config


def build_prompt() -> ChatPromptTemplate:
    """Build the QA prompt template."""
    template = """You are a helpful assistant answering questions based on the provided context.

Context:
{context}

---

Question: {question}

Answer the question based on the context above. If the context doesn't contain enough information, say so clearly. Answer in the same language as the question."""
    return ChatPromptTemplate.from_template(template)


def build_llm():
    """Build DeepSeek Chat LLM."""
    return ChatOpenAI(
        model=config.deepseek_model,
        api_key=config.deepseek_api_key,
        base_url=config.deepseek_api_base,
        temperature=0.3,
        streaming=True,
    )


def build_chain(retriever):
    """Build the complete LCEL RAG chain."""
    def format_docs(docs):
        return "\n\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
            for d in docs
        )

    prompt = build_prompt()
    llm = build_llm()

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
```

- [ ] **Step 4: Write app/query.py**

```python
"""Query interface: ask questions against the knowledge base."""

from app.config import config
from app.ingest import build_vector_store
from app.chain import build_chain


def get_retriever():
    """Get the ChromaDB retriever."""
    vector_store = build_vector_store()
    return vector_store.as_retriever(
        search_kwargs={"k": config.retrieval_top_k}
    )


def ask(question: str) -> str:
    """Ask a question and get an answer with sources."""
    retriever = get_retriever()
    chain = build_chain(retriever)

    result = chain.invoke(question)
    return result


def ask_stream(question: str):
    """Ask a question and stream the answer token by token."""
    retriever = get_retriever()
    chain = build_chain(retriever)

    for chunk in chain.stream(question):
        yield chunk
```

- [ ] **Step 5: Run unit test**

```bash
cd rag-project
python -m pytest tests/test_query.py -v
```
Expected: 1 passed

- [ ] **Step 6: Do a quick manual verification**

```bash
cd rag-project
python -c "from app.query import ask; print(ask('What is a Python decorator?'))"
```
Expected: An answer based on the hello.md content

- [ ] **Step 7: Commit**

```bash
git add app/chain.py app/query.py tests/test_query.py
git commit -m "feat: add RAG chain and query interface"
```

---

### Task 5: CLI Entry Point

**Files:**
- Create: `main.py`
- Modify: `app/ingest.py` (expose `run_ingest` publicly, already done)

**Interfaces:**
- Consumes: `run_ingest` from `app/ingest`, `ask_stream` from `app/query`
- Produces: CLI with `--ingest` and `--query` commands

- [ ] **Step 1: Write main.py**

```python
"""CLI entry point for the RAG knowledge base.

Usage:
    python main.py --ingest       Build/update vector store
    python main.py --query        Interactive Q&A session
"""

import argparse
import sys

from app.ingest import run_ingest
from app.query import ask, ask_stream
from app.config import config


def main():
    parser = argparse.ArgumentParser(
        description="RAG Knowledge Base - Q&A with your Markdown notes"
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Index all .md files in data/ into ChromaDB",
    )
    parser.add_argument(
        "--query",
        action="store_true",
        help="Start interactive Q&A session",
    )

    args = parser.parse_args()

    if not args.ingest and not args.query:
        parser.print_help()
        return

    if args.ingest:
        if not config.deepseek_api_key or config.deepseek_api_key == "sk-your-deepseek-api-key-here":
            print("ERROR: DEEPSEEK_API_KEY not configured.")
            print("Set it in .env file or as environment variable.")
            sys.exit(1)
        num_chunks = run_ingest()
        if num_chunks > 0:
            print(f"\nDone! Indexed {num_chunks} chunks.")

    if args.query:
        if not config.deepseek_api_key or config.deepseek_api_key == "sk-your-deepseek-api-key-here":
            print("ERROR: DEEPSEEK_API_KEY not configured.")
            print("Set it in .env file or as environment variable.")
            sys.exit(1)
        print("RAG Knowledge Base - Interactive Q&A")
        print("Type 'exit' to quit, 'clear' to clear screen")
        print("-" * 50)

        while True:
            try:
                question = input("\nYou: ").strip()
                if not question:
                    continue
                if question.lower() == "exit":
                    print("Goodbye!")
                    break
                if question.lower() == "clear":
                    import os
                    os.system("cls" if os.name == "nt" else "clear")
                    continue

                print("\nAssistant: ", end="", flush=True)
                full_response = ""
                for chunk in ask_stream(question):
                    print(chunk, end="", flush=True)
                    full_response += chunk
                print()

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test help output**

```bash
cd rag-project
python main.py --help
```
Expected: Shows usage with --ingest and --query flags

- [ ] **Step 3: Test --ingest via CLI**

```bash
cd rag-project
python main.py --ingest
```
Expected: "Loading documents from...", "Split into...", etc.

- [ ] **Step 4: Test --query via CLI (non-interactive check with echo)**

```bash
cd rag-project
echo "What is a decorator?" | python main.py --query
```
Expected: Should answer the question based on hello.md (might error on EOF, that's okay for initial version)

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "feat: add CLI entry point with --ingest and --query"
```

---

### Task 6: Error Handling and Polish

**Files:**
- Modify: `app/query.py` (add error handling to ask/ask_stream)
- Modify: `main.py` (better error messages)

**Interfaces:**
- No new interfaces

- [ ] **Step 1: Add error handling to query.py**

Update `app/query.py` to handle empty vector store:

```python
def get_retriever():
    vector_store = build_vector_store()
    count = vector_store._collection.count()
    if count == 0:
        raise ValueError(
            "Vector store is empty. Run 'python main.py --ingest' first."
        )
    return vector_store.as_retriever(search_kwargs={"k": config.retrieval_top_k})
```

- [ ] **Step 2: Add graceful API error handling**

Update `app/chain.py` to handle API errors gracefully in `build_llm`:

```python
def build_llm():
    if not config.deepseek_api_key:
        raise ValueError("DEEPSEEK_API_KEY is not set")
    return ChatOpenAI(...)
```

- [ ] **Step 3: Write summary tests**

Add to `tests/test_query.py`:

```python
def test_ask_raises_on_empty_store():
    import tempfile
    import os
    from app.config import config
    original = config.chroma_persist_dir
    with tempfile.TemporaryDirectory() as tmpdir:
        config.chroma_persist_dir = tmpdir
        try:
            from app.query import ask
            ask("test")
            assert False, "Should have raised"
        except ValueError as e:
            assert "empty" in str(e).lower() or "--ingest" in str(e)
        finally:
            config.chroma_persist_dir = original
```

- [ ] **Step 4: Run all tests**

```bash
cd rag-project
python -m pytest tests/ -v
```
Expected: All tests pass

- [ ] **Step 5: Tag the Phase 1 milestone**

```bash
git tag -a phase1-minimal-rag -m "Phase 1: Minimal RAG complete"
```

- [ ] **Step 6: Commit**

```bash
git add app/ tests/
git commit -m "feat: add error handling and polish for Phase 1"
```

---

## Phase 1 Completion Checklist

- [ ] Task 1 completed: pyproject.toml, directories, sample data
- [ ] Task 2 completed: config.py with env loading
- [ ] Task 3 completed: ingest pipeline loads -> chunks -> embeds -> stores
- [ ] Task 4 completed: chain.py + query.py with LCEL RAG chain
- [ ] Task 5 completed: main.py CLI with --ingest and --query
- [ ] Task 6 completed: error handling, tests pass, git tagged
