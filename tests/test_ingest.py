"""Tests for the ingestion pipeline."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.ingest import load_markdown_files, split_documents


def test_load_markdown_files_finds_md():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test.md")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("# Hello\n\nWorld")

        non_md = os.path.join(tmpdir, "notes.txt")
        with open(non_md, "w") as f:
            f.write("ignore me")

        docs = load_markdown_files(tmpdir)
        assert len(docs) == 1
        assert docs[0].metadata["source"].endswith("test.md")


def test_split_documents_creates_chunks():
    from langchain_core.documents import Document

    docs = [Document(page_content="A " * 1000, metadata={"source": "test.md"})]
    chunks = split_documents(docs)
    assert len(chunks) > 1
    assert all("source" in c.metadata for c in chunks)
