"""Simple vector store using numpy for similarity search."""

import json
import os
import uuid
from pathlib import Path
from typing import List, Tuple

import numpy as np
from langchain_core.documents import Document

_DATA_PATH = Path(__file__).resolve().parent.parent / "vector_store.json"


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def add_documents(docs: List[Document], embeddings: List[List[float]]) -> None:
    items = []
    for doc, emb in zip(docs, embeddings):
        items.append({
            "id": str(uuid.uuid4()),
            "text": doc.page_content,
            "metadata": doc.metadata,
            "embedding": emb,
        })

    existing = []
    if _DATA_PATH.exists():
        with open(_DATA_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)

    existing.extend(items)
    with open(_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False)


def search(query_embedding: List[float], k: int = 4) -> List[Document]:
    if not _DATA_PATH.exists():
        return []

    with open(_DATA_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)

    if not items:
        return []

    q = _normalize(np.array(query_embedding, dtype=np.float32))
    vectors = np.array([item["embedding"] for item in items], dtype=np.float32)
    vectors = np.apply_along_axis(_normalize, 1, vectors)

    scores = vectors @ q
    top_idx = np.argsort(scores)[-k:][::-1]

    results = []
    for i in top_idx:
        if scores[i] > 0:
            results.append(Document(
                page_content=items[i]["text"],
                metadata=items[i].get("metadata", {}),
            ))
    return results


def count() -> int:
    if not _DATA_PATH.exists():
        return 0
    with open(_DATA_PATH, "r", encoding="utf-8") as f:
        return len(json.load(f))


def clear() -> None:
    if _DATA_PATH.exists():
        _DATA_PATH.unlink()
