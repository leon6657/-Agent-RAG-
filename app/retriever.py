"""Hybrid retriever: BM25 + vector search with weighted combination."""

import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from app import store

_DATA_PATH = Path(__file__).resolve().parent.parent / "vector_store.json"
_bm25 = None
_all_texts = None


def _load_all() -> Tuple[List[str], List[dict]]:
    """Load all stored texts and metadata."""
    if not _DATA_PATH.exists():
        return [], []
    with open(_DATA_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)
    texts = [item["text"] for item in items]
    return texts, items


def _get_bm25():
    global _bm25, _all_texts
    if _bm25 is None:
        texts, _ = _load_all()
        tokenized = [_tokenize(t) for t in texts]
        _bm25 = BM25Okapi(tokenized)
        _all_texts = texts
    return _bm25, _all_texts


def _tokenize(text: str) -> List[str]:
    """Simple Chinese tokenizer: split on whitespace and punctuation."""
    import re
    tokens = re.findall(r"[\w\u4e00-\u9fff]+", text.lower())
    return tokens


def search_bm25(query: str, k: int = 10) -> List[Tuple[str, float]]:
    """Search with BM25, return (text, score) pairs."""
    bm25, texts = _get_bm25()
    q_tokens = _tokenize(query)
    scores = bm25.get_scores(q_tokens)
    top_idx = np.argsort(scores)[-k:][::-1]
    results = []
    for i in top_idx:
        if scores[i] > 0:
            results.append((texts[i], float(scores[i])))
    return results


def search_hybrid(query: str, query_vector: List[float], k: int = 4,
                  alpha: float = 0.5) -> list:
    """Hybrid search combining BM25 and vector scores.
    
    alpha: weight for BM25 score (1-alpha for vector score).
    Returns top-k documents with combined scores in metadata.
    """
    # Get BM25 results
    bm25_results = search_bm25(query, k=k * 3)
    bm25_dict = {t: s for t, s in bm25_results}
    
    # Get vector results
    vec_docs = store.search(query_vector, k=k * 3)
    vec_texts = [d.page_content for d in vec_docs]
    
    # Compute vector scores (re-normalized to 0-1)
    q = np.array(query_vector)
    q = q / np.linalg.norm(q)
    vec_scores = {}
    for doc in vec_docs:
        all_items = json.loads(open(_DATA_PATH, "r", encoding="utf-8").read())
        for item in all_items:
            if item["text"] == doc.page_content:
                v = np.array(item["embedding"])
                v = v / np.linalg.norm(v)
                sim = float(q @ v)
                vec_scores[doc.page_content] = max(0, sim)
                break
    
    # Combine scores
    all_texts = list(set(list(bm25_dict.keys()) + list(vec_scores.keys())))
    
    # Normalize BM25 scores to 0-1
    if bm25_dict:
        max_b = max(bm25_dict.values())
        if max_b > 0:
            bm25_dict = {t: s / max_b for t, s in bm25_dict.items()}
    
    # Normalize vector scores to 0-1
    if vec_scores:
        max_v = max(vec_scores.values())
        if max_v > 0:
            vec_scores = {t: s / max_v for t, s in vec_scores.items()}
    
    combined = {}
    for t in all_texts:
        b = bm25_dict.get(t, 0)
        v = vec_scores.get(t, 0)
        combined[t] = alpha * b + (1 - alpha) * v
    
    # Sort and return top-k
    sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
    
    from langchain_core.documents import Document
    result_docs = []
    for text, score in sorted_results:
        meta = {}
        for item in json.loads(open(_DATA_PATH, "r", encoding="utf-8").read()):
            if item["text"] == text:
                meta = item.get("metadata", {})
                break
        meta["score"] = round(score, 4)
        meta["method"] = "hybrid"
        result_docs.append(Document(page_content=text, metadata=meta))
    
    return result_docs


def reset():
    """Clear BM25 cache (call after re-ingesting)."""
    global _bm25, _all_texts
    _bm25 = None
    _all_texts = None
