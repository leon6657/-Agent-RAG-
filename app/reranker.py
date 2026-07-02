"""Simple reranker using keyword overlap and position."""

import re
from typing import List

from langchain_core.documents import Document


def _extract_keywords(text: str) -> set:
    """Extract meaningful keywords from text."""
    tokens = re.findall(r"[\w\u4e00-\u9fff]+", text.lower())
    return set(tokens)


def rerank(query: str, documents: List[Document], top_k: int = 4) -> List[Document]:
    """Rerank documents by query relevance.
    
    Scoring factors:
    - Keyword overlap between query and document
    - Original position (as a tiebreaker)
    """
    query_keywords = _extract_keywords(query)
    if not query_keywords:
        return documents[:top_k]
    
    scored = []
    for i, doc in enumerate(documents):
        doc_keywords = _extract_keywords(doc.page_content)
        overlap = len(query_keywords & doc_keywords)
        # Combined score: keyword overlap + position bonus
        score = overlap * 10 + max(0, 10 - i)
        scored.append((score, i, doc))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    reranked = [doc for _, _, doc in scored]
    reranked[0].metadata["rerank_score"] = scored[0][0]
    return reranked[:top_k]
