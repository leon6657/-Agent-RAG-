"""Evaluation metrics for retrieval quality."""

from typing import Any, List


def _text(doc: Any) -> str:
    return getattr(doc, "page_content", str(doc))


def _terms(relevant: Any) -> List[str]:
    if relevant is None:
        return []
    if isinstance(relevant, str):
        return [relevant]
    terms = []
    for item in relevant:
        if isinstance(item, str):
            terms.append(item)
        else:
            terms.extend(_terms(item))
    return terms


def _matches(doc: Any, relevant: Any) -> bool:
    content = _text(doc).lower()
    return any(term.lower() in content for term in _terms(relevant))


def recall_at_k(retrieved: List[Any], relevant: Any, k: int = 3) -> float:
    if not _terms(relevant):
        return 0.0
    return 1.0 if any(_matches(doc, relevant) for doc in retrieved[:k]) else 0.0


def mrr(retrieved: List[Any], relevant: Any) -> float:
    for rank, doc in enumerate(retrieved, 1):
        if _matches(doc, relevant):
            return 1.0 / rank
    return 0.0


def precision_at_k(retrieved: List[Any], relevant: Any, k: int = 3) -> float:
    if not retrieved or k == 0:
        return 0.0
    return (1.0 / k) if any(_matches(doc, relevant) for doc in retrieved[:k]) else 0.0


def evaluate_all(retrieved_list: List[List[Any]], relevant_list: List[Any], k: int = 3) -> dict:
    n = len(retrieved_list)
    recall = [recall_at_k(r, rel, k) for r, rel in zip(retrieved_list, relevant_list)]
    mrr_s = [mrr(r, rel) for r, rel in zip(retrieved_list, relevant_list)]
    precision = [precision_at_k(r, rel, k) for r, rel in zip(retrieved_list, relevant_list)]
    return {
        "k": k,
        "count": n,
        "recall_at_k": sum(recall) / n if n else 0,
        "mrr": sum(mrr_s) / n if n else 0,
        "precision_at_k": sum(precision) / n if n else 0,
    }
