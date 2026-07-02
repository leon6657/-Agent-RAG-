"""Evaluation metrics for retrieval quality."""

from typing import List


def recall_at_k(retrieved: List[str], relevant: List[str], k: int = 3) -> float:
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for r in relevant if any(r.lower() in d.lower() for d in top_k))
    return hits / len(relevant)


def mrr(retrieved: List[str], relevant: List[str]) -> float:
    for rank, doc in enumerate(retrieved, 1):
        if any(kw.lower() in doc.lower() for kw in relevant):
            return 1.0 / rank
    return 0.0


def precision_at_k(retrieved: List[str], relevant: List[str], k: int = 3) -> float:
    if not retrieved or k == 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for r in relevant if any(r.lower() in d.lower() for d in top_k))
    return hits / k


def evaluate_all(retrieved_list: List[List[str]], relevant_list: List[List[str]], k: int = 3) -> dict:
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
