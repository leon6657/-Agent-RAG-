"""Evaluation runner: run retrieval tests and compare results."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["HF_HOME"] = str(Path(__file__).resolve().parent.parent / ".hf_cache")

from evaluation.metrics import evaluate_all
from app.ingest import build_embeddings
from app import store
from app.retriever import search_hybrid
from app.query_rewriter import generate_queries
from app.reranker import rerank

_EMB_CACHE = None


def _get_emb():
    global _EMB_CACHE
    if _EMB_CACHE is None:
        _EMB_CACHE = build_embeddings()
    return _EMB_CACHE


def _search_baseline(query: str, k: int = 4) -> list:
    emb = _get_emb()
    vec = emb.embed_query(query)
    docs = store.search(vec, k=k)
    return [d.page_content for d in docs]


def _search_optimized_full(query: str, k: int = 4) -> list:
    """Full Phase 2 pipeline: Multi-Query -> Hybrid Search -> Rerank."""
    # Step 1: Multi-Query
    variations = generate_queries(query, n=3)
    all_queries = [query] + variations

    emb = _get_emb()
    all_docs = []
    seen = set()
    for q in all_queries:
        vec = emb.embed_query(q)
        docs = search_hybrid(q, vec, k=k, alpha=0.3)
        for d in docs:
            if d.page_content not in seen:
                seen.add(d.page_content)
                all_docs.append(d)

    # Step 2: Rerank
    reranked = rerank(query, all_docs, top_k=k)
    return [d.page_content for d in reranked]


def load_questions(path: str = "evaluation/questions.json") -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(search_fn, questions: list, k: int = 4) -> dict:
    retrieved_list = []
    relevant_list = []
    for q in questions:
        docs = search_fn(q["q"], k=k)
        retrieved_list.append(docs)
        relevant_list.append([q["a"]])
    results = evaluate_all(retrieved_list, relevant_list, k=k)
    results["questions"] = len(questions)
    return results


def print_report(baseline: dict, optimized: dict = None):
    print("=" * 55)
    print("  RETRIEVAL EVALUATION REPORT")
    print("=" * 55)

    def _print(name, data):
        print(f"  {name}:")
        print(f"    Questions:      {data['questions']}")
        print(f"    Recall@{data['k']}:     {data['recall_at_k']:.3f}")
        print(f"    MRR:            {data['mrr']:.3f}")
        print(f"    Precision@{data['k']}:   {data['precision_at_k']:.3f}")

    _print("BASELINE (Vector Only)", baseline)
    if optimized:
        _print("\nOPTIMIZED (Full Phase 2)", optimized)
        print(f"\n  Improvement:")
        print(f"    Recall@{baseline['k']}:     {optimized['recall_at_k']-baseline['recall_at_k']:+.3f}")
        print(f"    MRR:            {optimized['mrr']-baseline['mrr']:+.3f}")
        print(f"    Precision@{baseline['k']}:   {optimized['precision_at_k']-baseline['precision_at_k']:+.3f}")


if __name__ == "__main__":
    questions = load_questions()
    print(f"Loaded {len(questions)} questions\n")

    print("Running baseline (vector only)...")
    baseline = run_evaluation(_search_baseline, questions)

    print("\nRunning optimized (Phase 2 full pipeline)...")
    print("(this will take a while due to Multi-Query + DeepSeek API)")
    optimized = run_evaluation(_search_optimized_full, questions)

    print("\n" + "=" * 55)
    print_report(baseline, optimized)
