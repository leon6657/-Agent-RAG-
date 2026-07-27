"""Evaluation runner: run retrieval tests and compare results."""

import json
import os
import sys
from pathlib import Path
from typing import Any

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
    return store.search(vec, k=k)


def _search_optimized_full(query: str, k: int = 4) -> list:
    """Full Phase 2 pipeline: Multi-Query -> Hybrid Search -> Rerank."""
    variations = generate_queries(query, n=3)
    all_queries = [query] + variations

    emb = _get_emb()
    all_docs = []
    seen = set()
    original_vec = emb.embed_query(query)

    # Keep vector-only hits as a safety net so the optimized pipeline does not
    # drop strong semantic matches, especially for cross-lingual chunks.
    for doc in store.search(original_vec, k=k):
        seen.add(doc.page_content)
        all_docs.append(doc)

    for q in all_queries:
        vec = original_vec if q == query else emb.embed_query(q)
        docs = search_hybrid(q, vec, k=k, alpha=0.3)
        for doc in docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                all_docs.append(doc)

    return rerank(query, all_docs, top_k=k)


def load_questions(path: str = "evaluation/questions.json") -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _doc_text(doc: Any) -> str:
    return getattr(doc, "page_content", str(doc))


def _doc_source(doc: Any) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    source = metadata.get("filename") or metadata.get("source", "")
    return Path(str(source)).name


def _doc_score(doc: Any):
    metadata = getattr(doc, "metadata", {}) or {}
    return metadata.get("score")


def _expected_terms(expected: Any) -> list:
    if expected is None:
        return []
    if isinstance(expected, str):
        return [expected]
    terms = []
    for item in expected:
        if isinstance(item, str):
            terms.append(item)
        else:
            terms.extend(_expected_terms(item))
    return terms


def _find_relevant_rank(retrieved: list, expected: Any, k: int = 4):
    terms = [term.lower() for term in _expected_terms(expected)]
    for rank, doc in enumerate(retrieved[:k], 1):
        content = _doc_text(doc).lower()
        if any(term in content for term in terms):
            return rank
    return None


def _find_source_rank(retrieved: list, expected_source: str, k: int = 4):
    if not expected_source:
        return None
    expected_name = Path(expected_source).name.lower()
    for rank, doc in enumerate(retrieved[:k], 1):
        if _doc_source(doc).lower() == expected_name:
            return rank
    return None


def _retrieved_details(retrieved: list, k: int = 4) -> list:
    details = []
    for rank, doc in enumerate(retrieved[:k], 1):
        text = " ".join(_doc_text(doc).split())
        details.append({
            "rank": rank,
            "source": _doc_source(doc),
            "score": _doc_score(doc),
            "preview": text[:120],
        })
    return details


def _build_case(question: dict, retrieved: list, k: int = 4) -> dict:
    rank = _find_relevant_rank(retrieved, question["a"], k=k)
    source_rank = _find_source_rank(retrieved, question.get("s", ""), k=k)
    return {
        "id": question.get("id", ""),
        "question": question["q"],
        "expected": question["a"],
        "expected_source": question.get("s", ""),
        "hit": rank is not None,
        "rank": rank,
        "source_hit": source_rank is not None,
        "source_rank": source_rank,
        "retrieved_count": len(retrieved),
        "retrieved": _retrieved_details(retrieved, k=k),
    }


def run_evaluation(search_fn, questions: list, k: int = 4) -> dict:
    retrieved_list = []
    relevant_list = []
    cases = []
    for q in questions:
        docs = search_fn(q["q"], k=k)
        retrieved_list.append(docs)
        relevant_list.append(q["a"])
        cases.append(_build_case(q, docs, k=k))
    results = evaluate_all(retrieved_list, relevant_list, k=k)
    results["questions"] = len(questions)
    results["cases"] = cases
    results["misses"] = [case for case in cases if not case["hit"]]
    results["source_hit_at_k"] = (
        sum(1 for case in cases if case["source_hit"]) / len(cases) if cases else 0
    )
    return results


def write_markdown_report(path, baseline: dict, optimized: dict = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [("Baseline", baseline)]
    if optimized:
        rows.append(("Optimized", optimized))

    lines = [
        "# Retrieval Evaluation Report",
        "",
        "| Run | Questions | K | Recall@K | MRR | Precision@K | SourceHit@K | Misses |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, data in rows:
        lines.append(
            f"| {name} | {data['questions']} | {data['k']} | "
            f"{data['recall_at_k']:.3f} | {data['mrr']:.3f} | "
            f"{data['precision_at_k']:.3f} | {data.get('source_hit_at_k', 0):.3f} | "
            f"{len(data.get('misses', []))} |"
        )

    for name, data in rows:
        misses = data.get("misses", [])
        lines.extend(["", f"## {name} Misses", ""])
        if not misses:
            lines.append("No misses.")
            continue
        lines.extend([
            "| ID | Question | Expected | Source | Retrieved Count |",
            "| --- | --- | --- | --- | ---: |",
        ])
        for case in misses:
            question = str(case["question"]).replace("|", "\\|")
            expected = json.dumps(case["expected"], ensure_ascii=False).replace("|", "\\|")
            source = str(case["expected_source"]).replace("|", "\\|")
            lines.append(
                f"| {case['id']} | {question} | {expected} | {source} | {case['retrieved_count']} |"
            )

        lines.extend(["", f"### {name} Retrieved Details", ""])
        for case in misses:
            lines.append(f"**{case['id']} - {case['question']}**")
            lines.append("")
            lines.append("| Rank | Source | Score | Preview |")
            lines.append("| ---: | --- | ---: | --- |")
            for item in case.get("retrieved", []):
                preview = str(item["preview"]).replace("|", "\\|")
                source = str(item["source"]).replace("|", "\\|")
                score = "" if item["score"] is None else f"{item['score']:.4f}"
                lines.append(f"| {item['rank']} | {source} | {score} | {preview} |")
            lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        print(f"    SourceHit@{data['k']}:   {data.get('source_hit_at_k', 0):.3f}")

    _print("BASELINE (Vector Only)", baseline)
    if optimized:
        _print("\nOPTIMIZED (Full Phase 2)", optimized)
        print(f"\n  Improvement:")
        print(f"    Recall@{baseline['k']}:     {optimized['recall_at_k']-baseline['recall_at_k']:+.3f}")
        print(f"    MRR:            {optimized['mrr']-baseline['mrr']:+.3f}")
        print(f"    Precision@{baseline['k']}:   {optimized['precision_at_k']-baseline['precision_at_k']:+.3f}")
        print(f"    SourceHit@{baseline['k']}:   {optimized.get('source_hit_at_k', 0)-baseline.get('source_hit_at_k', 0):+.3f}")


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
    report_path = Path("evaluation/reports/latest.md")
    write_markdown_report(report_path, baseline, optimized)
    print(f"\nSaved detailed report to {report_path}")
