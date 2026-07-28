"""Run answer-quality evaluation against the RAG question set."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["HF_HOME"] = str(Path(__file__).resolve().parent.parent / ".hf_cache")

from evaluation.answer_quality import evaluate_answer_cases, write_answer_quality_report
from evaluation.runner import load_questions


def run_answer_evaluation(ask_fn, questions: list[dict]) -> dict:
    cases = []
    for question in questions:
        result = ask_fn(question["q"])
        cases.append(
            {
                "id": question.get("id", ""),
                "question": question["q"],
                "expected": question.get("a", []),
                "expected_source": question.get("s", ""),
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
                "mode": result.get("mode", ""),
                "top_score": result.get("top_score"),
            }
        )
    return evaluate_answer_cases(cases)


def print_answer_report(results: dict) -> None:
    print("=" * 55)
    print("  ANSWER QUALITY EVALUATION REPORT")
    print("=" * 55)
    print(f"  Questions:          {results['count']}")
    print(f"  Correctness:        {results['correctness']:.3f}")
    print(f"  Faithfulness:       {results['faithfulness']:.3f}")
    print(f"  Citation Coverage:  {results['citation_coverage']:.3f}")
    print(f"  Refusal Accuracy:   {results['refusal_accuracy']:.3f}")


if __name__ == "__main__":
    from app.query import ask_with_sources

    questions = load_questions()
    results = run_answer_evaluation(ask_with_sources, questions)
    report_path = Path("evaluation/reports/answer_quality.md")
    write_answer_quality_report(report_path, results)
    print_answer_report(results)
    print(f"\nSaved answer-quality report to {report_path}")
