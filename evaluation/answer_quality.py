"""Heuristic answer-quality evaluation for RAG outputs."""

from pathlib import Path
from typing import Any


def _terms(expected: Any) -> list[str]:
    if expected is None:
        return []
    if isinstance(expected, str):
        return [expected]
    terms = []
    for item in expected:
        if isinstance(item, str):
            terms.append(item)
        else:
            terms.extend(_terms(item))
    return terms


def _is_correct(answer: str, expected: Any) -> bool:
    terms = [term.lower() for term in _terms(expected)]
    if not terms:
        return False
    text = answer.lower()
    return any(term in text for term in terms)


def _source_text(sources: list[dict]) -> str:
    parts = []
    for source in sources:
        parts.append(str(source.get("preview", "")))
        parts.append(str(source.get("filename", "")))
        parts.append(str(source.get("source", "")))
    return " ".join(parts).lower()


def _is_faithful(answer: str, sources: list[dict], expected: Any) -> bool:
    source_text = _source_text(sources)
    if not source_text:
        return False
    return any(term.lower() in source_text for term in _terms(expected))


def _is_cited(answer: str, sources: list[dict]) -> bool:
    if not sources:
        return False
    text = answer.lower()
    return any(
        str(source.get("filename") or source.get("source") or "").lower() in text
        for source in sources
    )


def _is_refusal_case(case: dict) -> bool:
    return not _terms(case.get("expected") or case.get("a"))


def _is_correct_refusal(case: dict) -> bool:
    if not _is_refusal_case(case):
        return False
    answer = str(case.get("answer", "")).lower()
    mode = case.get("mode")
    refusal_markers = [
        "no_context",
        "not contain",
        "not enough",
        "does not contain",
        "\u6ca1\u6709\u627e\u5230",
        "\u4e0d\u8db3",
    ]
    return mode == "no_context" or any(marker in answer for marker in refusal_markers)


def _case_result(case: dict) -> dict:
    expected = case.get("expected", case.get("a"))
    answer = str(case.get("answer", ""))
    sources = case.get("sources", [])
    is_refusal = _is_refusal_case(case)
    return {
        "id": case.get("id", ""),
        "question": case.get("question", case.get("q", "")),
        "mode": case.get("mode", ""),
        "correct": None if is_refusal else _is_correct(answer, expected),
        "faithful": None if is_refusal else _is_faithful(answer, sources, expected),
        "cited": None if is_refusal else _is_cited(answer, sources),
        "correct_refusal": _is_correct_refusal(case) if is_refusal else None,
    }


def _mean(values: list[bool]) -> float:
    return sum(1 for value in values if value) / len(values) if values else 0.0


def evaluate_answer_cases(cases: list[dict]) -> dict:
    case_results = [_case_result(case) for case in cases]
    refusal_cases = [case for case in case_results if case["correct_refusal"] is not None]
    return {
        "count": len(case_results),
        "correctness": _mean([case["correct"] is True for case in case_results]),
        "faithfulness": _mean([case["faithful"] is True for case in case_results]),
        "citation_coverage": _mean([case["cited"] is True for case in case_results]),
        "refusal_accuracy": _mean([case["correct_refusal"] for case in refusal_cases]),
        "cases": case_results,
    }


def _yes_no(value) -> str:
    if value is None:
        return "n/a"
    return "yes" if value else "no"


def write_answer_quality_report(path, results: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Answer Quality Evaluation Report",
        "",
        "| Metric | Score |",
        "| --- | ---: |",
        f"| Correctness | {results['correctness']:.3f} |",
        f"| Faithfulness | {results['faithfulness']:.3f} |",
        f"| Citation Coverage | {results['citation_coverage']:.3f} |",
        f"| Refusal Accuracy | {results['refusal_accuracy']:.3f} |",
        "",
        "| ID | Mode | Correct | Faithful | Cited | Correct Refusal | Question |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for case in results.get("cases", []):
        question = str(case.get("question", "")).replace("|", "\\|")
        lines.append(
            f"| {case.get('id', '')} | {case.get('mode', '')} | "
            f"{_yes_no(case.get('correct'))} | {_yes_no(case.get('faithful'))} | "
            f"{_yes_no(case.get('cited'))} | {_yes_no(case.get('correct_refusal'))} | "
            f"{question} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
