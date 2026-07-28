from evaluation.answer_quality import evaluate_answer_cases, write_answer_quality_report


def test_evaluate_answer_cases_scores_answer_quality():
    cases = [
        {
            "id": "ok",
            "question": "What is RAG?",
            "expected": ["retrieval", "generation"],
            "answer": "RAG combines retrieval with generation.\n\nSources: rag.md",
            "sources": [
                {
                    "filename": "rag.md",
                    "preview": "RAG combines retrieval with generation.",
                }
            ],
            "mode": "rag",
        },
        {
            "id": "refusal",
            "question": "What is outside the notes?",
            "expected": [],
            "answer": "Knowledge base does not contain enough reliable evidence.",
            "sources": [],
            "mode": "no_context",
        },
    ]

    results = evaluate_answer_cases(cases)

    assert results["count"] == 2
    assert results["correctness"] == 0.5
    assert results["faithfulness"] == 0.5
    assert results["citation_coverage"] == 0.5
    assert results["refusal_accuracy"] == 1.0
    assert results["cases"][0]["correct"] is True
    assert results["cases"][0]["faithful"] is True
    assert results["cases"][0]["cited"] is True
    assert results["cases"][1]["correct_refusal"] is True


def test_write_answer_quality_report(tmp_path):
    results = {
        "count": 1,
        "correctness": 1.0,
        "faithfulness": 1.0,
        "citation_coverage": 1.0,
        "refusal_accuracy": 0.0,
        "cases": [
            {
                "id": "ok",
                "question": "What is RAG?",
                "correct": True,
                "faithful": True,
                "cited": True,
                "correct_refusal": None,
                "mode": "rag",
            }
        ],
    }
    path = tmp_path / "answer-quality.md"

    write_answer_quality_report(path, results)

    text = path.read_text(encoding="utf-8")
    assert "# Answer Quality Evaluation Report" in text
    assert "| Correctness | 1.000 |" in text
    assert "| ok | rag | yes | yes | yes | n/a | What is RAG? |" in text
