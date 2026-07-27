from langchain_core.documents import Document

from evaluation.runner import run_evaluation, write_markdown_report


def test_run_evaluation_records_case_diagnostics():
    questions = [
        {"id": "q1", "q": "What defines a Python function?", "a": ["def keyword", "function definition"], "s": "python-basics.md"},
        {"id": "q2", "q": "What is RAG?", "a": "retrieval augmented generation", "s": "rag.md"},
    ]

    def fake_search(query: str, k: int = 4) -> list:
        if "Python" in query:
            return [
                Document(
                    page_content="A Python function uses the def keyword.",
                    metadata={"filename": "python-basics.md", "score": 0.9},
                )
            ]
        return [Document(page_content="This chunk is unrelated.", metadata={"filename": "other.md", "score": 0.1})]

    results = run_evaluation(fake_search, questions, k=4)

    assert results["questions"] == 2
    assert results["recall_at_k"] == 0.5
    assert results["source_hit_at_k"] == 0.5
    assert results["cases"] == [
        {
            "id": "q1",
            "question": "What defines a Python function?",
            "expected": ["def keyword", "function definition"],
            "expected_source": "python-basics.md",
            "hit": True,
            "rank": 1,
            "source_hit": True,
            "source_rank": 1,
            "retrieved_count": 1,
            "retrieved": [
                {
                    "rank": 1,
                    "source": "python-basics.md",
                    "score": 0.9,
                    "preview": "A Python function uses the def keyword.",
                }
            ],
        },
        {
            "id": "q2",
            "question": "What is RAG?",
            "expected": "retrieval augmented generation",
            "expected_source": "rag.md",
            "hit": False,
            "rank": None,
            "source_hit": False,
            "source_rank": None,
            "retrieved_count": 1,
            "retrieved": [
                {
                    "rank": 1,
                    "source": "other.md",
                    "score": 0.1,
                    "preview": "This chunk is unrelated.",
                }
            ],
        },
    ]
    assert results["misses"] == [results["cases"][1]]


def test_write_markdown_report_saves_summary_and_misses(tmp_path):
    baseline = {
        "questions": 2,
        "k": 4,
        "recall_at_k": 0.5,
        "mrr": 0.5,
        "precision_at_k": 0.125,
        "source_hit_at_k": 0.5,
        "misses": [
            {
                "id": "q2",
                "question": "What is RAG?",
                "expected": "retrieval augmented generation",
                "expected_source": "rag.md",
                "hit": False,
                "rank": None,
                "source_hit": False,
                "source_rank": None,
                "retrieved_count": 1,
                "retrieved": [
                    {
                        "rank": 1,
                        "source": "other.md",
                        "score": 0.1,
                        "preview": "This chunk is unrelated.",
                    }
                ],
            }
        ],
    }
    optimized = {**baseline, "recall_at_k": 1.0, "mrr": 1.0, "precision_at_k": 0.25, "source_hit_at_k": 1.0, "misses": []}
    output = tmp_path / "report.md"

    write_markdown_report(output, baseline, optimized)

    text = output.read_text(encoding="utf-8")
    assert "# Retrieval Evaluation Report" in text
    assert "| Baseline | 2 | 4 | 0.500 | 0.500 | 0.125 | 0.500 | 1 |" in text
    assert "| Optimized | 2 | 4 | 1.000 | 1.000 | 0.250 | 1.000 | 0 |" in text
    assert "q2" in text
    assert "What is RAG?" in text
    assert "other.md" in text
    assert "This chunk is unrelated." in text
