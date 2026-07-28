from evaluation.answer_runner import run_answer_evaluation


def test_run_answer_evaluation_uses_ask_function_results():
    questions = [
        {
            "id": "q1",
            "q": "What is RAG?",
            "a": ["retrieval"],
            "s": "rag.md",
        }
    ]

    def fake_ask(question):
        assert question == "What is RAG?"
        return {
            "answer": "RAG uses retrieval.\n\nSources: rag.md",
            "sources": [{"filename": "rag.md", "preview": "RAG uses retrieval."}],
            "mode": "rag",
            "top_score": 0.88,
        }

    results = run_answer_evaluation(fake_ask, questions)

    assert results["count"] == 1
    assert results["correctness"] == 1.0
    assert results["cases"][0]["id"] == "q1"
    assert results["cases"][0]["mode"] == "rag"
