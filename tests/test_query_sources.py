from langchain_core.documents import Document

from app import query


def test_ask_with_sources_returns_answer_and_citations(monkeypatch):
    docs = [
        Document(
            page_content="RAG combines retrieval with generation.",
            metadata={"filename": "rag.md", "source": "data/rag.md", "score": 0.82},
        )
    ]

    class FakeChain:
        def __or__(self, other):
            return self

        def invoke(self, payload):
            assert "RAG combines retrieval" in payload["context"]
            return "RAG combines retrieval with generation."

    class FakePrompt:
        def __or__(self, other):
            return FakeChain()

    monkeypatch.setattr(query, "_retrieve_docs", lambda question: docs)
    monkeypatch.setattr(query.ChatPromptTemplate, "from_template", lambda template: FakePrompt())
    monkeypatch.setattr(query, "build_llm", lambda: object())
    monkeypatch.setattr(query.config, "rag_min_score", 0.35)

    result = query.ask_with_sources("What is RAG?")

    assert result["mode"] == "rag"
    assert result["answer"] == "RAG combines retrieval with generation."
    assert result["top_score"] == 0.82
    assert result["sources"] == [
        {
            "source": "data/rag.md",
            "filename": "rag.md",
            "score": 0.82,
            "preview": "RAG combines retrieval with generation.",
        }
    ]


def test_ask_with_sources_refuses_when_context_score_is_low(monkeypatch):
    docs = [
        Document(
            page_content="This chunk is unrelated.",
            metadata={"filename": "other.md", "score": 0.12},
        )
    ]
    called = {"llm": False}

    def fake_llm():
        called["llm"] = True
        return object()

    monkeypatch.setattr(query, "_retrieve_docs", lambda question: docs)
    monkeypatch.setattr(query, "build_llm", fake_llm)
    monkeypatch.setattr(query.config, "rag_min_score", 0.35)

    result = query.ask_with_sources("How do I deploy this?")

    assert result["mode"] == "no_context"
    assert result["answer"] == query.NO_CONTEXT_ANSWER
    assert result["top_score"] == 0.12
    assert result["sources"][0]["filename"] == "other.md"
    assert called["llm"] is False
