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
    assert result["answer"] == "RAG combines retrieval with generation.\n\nSources: rag.md"
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


def test_ask_with_sources_does_not_duplicate_existing_source_line(monkeypatch):
    docs = [
        Document(
            page_content="LCEL composes LangChain runnables with pipe syntax.",
            metadata={"filename": "langchain.md", "score": 0.77},
        )
    ]

    class FakeChain:
        def __or__(self, other):
            return self

        def invoke(self, payload):
            return "LCEL composes runnables.\n\nSources: langchain.md"

    class FakePrompt:
        def __or__(self, other):
            return FakeChain()

    monkeypatch.setattr(query, "_retrieve_docs", lambda question: docs)
    monkeypatch.setattr(query.ChatPromptTemplate, "from_template", lambda template: FakePrompt())
    monkeypatch.setattr(query, "build_llm", lambda: object())
    monkeypatch.setattr(query.config, "rag_min_score", 0.35)

    result = query.ask_with_sources("What is LCEL?")

    assert result["answer"].count("Sources:") == 1
    assert result["answer"].endswith("Sources: langchain.md")
