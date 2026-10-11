from fastapi.testclient import TestClient
from langchain_core.documents import Document

import api


def test_query_endpoint_returns_sources(monkeypatch):
    expected = {
        "answer": "Use dict.get(key, default).",
        "mode": "rag",
        "top_score": 0.91,
        "sources": [
            {
                "source": "data/data-structures.md",
                "filename": "data-structures.md",
                "score": 0.91,
                "preview": "dict.get(key, default) returns a fallback.",
            }
        ],
    }

    def fake_ask_with_sources(question):
        assert question == "How do I use dict.get?"
        return expected

    monkeypatch.setattr("app.query.ask_with_sources", fake_ask_with_sources)

    client = TestClient(api.app)
    response = client.post("/query", json={"question": "How do I use dict.get?"})

    assert response.status_code == 200
    assert response.json() == expected


def test_query_stream_endpoint_returns_ndjson_events(monkeypatch):
    def fake_stream(question):
        assert question == "What is RAG?"
        yield {
            "event": "meta",
            "mode": "rag",
            "top_score": 0.88,
            "sources": [{"filename": "rag.md", "score": 0.88, "preview": "RAG"}],
        }
        yield {"event": "token", "text": "RAG "}
        yield {"event": "token", "text": "answer"}
        yield {"event": "done"}

    monkeypatch.setattr("app.query.ask_stream_events", fake_stream)

    client = TestClient(api.app)
    response = client.post("/query/stream", json={"question": "What is RAG?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    lines = response.text.strip().splitlines()
    assert '"event": "meta"' in lines[0]
    assert '"text": "RAG "' in lines[1]
    assert '"event": "done"' in lines[-1]


def test_chat_stream_endpoint_returns_ndjson_events(monkeypatch):
    def fake_stream(question):
        assert question == "Tell me something"
        yield {"event": "meta", "mode": "agent", "sources": [], "top_score": None}
        yield {"event": "token", "text": "Agent "}
        yield {"event": "token", "text": "answer"}
        yield {"event": "done"}

    monkeypatch.setattr("app.agent.chat_stream_events", fake_stream)

    client = TestClient(api.app)
    response = client.post("/chat/stream", json={"question": "Tell me something"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    lines = response.text.strip().splitlines()
    assert '"mode": "agent"' in lines[0]
    assert '"text": "Agent "' in lines[1]
    assert '"event": "done"' in lines[-1]


def test_chat_stream_preserves_tool_events(monkeypatch):
    def fake_stream(question):
        assert question == "现在几点？"
        yield {
            "event": "status",
            "stage": "planning",
            "message": "正在分析问题并选择工具...",
        }
        yield {
            "event": "tool",
            "name": "get_current_time",
            "status": "running",
            "message": "正在执行：获取当前时间...",
        }
        yield {
            "event": "meta",
            "mode": "tool_agent",
            "sources": [],
            "top_score": None,
            "tools": ["获取当前时间"],
            "retrieval_ms": 0,
        }
        yield {"event": "token", "text": "当前时间是 10:00。"}
        yield {
            "event": "done",
            "retrieval_ms": 0,
            "generation_ms": 10,
            "total_ms": 10,
            "tools": ["获取当前时间"],
        }

    monkeypatch.setattr("app.agent.chat_stream_events", fake_stream)
    response = TestClient(api.app).post("/chat/stream", json={"question": "现在几点？"})

    assert response.status_code == 200
    assert '"event": "tool"' in response.text
    assert '"mode": "tool_agent"' in response.text
