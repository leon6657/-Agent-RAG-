from fastapi.testclient import TestClient

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
