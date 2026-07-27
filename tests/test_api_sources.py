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
