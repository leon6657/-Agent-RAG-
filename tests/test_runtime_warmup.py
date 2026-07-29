from fastapi.testclient import TestClient

import api


def test_embedding_runtime_reuses_model_and_query_vectors(monkeypatch):
    from app import runtime

    runtime.reset_runtime_caches()
    calls = {"build": 0, "embed": 0}

    class FakeEmbeddings:
        def embed_query(self, query):
            calls["embed"] += 1
            return [float(len(query))]

    def fake_build_embeddings():
        calls["build"] += 1
        return FakeEmbeddings()

    monkeypatch.setattr(runtime, "build_embeddings", fake_build_embeddings)

    assert runtime.embed_query_cached("same question") == [13.0]
    assert runtime.embed_query_cached("same question") == [13.0]
    assert runtime.embed_query_cached("other") == [5.0]

    assert calls == {"build": 1, "embed": 2}


def test_warmup_endpoint_returns_runtime_status(monkeypatch):
    calls = {"warmup": 0}

    def fake_warmup():
        calls["warmup"] += 1
        return {
            "embedding_model": "ready",
            "vector_store": "ready",
            "vector_count": 83,
        }

    monkeypatch.setattr("app.runtime.warmup_runtime", fake_warmup)

    response = TestClient(api.app).post("/warmup")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "embedding_model": "ready",
        "vector_store": "ready",
        "vector_count": 83,
    }
    assert calls["warmup"] == 1
