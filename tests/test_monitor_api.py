from pathlib import Path

from fastapi.testclient import TestClient

import api


def test_reports_endpoint_reads_markdown_reports(tmp_path, monkeypatch):
    reports_dir = tmp_path / "evaluation" / "reports"
    reports_dir.mkdir(parents=True)
    (reports_dir / "latest.md").write_text("# Retrieval\n\nok", encoding="utf-8")
    (reports_dir / "answer_quality.md").write_text("# Answer\n\nok", encoding="utf-8")
    monkeypatch.setattr(api, "ROOT_DIR", tmp_path)

    response = TestClient(api.app).get("/reports")

    assert response.status_code == 200
    data = response.json()
    assert data["reports"]["latest.md"].startswith("# Retrieval")
    assert data["reports"]["answer_quality.md"].startswith("# Answer")


def test_kb_files_endpoint_lists_markdown_files(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "rag.md").write_text("RAG notes", encoding="utf-8")
    (data_dir / "ignore.txt").write_text("ignore", encoding="utf-8")
    monkeypatch.setattr(api.config, "data_dir", str(data_dir))

    response = TestClient(api.app).get("/kb/files")

    assert response.status_code == 200
    assert response.json()["files"][0]["name"] == "rag.md"
    assert response.json()["files"][0]["size"] > 0


def test_recent_logs_endpoint_returns_tail(tmp_path, monkeypatch):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "rag.log").write_text("a\nb\nc\n", encoding="utf-8")
    monkeypatch.setattr(api, "ROOT_DIR", tmp_path)

    response = TestClient(api.app).get("/logs/recent?limit=2")

    assert response.status_code == 200
    assert response.json() == {"lines": ["b", "c"]}


def test_recent_logs_endpoint_returns_helpful_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "ROOT_DIR", tmp_path)

    response = TestClient(api.app).get("/logs/recent")

    assert response.status_code == 200
    assert "暂未产生日志" in response.json()["lines"][0]


def test_metrics_endpoint_combines_report_store_and_last_interaction(tmp_path, monkeypatch):
    reports_dir = tmp_path / "evaluation" / "reports"
    reports_dir.mkdir(parents=True)
    (reports_dir / "latest.md").write_text(
        "\n".join(
            [
                "# Retrieval Evaluation Report",
                "",
                "| Run | Questions | K | Recall@K | MRR | Precision@K | SourceHit@K | Misses |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                "| Baseline | 30 | 4 | 0.900 | 0.800 | 0.200 | 0.900 | 3 |",
                "| Optimized | 30 | 4 | 1.000 | 0.944 | 0.250 | 1.000 | 0 |",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "ROOT_DIR", tmp_path)
    monkeypatch.setattr("app.store.count", lambda: 344)
    api.record_interaction(
        question="LCEL 是什么？",
        mode="rag",
        top_score=0.8123,
        sources_count=4,
    )

    response = TestClient(api.app).get("/metrics")

    assert response.status_code == 200
    data = response.json()
    assert data["retrieval"]["run"] == "Optimized"
    assert data["retrieval"]["recall"] == 1.0
    assert data["retrieval"]["mrr"] == 0.944
    assert data["knowledge_base"]["chunks"] == 344
    assert data["last_answer"]["question"] == "LCEL 是什么？"
    assert data["last_answer"]["top_score"] == 0.8123


def test_query_endpoint_records_recent_log_and_metric_state(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(
        "app.query.ask_with_sources",
        lambda question: {
            "answer": "ok",
            "sources": [{"filename": "demo.md", "score": 0.91}],
            "mode": "rag",
            "top_score": 0.91,
        },
    )

    response = TestClient(api.app).post("/query", json={"question": "测试问题"})

    assert response.status_code == 200
    lines = TestClient(api.app).get("/logs/recent").json()["lines"]
    assert any("query" in line and "测试问题" in line for line in lines)
    metrics = TestClient(api.app).get("/metrics").json()
    assert metrics["last_answer"]["question"] == "测试问题"
    assert metrics["last_answer"]["sources_count"] == 1


def test_config_endpoint_is_read_only_and_redacts_secret(monkeypatch):
    monkeypatch.setattr(api.config, "deepseek_api_key", "sk-secret-value")

    response = TestClient(api.app).get("/config")

    assert response.status_code == 200
    data = response.json()
    assert data["deepseek_api_key"] == "set"
    assert data["retrieval_top_k"] == api.config.retrieval_top_k


def test_ingest_endpoint_returns_status(monkeypatch):
    monkeypatch.setattr("app.ingest.run_ingest", lambda: 7)

    response = TestClient(api.app).post("/ingest")

    assert response.status_code == 200
    assert response.json() == {"chunks": 7, "status": "ok"}
