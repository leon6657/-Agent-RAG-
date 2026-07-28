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
