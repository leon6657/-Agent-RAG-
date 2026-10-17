import json
import math

import pytest
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


@pytest.mark.parametrize(
    ("endpoint", "stream_path", "question"),
    [
        ("/query/stream", "app.query.ask_stream_events", "严格 RAG 问题"),
        ("/chat/stream", "app.agent.chat_stream_events", "Agent 问题"),
    ],
)
def test_stream_endpoints_replace_upstream_iteration_errors_with_valid_json_error(
    monkeypatch, endpoint, stream_path, question
):
    def fake_stream(received_question):
        assert received_question == question
        yield {"event": "status", "message": "正在处理..."}
        raise RuntimeError("internal diagnostic detail")

    monkeypatch.setattr(stream_path, fake_stream)

    response = TestClient(api.app).post(endpoint, json={"question": question})

    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.strip().splitlines()]
    assert events == [
        {"event": "status", "message": "正在处理..."},
        {"event": "error", "message": "流式响应服务暂时不可用，请稍后重试。"},
    ]
    assert "internal diagnostic detail" not in response.text


@pytest.mark.parametrize(
    ("endpoint", "stream_path", "question"),
    [
        ("/query/stream", "app.query.ask_stream_events", "严格 RAG 问题"),
        ("/chat/stream", "app.agent.chat_stream_events", "Agent 问题"),
    ],
)
@pytest.mark.parametrize("non_finite_value", [math.nan, math.inf, -math.inf])
def test_stream_endpoints_replace_non_finite_events_with_valid_json_error(
    monkeypatch, endpoint, stream_path, question, non_finite_value
):
    def fake_stream(received_question):
        assert received_question == question
        yield {
            "event": "meta",
            "mode": "rag",
            "top_score": non_finite_value,
            "sources": [{"score": non_finite_value}],
        }

    monkeypatch.setattr(stream_path, fake_stream)

    response = TestClient(api.app).post(endpoint, json={"question": question})

    assert response.status_code == 200
    lines = response.text.strip().splitlines()
    events = [json.loads(line) for line in lines]
    assert events == [{"event": "error", "message": "流式响应数据异常，请稍后重试。"}]
    assert all(math.isfinite(value) for event in events for value in _floats(event))


@pytest.mark.parametrize("endpoint", ["/query", "/query/stream"])
def test_strict_rag_endpoints_use_real_query_path_without_tavily(monkeypatch, endpoint):
    class FakeChain:
        def __or__(self, other):
            return self

        def invoke(self, payload):
            assert "仅查询本地知识库" == payload["question"]
            return "本地回答"

        def stream(self, payload):
            assert "仅查询本地知识库" == payload["question"]
            yield "本地流式回答"

    class FakePrompt:
        def __or__(self, other):
            return FakeChain()

    def fail_if_tavily_or_web_search_called(*args, **kwargs):
        raise AssertionError("strict RAG route must not call Tavily or web search")

    doc = Document(
        page_content="仅来自本地知识库的内容。",
        metadata={"source": "data/local.md", "filename": "local.md", "score": 0.8},
    )
    monkeypatch.setattr("app.web_search.search", fail_if_tavily_or_web_search_called)
    monkeypatch.setattr("app.web_search.requests.post", fail_if_tavily_or_web_search_called)
    monkeypatch.setattr("app.query.store.count", lambda: 1)
    monkeypatch.setattr("app.query.embed_query_cached", lambda question: [0.1])
    monkeypatch.setattr("app.query.store.search_cached", lambda vector, k: [doc])
    monkeypatch.setattr("app.query.build_llm", lambda: object())
    monkeypatch.setattr(
        "app.query.ChatPromptTemplate.from_template", lambda template: FakePrompt()
    )
    monkeypatch.setattr("app.query.config.rag_min_score", 0.35)

    response = TestClient(api.app).post(endpoint, json={"question": "仅查询本地知识库"})

    assert response.status_code == 200
    if endpoint == "/query":
        assert response.json()["answer"] == "本地回答\n\nSources: local.md"
        assert response.json()["mode"] == "rag"
    else:
        events = [json.loads(line) for line in response.text.strip().splitlines()]
        assert any(event == {"event": "token", "text": "本地流式回答"} for event in events)
        assert events[-1]["event"] == "done"


def _floats(value):
    if isinstance(value, float):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _floats(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _floats(nested)
