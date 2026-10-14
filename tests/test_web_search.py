import pytest
import requests

from app import web_search


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_search_posts_normalized_tavily_results(monkeypatch):
    calls = {}

    def fake_post(url, *, headers, json, timeout):
        calls.update(url=url, headers=headers, json=json, timeout=timeout)
        return FakeResponse({"results": [{
            "title": "RAG",
            "url": "https://example.com/rag",
            "content": "摘要",
            "score": 0.9,
        }]})

    monkeypatch.setattr(web_search.config, "tavily_api_key", "test-key")
    monkeypatch.setattr(web_search.requests, "post", fake_post)

    results = web_search.search("RAG 最新进展")

    assert calls["url"] == web_search.TAVILY_SEARCH_URL
    assert calls["headers"]["Authorization"] == "Bearer test-key"
    assert calls["json"]["query"] == "RAG 最新进展"
    assert calls["json"]["max_results"] == web_search.config.tavily_max_results
    assert calls["timeout"] == web_search.config.tavily_timeout_seconds
    assert results == [{"title": "RAG", "url": "https://example.com/rag", "content": "摘要", "score": 0.9}]


def test_search_rejects_missing_api_key(monkeypatch):
    monkeypatch.setattr(web_search.config, "tavily_api_key", "")

    with pytest.raises(web_search.WebSearchError, match="TAVILY_API_KEY"):
        web_search.search("RAG")


@pytest.mark.parametrize(
    ("error", "code", "message"),
    [
        (requests.Timeout(), "timeout", "联网搜索请求超时"),
        (requests.RequestException(), "request_error", "联网搜索服务暂不可用"),
    ],
)
def test_search_translates_request_errors(monkeypatch, error, code, message):
    monkeypatch.setattr(web_search.config, "tavily_api_key", "test-key")
    monkeypatch.setattr(
        web_search.requests,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(web_search.WebSearchError, match=message) as raised:
        web_search.search("RAG")

    assert raised.value.code == code


def test_search_rejects_invalid_payload(monkeypatch):
    monkeypatch.setattr(web_search.config, "tavily_api_key", "test-key")
    monkeypatch.setattr(
        web_search.requests, "post", lambda *args, **kwargs: FakeResponse({"results": {}})
    )

    with pytest.raises(web_search.WebSearchError, match="结果格式异常"):
        web_search.search("RAG")


def test_search_rejects_non_numeric_result_score(monkeypatch):
    monkeypatch.setattr(web_search.config, "tavily_api_key", "test-key")
    monkeypatch.setattr(
        web_search.requests,
        "post",
        lambda *args, **kwargs: FakeResponse({"results": [{
            "url": "https://example.com/rag",
            "score": "not-a-number",
        }]}),
    )

    with pytest.raises(web_search.WebSearchError, match="结果格式异常") as raised:
        web_search.search("RAG")

    assert raised.value.code == "invalid_response"


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_search_rejects_non_finite_result_score(monkeypatch, score):
    monkeypatch.setattr(web_search.config, "tavily_api_key", "test-key")
    monkeypatch.setattr(
        web_search.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(
            {"results": [{"url": "https://example.com/rag", "score": score}]}
        ),
    )

    with pytest.raises(web_search.WebSearchError, match="结果格式异常") as raised:
        web_search.search("RAG")

    assert raised.value.code == "invalid_response"


def test_search_returns_empty_list_for_empty_results(monkeypatch):
    monkeypatch.setattr(web_search.config, "tavily_api_key", "test-key")
    monkeypatch.setattr(
        web_search.requests, "post", lambda *args, **kwargs: FakeResponse({"results": []})
    )

    assert web_search.search("RAG") == []
