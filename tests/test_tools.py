from langchain_core.documents import Document

from app import tools
from app.web_search import WebSearchError


def test_search_knowledge_base_returns_sources(monkeypatch):
    doc = Document(
        page_content="LCEL 是 LangChain 的表达式语言。",
        metadata={"source": "data/lcel.md", "filename": "lcel.md", "score": 0.88},
    )
    monkeypatch.setattr("app.query._retrieve_docs", lambda query: [doc])

    result = tools.execute_with_metadata(
        "search_knowledge_base", '{"query": "LCEL 是什么"}'
    )

    assert result["error"] is None
    assert result["name"] == "search_knowledge_base"
    assert "LCEL" in result["content"]
    assert result["top_score"] == 0.88
    assert result["sources"][0]["filename"] == "lcel.md"


def test_search_web_returns_web_sources(monkeypatch):
    monkeypatch.setattr(
        "app.web_search.search",
        lambda query: [{
            "title": "RAG 文章",
            "url": "https://example.com/rag",
            "content": "RAG 的公开摘要。",
            "score": 0.91,
        }],
    )

    result = tools.execute_with_metadata("search_web", '{"query":"RAG"}')

    assert result["error"] is None
    assert result["sources"] == [{
        "source": "https://example.com/rag",
        "filename": "网页：RAG 文章",
        "title": "RAG 文章",
        "url": "https://example.com/rag",
        "preview": "RAG 的公开摘要。",
        "score": 0.91,
        "source_type": "web",
    }]
    assert "https://example.com/rag" in result["content"]


def test_search_web_returns_controlled_tavily_error(monkeypatch):
    monkeypatch.setattr(
        "app.web_search.search",
        lambda query: (_ for _ in ()).throw(
            WebSearchError("missing_api_key", "联网搜索未配置，请设置 TAVILY_API_KEY。")
        ),
    )

    result = tools.execute_with_metadata("search_web", '{"query":"RAG"}')

    assert result["error"] == "missing_api_key"
    assert result["content"] == "联网搜索未配置，请设置 TAVILY_API_KEY。"


def test_search_web_returns_empty_result_message(monkeypatch):
    monkeypatch.setattr("app.web_search.search", lambda query: [])

    result = tools.execute_with_metadata("search_web", '{"query":"RAG"}')

    assert result["error"] is None
    assert result["sources"] == []
    assert result["content"] == "联网搜索未找到可用结果。"


def test_invalid_tool_arguments_return_controlled_error():
    result = tools.execute_with_metadata("get_current_time", "{")

    assert result["content"].startswith("工具参数错误")
    assert result["error"] == "invalid_arguments"


def test_unknown_tool_returns_controlled_error():
    result = tools.execute_with_metadata("unknown", "{}")

    assert result["content"].startswith("未知工具")
    assert result["error"] == "unknown_tool"
