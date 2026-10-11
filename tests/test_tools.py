from langchain_core.documents import Document

from app import tools


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


def test_invalid_tool_arguments_return_controlled_error():
    result = tools.execute_with_metadata("get_current_time", "{")

    assert result["content"].startswith("工具参数错误")
    assert result["error"] == "invalid_arguments"


def test_unknown_tool_returns_controlled_error():
    result = tools.execute_with_metadata("unknown", "{}")

    assert result["content"].startswith("未知工具")
    assert result["error"] == "unknown_tool"
