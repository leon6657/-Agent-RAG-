"""Tool definitions for function calling (OpenAI-compatible format)."""

import datetime
import json
import logging

_TOOL_REGISTRY = {}
logger = logging.getLogger(__name__)


def _register(func, name=None):
    _TOOL_REGISTRY[name or func.__name__] = func
    return func


@_register
def get_current_time() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")


@_register
def search_knowledge_base(query: str) -> str:
    from app.query import _search
    try:
        return _search(query)
    except ValueError:
        return "No relevant notes found."
    except Exception as e:
        return f"Search error: {e}"


def get_definitions():
    return [
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current date and time / 获取当前日期和时间",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_knowledge_base",
                "description": "Search local notes / 搜索本地笔记",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search keywords"}
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search public web information for timely or external facts / 搜索公开网页信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search keywords"}
                    },
                    "required": ["query"],
                },
            },
        },
    ]


def _result(
    name: str,
    content: str,
    *,
    sources: list[dict] | None = None,
    top_score: float | None = None,
    error: str | None = None,
) -> dict:
    return {
        "name": name,
        "content": content,
        "sources": sources or [],
        "top_score": top_score,
        "error": error,
    }


def _search_knowledge_base_with_metadata(query: str) -> dict:
    from app.query import _format_context, _retrieve_docs, _source_payload, _top_score

    docs = _retrieve_docs(query)
    return _result(
        "search_knowledge_base",
        _format_context(docs),
        sources=_source_payload(docs),
        top_score=_top_score(docs),
    )


def _search_web_with_metadata(query: str) -> dict:
    from app.web_search import search

    results = search(query)
    if not results:
        return _result("search_web", "联网搜索未找到可用结果。")

    sources = [
        {
            "source": result["url"],
            "filename": f"网页：{result['title']}",
            "title": result["title"],
            "url": result["url"],
            "preview": result["content"],
            "score": result["score"],
            "source_type": "web",
        }
        for result in results
    ]
    content = "\n\n".join(
        f"[网页来源 {index}] {result['title']}\n"
        f"URL: {result['url']}\n"
        f"摘要: {result['content']}"
        for index, result in enumerate(results, start=1)
    )
    return _result("search_web", content, sources=sources)


def execute_with_metadata(name: str, arguments: str) -> dict:
    try:
        args = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError:
        return _result(name, "工具参数错误：参数必须是合法 JSON。", error="invalid_arguments")

    if name == "search_knowledge_base":
        try:
            return _search_knowledge_base_with_metadata(**args)
        except Exception:
            logger.error("Tool execution failed: %s", name)
            return _result(name, "工具执行失败，请稍后重试。", error="execution_error")

    if name == "search_web":
        from app.web_search import WebSearchError

        try:
            return _search_web_with_metadata(**args)
        except WebSearchError as exc:
            return _result(name, str(exc), error=exc.code)
        except Exception:
            logger.error("Tool execution failed: %s", name)
            return _result(name, "工具执行失败，请稍后重试。", error="execution_error")

    func = _TOOL_REGISTRY.get(name)
    if not func:
        return _result(name, f"未知工具：{name}", error="unknown_tool")
    try:
        return _result(name, str(func(**args)))
    except Exception:
        logger.error("Tool execution failed: %s", name)
        return _result(name, "工具执行失败，请稍后重试。", error="execution_error")


def execute(name: str, arguments: str) -> str:
    return execute_with_metadata(name, arguments)["content"]
