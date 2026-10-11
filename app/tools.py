"""Tool definitions for function calling (OpenAI-compatible format)."""

import datetime
import json

_TOOL_REGISTRY = {}


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


def execute_with_metadata(name: str, arguments: str) -> dict:
    try:
        args = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError:
        return _result(name, "工具参数错误：参数必须是合法 JSON。", error="invalid_arguments")

    if name == "search_knowledge_base":
        try:
            return _search_knowledge_base_with_metadata(**args)
        except Exception as exc:
            return _result(name, f"工具执行失败：{exc}", error="execution_error")

    func = _TOOL_REGISTRY.get(name)
    if not func:
        return _result(name, f"未知工具：{name}", error="unknown_tool")
    try:
        return _result(name, str(func(**args)))
    except Exception as exc:
        return _result(name, f"工具执行失败：{exc}", error="execution_error")


def execute(name: str, arguments: str) -> str:
    return execute_with_metadata(name, arguments)["content"]
