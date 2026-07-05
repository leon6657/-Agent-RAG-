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
                "description": "Get the current date and time",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_knowledge_base",
                "description": "Search local notes",
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


def execute(name: str, arguments: str) -> str:
    args = json.loads(arguments) if arguments else {}
    func = _TOOL_REGISTRY.get(name)
    if not func:
        return f"Unknown tool: {name}"
    try:
        return func(**args)
    except Exception as e:
        return f"Execution error: {e}"
