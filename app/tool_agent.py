"""Tool-calling Agent with streaming events for the web application."""

from time import perf_counter

import openai

from app import tools
from app.config import config
from app.memory import SimpleMemory

_SYSTEM_PROMPT = """你是一个可以使用工具的中文助手。

当回答依赖项目资料或用户知识库时，优先调用 search_knowledge_base。
当回答依赖时效性信息或外部公开事实时，调用 search_web。
当被问及当前时间或日期时，调用 get_current_time。
一般性对话可直接回答。模型自主选择工具；需要时必须选择合适的工具。

请使用中文回答，结构清晰、便于阅读。不要使用 ### 标题，也不要堆叠无意义的星号。
如需列举多个步骤或要点，使用连续编号。"""

_MAX_ITERATIONS = 3
_TOOL_LABELS = {
    "search_knowledge_base": "知识库检索",
    "search_web": "联网搜索",
    "get_current_time": "获取当前时间",
}
memory = SimpleMemory(window_size=5)


def _get_client():
    return openai.OpenAI(
        api_key=config.deepseek_api_key,
        base_url=config.deepseek_api_base,
    )


def _build_messages(question: str) -> list:
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    history = memory.get_history()
    if history:
        messages.append({"role": "user", "content": f"Previous conversation:\n{history}"})
        messages.append({"role": "assistant", "content": "Got it. Continue."})
    messages.append({"role": "user", "content": question})
    return messages


def _tool_message(tool_call) -> dict:
    return {
        "id": tool_call.id,
        "type": tool_call.type,
        "function": {
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments,
        },
    }


def _stream_final_answer(client, messages):
    stream = client.chat.completions.create(
        model=config.deepseek_model,
        messages=messages,
        temperature=0.3,
        max_tokens=2048,
        tool_choice="none",
        stream=True,
    )
    for chunk in stream:
        text = getattr(chunk.choices[0].delta, "content", None)
        if text:
            yield text


def _merge_sources(existing: list[dict], incoming: list[dict]) -> list[dict]:
    def source_key(source: dict):
        if source.get("source_type") == "web":
            return ("web", source.get("url"))
        return ("local", source.get("source") or source.get("filename"))

    merged = list(existing)
    seen = {source_key(source) for source in existing}
    for source in incoming:
        key = source_key(source)
        if key not in seen:
            merged.append(source)
            seen.add(key)
    return merged


def chat_stream_events(message: str):
    """Yield NDJSON-ready events while the model selects and uses tools."""
    total_started = perf_counter()
    sources: list[dict] = []
    tool_labels: list[str] = []
    top_score: float | None = None
    retrieval_ms = 0

    yield {
        "event": "status",
        "stage": "planning",
        "message": "正在分析问题并选择工具...",
    }

    try:
        client = _get_client()
        messages = _build_messages(message)

        for _ in range(_MAX_ITERATIONS):
            decision = client.chat.completions.create(
                model=config.deepseek_model,
                messages=messages,
                temperature=0.3,
                max_tokens=512,
                tools=tools.get_definitions(),
                tool_choice="auto",
            ).choices[0].message
            tool_calls = getattr(decision, "tool_calls", None) or []
            if not tool_calls:
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": decision.content or "",
                    "tool_calls": [_tool_message(call) for call in tool_calls],
                }
            )

            for call in tool_calls:
                name = call.function.name
                label = _TOOL_LABELS.get(name, name)
                yield {
                    "event": "tool",
                    "name": name,
                    "status": "running",
                    "message": f"正在执行：{label}...",
                }
                started = perf_counter()
                result = tools.execute_with_metadata(name, call.function.arguments)

                if name in {"search_knowledge_base", "search_web"}:
                    sources = _merge_sources(sources, result["sources"])

                if name == "search_knowledge_base":
                    retrieval_ms += int((perf_counter() - started) * 1000)
                    result_top_score = result["top_score"]
                    if result_top_score is not None:
                        top_score = max(top_score or result_top_score, result_top_score)

                if label not in tool_labels:
                    tool_labels.append(label)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result["content"],
                    }
                )
                yield {
                    "event": "tool",
                    "name": name,
                    "status": "completed",
                    "message": f"已完成：{label}",
                }
        else:
            yield {
                "event": "error",
                "message": "工具调用已达到三轮上限，请换一种问法后重试。",
            }
            return

        yield {
            "event": "meta",
            "mode": "tool_agent",
            "sources": sources,
            "top_score": top_score,
            "tools": tool_labels,
            "retrieval_ms": retrieval_ms,
        }
        yield {
            "event": "status",
            "stage": "generating",
            "message": "正在基于工具结果生成回答...",
        }

        answer_parts = []
        generation_started = perf_counter()
        for text in _stream_final_answer(client, messages):
            answer_parts.append(text)
            yield {"event": "token", "text": text}

        answer = "".join(answer_parts)
        memory.add_user(message)
        memory.add_assistant(answer)
        yield {
            "event": "done",
            "retrieval_ms": retrieval_ms,
            "generation_ms": int((perf_counter() - generation_started) * 1000),
            "total_ms": int((perf_counter() - total_started) * 1000),
            "tools": tool_labels,
        }
    except Exception as exc:
        yield {"event": "error", "message": f"Agent 执行失败：{exc}"}


def chat(message: str) -> str:
    answer_parts = []
    error_message = ""
    for event in chat_stream_events(message):
        if event["event"] == "token":
            answer_parts.append(event["text"])
        elif event["event"] == "error":
            error_message = event["message"]
    return "".join(answer_parts) or error_message
