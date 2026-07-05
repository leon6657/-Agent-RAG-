"""Tool agent using OpenAI-compatible function calling."""

import json
import openai

from app.config import config
from app.memory import SimpleMemory
from app import tools

_SYSTEM_PROMPT = """You are a helpful assistant with access to tools.

When you need information from the user's notes, use search_knowledge_base.
When asked about the current time, use get_current_time.
For general conversation, answer directly.

Always use the appropriate tool when needed."""

_MAX_ITERATIONS = 3
memory = SimpleMemory(window_size=5)


def _get_client():
    return openai.OpenAI(
        api_key=config.deepseek_api_key,
        base_url=config.deepseek_api_base,
    )


def _build_messages(question: str) -> list:
    msgs = [{"role": "system", "content": _SYSTEM_PROMPT}]
    history = memory.get_history()
    if history:
        msgs.append({"role": "user", "content": f"Previous conversation:\n{history}"})
        msgs.append({"role": "assistant", "content": "Got it. Continue."})
    msgs.append({"role": "user", "content": question})
    return msgs


def chat(message: str) -> str:
    client = _get_client()
    messages = _build_messages(message)

    for iteration in range(_MAX_ITERATIONS):
        kwargs = {
            "model": config.deepseek_model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 2048,
        }
        kwargs["tools"] = tools.get_definitions()
        kwargs["tool_choice"] = "auto"

        resp = client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message

        if not msg.tool_calls:
            result = msg.content or ""
            memory.add_user(message)
            memory.add_assistant(result)
            return result

        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            args = tc.function.arguments
            output = tools.execute(name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(output),
            })

    return "Max iterations reached."
