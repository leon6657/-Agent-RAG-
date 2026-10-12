from types import SimpleNamespace

from app import tool_agent


def test_system_prompt_requires_chinese_readable_format():
    assert "请使用中文回答" in tool_agent._SYSTEM_PROMPT
    assert "不要使用 ###" in tool_agent._SYSTEM_PROMPT
    assert "连续编号" in tool_agent._SYSTEM_PROMPT


def _tool_call(name, arguments, call_id="call_1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _response(*, content=None, tool_calls=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls or [])
            )
        ]
    )


def _stream_response(text):
    return iter(
        [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=text))]
            )
        ]
    )


def test_stream_calls_kb_tool_and_exposes_sources(monkeypatch):
    responses = iter(
        [
            _response(
                tool_calls=[
                    _tool_call("search_knowledge_base", '{"query":"LCEL"}')
                ]
            ),
            _response(),
        ]
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: next(responses)
                if not kwargs.get("stream")
                else _stream_response("LCEL 是表达式语言。")
            )
        )
    )
    monkeypatch.setattr(tool_agent, "_get_client", lambda: client)
    monkeypatch.setattr(
        tool_agent.tools,
        "execute_with_metadata",
        lambda name, arguments: {
            "name": name,
            "content": "[data/lcel.md]\nLCEL 是表达式语言。",
            "sources": [
                {"filename": "lcel.md", "score": 0.88, "preview": "LCEL"}
            ],
            "top_score": 0.88,
            "error": None,
        },
    )

    events = list(tool_agent.chat_stream_events("LCEL 是什么？"))

    assert any(
        event["event"] == "tool" and event["name"] == "search_knowledge_base"
        for event in events
    )
    meta = next(event for event in events if event["event"] == "meta")
    assert meta["mode"] == "tool_agent"
    assert meta["sources"][0]["filename"] == "lcel.md"
    assert meta["tools"] == ["知识库检索"]


def test_stream_exposes_web_tool_sources(monkeypatch):
    responses = iter(
        [
            _response(
                tool_calls=[_tool_call("search_web", '{"query":"RAG 最新进展"}')]
            ),
            _response(),
        ]
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: next(responses)
                if not kwargs.get("stream")
                else _stream_response("已检索网页。")
            )
        )
    )
    monkeypatch.setattr(tool_agent, "_get_client", lambda: client)
    monkeypatch.setattr(
        tool_agent.tools,
        "execute_with_metadata",
        lambda name, arguments: {
            "name": name,
            "content": "[网页来源 1] RAG",
            "sources": [{"url": "https://example.com/rag", "source_type": "web"}],
            "top_score": 0.8,
            "error": None,
        },
    )

    events = list(tool_agent.chat_stream_events("近期 RAG 进展"))

    assert any(
        event["event"] == "tool" and event["message"] == "正在执行：联网搜索..."
        for event in events
    )
    meta = next(event for event in events if event["event"] == "meta")
    assert meta["tools"] == ["联网搜索"]
    assert meta["sources"][0]["source_type"] == "web"


def test_merge_sources_deduplicates_web_sources_by_url():
    first = {"url": "https://example.com/rag", "source_type": "web"}
    duplicate = dict(first)
    second = {"url": "https://example.com/agent", "source_type": "web"}

    merged = tool_agent._merge_sources([first], [duplicate, second])

    assert merged == [first, second]


def test_stream_answers_directly_when_model_needs_no_tool(monkeypatch):
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: _response()
                if not kwargs.get("stream")
                else _stream_response("你好。")
            )
        )
    )
    monkeypatch.setattr(tool_agent, "_get_client", lambda: client)

    events = list(tool_agent.chat_stream_events("你好"))

    assert {"event": "token", "text": "你好。"} in events
    meta = next(event for event in events if event["event"] == "meta")
    assert meta["tools"] == []


def test_stream_returns_time_tool_to_model(monkeypatch):
    responses = iter(
        [_response(tool_calls=[_tool_call("get_current_time", "{}")]), _response()]
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: next(responses)
                if not kwargs.get("stream")
                else _stream_response("当前时间是 10:00。")
            )
        )
    )
    monkeypatch.setattr(tool_agent, "_get_client", lambda: client)
    monkeypatch.setattr(
        tool_agent.tools,
        "execute_with_metadata",
        lambda name, arguments: {
            "name": name,
            "content": "2026-10-11 10:00:00",
            "sources": [],
            "top_score": None,
            "error": None,
        },
    )

    events = list(tool_agent.chat_stream_events("现在几点？"))

    meta = next(event for event in events if event["event"] == "meta")
    assert meta["tools"] == ["获取当前时间"]
    assert any(event["event"] == "token" for event in events)


def test_stream_stops_after_maximum_tool_iterations(monkeypatch):
    endless = _response(tool_calls=[_tool_call("get_current_time", "{}")])
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: endless))
    )
    monkeypatch.setattr(tool_agent, "_get_client", lambda: client)
    monkeypatch.setattr(
        tool_agent.tools,
        "execute_with_metadata",
        lambda name, arguments: {
            "name": name,
            "content": "2026-10-11",
            "sources": [],
            "top_score": None,
            "error": None,
        },
    )

    events = list(tool_agent.chat_stream_events("现在几点？"))

    assert events[-1]["event"] == "error"
    assert "三轮" in events[-1]["message"]
