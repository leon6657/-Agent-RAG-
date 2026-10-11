# V2 网页端真实工具调用 Agent 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 将网页 Agent 模式接入可追踪、最多三轮的 Function Calling 工具循环，并保留现有的流式回答与来源展示。

**架构：** `app.tool_agent` 成为唯一的工具调用执行器：先以非流式请求让模型选择工具，执行只读工具并把结果回填消息，再以流式请求生成最终答案。`app.tools` 返回同时包含文本、来源和相似度的结构化结果；`app.agent` 只保留网页 API 的兼容导出。网页根据新增的 `tool` 事件展示实时状态，并在最终元数据中显示工具轨迹。

**技术栈：** Python 3.11、FastAPI、OpenAI 兼容 Chat Completions、DeepSeek、LangChain、pytest、原生 JavaScript。

## 全局约束

- 仅开放 `search_knowledge_base` 与 `get_current_time` 两个只读工具。
- 每次提问最多进行 3 轮模型决策与工具调用。
- 不修改严格 RAG 的 `/query` 与 `/query/stream` 行为。
- 继续使用 NDJSON 事件协议和中文网页文案。
- 不在本阶段接入联网搜索、写入工具、多 Agent 或持久化记忆。

---

## 文件职责

- `app/tools.py`：工具定义、参数解析、执行，以及带来源元数据的结构化工具结果。
- `app/tool_agent.py`：对话消息构造、工具调用循环、流式事件输出与短期记忆。
- `app/agent.py`：保留 `/chat` 既有导入路径，转发至 `app.tool_agent`。
- `static/index.html`：处理 `tool` 事件，在回答元数据中展示本次调用过的工具。
- `tests/test_tools.py`：工具的结构化结果、未知工具和非法 JSON 测试。
- `tests/test_tool_agent.py`：工具选择、来源传递、时间工具、轮次上限和异常流程测试。
- `tests/test_api_sources.py`：网页 Agent 的 NDJSON 兼容性测试。
- `tests/test_static_workbench.py`：网页端工具事件处理和工具轨迹文案测试。

### Task 1: 让工具返回可供 API 使用的结构化结果

**文件：**
- 新建：`tests/test_tools.py`
- 修改：`app/tools.py`
- 测试：`tests/test_tools.py`

**接口：**
- 产生：`execute_with_metadata(name: str, arguments: str) -> dict`
- 产生字段：`name`、`content`、`sources`、`top_score`、`error`
- 兼容：保留 `execute(name, arguments) -> str`，供已有 CLI 调用。

- [ ] **步骤 1：编写失败测试**

```python
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
```

- [ ] **步骤 2：运行测试，确认当前失败**

运行：

```powershell
.\.venv-conda\python.exe -m pytest tests\test_tools.py -q
```

预期：失败，提示 `execute_with_metadata` 尚未定义。

- [ ] **步骤 3：实现最小结构化工具接口**

在 `app/tools.py` 中新增默认结果和检索工具专用执行函数：

```python
def _result(name: str, content: str, *, sources=None, top_score=None, error=None) -> dict:
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
```

- [ ] **步骤 4：运行聚焦测试，确认通过**

运行：

```powershell
.\.venv-conda\python.exe -m pytest tests\test_tools.py -q
```

预期：3 个测试通过。

- [ ] **步骤 5：提交本任务**

```powershell
git add app/tools.py tests/test_tools.py
git commit -m "feat: add structured tool execution results"
```

### Task 2: 实现受控的网页工具调用循环

**文件：**
- 新建：`tests/test_tool_agent.py`
- 修改：`app/tool_agent.py`
- 修改：`app/agent.py`
- 测试：`tests/test_tool_agent.py`

**接口：**
- 产生：`chat_stream_events(message: str)`，依次输出 `status`、`tool`、`meta`、`token`、`done` 事件。
- `meta` 字段包含：`mode="tool_agent"`、`sources`、`top_score`、`tools`、`retrieval_ms`。
- `tool` 字段包含：`name`、`status`、`message`。
- 消费：`tools.execute_with_metadata(name, arguments)`。

- [ ] **步骤 1：编写失败测试**

```python
from types import SimpleNamespace

from app import tool_agent


def _tool_call(name, arguments, call_id="call_1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _response(*, content=None, tool_calls=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls or []))]
    )


def test_stream_calls_kb_tool_and_exposes_sources(monkeypatch):
    responses = iter([_response(tool_calls=[_tool_call("search_knowledge_base", '{"query":"LCEL"}')])])
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: next(responses) if not kwargs.get("stream") else iter([
                    SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="LCEL 是表达式语言。"))])
                ])
            )
        )
    )
    monkeypatch.setattr(tool_agent, "_get_client", lambda: client)
    monkeypatch.setattr(
        tool_agent.tools,
        "execute_with_metadata",
        lambda name, arguments: {
            "name": name,
            "content": "[data/lcel.md]\\nLCEL 是表达式语言。",
            "sources": [{"filename": "lcel.md", "score": 0.88, "preview": "LCEL"}],
            "top_score": 0.88,
            "error": None,
        },
    )

    events = list(tool_agent.chat_stream_events("LCEL 是什么？"))

    assert any(event["event"] == "tool" and event["name"] == "search_knowledge_base" for event in events)
    meta = next(event for event in events if event["event"] == "meta")
    assert meta["mode"] == "tool_agent"
    assert meta["sources"][0]["filename"] == "lcel.md"
    assert meta["tools"] == ["知识库检索"]


def test_stream_stops_after_maximum_tool_iterations(monkeypatch):
    endless = _response(tool_calls=[_tool_call("get_current_time", "{}")])
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: endless))
    )
    monkeypatch.setattr(tool_agent, "_get_client", lambda: client)
    monkeypatch.setattr(
        tool_agent.tools,
        "execute_with_metadata",
        lambda name, arguments: {"name": name, "content": "2026-10-11", "sources": [], "top_score": None, "error": None},
    )

    events = list(tool_agent.chat_stream_events("现在几点？"))

    assert events[-1]["event"] == "error"
    assert "三轮" in events[-1]["message"]
```

- [ ] **步骤 2：运行测试，确认当前失败**

运行：

```powershell
.\.venv-conda\python.exe -m pytest tests\test_tool_agent.py -q
```

预期：失败，提示 `chat_stream_events` 尚未定义。

- [ ] **步骤 3：实现工具调用与流式输出**

在 `app/tool_agent.py` 中保留 `_get_client()`、`_build_messages()` 与 `memory`，并新增下列核心逻辑：

```python
_TOOL_LABELS = {
    "search_knowledge_base": "知识库检索",
    "get_current_time": "获取当前时间",
}


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


def chat_stream_events(message: str):
    total_started = perf_counter()
    client = _get_client()
    messages = _build_messages(message)
    sources, tool_labels, top_score = [], [], None
    retrieval_ms = 0
    yield {"event": "status", "stage": "planning", "message": "正在分析问题并选择工具..."}

    for _ in range(_MAX_ITERATIONS):
        decision = client.chat.completions.create(
            model=config.deepseek_model,
            messages=messages,
            temperature=0.3,
            max_tokens=512,
            tools=tools.get_definitions(),
            tool_choice="auto",
        ).choices[0].message
        if not decision.tool_calls:
            break
        messages.append({"role": "assistant", "content": decision.content or "", "tool_calls": [_tool_message(call) for call in decision.tool_calls]})
        for call in decision.tool_calls:
            name = call.function.name
            label = _TOOL_LABELS.get(name, name)
            yield {"event": "tool", "name": name, "status": "running", "message": f"正在执行：{label}..."}
            started = perf_counter()
            result = tools.execute_with_metadata(name, call.function.arguments)
            if name == "search_knowledge_base":
                retrieval_ms += int((perf_counter() - started) * 1000)
                sources = result["sources"]
                top_score = result["top_score"]
            if label not in tool_labels:
                tool_labels.append(label)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result["content"]})
            yield {"event": "tool", "name": name, "status": "completed", "message": f"已完成：{label}"}
    else:
        yield {"event": "error", "message": "工具调用已达到三轮上限，请换一种问法后重试。"}
        return

    yield {"event": "meta", "mode": "tool_agent", "sources": sources, "top_score": top_score, "tools": tool_labels, "retrieval_ms": retrieval_ms}
    yield {"event": "status", "stage": "generating", "message": "正在基于工具结果生成回答..."}
    answer_parts, generation_started = [], perf_counter()
    for text in _stream_final_answer(client, messages):
        answer_parts.append(text)
        yield {"event": "token", "text": text}
    answer = "".join(answer_parts)
    memory.add_user(message)
    memory.add_assistant(answer)
    yield {"event": "done", "retrieval_ms": retrieval_ms, "generation_ms": int((perf_counter() - generation_started) * 1000), "total_ms": int((perf_counter() - total_started) * 1000), "tools": tool_labels}
```

将 `chat(message)` 改为收集 `chat_stream_events(message)` 中的 `token` 文本；将
`app/agent.py` 收敛为：

```python
"""网页 Agent 的兼容入口。"""

from app.tool_agent import chat, chat_stream_events

__all__ = ["chat", "chat_stream_events"]
```

- [ ] **步骤 4：运行聚焦测试，确认通过**

运行：

```powershell
.\.venv-conda\python.exe -m pytest tests\test_tool_agent.py tests\test_tools.py -q
```

预期：工具选择、来源传递和三轮上限测试全部通过。

- [ ] **步骤 5：提交本任务**

```powershell
git add app/tools.py app/tool_agent.py app/agent.py tests/test_tools.py tests/test_tool_agent.py
git commit -m "feat: add streaming web tool agent loop"
```

### Task 3: 在网页端展示工具轨迹并保持 API 契约

**文件：**
- 修改：`static/index.html`
- 修改：`tests/test_api_sources.py`
- 修改：`tests/test_static_workbench.py`
- 测试：`tests/test_api_sources.py`、`tests/test_static_workbench.py`

**接口：**
- 消费：NDJSON 的 `tool` 事件，以及 `meta`、`done` 中的 `tools: list[str]`。
- 保持：`status`、`meta`、`token`、`done` 的既有处理。

- [ ] **步骤 1：编写失败测试**

在 `tests/test_api_sources.py` 增加：

```python
def test_chat_stream_preserves_tool_events(monkeypatch):
    def fake_stream(question):
        yield {"event": "status", "stage": "planning", "message": "正在分析问题并选择工具..."}
        yield {"event": "tool", "name": "get_current_time", "status": "running", "message": "正在执行：获取当前时间..."}
        yield {"event": "meta", "mode": "tool_agent", "sources": [], "top_score": None, "tools": ["获取当前时间"], "retrieval_ms": 0}
        yield {"event": "token", "text": "当前时间是 10:00。"}
        yield {"event": "done", "retrieval_ms": 0, "generation_ms": 10, "total_ms": 10, "tools": ["获取当前时间"]}

    monkeypatch.setattr("app.agent.chat_stream_events", fake_stream)
    response = TestClient(api.app).post("/chat/stream", json={"question": "现在几点？"})

    assert response.status_code == 200
    assert '"event": "tool"' in response.text
    assert '"mode": "tool_agent"' in response.text
```

同时删除或改写旧的 `test_agent_stream_returns_sources_when_using_kb`、
`test_agent_stream_keeps_retrieved_context_for_natural_language_question` 与
`test_agent_stream_falls_back_when_retrieval_score_is_low`。这些测试依赖
`app.agent.store`、`app.agent._stream_llm` 等 v1 内部实现；v2 应改为
mock `app.tool_agent._get_client` 与 `app.tool_agent.tools.execute_with_metadata`，
并断言 `tool_agent` 模式、`tool` 事件和来源元数据。

在 `tests/test_static_workbench.py` 增加：

```python
def test_workbench_displays_agent_tool_events():
    html = Path("static/index.html").read_text(encoding="utf-8")

    assert 'event.event === "tool"' in html
    assert '工具：' in html
    assert '正在执行：' in html
```

- [ ] **步骤 2：运行测试，确认当前失败**

运行：

```powershell
.\.venv-conda\python.exe -m pytest tests\test_api_sources.py tests\test_static_workbench.py -q
```

预期：静态网页测试失败，因为尚未处理 `tool` 事件与“工具：”文案。

- [ ] **步骤 3：实现网页工具状态与轨迹展示**

在 `static/index.html` 中扩展 `buildMetaText` 和 `handleStreamEvent`：

```javascript
function buildMetaText(meta) {
  if (!meta) return "Assistant";
  const parts = [meta.mode || "assistant"];
  if (typeof meta.top_score === "number") parts.push("top " + meta.top_score.toFixed(3));
  if (Array.isArray(meta.sources)) parts.push(meta.sources.length + " sources");
  if (Array.isArray(meta.tools) && meta.tools.length) parts.push("工具：" + meta.tools.join("、"));
  const timing = buildTimingText(meta);
  if (timing) parts.push(timing);
  return parts.join(" | ");
}

function handleStreamEvent(event, assistant, meta) {
  if (event.event === "status") {
    setAssistantStatus(assistant, event.message);
  } else if (event.event === "tool") {
    setAssistantStatus(assistant, event.message);
  } else if (event.event === "meta") {
    assistant.row.textContent = buildMetaText(event);
    updateEvidence(event);
    return event;
  } else if (event.event === "token") {
    appendAssistantToken(assistant, event.text);
  } else if (event.event === "done") {
    const nextMeta = Object.assign({}, meta, event);
    assistant.row.textContent = buildMetaText(nextMeta);
    return nextMeta;
  }
  return meta;
}
```

并在 `updateEvidence` 中为 `data.mode === "tool_agent"` 增加分支：知识库工具返回来源时显示“Agent 已调用知识库检索并返回来源”，未调用检索工具时显示“本次 Agent 未使用知识库检索工具”。保留其他模式原有文案。

- [ ] **步骤 4：运行聚焦测试，确认通过**

运行：

```powershell
.\.venv-conda\python.exe -m pytest tests\test_api_sources.py tests\test_static_workbench.py -q
```

预期：API NDJSON 与静态网页测试全部通过。

- [ ] **步骤 5：提交本任务**

```powershell
git add static/index.html tests/test_api_sources.py tests/test_static_workbench.py
git commit -m "feat: display tool activity in agent chat"
```

### Task 4: 进行全量回归并记录验证结果

**文件：**
- 修改：无
- 测试：`tests/`

**接口：**
- 验证：严格 RAG 与网页 Agent 接口均保持兼容。

- [ ] **步骤 1：运行完整测试套件**

```powershell
.\.venv-conda\python.exe -m pytest -q
```

预期：全部测试通过；如有第三方弃用警告，单独记录但不将其视为测试失败。

- [ ] **步骤 2：本地启动并人工验证**

```powershell
.\.venv-conda\python.exe main.py --serve
```

打开 `http://127.0.0.1:8000`，在 Agent 模式依次提问“LCEL 是什么？”和“现在几点？”。确认聊天区依次显示分析、工具执行、生成回答；知识库问题在右侧显示来源，时间问题显示“工具：获取当前时间”。
