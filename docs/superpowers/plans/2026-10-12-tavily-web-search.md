# Tavily Web Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 Agent 模式增加 Tavily 只读联网搜索工具，并在流式对话和证据面板中展示可靠的网页来源。

**Architecture:** 新建 `app.web_search` 负责唯一的 Tavily HTTP 调用、响应验证与错误分类；`app.tools` 负责将其转换成统一的工具元数据；`app.tool_agent` 只负责工具选择与流式事件。前端继续接收现有 `meta.sources`，通过 `source_type="web"` 和安全 HTTPS 链接区分网页证据。

**Tech Stack:** Python 3.11、FastAPI、OpenAI-compatible Function Calling、`requests`、Tavily Search API、pytest、原生 JavaScript。

## Global Constraints

- 仅允许向 `https://api.tavily.com/search` 发送只读 POST 请求。
- `TAVILY_API_KEY` 仅存在于本地或服务器 `.env`，不得写入源代码、测试、README 示例中的真实值或 Git 历史。
- 默认 `TAVILY_MAX_RESULTS=5`、`TAVILY_TIMEOUT_SECONDS=10`，模型不得覆盖结果数量上限。
- 严格 RAG 模式、向量库格式和既有 `/query/stream` 协议不得改变。
- 所有面向用户的状态与错误使用中文；回答继续避免 `###` 与无序的重复编号。
- 使用测试先行：每个任务先运行失败测试，再实现最小修复并单独提交。

---

## File Structure

- Create: `app/web_search.py` - Tavily HTTP 客户端、响应规范化与可控异常。
- Modify: `app/config.py` - Tavily 环境变量配置。
- Modify: `app/tools.py` - `search_web` 工具定义与统一来源元数据。
- Modify: `app/tool_agent.py` - 联网搜索标签、系统提示词与来源合并键。
- Modify: `static/index.html` - 网页来源证据说明与安全 HTTPS 外链。
- Modify: `.env.example` - 不含真实值的 Tavily 配置模板。
- Modify: `README.md` - 本地配置、工具行为与 Docker 环境变量说明。
- Modify: `tests/test_config.py` - Tavily 配置默认值与环境变量覆盖。
- Create: `tests/test_web_search.py` - Tavily 客户端成功、缺失密钥、超时、HTTP/格式/空结果测试。
- Modify: `tests/test_tools.py` - `search_web` 元数据与工具定义测试。
- Modify: `tests/test_tool_agent.py` - Agent 调用联网搜索时的流式事件与来源测试。
- Modify: `tests/test_static_workbench.py` - 前端网页来源提示与安全链接回归测试。

## Shared Interfaces

```python
# app/web_search.py
class WebSearchError(RuntimeError):
    def __init__(self, code: str, message: str) -> None: ...

def search(query: str) -> list[dict]: ...

# app/tools.py
def execute_with_metadata(name: str, arguments: str) -> dict: ...
# search_web 成功时：
# {"name": "search_web", "content": str, "sources": list[dict],
#  "top_score": float | None, "error": str | None}
```

网页来源结构：

```python
{
    "source": "https://example.com/article",
    "filename": "网页：示例标题",
    "title": "示例标题",
    "url": "https://example.com/article",
    "preview": "摘要文本",
    "score": 0.87,
    "source_type": "web",
}
```

### Task 1: Tavily 配置与独立搜索客户端

**Files:**
- Create: `app/web_search.py`
- Modify: `app/config.py`
- Modify: `tests/test_config.py`
- Create: `tests/test_web_search.py`

**Interfaces:**
- Consumes: `app.config.config.tavily_api_key`、`tavily_max_results`、`tavily_timeout_seconds`。
- Produces: `web_search.search(query: str) -> list[dict]`，或抛出 `WebSearchError(code, message)`。

- [ ] **Step 1: 写入失败的配置与客户端测试**

```python
# tests/test_config.py
def test_tavily_config_can_be_configured_from_environment(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setenv("TAVILY_MAX_RESULTS", "3")
    monkeypatch.setenv("TAVILY_TIMEOUT_SECONDS", "7")

    cfg = Config()

    assert cfg.tavily_api_key == "test-key"
    assert cfg.tavily_max_results == 3
    assert cfg.tavily_timeout_seconds == 7

# tests/test_web_search.py
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
    assert calls["json"]["max_results"] == web_search.config.tavily_max_results
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


def test_search_returns_empty_list_for_empty_results(monkeypatch):
    monkeypatch.setattr(web_search.config, "tavily_api_key", "test-key")
    monkeypatch.setattr(
        web_search.requests, "post", lambda *args, **kwargs: FakeResponse({"results": []})
    )

    assert web_search.search("RAG") == []
```

- [ ] **Step 2: 运行测试，确认尚未实现而失败**

Run: `..\RAG-project\.venv-conda\python.exe -m pytest tests/test_config.py tests/test_web_search.py -q`

Expected: 失败，原因是 `Config` 没有 Tavily 字段且 `app.web_search` 不存在。

- [ ] **Step 3: 实现最小配置和 HTTP 客户端**

在 `app/config.py` 的 `Config` 中增加：

```python
tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
tavily_max_results: int = field(default_factory=lambda: int(os.getenv("TAVILY_MAX_RESULTS", "5")))
tavily_timeout_seconds: int = field(
    default_factory=lambda: int(os.getenv("TAVILY_TIMEOUT_SECONDS", "10"))
)
```

创建 `app/web_search.py`：

```python
"""Tavily 搜索客户端，不包含 Agent 或前端逻辑。"""

import requests

from app.config import config

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


class WebSearchError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def search(query: str) -> list[dict]:
    if not config.tavily_api_key:
        raise WebSearchError("missing_api_key", "联网搜索未配置，请设置 TAVILY_API_KEY。")
    try:
        response = requests.post(
            TAVILY_SEARCH_URL,
            headers={"Authorization": f"Bearer {config.tavily_api_key}"},
            json={"query": query, "max_results": config.tavily_max_results},
            timeout=config.tavily_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.Timeout as exc:
        raise WebSearchError("timeout", "联网搜索请求超时，请稍后重试。") from exc
    except requests.RequestException as exc:
        raise WebSearchError("request_error", "联网搜索服务暂不可用，请稍后重试。") from exc
    except ValueError as exc:
        raise WebSearchError("invalid_response", "联网搜索结果格式异常，请稍后重试。") from exc

    raw_results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(raw_results, list):
        raise WebSearchError("invalid_response", "联网搜索结果格式异常，请稍后重试。")
    return [
        {
            "title": str(item.get("title") or "未命名网页"),
            "url": str(item.get("url") or ""),
            "content": str(item.get("content") or ""),
            "score": float(item.get("score") or 0),
        }
        for item in raw_results[: config.tavily_max_results]
        if isinstance(item, dict) and item.get("url")
    ]
```

所有客户端测试通过模拟 `requests.post` 覆盖，不发送真实网络请求。

- [ ] **Step 4: 运行定向测试，确认通过**

Run: `..\RAG-project\.venv-conda\python.exe -m pytest tests/test_config.py tests/test_web_search.py -q`

Expected: PASS，且不会访问真实 Tavily API。

- [ ] **Step 5: 提交任务**

```powershell
git add app/config.py app/web_search.py tests/test_config.py tests/test_web_search.py
git commit -m "feat: add Tavily web search client"
```

### Task 2: 将联网搜索注册为统一工具

**Files:**
- Modify: `app/tools.py`
- Modify: `tests/test_tools.py`

**Interfaces:**
- Consumes: `app.web_search.search(query)` 的标准化结果或 `WebSearchError`。
- Produces: `execute_with_metadata("search_web", arguments)` 的统一来源元数据与中文错误。

- [ ] **Step 1: 写入失败的工具测试**

```python
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
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `..\RAG-project\.venv-conda\python.exe -m pytest tests/test_tools.py -q`

Expected: 失败，因为 `search_web` 还未被 `execute_with_metadata` 识别。

- [ ] **Step 3: 实现工具定义与元数据转换**

在 `get_definitions()` 增加：

```python
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
}
```

新增 `_search_web_with_metadata(query: str) -> dict`：调用 `web_search.search`，把每项映射为共享来源结构，并用下列文本传给模型：

```text
[网页来源 1] 标题
URL: https://example.com/rag
摘要: RAG 的公开摘要。
```

在 `execute_with_metadata()` 的知识库分支后加入 `search_web` 分支；只捕获 `WebSearchError` 并将其 `code` 写入 `error` 字段，其他异常返回既有 `execution_error`。空结果返回 `error=None` 和“联网搜索未找到可用结果。”，不构造来源。

- [ ] **Step 4: 运行定向测试，确认通过**

Run: `..\RAG-project\.venv-conda\python.exe -m pytest tests/test_tools.py -q`

Expected: PASS，既有知识库、参数错误和未知工具测试仍通过。

- [ ] **Step 5: 提交任务**

```powershell
git add app/tools.py tests/test_tools.py
git commit -m "feat: expose web search as an agent tool"
```

### Task 3: Agent 工具轨迹、网页来源合并与证据面板

**Files:**
- Modify: `app/tool_agent.py`
- Modify: `static/index.html`
- Modify: `tests/test_tool_agent.py`
- Modify: `tests/test_static_workbench.py`

**Interfaces:**
- Consumes: 工具层的 `search_web` 结果和 `source_type="web"` 来源。
- Produces: `tool` 事件中的“联网搜索”、包含网页来源的 `meta` 事件，以及安全外链证据卡片。

- [ ] **Step 1: 写入失败的 Agent 与前端测试**

```python
# tests/test_tool_agent.py
def test_stream_exposes_web_tool_sources(monkeypatch):
    responses = iter([
        _response(tool_calls=[_tool_call("search_web", '{"query":"RAG 最新进展"}')]),
        _response(),
    ])
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
    monkeypatch.setattr(tool_agent.tools, "execute_with_metadata", lambda name, arguments: {
        "name": name,
        "content": "[网页来源 1] RAG",
        "sources": [{"url": "https://example.com/rag", "source_type": "web"}],
        "top_score": 0.8,
        "error": None,
    })

    events = list(tool_agent.chat_stream_events("近期 RAG 进展"))

    assert any(event["message"] == "正在执行：联网搜索..." for event in events)
    meta = next(event for event in events if event["event"] == "meta")
    assert meta["tools"] == ["联网搜索"]
    assert meta["sources"][0]["source_type"] == "web"

# tests/test_static_workbench.py
def test_workbench_displays_safe_web_sources():
    html = Path("static/index.html").read_text(encoding="utf-8")

    assert "source_type === \"web\"" in html
    assert "function isSafeExternalUrl" in html
    assert 'link.rel = "noopener noreferrer"' in html
    assert "联网搜索" in html
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `..\RAG-project\.venv-conda\python.exe -m pytest tests/test_tool_agent.py tests/test_static_workbench.py -q`

Expected: 失败，因为 Agent 标签和网页来源的安全链接逻辑尚不存在。

- [ ] **Step 3: 实现 Agent 和前端最小改动**

在 `app/tool_agent.py`：

```python
_TOOL_LABELS = {
    "search_knowledge_base": "知识库检索",
    "search_web": "联网搜索",
    "get_current_time": "获取当前时间",
}
```

将系统提示词扩展为：项目资料优先 `search_knowledge_base`；时效性或外部公开事实使用 `search_web`；并保持“模型自主选择工具”的原则。

调整 `_merge_sources()` 的去重键：网页来源使用 `source["url"]`，本地来源继续使用 `source` 或 `filename`。这样同一网页不会重复出现，且不会错误合并同名本地文件与网页。

在 `static/index.html` 增加：

```javascript
function isSafeExternalUrl(value) {
  try {
    return new URL(value).protocol === "https:";
  } catch (_) {
    return false;
  }
}
```

在 `updateEvidence(data)` 的 `tool_agent` 分支中，根据 `data.tools.includes("联网搜索")` 显示“Agent 已调用联网搜索，并返回网页来源。”。渲染来源卡片时，仅对 `src.source_type === "web" && isSafeExternalUrl(src.url)` 创建 `<a>`，并设置：

```javascript
link.href = src.url;
link.target = "_blank";
link.rel = "noopener noreferrer";
link.textContent = src.title || src.filename;
```

其他来源维持现有纯文本渲染；禁止使用 `innerHTML` 拼接 Tavily 返回内容。

- [ ] **Step 4: 运行定向测试，确认通过**

Run: `..\RAG-project\.venv-conda\python.exe -m pytest tests/test_tool_agent.py tests/test_static_workbench.py -q`

Expected: PASS，既有知识库、时间工具、错误事件和移动端静态测试继续通过。

- [ ] **Step 5: 提交任务**

```powershell
git add app/tool_agent.py static/index.html tests/test_tool_agent.py tests/test_static_workbench.py
git commit -m "feat: show web search evidence in agent chat"
```

### Task 4: 配置示例、文档与本地验收

**Files:**
- Modify: `.env.example`
- Modify: `README.md`
- Modify: `tests/test_deployment_demo.py`

**Interfaces:**
- Consumes: `TAVILY_API_KEY`、`TAVILY_MAX_RESULTS`、`TAVILY_TIMEOUT_SECONDS`。
- Produces: 可复现的本地配置说明和 Docker 环境变量说明；不包含真实密钥。

- [ ] **Step 1: 写入失败的文档回归测试**

```python
def test_demo_documentation_mentions_optional_tavily_configuration():
    readme = Path("README.md").read_text(encoding="utf-8")
    env_example = Path(".env.example").read_text(encoding="utf-8")

    assert "TAVILY_API_KEY" in readme
    assert "TAVILY_API_KEY=" in env_example
    assert "tvly-" not in env_example
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `..\RAG-project\.venv-conda\python.exe -m pytest tests/test_deployment_demo.py -q`

Expected: 失败，因为当前文档没有 Tavily 配置说明。

- [ ] **Step 3: 更新示例与 README**

在 `.env.example` 增加：

```text
# 可选：仅 Agent 联网搜索工具使用；留空时联网搜索会返回配置提示。
TAVILY_API_KEY=
TAVILY_MAX_RESULTS=5
TAVILY_TIMEOUT_SECONDS=10
```

在 README 的本地 `.env` 和 Docker `docker run` 示例补充同名变量；说明：严格 RAG 不联网，Agent 模式仅在模型调用 `search_web` 时请求 Tavily，真实 Key 不应提交。更新功能说明，准确表述联网搜索为 V2.1 的可选工具，而非默认启用的能力。

在开发机的实际 `.env` 中新增 `TAVILY_API_KEY`，仅通过编辑本地被忽略的文件完成，绝不使用 `git add .env`。

- [ ] **Step 4: 运行完整自动化验证**

Run: `..\RAG-project\.venv-conda\python.exe -m pytest -q -rA`

Expected: PASS，所有既有和新增测试通过。

- [ ] **Step 5: 执行本地真实验收**

Run: `..\RAG-project\.venv-conda\python.exe main.py --serve`

在另一个 PowerShell 窗口运行：

```powershell
curl.exe -s -N -H "Content-Type: application/json" --data-raw "{\"question\":\"近期 RAG 领域有哪些值得关注的进展？\"}" http://127.0.0.1:8000/chat/stream
```

Expected: 输出依次包含 `planning`、`search_web` 的 `tool` 事件、带“联网搜索”的 `meta`、至少一个 `source_type: "web"` 来源、`token` 与 `done`。浏览器证据面板可打开 HTTPS 网页链接。

- [ ] **Step 6: 提交任务**

```powershell
git add .env.example README.md tests/test_deployment_demo.py
git commit -m "docs: describe optional Tavily web search setup"
```

## Final Verification

- [ ] 运行 `git diff --check`，预期无输出。
- [ ] 运行 `..\RAG-project\.venv-conda\python.exe -m pytest -q -rA`，预期全部通过。
- [ ] 运行 `git status --short`，确认仅出现预期的已忽略 `.env`、向量库或运行日志，不得出现被跟踪的密钥文件。
- [ ] 使用 `git log --oneline -4` 确认四个任务的提交记录完整。
