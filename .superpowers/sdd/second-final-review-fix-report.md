# 第二轮最终审查修复报告

## 范围与约束

- 修改仅发生在 `E:\AGitHub-project\RAG-Agent-v2` 的 `v2-agent` 分支。
- 未读取、写入或输出 API 密钥；测试替换 Tavily 搜索函数，不发起真实网络请求。
- 未读取、写入或修改 `E:\AGitHub-project\RAG-project`，未做部署。

## 改动

1. `api.py`
   - 抽取 `_safe_ndjson_events()`，统一用于 `/query/stream` 与 `/chat/stream`。
   - 所有 SSE NDJSON 事件使用 `json.dumps(..., allow_nan=False)`。
   - 若事件含 `NaN`、`Infinity`、`-Infinity` 或其他不可 JSON 序列化值，输出固定的中文 `error` 事件 `流式响应数据异常，请稍后重试。`，并停止该流，避免浏览器收到非法 JSON。
   - 仅在事件成功序列化后保存 `meta`，使运行记录不接收无效元数据。
2. `tests/test_api_sources.py`
   - 覆盖 `/query/stream` 和 `/chat/stream` 的三类非有限值，并以标准 `json.loads` 解析每一个响应行，断言只返回受控错误事件且无非有限浮点数。
   - 替换 `app.web_search.search` 为失败函数，锁定 `/query` 与 `/query/stream` 不触达 Tavily。
3. `tests/test_config.py`
   - 补齐 `TAVILY_MAX_RESULTS` 的非整数输入，以及 `TAVILY_TIMEOUT_SECONDS` 的 `0`、负数输入；现有配置实现均受控回退到默认值。
4. `tests/test_static_workbench.py`
   - 使用 Node 最小 DOM 夹具执行实际 `updateEvidence()` 渲染逻辑，断言只有 HTTPS 网页来源生成 `<a>`，且同时设置 `target="_blank"` 与 `rel="noopener noreferrer"`；`http:` 和 `javascript:` 来源保持非 anchor。

## TDD 与实际测试输出

先新增测试并运行相关测试组，获得预期 SSE 红灯：

```text
E:\AGitHub-project\RAG-Agent-v2\.venv\Scripts\python.exe -m pytest -q tests/test_api_sources.py tests/test_config.py tests/test_static_workbench.py
3 failed, 32 passed, 1 warning in 7.00s
```

其中两个失败证明 `/query/stream` 与 `/chat/stream` 原先会输出 `NaN`/`Infinity`；另一个失败是新 Node 测试夹具缺少 `document.createElement`，补齐夹具后前端回归测试通过。实现 SSE 防护后，定向验证为：

```text
35 passed, 1 warning in 3.55s
```

相关完整回归组：

```text
E:\AGitHub-project\RAG-Agent-v2\.venv\Scripts\python.exe -m pytest -q tests/test_api_sources.py tests/test_config.py tests/test_static_workbench.py tests/test_tools.py tests/test_tool_agent.py tests/test_web_search.py
69 passed, 1 warning in 4.94s
```

全量测试：

```text
E:\AGitHub-project\RAG-Agent-v2\.venv\Scripts\python.exe -m pytest -q
102 passed, 1 warning in 6.24s
```

唯一警告为既有 `app/ingest.py:10` 引入的 `langchain-community` 弃用提示，不由本次改动引入。`git diff --check` 无空白错误。

## 提交

- `2b9c1d4 fix: harden final review regressions`

## 风险

- 无功能性阻塞风险。序列化异常时会主动终止当前 SSE 流而非继续发送可能不安全的后续事件；客户端会接收受控中文错误事件。
- 严格 RAG 路由的 Tavily 隔离由端点级测试固定；Agent 仍可由模型自主决定是否调用工具。
