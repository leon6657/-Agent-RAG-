# 第三轮最终审查修复报告

## 范围与约束

- 修改、测试和提交仅在 `E:\AGitHub-project\RAG-Agent-v2` 的 `v2-agent` 分支完成。
- 未读取、写入或输出 API 密钥；未访问 `E:\AGitHub-project\RAG-project`；未执行部署。
- 严格 RAG 测试对 Tavily 和 web search 设置本地失败哨兵，未发生网络请求。

## 变更

1. `api.py`
   - `_safe_ndjson_events()` 现在捕获上游生成器迭代中的 `Exception`，并追加固定、可 JSON 解析的中文 error SSE 事件：`流式响应服务暂时不可用，请稍后重试。`。
   - `/query/stream` 与 `/chat/stream` 共用该边界，因此两类流均不会因上游在已 yield 事件后抛异常而断流或暴露原始异常文本。
   - 保留非有限值防护；正常事件与两种受控 error 事件均使用 `json.dumps(..., allow_nan=False)`。
2. `tests/test_api_sources.py`
   - 为 `/query/stream` 和 `/chat/stream` 添加上游生成器在 yield 后抛异常的回归测试，逐行 `json.loads()` 验证 HTTP NDJSON 内容，并断言原始诊断文本不出现。
   - 严格 RAG 覆盖不再替换 `api` 导入的 `ask_*` 函数：端点真实进入 `app.query.ask_with_sources()` 与 `app.query.ask_stream_events()`，本地 stub 仅替换向量存储、嵌入和 LLM 依赖。
   - 测试同时将 `app.web_search.search` 与 Tavily HTTP `requests.post` 设为立即失败，覆盖 `/query` 和 `/query/stream` 的无联网路径。
3. `tests/test_tool_agent.py`
   - Agent 直接回答且无需工具的回归测试现在记录模型决策调用，并断言 `tool_choice == "auto"`，以锁定模型自主选择调用或不调用工具的能力。

## TDD 与测试输出

先只写测试后运行重点测试组，得到预期红灯：

```text
.\\.venv\\Scripts\\python.exe -m pytest -q tests/test_api_sources.py tests/test_tool_agent.py
....FF....................                                               [100%]
2 failed, 24 passed, 1 warning in 5.04s
```

两个失败均来自 `api.py:_safe_ndjson_events` 的 `for event in events`：上游生成器在发送首个事件后抛出的 `RuntimeError("internal diagnostic detail")` 被 Starlette 传播，验证了缺少迭代边界异常处理。

实现最小修复后，重点测试组输出：

```text
.\\.venv\\Scripts\\python.exe -m pytest -q tests/test_api_sources.py tests/test_tool_agent.py
..........................                                               [100%]
26 passed, 1 warning in 2.91s
```

全量测试输出：

```text
.\\.venv\\Scripts\\python.exe -m pytest -q
........................................................................ [ 69%]
................................                                         [100%]
104 passed, 1 warning in 4.16s
```

`git diff --check` 未报告空白错误；Git 仅提示两个修改过的测试文件在后续 Git 操作时会从 LF 转为 CRLF。

唯一警告为既有 `app/ingest.py:10` 的 `langchain-community` 弃用提示，不由本次修改引入。

## 提交

- 实现提交：`fix: contain upstream SSE iteration errors`
- 本报告提交：`docs: record third final review fixes`

## 剩余风险

- API 层能受控处理同步生成器迭代异常；异步 SSE 生产者尚未存在于当前接口，未来若引入需要复用同等的迭代异常边界。
- 本地严格 RAG 测试覆盖检索与 LLM stub 的真实调用路径，并禁止 Tavily/web search；不替代对真实模型或真实向量库的集成验收，后者不在本轮允许的联网和密钥范围内。
