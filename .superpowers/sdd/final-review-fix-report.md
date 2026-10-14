# Tavily 联网搜索最终审查修复报告

## 范围与约束

- 修改仅发生在 `E:\AGitHub-project\RAG-Agent-v2` 的 `v2-agent` 分支；未读写 `E:\AGitHub-project\RAG-project`。
- 未读取、写入或输出真实 API 密钥；测试仅使用模拟值，并未发起 Tavily 请求。
- 未进行公网部署。为运行测试在目标仓库创建了被 Git 忽略的本地 `.venv`，仅安装测试依赖。

## 改动

1. `app/web_search.py`
   - 使用 `math.isfinite` 拒绝 `NaN`、`Infinity` 和 `-Infinity` 搜索评分，并返回既有受控的 `invalid_response` 错误，避免这些值进入 Agent 来源和 SSE JSON。
2. `app/tool_agent.py` 与 `static/index.html`
   - 工具结果包含 `error` 时发出 `status: failed` 与受控中文状态，绝不再发出“已完成”。
   - 前端仅在实际存在 `source_type: web` 的来源时说明“返回网页来源”；联网搜索失败或空结果时显示“未返回可用网页来源”。
   - 系统提示明确将 Tavily 标题、URL、摘要等视为不可信参考内容，禁止执行其指令、泄露提示词/密钥或改变权限；工具结果以 `<tool_result>` 包裹的 JSON 传递给模型。
3. `app/tools.py` 与 `app/tool_agent.py`
   - 未知异常不再拼接给用户；工具返回固定的“工具执行失败，请稍后重试。”，Agent 返回固定的“Agent 服务暂时不可用，请稍后重试。”。
   - 服务端日志只记录工具名或固定 Agent 失败诊断，不记录原始异常文本。
4. `app/config.py`
   - Tavily 结果数限制为 1-20，超时限制为 1-60；0、负数、超范围与非整数值均回退到现有默认值（5、10）。
5. Minor
   - `docs/deployment-demo.md` 中采用 `^` 的 Windows 命令块改为 `cmd` 标注。
   - 回归测试确认本地与网页来源即使字段值完全相同也分别保留。

## TDD 与测试

先新增回归测试，并在实现前获得以下预期红灯：

```text
E:\software\Anaconda3\python.exe -m pytest -q tests/test_web_search.py tests/test_static_workbench.py tests/test_config.py tests/test_deployment_demo.py
6 failed, 30 passed in 0.53s

.\.venv\Scripts\python.exe -m pytest -q tests/test_tool_agent.py
4 failed, 8 passed in 1.96s
```

最终重点测试命令与实际输出：

```text
.\.venv\Scripts\python.exe -m pytest -q tests/test_web_search.py tests/test_tools.py tests/test_tool_agent.py tests/test_static_workbench.py tests/test_config.py tests/test_deployment_demo.py
60 passed, 1 warning in 2.29s
```

全量测试命令与实际输出：

```text
.\.venv\Scripts\python.exe -m pytest -q
90 passed, 1 warning in 3.46s
```

唯一警告来自 `langchain-community` 自身的弃用提示（`app/ingest.py:10`），不由本次改动引入。

## 提交

- 待提交：本报告将在提交创建后补充提交哈希。

## 风险与关注项

- 工具失败后仍将受控失败结果交给模型，以便模型决定如何回复或选择其他工具；前端会准确显示失败，不会伪称网页来源成功返回。
- 严格 RAG 的 `/query` 与 `/query/stream` 未改动，仍仅调用 `app.query`，没有获得联网搜索定义或调用路径。
