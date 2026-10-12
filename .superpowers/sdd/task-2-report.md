# Task 2 Report: 将联网搜索注册为统一工具

## 变更

- 在 `app/tools.py` 的函数定义中注册 `search_web`。
- 新增 `_search_web_with_metadata(query)`：调用 `app.web_search.search`，将标准化网页结果转换为统一来源元数据，并生成供模型使用的网页上下文文本。
- 在 `execute_with_metadata()` 中增加 `search_web` 专用分支：仅将 `WebSearchError` 转为对应错误码和中文消息；其余异常保持既有 `execution_error` 行为。
- 空搜索结果返回 `error=None`、无来源和“联网搜索未找到可用结果。”。
- 在 `tests/test_tools.py` 增加网页来源、受控 Tavily 错误及空结果测试。

## TDD 记录

### RED

1. 写入网页来源与 `WebSearchError` 测试后运行：
   `..\\RAG-project\\.venv-conda\\python.exe -m pytest tests/test_tools.py -q`
2. 结果：`2 failed, 3 passed`。失败原因符合预期：`search_web` 尚未被识别，返回 `unknown_tool`。
3. 加入空结果测试后再次运行相同命令。
4. 结果：`3 failed, 3 passed`。失败原因仍符合预期：`search_web` 返回 `unknown_tool`。

### GREEN

1. 以最小实现注册工具、转换元数据和处理受控错误。
2. 定向测试：`6 passed, 1 warning`。
3. 完整测试：`68 passed, 2 warnings`。

## 文件

- 修改：`app/tools.py`
- 修改：`tests/test_tools.py`
- 新增：`.superpowers/sdd/task-2-report.md`

## 自审

- `search_web` 定义的名称、描述和 `query` 参数符合任务摘要。
- 成功结果的来源字段、文本中的标题/URL/摘要、错误码传播和空结果行为均有测试覆盖。
- 未改动 `app/web_search.py`、`app/tool_agent.py`、前端、README 或环境文件。
- 未向代码、测试、命令输出或本报告写入任何密钥。
- `git diff --check` 通过。

## 顾虑

- 测试环境仍报告既有依赖弃用警告：`langchain-community` 与 Starlette `TestClient`/`httpx`；本任务没有修改这些依赖或其调用。

---

## Task 2 小型修复：工具定义回归测试

### 修复细节

- 仅修改 `tests/test_tools.py`。
- 新增 `test_search_web_definition_requires_query()`：调用真实的 `tools.get_definitions()`，按 `function.name == "search_web"` 定位工具定义，并断言 `parameters.required == ["query"]`。
- 未改动 `app/tool_agent.py`、工具实现、`.env`、README 或其它源文件。

### RED 证据

1. 先以受控失配断言 `parameters.required == []` 运行：
   `..\\RAG-project\\.venv-conda\\python.exe -m pytest tests/test_tools.py -q`
2. 输出：`1 failed, 6 passed, 1 warning`。
3. 失败位于新增测试，实际返回值为 `['query']`，与受控期望 `[]` 不符：
   `AssertionError: assert ['query'] == []`。

### GREEN 证据

1. 将断言调整为审查要求的 `parameters.required == ["query"]`；测试仍读取真实 `tools.get_definitions()` 返回值。
2. 运行：`..\\RAG-project\\.venv-conda\\python.exe -m pytest tests/test_tools.py -q`
3. 输出：`7 passed, 1 warning in 14.05s`。

### 测试输出与顾虑

- 唯一警告为既有 `langchain-community` 弃用警告，来自 `app/ingest.py:10`；本次未修改该依赖或调用。
