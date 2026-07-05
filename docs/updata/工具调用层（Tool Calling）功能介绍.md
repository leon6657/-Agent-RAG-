## 工具调用层（Tool Calling）功能介绍

### 解决的问题

之前的 Agent 用 if-else 硬编码决策路径：

```
def chat(message):
    if _has_relevant(message):   # 代码判断
        context = _search_kb(...)  # 走知识库
    else:
        response = _search_web(...)  # 走联网搜索
```

**题：** 每加一种能力就要改 if-else，工具和决策逻辑耦合在一起。

### 现在的方案

采用 OpenAI 标准的 **Function Calling** 模式，让 LLM 自己决定调什么工具：

```
用户提问
    │
    ▼
DeepSeek 收到问题 + 工具列表
    │
    ├─ 判断：这个问题需要查知识库
    │   → 调用 search_knowledge_base(query="...")
    │   → 拿到结果 → 生成回答
    │
    ├─ 判断：这个问题需要看时间
    │   → 调用 get_current_time()
    │   → 拿到时间 → 生成回答
    │
    └─ 判断：普通聊天
        → 直接回答
```

### 三个核心文件

| 文件                | 职责                                                         |
| ------------------- | ------------------------------------------------------------ |
| `app/tools.py`      | **工具注册中心** — 定义工具 + 实现函数                       |
| `app/tool_agent.py` | **工具执行器** — 调 DeepSeek API → 解析 tool_calls → 执行 → 回填结果 → 循环 |
| `main.py --tool`    | **CLI 入口**                                                 |

### 工具的定义方式

每个工具包含三部分：

```
# 1. JSON schema（给 LLM 看，让它知道有这个工具）
{
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": "搜索本地笔记",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"}
            },
            "required": ["query"]
        }
    }
}

# 2. 实现函数（给 Python 执行）
@_register
def search_knowledge_base(query: str) -> str:
    from app.query import _search
    return _search(query)
```

`@_register` 装饰器自动注册，不需要手动维护映射表。

### 执行流程

```
tool_agent.chat("Python装饰器是什么？")
    │
    ├─ 1. 构建 messages（system + history + user）
    ├─ 2. 调 DeepSeek API，传入 tools 定义
    ├─ 3. DeepSeek 返回：tool_calls = [search_knowledge_base]
    ├─ 4. 解析 tool_call，执行 tools.execute("search_knowledge_base", ...)
    ├─ 5. 结果回填到 messages
    ├─ 6. 再次调 DeepSeek API（带工具结果）
    ├─ 7. DeepSeek 返回最终回答
    └─ 8. 输出到用户
```

支持最多 3 轮工具调用（避免无限循环）。

### 当前已有工具

| 工具                    | 触发场景                     | 说明                           |
| ----------------------- | ---------------------------- | ------------------------------ |
| `get_current_time`      | "现在几点"、"今天几号"       | 返回 `2026-07-05 16:42 Sunday` |
| `search_knowledge_base` | "Python装饰器"、"赵日天是谁" | 搜索 43 个笔记片段             |

### 加新工具的方式

在 `app/tools.py` 里加一个函数 + 一条注册即可：

```
@_register
def calculator(expression: str) -> str:
    """执行数学计算"""
    return str(eval(expression))
```

然后在 `get_definitions()` 里加对应的 JSON schema。LLM 会自动学会在合适的场景调用它。

### 使用方式

```
python main.py --tool
```

跟 `--chat` 的区别：

| 模式     | 决策方式         | 扩展性             |
| -------- | ---------------- | ------------------ |
| `--chat` | if-else 代码判断 | 加能力要改代码     |
| `--tool` | LLM 自主选择工具 | 加能力加个函数就行 |