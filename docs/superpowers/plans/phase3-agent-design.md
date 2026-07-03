# Phase 3：Agent 智能体详细设计

> 对应 Git 标签：`phase3-agent`

---

## 1. 概述

将知识库封装为 Agent，实现三层智能判断：
1. 是否与笔记内容相关 → 搜索知识库
2. 是否需要实时信息 → 联网搜索（DeepSeek enable_search）
3. 其他 → 自由对话

## 2. 架构

```
用户输入
    │
    ▼
┌─────────────────────────────┐
│         Agent               │
│                             │
│  1. 搜索知识库               │
│     计算余弦相似度            │
│                             │
│  ┌──── 相似度 ≥ 0.35 ────┐  │
│  │  基于笔记生成回答       │  │
│  └────────────────────────┘  │
│                             │
│  ┌──── 相似度 < 0.35 ────┐  │
│  │  尝试联网搜索           │  │
│  │  失败 → 自由对话        │  │
│  └────────────────────────┘  │
└─────────────────────────────┘
    │
    ▼
  输出 + 更新记忆
```

## 3. 组件设计

### 3.1 SimpleMemory (`app/memory.py`)

- 滑动窗口：保留最近 5 轮对话（10 条消息）
- 每次对话保存 (role, content) 元组
- `get_history()` 输出格式化字符串供 LLM 使用
- 第 1 版使用 Python 列表实现，无外部依赖

### 3.2 Agent (`app/agent.py`)

**核心流程：**

```
chat(message)
  │
  ├─ _search_kb(message)
  │    使用 BGE 模型将 message 向量化
  │    查 store.search_top_score()
  │
  ├─ 得分 ≥ 0.35？
  │    是 → 搜索知识库 → _KB_PROMPT → LLM
  │    否 → _call_with_search()
  │
  └─ _call_with_search()
       调用 DeepSeek API enable_search=True
       失败 → _CHAT_PROMPT → LLM
```

**三个 Prompt 模板：**

| Prompt | 用途 | 关键变量 |
|--------|------|---------|
| `_KB_PROMPT` | 基于知识库回答 | context, question, current_date |
| `_CHAT_PROMPT` | 自由对话 | history, question, current_date |
| （无需 `_SELECT_PROMPT`） | 不再使用 | — |

## 4. 关键决策：向量阈值 vs LLM 判断

### 问题
最初让 LLM 决定"是否搜索知识库"，但 LLM 判断不准确：
- "赵日天是谁" → LLM 判断不需要搜索，实际笔记里有大量相关内容
- 搜索操作耗时（~5s BGE 加载），判断错了浪费时间还错过答案

### 解决方案
改为 **先搜索，再用量化指标判断**：
1. 永远先搜索知识库（用 BGE 将问题向量化）
2. 计算最高余弦相似度 `store.search_top_score()`
3. 阈值 0.35：超过则认为有相关内容
4. 根据阈值结果决定后续路径

**阈值调试数据：**
- "赵日天是谁" → 0.56（搜索）
- "Python装饰器" → 0.50（搜索）
- "陕西师范大学院长" → 0.28（不搜索）
- "今天是几月几日" → 0.15（不搜索）

## 5. 联网搜索实现

利用 DeepSeek API 的内置搜索能力：

```python
requests.post("https://api.deepseek.com/v1/chat/completions",
    json={
        "model": "deepseek-chat",
        "messages": [...],
        "enable_search": True,  # 关键参数
    },
    headers={"Authorization": f"Bearer {api_key}"})
```

优势：
- 不需要额外 API Key
- 不需要额外费用
- DeepSeek 自动处理搜索和摘要

局限：
- 搜索结果不可控，可能不准确（如"陕西师范大学院长"返回了李葆华而非姚若侠）
- 对于需要确保准确的信息，应写入知识库

## 6. 对话记忆

每次 `chat()` 调用：
1. 读取当前历史 `memory.get_history()`
2. 传给 LLM 作为上下文
3. 将用户消息和 AI 响应存入 `memory`

### 日期处理
LLM 不知道当前日期，在 Prompt 中注入：
```
Current date: {current_date}
```
`current_date = date.today().isoformat()`

## 7. 导入顺序处理

`langchain_community` 必须在 `langchain_openai` 之前导入，否则 OpenTelemetry 初始化冲突导致静默崩溃。

```python
# agent.py 第一行
from app.ingest import build_embeddings  # 强制先导入 community
# 之后其他导入
from app.chain import build_llm          # 然后导入 openai
```

## 8. 输出

```bash
python main.py --chat
```
