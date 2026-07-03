# Prompt 版本管理说明

> 将提示词从硬编码的 Python 变量迁移到外部 JSON 文件，支持版本迭代、使用审计和热加载。

---

## 文件结构

```
.prompts/
+-- prompts.json          # prompt 模板仓库（多版本支持）
+-- prompt_usage.log      # 使用审计日志（自动生成）
```

## 五个 Prompt 说明

| 名称 | 用途 | 关键变量 |
|------|------|---------|
| query | 严格 RAG 问答 | {context}, {question} |
| chat | Agent 自由对话 | {current_date}, {history}, {question} |
| kb_search | 知识库检索回答 | {current_date}, {context}, {question} |
| multi_query | 查询改写 | {n}, {question} |
| agent_select | 判断是否搜索知识库 | {current_date}, {question} |

## API 参考

```python
from app.prompt_manager import get, save, create_version, list_versions

prompt = get("query")              # 当前版本
prompt = get("query", "v2")        # 指定版本
list_versions()                    # ["v1", "v2"]
create_version("v1", "v2")         # 从 v1 复制
create_version("v2")               # 自动命名
save("v2", "query", "新模板...")
```

## 使用审计

每次 get() 自动记录到 .prompts/prompt_usage.log

```
[2026-07-03T14:30:00] version=v1 prompt=query
[2026-07-03T14:30:15] version=v2 prompt=chat
```

## 版本迭代流程

1. 基于 v1 创建 v2
   create_version("v1", "v2")

2. 修改 v2 prompt
   save("v2", "query", "新模板...")

3. 代码中指定版本
   prompt = get("query", "v2")

4. 回退
   prompt = get("query", "v1")

## 与代码的集成方式

当前 prompt 在代码中硬编码。使用 prompt_manager 替换：

```python
from app.prompt_manager import get
_KB_PROMPT = get("kb_search")
```

## 当前版本

| 版本 | 特点 |
|------|------|
| v1 | 详细版，530-551 字/条 |
| v2 | 精炼版，386-502 字/条 |
