# RAG 知识库 — 四阶段演进总结

> 从零搭建一个以 Markdown 笔记为知识库的 RAG 系统，逐步优化检索、加入 Agent、引入联网搜索。
> 完整记录了所有踩过的坑和对应的解决方案。

---

## 项目结构

```
rag-project/
├── data/                       # 你的笔记（11 篇 .md 文件）
├── app/
│   ├── config.py               # 配置管理（API Key、路径、参数）
│   ├── store.py                # numpy 向量存储（JSON 持久化 + 余弦相似度搜索）
│   ├── ingest.py               # 文档加载 → 分块 → BGE 嵌入 → 存储
│   ├── query.py                # 问答接口（先搜索再调用 LLM）
│   ├── chain.py                # LCEL 链定义（ChatOpenAI + Prompt）
│   ├── retriever.py            # BM25 + 向量混合检索
│   ├── reranker.py             # 关键词重叠重排序
│   ├── query_rewriter.py       # Multi-Query 查询改写
│   ├── memory.py               # 对话记忆（滑动窗口 5 轮）
│   └── agent.py                # Agent（知识库 → 联网搜索 → 自由对话）
├── evaluation/
│   ├── questions.json          # 20 个 QA 对
│   ├── metrics.py              # Recall@k / MRR / Precision
│   └── runner.py               # 评测运行 + 对比报告
├── main.py                     # CLI 入口（--ingest / --query / --chat / --serve）
├── vector_store.json           # 43 个文档片段（已嵌入）
└── .hf_cache/                  # BGE 模型缓存（离线可用）
```

---

## 四阶段演进路线

### Phase 1：最小 RAG

**目标：** 跑通文档加载 → 分块 → 嵌入 → 检索 → 生成的完整链路。

**选型：** LangChain + BGE 嵌入 + DeepSeek Chat + numpy 向量存储

**输出：** `python main.py --query` 可交互问答

### Phase 2：优化 RAG

**新增：**
- BM25 + 向量混合检索（`retriever.py`）
- 关键词重排序（`reranker.py`）
- Multi-Query 查询改写（`query_rewriter.py`）
- 评测体系（`evaluation/`）

**评测结果：** Recall@4 从 0.350 提升到 0.575（+64%）

### Phase 3：Agent

**新增：**
- `memory.py` — 对话记忆（滑动窗口 5 轮）
- `agent.py` — Agent（自动判断：笔记相关→搜索知识库，实时信息→联网搜索，日常对话→自由聊天）

**核心改进：** 从"LLM 判断是否搜索"改为"先搜索，再用余弦相似度阈值（0.35）判断是否相关"

**输出：** `python main.py --chat` Agent 模式

### Phase 4：项目闭环

**新增：**
- `api.py` — FastAPI Web 服务（端口 8000）
- `app/cache.py` — 响应缓存（diskcache，Redis-ready 接口）
- `app/logging_setup.py` — 日志监控（滚动文件 + 控制台 + 请求 ID）
- `app/state.py` + `app/graph.py` — LangGraph 有向状态图
- `static/index.html` — 聊天界面 Web UI

**关键改进：**
- 分离前后端，前端独立为 HTML 文件
- 聊天历史记录保留
- Enter 发送 / Shift+Enter 换行
- 相同问题秒回（缓存）
- 图结构 Agent 取代 if-else 判断

**输出：** `python main.py --serve` 启动 Web 服务

---

## 提出的问题 & 踩过的坑

### 1. RAG 无法回答知识库外的问题

**问题：** "今天是几月几日" — RAG 回答不了，因为笔记里没有日期信息。

**原因：** `--query` 模式严格基于知识库，这是设计如此——宁可说不知道，也不编造。

**解决：** 引入 `--chat` 模式（Agent），自动判断：
- 笔记相关内容 → 搜索知识库
- 实时信息 → 联网搜索（DeepSeek enable_search）
- 日常对话 → 自由聊天

---

### 2. Agent 判断是否搜索知识库不准确

**问题：** "赵日天是谁" — Agent 错误判断不需要搜索，直接用网络梗回答，没有用到笔记中的内容。

**之前的做法：** 让 LLM 决定"要不要搜索"——但 LLM 判断不准。

**解决：** **改为先搜索，再判断**。永远先搜索知识库，计算查询与文档的余弦相似度，只有最高分超过阈值（0.35）时才认为知识库有相关内容。否则回退到联网搜索或自由对话。

```
问：赵日天是谁？
  → 搜索知识库 → 相似度 0.56 > 0.35
  → 基于笔记回答："一只十四斤重的橘猫..." ✅

问：陕西师范大学院长是谁？
  → 搜索知识库 → 相似度 < 0.35
  → 联网搜索 → 回答 ✅
```

---

### 3. LLM 日期幻觉

**问题：** Agent 回答的日期是错的（回答 2025 年，实际是 2026 年）。

**原因：** DeepSeek 不知道当前日期，只能"猜"一个。每次猜的结果还不一样。

**解决：** 在 Agent 的所有 prompt 中传入当前日期：

```
Current date: {current_date}
```

修复后：
```
问：今天是几月几日？
  → 知识库没有 → 自由对话
  → 回答："2026年7月2日" ✅（来自系统时间）
```

---

### 4. 联网搜索不准确

**问题：** "陕西师范大学的人工智能与计算机学院的院长是谁" — DeepSeek 联网搜索返回"李葆华"，但正确答案是"姚若侠教授"。

**原因：** DeepSeek 搜索网络后拿到多个网页片段，它没有能力核实哪个信息是正确的。它从某个片段看到"李葆华"就当成答案了。

**本质问题：** LLM 被训练成"即使不确定也要给出合理的回答"，不会说"我在不同地方看到了不同的人名"。

**最佳实践：** 对需要确保准确的信息，写入知识库：

```markdown
# data/school-info.md
陕西师范大学人工智能与计算机学院院长：姚若侠教授
```

之后无论 `--query` 还是 `--chat` 都会返回正确结果。

---

### 5. 后续指令问题（"去查"）

**问题：** 问完问题后说"去查"，Agent 无法理解这是让它搜索之前的问题。

**原因：** Agent 没有上下文理解能力——"去查"被当成独立问题处理，搜索了"去查"这个关键词本身。

**现状：** 这是一个自然语言理解的难题，需要更复杂的对话管理。当前的最佳做法是一次性把问题问清楚。

---

## 技术坑 & 解决方案

### 6. ChromaDB ONNX 模型权限问题

**症状：** ChromaDB 1.5.x 创建集合时，默认 ONNX 嵌入模型试图写入 `C:\Users\dad\.cache\chroma`，沙箱环境无权限，导致静默崩溃。

**根因：** ChromaDB 1.5.x 的 `PersistentClient` 在内部使用 gRPC 服务，且默认嵌入函数需要下载 ONNX 模型。即使传入预计算的向量，在某些代码路径下仍会触发 ONNX 下载。

**解决：** 彻底移除 ChromaDB，改用 **numpy 向量库**（JSON 存储 + 余弦相似度搜索）。零外部依赖，无权限问题。

```python
# app/store.py — 核心代码仅 60 行
def search(query_embedding, k=4):
    vectors = np.array([item["embedding"] for item in items])
    scores = vectors @ q  # 余弦相似度
    top_idx = np.argsort(scores)[-k:][::-1]
    return [Document(...) for i in top_idx]
```

---

### 7. BGE 模型下载被墙

**症状：** `sentence-transformers` 首次加载 BGE 模型时连接 `huggingface.co` 超时。

**解决：** 使用国内镜像 `hf-mirror.com` 下载，本地缓存，`_get_model_path()` 直接指向本地快照路径。

---

### 8. langchain_openai 和 langchain_community 导入冲突

**症状：** `from langchain_openai import ChatOpenAI` 后再导入 `langchain_community` 会静默崩溃。

**根因：** 两个包都使用 OpenTelemetry 进行 instrumentation，导入顺序不对会导致初始化冲突。

**解决：** 永远确保 `langchain_community` 在 `langchain_openai` 之前导入。在关键文件中添加：

```python
from app.ingest import build_embeddings  # 强制先导入 langchain_community
from app.chain import build_llm          # 然后导入 langchain_openai
```

---

### 9. LangChain RunnableParallel 线程池挂起

**症状：** LangChain 的 LCEL 链使用 `RunnableParallel` 时，内部用 `ThreadPoolExecutor` 执行两个分支。当其中一个分支调用 ChromaDB 或 BGE 模型时，线程池挂起。

**解决：** 先计算搜索上下文（在主线程中），再调用 LLM（在主线程中），完全避免 RunnableParallel：

```python
# 之前（挂起）：
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

# 之后（正常）：
context = _search(question)  # 主线程搜索
chain = prompt | llm | StrOutputParser()  # 简单链
return chain.invoke({"context": context, "question": question})
```

---

### 10. 沙箱环境文件操作问题

**症状：** Codex 沙箱的 `apply_patch` 工具频繁失败，`ShellExecuteExW` 找不到模块。

**解决：** 使用 PowerShell 的 `@'"..."'@` 单引号 here-string 写文件，或使用 Python 直接写。

**PowerShell 编码注意事项：**
- `@'...'@`（单引号 here-string）不会转义任何字符，适合写 Python 代码
- 但 `@'...'@` 中的 `"` 和 `'''` 都是字面量，可能导致 PowerShell 解析异常
- 对于含中文或复杂引号的内容，优先使用 Python 的 `open().write()`

---

## 使用方式

```bash
# 1. 激活虚拟环境
.venv\Scripts\activate

# 2. 检查配置（.env 文件需有 DEEPSEEK_API_KEY）
#    获取 Key：https://platform.deepseek.com

# 3. 建库（索引 data/ 下所有 .md 文件）
python main.py --ingest

# 4. 严格 RAG 问答（只从笔记提取，没有就说不知道）
python main.py --query

# 5. Agent 模式（自动判断：知识库 / 联网搜索 / 自由对话）
python main.py --chat

# 6. 启动 Web 服务（浏览器访问 http://localhost:8000）
python main.py --serve

# 7. 运行评测
python evaluation/runner.py
```

---

## 技术栈

| 组件 | 选型 | 用途 |
|------|------|------|
| LLM | DeepSeek Chat | 问答生成 + 联网搜索 |
| Embedding | BAAI/bge-small-zh-v1.5 | 中文文本向量化 |
| 向量存储 | numpy + JSON | 余弦相似度搜索 |
| 框架 | LangChain | LCEL 链 + Agent 工具 |
| Web 服务 | FastAPI + Uvicorn | REST 接口（端口 8000）|
| 缓存 | diskcache | 响应缓存（支持切换 Redis）|
| 状态图 | LangGraph | Agent 有向状态图 |
| CLI | argparse | 命令行入口 |

---

## Git 标签

```bash
phase1-minimal-rag   # Phase 1：最小 RAG
phase2-optimized-rag # Phase 2：优化 RAG
phase3-agent         # Phase 3：Agent
```

---

## 后续方向

- **Phase 4 已完成** — FastAPI 封装 + 缓存优化 + 日志监控 + LangGraph 状态图
- **更好的评测** — 更细粒度的评估指标
- **多文档格式** — 支持 PDF、网页等
- **知识库管理** — 增删改查 CRUD 界面
