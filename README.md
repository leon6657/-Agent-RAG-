# RAG 知识库 — 四阶段演进项目

> 从零搭建的 Markdown 笔记 RAG 系统，经历四阶段演进：最小原型 → 优化检索 → Agent 智能体 → 工程化闭环。
> 完整解决了 ChromaDB 兼容性、LangChain 导入冲突、Agent 决策逻辑、联网搜索等关键技术难题。

> **详细演进记录：** [docs/project-summary.md](docs/project-summary.md) · **完整使用说明：** [USAGE.md](USAGE.md) · **设计文档：** [docs/](docs/superpowers/specs/)

---

## 技术栈

| 类别 | 技术 | 用途 |
|------|------|------|
| **LLM** | DeepSeek Chat (API) | 问答生成 + 联网搜索 |
| **Embedding** | BAAI/bge-small-zh-v1.5 | 中文语义向量化 |
| **向量存储** | numpy + JSON | 余弦相似度搜索（43 个片段） |
| **Agent 框架** | LangChain LCEL | 检索链 + LLM 调用 |
| **状态图** | LangGraph | Agent 有向状态图 |
| **Web 服务** | FastAPI + Uvicorn | REST API + 聊天界面 |
| **缓存** | diskcache（Redis-ready） | 响应缓存，相同问题秒回 |
| **日志** | RotatingFileHandler | 滚动文件日志 + 请求 ID 追踪 |
| **前端** | 纯 HTML/CSS/JS | 聊天气泡界面 |
| **评测** | 自定义指标 | Recall@k / MRR / Precision |
| **评分** | BM25Okapi | 关键词混合检索 |
| **部署** | Python 虚拟环境 | 离线可运行 |

---

## 架构

```
浏览器 / 终端
    │
    ▼
┌────────────────────────────────────────┐
│          FastAPI (端口 8000)            │
│  /query  /chat  /ingest  /health       │
└──────────────┬─────────────────────────┘
               │
    ┌──────────┴──────────┐
    ▼                     ▼
┌──────────┐        ┌──────────┐
│ 严格 RAG │        │  Agent    │
│ (query)  │        │ (chat)   │
│ 仅笔记   │        │ 笔记+联网  │
└──────────┘        └──────────┘
    │                     │
    └──────────┬──────────┘
               ▼
┌──────────────────────────────┐
│      向量检索 + LLM 生成      │
│  BGE → 相似度搜索 → DeepSeek  │
└──────────────────────────────┘
```

---

## 关键成果

### 1. 检索优化（Phase 2）
**Recall@4 从 0.350 提升到 0.575（+64%）**
- 引入 BM25 混合检索
- 关键词重排序
- Multi-Query 查询改写

### 2. Agent 智能判断（Phase 3）
从"LLM 决定是否搜索"改为**先搜索再用向量相似度阈值判断**（0.35），解决了 LLM 判断不准导致跳过有内容笔记的问题。

### 3. 联网搜索（Phase 3）
利用 DeepSeek API 的 `enable_search=True` 参数，**无需额外 API Key** 即可实现联网搜索。

### 4. 缓存优化（Phase 4）
diskcache 响应缓存，相同问题在 TTL（1 小时）内直接返回，**省去 BGE 加载 + LLM 调用时间**。接口兼容 Redis，可无缝切换。

### 5. LangGraph 状态图（Phase 4）
用 LangGraph 替代 if-else 实现 Agent 控制流，支持状态追踪和循环控制。

### 6. 项目闭环（Phase 4）
- FastAPI 提供 REST 接口，可被其他服务调用
- 旋转文件日志 + 请求 ID 追踪
- Web UI 聊天界面（Enter 发送 / Shift+Enter 换行）

---

## 解决的问题

| 问题 | 根因 | 解决方案 |
|------|------|---------|
| ChromaDB 静默崩溃 | ONNX 默认模型无缓存目录权限 | 改用 numpy 向量库 |
| langchain 导入崩溃 | OpenTelemetry 初始化冲突 | `langchain_community` 先于 `langchain_openai` 导入 |
| 链执行挂起 | RunnableParallel 线程池调用 BGE | 先搜索再调 LLM |
| Agent 判断不准确 | LLM 决定是否搜索 | 改为向量阈值判断 |
| 日期幻觉 | LLM 不知道当前日期 | Prompt 中注入 `current_date` |
| BGE 模型被墙 | huggingface.co 无法访问 | hf-mirror.com 镜像 + 本地缓存 |

---

## 项目结构

```
rag-project/
├── app/
│   ├── config.py / chain.py      # 配置 + LLM 链
│   ├── ingest.py / store.py      # 文档导入 + 向量存储
│   ├── query.py                  # RAG 问答接口
│   ├── retriever.py / reranker.py# 混合检索 + 重排序
│   ├── query_rewriter.py         # 查询改写
│   ├── memory.py / agent.py      # 对话记忆 + Agent
│   ├── cache.py                  # 响应缓存（diskcache）
│   ├── logging_setup.py          # 日志监控
│   ├── state.py / graph.py       # LangGraph 状态图
├── evaluation/                   # 评测体系（20 QA 对）
│   ├── metrics.py / runner.py
├── static/index.html             # Web 聊天界面
├── api.py                        # FastAPI 服务
├── main.py                       # CLI 入口
├── vector_store.json             # 43 个文档片段
├── USAGE.md                      # 使用说明
└── docs/                         # 设计文档
```

---

## 快速开始

```bash
# 1. 激活环境
.venv\Scripts\activate

# 2. 配置 API Key（.env 文件）
DEEPSEEK_API_KEY=sk-your-key

# 3. 建库
python main.py --ingest

# 4. 使用
python main.py --query     # 严格 RAG
python main.py --chat      # Agent 模式
python main.py --serve     # Web 服务（浏览器打开 http://localhost:8000）
```

---

## Git 标签

```bash
phase1-minimal-rag       # 最小 RAG
phase2-optimized-rag     # 优化检索（Recall +64%）
phase3-agent             # Agent 智能体
phase4-project-closure   # 工程化闭环
```

---

## 设计文档

- [整体设计](docs/superpowers/specs/rag-knowledge-base-design.md)
- [Phase 1 minimal-RAG ](docs\superpowers\plans\phase1-minimal-rag.md)
- [Phase 2 optimized-RAG ](docs\superpowers\plans\phase2-optimized-rag.md)
- [Phase 3 Agent 设计](docs\superpowers\plans\phase3-agent-design.md)
- [Phase 4 项目闭环设计](docs\superpowers\plans\phase4-closure-design.md)
- [项目总结](docs\project-summary.md)

