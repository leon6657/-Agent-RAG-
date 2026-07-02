# RAG 知识库 — 从零到 LangGraph 的四阶段演进

一个基于 Markdown 笔记的最小 RAG（检索增强生成）知识库，分四个阶段逐步演进：

**Phase 1 最小 RAG → Phase 2 优化检索 → Phase 3 Agent → Phase 4 LangGraph**

当前已完成 **Phase 1**，一个可运行的最小 RAG 问答系统。

> 详细演进记录和踩坑总结：[docs/project-summary.md](docs/project-summary.md)

---

## 快速开始

### 环境要求

- Python >= 3.9
- DeepSeek API Key（[注册获取](https://platform.deepseek.com)）

### 安装

```bash
# 进入项目目录
cd rag-project

# 激活虚拟环境
.venv\Scripts\activate

# 配置 API Key
# 编辑 .env 文件，填入你的 DeepSeek API Key
# DEEPSEEK_API_KEY=sk-xxx
```

### 建库

把你的 Markdown 笔记放到 `data/` 目录下，然后运行：

```bash
python main.py --ingest
```

这会扫描 `data/` 下所有 `.md` 文件，分块后用 BGE 模型生成嵌入向量，存入本地向量库。

### 问答

```bash
python main.py --query
```

输入问题后等待 15-20 秒（首次加载模型 + 导入依赖），后续问题响应更快。

```
You: Python 装饰器是什么？
Assistant: 根据提供的上下文，Python 装饰器是一种函数，它接受另一个函数并扩展其行为...
```

---

## 项目结构

```
rag-project/
├── data/                  # 你的 Markdown 笔记（知识库源）
├── app/
│   ├── config.py          # 配置管理（API Key、分块参数）
│   ├── ingest.py          # 文档加载 → 分块 → 嵌入 → 存储
│   ├── query.py           # 检索 + LLM 生成回答
│   ├── chain.py           # LCEL 链定义（DeepSeek Chat）
│   └── store.py           # numpy 向量存储（JSON + 余弦相似度搜索）
├── main.py                # CLI 入口（--ingest / --query）
├── .env                   # DEEPSEEK_API_KEY
├── vector_store.json      # 向量库（持久化文件）
├── .hf_cache/             # BGE 模型缓存（离线可用）
├── docs/
│   └── superpowers/
│       ├── specs/         # 设计文档
│       └── plans/         # 实现计划
└── tests/                 # 单元测试
```

---

## 技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| LLM | DeepSeek Chat | OpenAI 兼容 API，成本低，中文强 |
| Embedding | BAAI/bge-small-zh-v1.5 | 纯本地运行，~30MB，中文效果好 |
| 向量存储 | numpy + JSON | 余弦相似度搜索，零外部依赖 |
| 框架 | LangChain LCEL | 链式编排，可逐步演进 |
| 文档格式 | Markdown .md | 你的笔记格式 |

---

## 四阶段路线

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | 最小 RAG：文档 → 分块 → 向量检索 → LLM 生成 | ✅ 完成 |
| Phase 2 | 优化 RAG：重排序、查询改写、混合检索、评测 | ⏳ 待开始 |
| Phase 3 | Agent：知识库 Tool、对话记忆、ReAct 循环 | ⏳ 待开始 |
| Phase 4 | LangGraph：有向图、状态管理、条件路由 | ⏳ 待开始 |

---

## 踩坑记录

- **ChromaDB 1.5.x ONNX 模型权限问题**：在受限环境中 ChromaDB 的默认 ONNX 嵌入模型无法写入系统缓存目录，改用 numpy 向量库解决
- **langchain_openai 与 langchain_community 导入冲突**：OpenTelemetry 全局 instrumentation 冲突，通过延迟导入 `build_llm` 解决
- **LangChain RunnableParallel 线程池冻结**：BGE 模型加载 + ChromaDB 客户端创建在线程池中会挂起，改为先搜索后调用 LLM
- **HuggingFace 模型下载被墙**：使用 `hf-mirror.com` 镜像 + 本地持久缓存

---

## 运行测试

```bash
.venv\Scripts\activate
pytest tests/ -v
```

当前 6 个测试，全部通过。

---

## 许可证

MIT
