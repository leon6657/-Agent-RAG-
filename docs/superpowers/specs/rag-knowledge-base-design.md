# RAG 知识库 --- 四阶段演进设计

## 概述

从零搭建一个以 Markdown 笔记为知识库的 RAG 系统，按四阶段逐步演进：
**Phase 1 最小 RAG -> Phase 2 优化检索 -> Phase 3 Agent -> Phase 4 LangGraph**

目标是让这套项目既可作为可演示的个人作品，也能在面试中清晰讲述"为什么这样演进"的技术叙事。

---

## Phase 1：最小 RAG

### 目标
3-5 天内跑通一个能用的问答系统：用户问一个关于笔记的问题，返回带上下文引用的回答。

### 技术选型

| 组件 | 选型 | 理由 |
|------|------|------|
| LLM | DeepSeek Chat (API) | OpenAI 兼容接口，成本低，中文强 |
| Embedding | BAAI/bge-small-zh-v1.5 (本地) | 纯本地运行，中文效果好，~30MB |
| 向量库 | ChromaDB | 纯 Python，本地持久化，无需额外服务 |
| 分块 | RecursiveCharacterTextSplitter | LangChain 内置，chunk_size=500，overlap=50 |
| 文档格式 | Markdown .md | 你的笔记格式 |
| 框架 | LangChain (LCEL) | 第一天就用，后续替换内部组件 |

### 项目结构

```
rag-project/
+-- data/                  # 源文档（你的 .md 笔记）
+-- chroma_db/             # ChromaDB 持久化目录
+-- app/
|   +-- __init__.py
|   +-- config.py          # 配置管理（API Key、模型名、路径）
|   +-- ingest.py          # 文档加载 -> 分块 -> 向量化 -> 入库
|   +-- chain.py           # LCEL chain 组装
|   +-- query.py           # ask() 接口
+-- main.py                # CLI 入口
+-- .env                   # DEEPSEEK_API_KEY
+-- .gitignore
+-- pyproject.toml
+-- README.md              # 收尾阶段写
```

### 数据流

```
用户输入问题
    |
    v
ask(question)
    |
    +- 1. 检索：ChromaDB.similarity_search(question, k=4)
    |    返回 chunk list
    +- 2. 组装：prompt_template.format(context=chunks, question=question)
    +- 3. 生成：DeepSeek Chat stream 输出回答
    +- 4. 输出：终端打印回答 + 来源引用
```

### CLI 命令

python main.py --ingest     # 扫描 data/ 目录，构建/更新向量库
python main.py --query      # 交互式问答模式

### 边界处理

- 空库时查询 -> 提示"请先运行 --ingest"
- DeepSeek API 不可用 -> 降级提示，不崩溃
- 中文 chunk 截断在字符边界 -> RecursiveCharacterTextSplitter 按 ["\n\n", "\n", "。", "！", "？", ""] 切分
- 流式输出 -> stdout flush 实时显示

---

## Phase 2：优化 RAG

### 目标
将检索质量提升到可实用水平，引入评测体系。

### 新增/修改组件

| 组件 | 作用 |
|------|------|
| retriever.py | 封装多路检索器（向量 + BM25） |
| reranker.py | Cross-encoder 重排序（BAAI/bge-reranker-v2-m3） |
| query_rewriter.py | HyDE (Hypothetical Document Embedding) + Multi-Query |
| chain.py 升级 | 集成改写->检索->重排->生成管线 |
| evaluation/ | 评测集 + 指标（Recall@k、MRR、Answer Relevancy） |

### 优化的检索流程

```
原始问题
    |
    +- HyDE：生成虚拟文档 -> 用文档 embedding 检索
    +- Multi-Query：生成 N 个相关问法 -> 分别检索后去重
    |
    +- 合并结果 -> Cross-encoder 重排序 -> Top-K -> 生成
```

### 评测

- 手动标注 20-30 个 QA 对，覆盖笔记各章节
- 对比优化前后的 Recall@3 / MRR
- 输出报告，量化"优化了多少"

---

## Phase 3：Agent

### 目标
将知识库包装为 Tool，让 LLM 自主决定是否检索、什么时机检索，并支持多轮对话。

### 新增组件

| 组件 | 作用 |
|------|------|
| agent.py | Tool + AgentExecutor（ReAct 模式） |
| memory.py | ConversationBufferWindowMemory，保留最近 5 轮 |

### Agent 工具

1. knowledge_base(query: str) --- 检索知识库并返回摘要
2. chat(topic: str) --- 自由对话，不触发检索

### 能力

- 用户问"Python 里装饰器怎么用" -> Agent 决定去知识库检索
- 用户追问"那闭包呢？" -> 结合上文继续检索
- 用户说"帮我写个例子" -> Agent 自由对话，不检索

---

## Phase 4：LangGraph

### 目标
将 Agent 的 ReAct 循环拆为有向图，显式控制状态流。

### 新增组件

| 组件 | 作用 |
|------|------|
| state.py | AgentState（messages、source、retrieval_count） |
| graph.py | LangGraph 状态图定义 |

### 图结构

```
                    +----------+
                    |  入口节点  |
                    +----+-----+
                         |
                         v
                 +---------------+
                 |  意图判断节点    |
                 +-------+-------+
                         |
             +-----------+-----------+
             |                       |
             v                       v
     +---------------+      +---------------+
     |  检索知识库    |      |  直接回复     |
     +-------+------+      +-------+-------+
             |                      |
             +----------+-----------+
                        |
                        v
                +---------------+
                |  生成最终回答   |
                +-------+-------+
                        |
                        v
                +---------------+
                |  是否继续？    |---> 回到意图判断
                +---------------+
```

### 状态管理
- 显式的 retrieval_count 防止无限循环
- 每一步的 source_documents 可追踪
- 条件边控制路由逻辑

---

## 收尾

### README 内容
- 项目背景与动机
- 四阶段演进示意图
- 快速开始（安装 -> ingest -> query）
- 技术栈一览
- 面试 FAQ：为什么选这些组件？Phase 2 优化了多少？LangGraph 比 AgentExecutor 好在哪里？

### 面试叙事线

> 我先用 LangChain + ChromaDB + BGE 搭了一个最小 RAG，跑通后发现某些问题检索不准，于是 Phase 2 加了 hyde、cross-encoder rerank 和混合检索，用评测集证明 Recall 提升了 X%。然后封装成 Tool 做了 Agent，最后迁移到 LangGraph 做更细粒度的控制流...

---

## 约束条件

- 本地开发环境：Windows（已在用）
- Python >= 3.10
- DeepSeek API：需注册获取 Key
- BGE：首次运行自动下载模型
- ChromaDB：持久化到本地目录，纳入 .gitignore
- 所有阶段产物都在一个 repo 内，通过 git 标签标记阶段边界
