# Phase 2: 优化 RAG — 详细设计文档

## 概述

Phase 2 在 Phase 1 最小 RAG 的基础上，引入了完整的评测体系、混合检索、查询改写和重排序机制。

## 1. 评测体系

### 文件结构

评测体系位于 evaluation/ 目录下：

- metrics.py: Recall@k、MRR、Precision 指标计算
- questions.json: 20 个 QA 对，覆盖所有笔记主题
- runner.py: 评测运行器，对比基线和优化版

### 20 个 QA 对

基于 data/ 目录下的 11 篇技术笔记，编制了覆盖以下主题的问题：

- Python 函数基础（定义、参数、作用域）
- Python 面向对象（类、继承、魔术方法）
- Python 数据结构（列表、元组、字典、集合）
- 机器学习基础（监督、非监督、过/欠拟合、特征工程）

每个 QA 对包含 id、question、answer（关键词）、source（来源文档）。

### 指标定义

Recall@k：查询的关键词是否出现在 top-k 检索结果中
MRR：第一个相关结果的排名的倒数
Precision@k：top-k 结果中相关结果的比例

### 运行评测

```bash
python evaluation/runner.py
```

### 评测结果（Phase 2）

```
Baseline (Vector Only):
  Recall@4:    0.250    MRR: 0.200    Prec: 0.062
Optimized (Hybrid+Rerank):
  Recall@4:    0.250    MRR: 0.192    Prec: 0.062
```

两者持平的原因：BGE 模型本身语义检索能力强；BM25 在中文技术笔记这种结构化文本上未能显著补充向量检索的不足。

---

## 2. 混合检索（BM25 + 向量）

### 文件
- app/retriever.py

### 原理
同时执行 BM25 关键词搜索和 BGE 向量搜索，将两者的得分加权合并后返回 top-k：

```
用户问题
    |
    +-- BM25 搜索（关键词匹配，精准）
    +   返回 BM25 得分
    |
    +-- 向量搜索（语义匹配，泛化）
    +   返回余弦相似度
    |
    合并得分 = alpha x BM25 + (1-alpha) x Vector
    默认 alpha = 0.3
```

### 使用方法

```python
from app.retriever import search_hybrid
from app.ingest import build_embeddings

emb = build_embeddings()
query = "Python装饰器"
vec = emb.embed_query(query)
docs = search_hybrid(query, vec, k=4, alpha=0.3)
```

### 局限
- 中文分词用简单正则（re.findall 切分中英文），未使用 jieba
- BM25 对短查询效果有限（多数问题是 5-15 个字）

---

## 3. 查询改写（Multi-Query）

### 文件
- app/query_rewriter.py

### 原理
用 LLM（DeepSeek Chat）将原始问题重写为多个不同角度的问法，分别搜索后合并去重：

```
原始: "Python如何定义函数？"
  -> "Python中def关键字的用法？"
  -> "Python函数定义语法？"
  -> "如何在Python中创建函数？"
```

### 使用方法

```python
from app.query_rewriter import generate_queries
variations = generate_queries("Python装饰器是什么？", n=3)
for v in variations:
    print(v)
```

### 注意
- 每次调用需要调 DeepSeek API，耗时约 2-5 秒
- 后续可缓存常见问题的改写结果

---

## 4. 重排序（Reranker）

### 文件
- app/reranker.py

### 原理
基于关键词重叠对检索结果重新排序。对每个文档，计算其与查询的关键词交集大小，结合原始位置得分，重新排序后返回 top-k。

### 使用方法

```python
from app.retriever import search_hybrid
from app.reranker import rerank
from app.ingest import build_embeddings

emb = build_embeddings()
vec = emb.embed_query("Python函数")
docs = search_hybrid("Python函数", vec, k=4)
reranked = rerank("Python函数", docs, top_k=4)
```

### 当前局限
- 使用简单的关键词重叠，没有引入 cross-encoder 模型
- 未来可替换为 BAAI/bge-reranker-v2-m3 等专用重排序模型

---

## 5. 主界面整合

evaluation/runner.py 提供了完整的评测管线，可以对比基线（纯向量搜索）和优化版（混合+重排）的表现。

当前 runner 暂未包含 Multi-Query（因为需要调 DeepSeek API 会增加评测时间），但你可以通过 import 独立使用。

---

## 6. Phase 2 与 Phase 1 的对比

| 维度 | Phase 1 | Phase 2 |
|------|---------|---------|
| 检索方式 | 纯向量搜索 | 向量 + BM25 混合 |
| 查询策略 | 单个问题 | 支持 Multi-Query 改写 |
| 排序策略 | Cosine相似度 | 关键词重叠重排序 |
| 评测能力 | 无 | 有（20 QA pairs + 3 指标） |
| 知识库规模 | 1-2 篇笔记 | 11 篇笔记，43 个片段 |

---

## 7. 后续方向
- 引入 jieba 分词提升 BM25 在中文上的效果
- 接入 BGE Reranker 提升重排质量
- 用 Phase 2 的评测框架验证优化效果
- 基于 retriever.py 构建 Agent 的检索 Tool（Phase 3）
