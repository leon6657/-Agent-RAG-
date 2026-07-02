# Embedding 模型与向量检索

## 什么是文本嵌入

文本嵌入（Text Embedding）是将自然语言文本转换为固定长度的浮点数向量的过程。一个好的 Embedding 模型应该使得语义相似的文本在向量空间中距离相近，语义不相关的文本距离较远。

## 余弦相似度

衡量两个向量相似程度最常用的指标是余弦相似度（Cosine Similarity），计算两个向量之间夹角的余弦值。值域为 [-1, 1]，越接近 1 表示越相似。

## BGE 系列模型

BGE（BAAI General Embedding）是由北京智源人工智能研究院（BAAI）发布的一系列 Embedding 模型。BGE 模型在中文和英文场景下都有出色的表现，是开源社区最广泛使用的 Embedding 模型之一。

常用模型：
- **BAAI/bge-small-zh-v1.5**：轻量级中文模型，向量维度 512，推理速度快，适合本地部署和原型开发
- **BAAI/bge-base-zh-v1.5**：中等规模中文模型，向量维度 768，效果更好但速度略慢
- **BAAI/bge-large-zh-v1.5**：大规模中文模型，效果最佳但资源消耗最高

## 向量维度的意义

向量维度决定了模型能编码的信息量上限。维度越高，模型表达力越强，但存储和检索的计算成本也更高。bge-small（512维）对于大多数 RAG 应用已经足够。

## 本地 Embedding vs API Embedding

在实际工程中，Embedding 可以通过本地模型或云端 API 两种方式实现：

- **本地模型**：如 BGE、Sentence-BERT，无需网络，零成本，适合开发和实验
- **API 服务**：如 OpenAI Embeddings、DeepSeek Embeddings，使用方便但产生费用

在简历项目中，使用本地 BGE 模型可以展示对模型本身的理解，同时避免 API 费用。

## 混合检索（Hybrid Search）

将向量语义检索与传统的关键词检索（如 BM25）相结合，可以同时利用语义匹配和精确匹配的优势，提升检索的召回率。混合检索是 RAG 优化中的重要手段。
