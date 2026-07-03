# Phase 1：最小 RAG 实现计划

**目标：** 构建最小 RAG 系统：Markdown 笔记 → BGE 嵌入 → 向量检索 → DeepSeek 生成回答

**技术栈：** Python 3.10+, LangChain, BGE (sentence-transformers), DeepSeek Chat, numpy 向量存储

---

## 全局约束

- Python >= 3.10
- DeepSeek API Key 存于 `.env`
- BGE 模型：`BAAI/bge-small-zh-v1.5`（自动下载缓存）
- 分块：chunk_size=500, overlap=50
- 检索：top_k=4
- 代码在 `app/` 包内，CLI 入口 `main.py`
- BGE 模型使用 hf-mirror.com 镜像下载（国内网络）

---

### Task 1：项目脚手架

**文件：** `pyproject.toml`, `.env`, `app/__init__.py`, `data/hello.md`

**步骤：**

```bash
mkdir -p app data evaluation tests notebooks
```

`pyproject.toml` 依赖：
```toml
dependencies = [
    "langchain-core>=0.3",
    "langchain-community>=0.3",
    "langchain-openai>=0.2",
    "chromadb>=0.5",
    "sentence-transformers>=3.0",
    "python-dotenv>=1.0",
    "pytest>=8.0",
]
```

`.env` 文件：
```
DEEPSEEK_API_KEY=sk-your-key-here
```

---

### Task 2：Config 配置模块

**文件：** `app/config.py`

从 `.env` 加载配置：
- `deepseek_api_key`, `deepseek_api_base`, `embedding_model_name`
- `chunk_size=500`, `chunk_overlap=50`, `retrieval_top_k=4`
- `chroma_persist_dir`, `data_dir`

**注意：** 缺失 API Key 时发出警告，不崩溃。

---

### Task 3：数据导入管线

**文件：** `app/ingest.py`

流程：
1. `load_markdown_files()` — 扫描 `data/` 下所有 `.md` 文件
2. `split_documents()` — 按 chunk_size=500 分块
3. `build_embeddings()` — 加载 BGE 模型（本地缓存路径）
4. `build_vector_store()` — 将向量存入 ChromaDB/numpy

**BGE 模型加载：**
```python
def build_embeddings():
    model_path = _get_model_path()  # 从本地缓存获取
    return HuggingFaceBgeEmbeddings(
        model_name=model_path,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
```

---

### Task 4：Chain + Query

**文件：** `app/chain.py`, `app/query.py`

**RAG 链（LCEL）：**
```python
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)
```

**Prompt 模板：**
```
Based on the context below, answer the question.
Context: {context}
Question: {question}
Answer:
```

**注意：** 因为 LangChain 的 RunnableParallel 在线程池中调用 ChromaDB/BGE 会挂起，改为先搜索再调用 LLM：
```python
def ask(question):
    context = _search(question)  # 主线程搜索
    chain = prompt | llm | StrOutputParser()  # 简单链
    return chain.invoke({"context": context, "question": question})
```

---

### Task 5：CLI 入口

**文件：** `main.py`

```bash
python main.py --ingest    # 建库
python main.py --query     # 交互式问答
```

---

### Task 6：错误处理 + 收尾

- 空库查询 → 提示 "请先运行 --ingest"
- `query_instruction` 参数优化搜索质量
- `langchain_community` 必须在 `langchain_openai` 之前导入

---

## 常见坑

### 1. ChromaDB 权限问题
ChromaDB 1.5.x 默认 ONNX 模型试图写入 `C:\Users\dad\.cache\chroma`，无权限时静默崩溃。
→ **解决：** 改用 numpy 向量库替代 ChromaDB。

### 2. BGE 模型下载被墙
huggingface.co 在中国无法直接访问。
→ **解决：** 设置 `HF_ENDPOINT=https://hf-mirror.com`，模型缓存到 `.hf_cache/`。

### 3. langchain 导入顺序
`langchain_openai` 在 `langchain_community` 之前导入会导致 OpenTelemetry 冲突，静默崩溃。
→ **解决：** 永远先导入 `from app.ingest import build_embeddings`。

### 4. RunnableParallel 线程池挂起
LCEL 链的 RunnableParallel 内部使用 ThreadPoolExecutor，其中一个线程调用 ChromaDB/BGE 时挂起。
→ **解决：** 先搜索再调 LLM，避免并行执行。

### 5. PowerShell 文件编码
PowerShell 的 `@\'@` heredoc 和 `-c` 参数在处理中文和引号时会有问题。
→ **解决：** 优先用 Python 的 `open().write()` 写文件。
