# LangChain 框架基础

## LangChain 是什么

LangChain 是一个用于构建 LLM 应用的开源框架，提供了一套统一的抽象接口，简化了 RAG、Agent 等常见模式的开发。LangChain 支持多种 LLM 提供商、向量数据库、Embedding 模型，并且可以灵活组合。

## 核心抽象

### Document Loader（文档加载器）

负责从不同来源加载文档。支持 Markdown（TextLoader）、PDF（PyPDFLoader）、网页（WebBaseLoader）等。加载后的文档以 `Document` 对象表示，包含 `page_content`（文本内容）和 `metadata`（元数据）两个字段。

### Text Splitter（文本分块器）

将长文档切分为适合 Embedding 和检索的小片段。`RecursiveCharacterTextSplitter` 是最常用的分块器，它按优先级从高到低使用不同的分隔符（如标题、段落、句子）进行递归切分。

### Embedding 模型

提供统一的 Embedding 接口，支持切换不同的底层模型。LangChain 的 `HuggingFaceBgeEmbeddings` 可以直接加载 HuggingFace 上的 BGE 模型。

### Vector Store（向量存储）

提供统一的向量数据库接口。ChromaDB 是 LangChain 原生支持最完善的本地向量库，无需额外部署。其他选项包括 Qdrant、FAISS、Pinecone 等。

### Chain（链）

将多个组件串联起来形成可执行的流程。例如一个 RAG Chain 可能包含：Retriever → Prompt Template → LLM → Output Parser。

### Tool（工具）

Agent 可以调用的外部功能单元。LangChain 提供了丰富的内置 Tool，如计算器、搜索引擎、文档检索等，同时支持自定义 Tool。

## LangChain 与 LangGraph 的关系

LangChain 提供基础的 Chain 和 Agent 抽象，适用于大多数应用。LangGraph 则在 LangChain 之上增加了显式的状态图管理，让开发者可以用有向图的方式建模复杂的控制流，包括循环、分支、条件路由等。LangGraph 特别适合需要精细控制的多步 Agent 工作流。

## Message 类型

LangChain 使用标准化的消息类型表示对话：
- **SystemMessage**：系统级指令，设定助手的行为
- **HumanMessage**：用户输入
- **AIMessage**：模型的回复
- **FunctionMessage**：工具调用的返回结果
