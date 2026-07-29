---
**项目名称：基于 Agent + RAG 的知识库智能问答系统**
**技术栈：** Python、FastAPI、LangChain、LangGraph、BGE Embedding、DeepSeek Chat、numpy、diskcache、Redis

---

面向个人知识管理场景，设计并实现了一套基于 Agent + RAG 架构的智能问答后端系统，支持笔记检索、知识问答、Agent 自由对话与联网搜索等功能。项目经历了四个阶段的完整演进：从最小原型到工程化闭环。

- **多阶段 RAG 链路设计与优化**
  基于 LangChain LCEL 搭建"文档解析 → 语义分块 → BGE 向量化 → 相似度检索 → LLM 生成"的 RAG 链路。引入 BM25 混合检索、关键词重排序与 Multi-Query 查询改写策略，**Recall@4 从 0.350 提升至 0.575（+64%）**。构建小规模评测集（20 个 QA 对），从 Recall、MRR、Precision 等维度进行离线评估，基于 bad case 迭代 chunk 策略与 top-k 参数。

- **基于向量相似度的 Agent 决策机制**
  针对 LLM 判断"是否检索知识库"准确率低的问题，提出**"先搜索、后判断"的 Agent 决策架构**：使用 BGE 将用户问题向量化，计算与知识库的余弦相似度，以 0.35 为阈值自动路由至知识库检索、联网搜索或自由对话三种模式。解决了 Agent 跳过有内容笔记的关键问题。

- **LangGraph 多步骤工作流与状态管理**
  基于 LangGraph 构建 Agent 状态图，将 Agent 决策逻辑从 if-else 重构为**search_kb → generate 的有向图结构**，支持 retrieval_count 状态追踪与条件边路由，实现失败重试上限（最大 2 次）与中间结果传递，提升复杂查询场景下的执行稳定性。

- **FastAPI 服务封装与全链路优化**
  基于 FastAPI 搭建 REST 服务，封装 /query、/chat、/ingest、/health 等接口；引入 diskcache 响应缓存层（Redis 兼容接口），设计相同问题 TTL 缓存策略，**高频请求响应时间从 15s 降至秒级**；实现旋转文件日志 + 请求 ID 追踪的监控链路，覆盖节点耗时、异常信息与 token 消耗记录。

- **Web 交互界面与 Prompt 工程**
  设计并实现前后端分离的聊天式 Web 界面（纯 HTML/CSS/JS），支持聊天历史保留、Enter 发送 / Shift+Enter 换行、模式切换（严格 RAG / Agent），提升交互体验。围绕 Agent 决策、知识库问答、联网搜索、自由对话等场景设计结构化 Prompt 模板，支持版本管理与变量注入（如当前日期），有效消除 LLM 日期幻觉。

**项目成果：**
- 完成四阶段完整演进：最小 RAG → 优化检索（Recall +64%）→ Agent 智能体 → 工程化闭环
- 解决 ChromaDB ONNX 兼容性崩溃、langchain 导入顺序冲突、RunnableParallel 线程池挂起等 6 个关键技术难题
- 构建了覆盖 11 篇笔记、43 个文档片段的本地知识库，支持离线运行
- 设计了完整的缓存、日志、状态图与评测体系，具备可观测性与可扩展性
