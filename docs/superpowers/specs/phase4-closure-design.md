# Phase 4：项目闭环详细设计

> 对应 Git 标签：`phase4-project-closure`

---

## 1. 概述

将 RAG 系统从命令行工具升级为可部署的 Web 服务，同时补齐缓存优化、日志监控、LangGraph 状态图和 Web 界面。

## 2. 架构

```
┌─ 浏览器 ─────────────────────┐
│  http://localhost:8000        │
│  聊天界面 (static/index.html) │
└──────────────┬───────────────┘
               │ POST /query, /chat
               ▼
┌──────────────────────────────────┐
│  FastAPI 服务 (api.py)           │
│  端口 8000                       │
│  CORS 全开放                     │
├──────────────────────────────────┤
│  /query → app.query.ask()        │
│  /chat  → app.agent.chat()       │
│  /ingest → app.ingest.run()      │
│  /health → {"status": "ok"}      │
└──────────────────────────────────┘
               │
     ┌─────────┼─────────┐
     ▼         ▼         ▼
┌────────┐ ┌──────┐ ┌──────────┐
│ Cache  │ │Log   │ │ LangGraph│
│diskcache│ │文件+ │ │ 状态图    │
│Redis接口│ │控制台│ │ 条件边    │
└────────┘ └──────┘ └──────────┘
```

## 3. 组件设计

### 3.1 FastAPI 服务 (`api.py`)

**接口定义：**

| 方法 | 路径 | 参数 | 返回 | 延迟 |
|------|------|------|------|------|
| GET | / | — | HTML 页面 | 即时 |
| GET | /health | — | {"status":"ok"} | 即时 |
| POST | /query | {"question": "..."} | {"answer": "..."} | 5-15s |
| POST | /chat | {"question": "..."} | {"answer": "..."} | 3-15s |
| POST | /ingest | — | {"chunks": N} | 10-20s |

**技术选型：**
- FastAPI（异步框架，自动生成 OpenAPI 文档）
- Uvicorn（ASGI 服务器）
- Pydantic（请求/响应模型校验）

**CORS：** 全开放，允许前后端分离部署。

### 3.2 缓存层 (`app/cache.py`)

**设计目标：** 相同问题在 TTL 内直接返回缓存结果，省去 BGE 加载 + LLM 调用时间。

**技术选型：** diskcache（文件级键值存储，无需额外服务）

**缓存策略：**
```
Key: MD5(mode + question.lower())
Value: 响应文本
TTL: 3600 秒（1 小时）
```

**Redis-ready 接口：** 方法签名与 Redis 一致，可无缝切换：

```python
# 当前（diskcache）
from app.cache import get, set

# 切换 Redis（代码不变）
from app.cache import get, set  # 内部实现替换
# 见文件底部的 Redis 实现示例
```

### 3.3 日志系统 (`app/logging_setup.py`)

**结构：**

| 组件 | 配置 |
|------|------|
| 输出目标 | 文件 + 控制台 |
| 文件处理器 | RotatingFileHandler, 5MB 滚动, 保留 3 个备份 |
| 日志路径 | `logs/rag.log` |
| 格式 | `[时间] 级别 请求ID 消息` |
| 请求 ID | UUID hex[:12]，每次 HTTP 请求自动分配 |

**使用方式：**

```python
from app.logging_setup import logger
logger.info("Processing query", extra={"request_id": rid})
```

### 3.4 LangGraph 状态图 (`app/state.py` + `app/graph.py`)

**状态定义：**

```python
class AgentState(TypedDict):
    messages: List[dict]     # 对话历史
    source: str              # 信息来源（"kb"/"web"/"chat"）
    retrieval_count: int     # 已检索次数（防无限循环）
    context: str             # 检索到的上下文
    response: str            # 最终回答
```

**图结构：**

```
entry ──→ search_kb
              │
          retrieval_count
          ┌────┴────┐
          │ < 2     │ ≥ 2
          ▼         ▼
      search_kb  generate ──→ END
```

**关键设计：**
- `retrieval_count` 限制最大检索 2 次
- 条件边根据状态决定下一节点
- 编译后的图支持 `.invoke()` 接口

### 3.5 Web 界面 (`static/index.html`)

**设计目标：** 替代命令行交互，提供直观的聊天界面。

**技术方案：** 纯前端 HTML + CSS + JavaScript（无框架依赖）

**核心功能：**
- 聊天气泡布局（用户 / AI 分左右）
- Enter 发送，Shift+Enter 换行
- 输入框自动增高
- 自动滚动到底部
- 模式切换（严格 RAG / Agent）

**通信方式：** Fetch API POST JSON

## 4. 关键决策

### 4.1 为什么不把 HTML 嵌入 Python？
- 分开维护：前端改样式不碰后端代码
- 避免 PowerShell heredoc 大小限制
- 独立开发：可以用 IDE 的 HTML/CSS 预览

### 4.2 为什么用 diskcache 而非 Redis？
- 不需要安装 Redis 服务
- 零配置，即装即用
- 接口兼容，将来可以切换到 Redis：
  1. 修改 `app/cache.py` 的实现
  2. 在 `.env` 加 `REDIS_URL`
  3. 重启服务

### 4.3 LangGraph 为什么加到 Phase 4 而非 Phase 3？
Phase 3 的 Agent 用 if-else 流程可以工作。LangGraph 提供了：
- **可追踪性**：每一步的状态变化都记录
- **可控制性**：条件边比 if-else 更灵活
- **可扩展性**：加新节点不影响现有逻辑

## 5. 输出

```bash
python main.py --serve
# 浏览器访问 http://localhost:8000
```
