# RAG 知识库 — 完整使用说明

> 更新时间：Phase 4 项目闭环

---

## 一、快速启动

```bash
# 1. 进入项目目录
cd E:\AGitHub-project\RAG搭建

# 2. 激活虚拟环境（每次新开终端都要做）
.venv\Scripts\activate

# 3. 配置 API Key
#    编辑 .env 文件，填入你的 DeepSeek API Key
#    获取地址：https://platform.deepseek.com
#    DEEPSEEK_API_KEY=sk-your-key-here

# 4. 建库（首次使用或添加新笔记后需要）
python main.py --ingest

# 5. 选择使用方式
python main.py --query      # 严格 RAG（仅笔记）
python main.py --chat       # Agent（笔记+联网+聊天）
python main.py --serve      # Web 服务（推荐）
```

---

## 二、使用方式对比

| 方式 | 命令 | 能做什么 | 不能做什么 |
|------|------|---------|-----------|
| 严格 RAG | `--query` | 笔记相关问题 | 笔记外的问题→说不知道 |
| Agent | `--chat` | 笔记 + 联网 + 聊天 | 实时信息联网有误差 |
| Web 服务 | `--serve` | 浏览器可视化交互 | 需要启动服务 |

---

## 三、Web 服务（推荐）

### 3.1 启动

```bash
.venv\Scripts\activate
python main.py --serve
```

看到以下输出说明启动成功：
```
Uvicorn running on http://0.0.0.0:8000
```

### 3.2 浏览器使用

打开 `http://localhost:8000`，进入聊天界面：

```
┌─────────────────────────────────────┐
│  RAG 知识库                          │
│  ○ 严格 RAG    ○ Agent 模式          │
├─────────────────────────────────────┤
│  你：Python装饰器是什么              │
│  AI：根据笔记内容，装饰器是一个函数... │
│                                     │
│  你：今天是几月几日                  │
│  AI：2026年7月2日                    │
├─────────────────────────────────────┤
│ [输入框... Enter 发送]     [发送]    │
└─────────────────────────────────────┘
```

**快捷键：**
- `Enter` — 发送消息
- `Shift + Enter` — 换行
- 历史记录自动保留在页面中

### 3.3 REST API

所有接口返回 JSON，Content-Type: `application/json`。

**健康检查：**
```bash
# 浏览器直接打开
http://localhost:8000/health

# 返回：{"status": "ok", "version": "phase4"}
```

**知识库问答：**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"Python装饰器是什么\"}"

# 返回：{"answer": "根据笔记内容..."}
```

**Agent 聊天：**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"赵日天是谁\"}"

# 返回：{"answer": "赵日天是一只十四斤重的橘猫..."}
```

**重新建库：**
```bash
curl -X POST http://localhost:8000/ingest

# 返回：{"chunks": 43, "status": "ok"}
```

**PowerShell 的调用方式：**
```powershell
Invoke-RestMethod -Uri http://localhost:8000/query `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"question":"Python装饰器是什么"}'
```

**Python 调用：**
```python
import requests
r = requests.post("http://localhost:8000/query",
    json={"question": "Python装饰器是什么"})
print(r.json()["answer"])
```

---

## 四、终端交互

### 4.1 严格 RAG

```bash
python main.py --query
```

- 只从你的笔记中找答案
- 笔记里没有相关内容的，直接说不知道
- 适合需要准确引用的场景

### 4.2 Agent 模式

```bash
python main.py --chat
```

自动判断：
- 笔记相关内容 → 搜索知识库
- 实时/不确定信息 → 联网搜索
- 日常对话/闲聊 → 自由聊天

---

## 五、建库（什么时候需要 --ingest）

```bash
python main.py --ingest
```

| 场景 | 需要建库吗 |
|------|-----------|
| 第一次使用 | ✅ 必须 |
| data/ 里加了新笔记 | ✅ 必须 |
| 修改了笔记内容 | ✅ 必须 |
| 只是问问题 | ❌ 不需要 |
| 重启电脑后 | ❌ 不需要 |

当前库里已有 **43 个片段**，来自 11 篇笔记。

**笔记格式要求：**
- 只支持 `.md` 文件
- 放在 `data/` 目录下
- 建议用中文文件名

---

## 六、评测

```bash
.venv\Scripts\activate
python evaluation/runner.py
```

输出示例：
```
BASELINE (Vector Only):
    Recall@4:   0.350
    MRR:        0.310
OPTIMIZED (Hybrid):
    Recall@4:   0.575
    MRR:        0.572
```

评测集：20 个 QA 对，覆盖笔记各章节。

---

## 七、缓存与日志

### 缓存

```bash
# 缓存目录：.cache/responses/
# 每次问答的响应会自动缓存
# 相同问题在 1 小时内秒回

# 清空缓存
python -c "from app.cache import clear; clear()"
```

### 日志

```bash
# 日志目录：logs/rag.log
# 每次请求自动记录，5MB 滚动
# 包含请求 ID，方便追踪问题
```

---

## 八、常见问题

### 端口被占用
```bash
# 默认端口 8000 被占用时，找到旧进程关掉
netstat -ano | findstr :8000
taskkill /PID <进程ID> /F
```

### 乱码
中文乱码是控制台编码问题（GBK vs UTF-8），用 **Web 服务模式**（浏览器打开 `http://localhost:8000`）可以避免。

### 联网搜索不准确
DeepSeek 联网搜索的结果可能不准确。对于需要确保准确的信息，写入笔记后再用 `--query` 或 `--ingest`。

### 响应慢（首次 10-20s）
- 首次：导入依赖 + 加载 BGE 模型 + DeepSeek API ≈ 15s
- 后续：缓存命中后秒回
- 建议保持 Web 服务运行，避免频繁重启

### 服务挂了怎么办
Web 服务在终端前台运行，关闭终端或按 `Ctrl+C` 会停止。保持终端窗口打开即可。
