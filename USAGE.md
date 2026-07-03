# RAG 知识库 — 完整使用说明

---

## 一、启动服务（选择一种模式）

### 1.1 终端交互模式

```bash
# 1. 进入项目目录
cd E:\AGitHub-project\RAG搭建

# 2. 激活虚拟环境（每次打开新终端都要先做这一步）
.venv\Scripts\activate

# 3. 选一个模式运行
python main.py --query      # 严格知识库问答
python main.py --chat       # Agent 模式（知识库+联网搜索+自由对话）
python main.py --ingest     # 重新建库（加新笔记后需要）
```

### 1.2 Web 服务模式

```bash
# 终端 A：启动服务（保持这个窗口开着）
cd E:\AGitHub-project\RAG搭建
.venv\Scripts\activate
python main.py --serve
# 终端 B：调用接口（新开一个窗口，也要先 activate）
.venv\Scripts\activate
```

---

## 二、Web 接口说明

服务启动后访问 `http://localhost:8000`，提供以下接口：

### 2.1 健康检查

**浏览器：** 直接打开 `http://localhost:8000/health`
**返回：** `{"status":"ok","version":"phase4"}`

### 2.2 知识库问答

```powershell
# PowerShell
Invoke-RestMethod -Uri http://localhost:8000/query `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"question":"Python装饰器是什么"}'
```

```bash
# CMD
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d "{\"question\":\"Python装饰器是什么\"}"
```

```python
# Python
import requests
r = requests.post("http://localhost:8000/query",
    json={"question": "Python装饰器是什么"})
print(r.json()["answer"])
```

### 2.3 Agent 聊天

```powershell
Invoke-RestMethod -Uri http://localhost:8000/chat `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"question":"赵日天是谁"}'
```

### 2.4 重新建库

```powershell
Invoke-RestMethod -Uri http://localhost:8000/ingest -Method POST
```

---

## 三、什么时候需要建库

```bash
python main.py --ingest
```

| 场景 | 需要吗 |
|------|--------|
| 第一次使用 | ✅ 需要 |
| data/ 里加了新笔记 | ✅ 需要 |
| 改了笔记内容 | ✅ 需要 |
| 只是问问题 | ❌ 不需要 |
| 重启电脑后 | ❌ 不需要 |

当前库里已有 **43 个片段**，来自 11 篇笔记。

---

## 四、当前已知状态

```bash
# 查看所有 Git 标签
git tag

# 各阶段进度
phase1-minimal-rag     # 最小 RAG — 已完成
phase2-optimized-rag   # 优化检索 — 已完成
phase3-agent           # Agent — 已完成
phase4-project-closure # 项目闭环 — 已完成
```

---

## 五、常见问题

### Q: 浏览器和终端的关系？
服务跑在一个终端窗口里，浏览器是独立的。**服务窗口要一直开着**，关了浏览器就连不上了。

### Q: 每次打开终端都要 activate 吗？
是的。`activate` 只在当前窗口生效：
```bash
.venv\Scripts\activate   # 每次新开终端都要执行
```

### Q: 怎么关掉服务？
在运行服务的窗口按 `Ctrl + C`。

### Q: 笔记放哪里？
```
data/
├── python-basics.md
├── python-oop.md
├── ml-basics.md
├── data-structures.md
├── 丧尸末日之我的宠物会修仙.md
└── ...（你自己的笔记）
```

只支持 `.md` 格式。加新笔记后需要 `--ingest`。

### Q: 换端口？
```bash
python main.py --serve   # 默认 8000
# 要改端口去 api.py 改 port 参数
```

### Q: 端口被占用了？
说明之前启动的服务没关。找到那个终端窗口按 Ctrl+C 关掉，或者换个端口。
