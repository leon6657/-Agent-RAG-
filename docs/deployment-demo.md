# 固定知识库 Demo 部署说明

这个部署方案面向在线演示：知识库内容固定，镜像构建时根据 `data/` 自动生成 `vector_store.json`，线上只提供问答和展示，不依赖用户上传文档。

## 部署形态

- 后端：FastAPI + Uvicorn
- 前端：`static/index.html`
- 知识库：仓库中的 `data/*.md`
- 向量库：Docker 构建阶段执行 `python main.py --ingest` 自动生成
- 密钥：只通过云平台环境变量配置，不写入仓库

## 必需环境变量

```text
DEEPSEEK_API_KEY=你的 DeepSeek API Key
DEEPSEEK_MODEL=deepseek-v4-flash
RAG_MIN_SCORE=0.35
```

云平台通常会自动提供 `PORT`。如果需要手动填写，设置为 `8000` 即可。

## 本地 Docker 验证

```bash
docker build -t rag-workbench-demo .
docker run --rm -p 8000:8000 ^
  -e DEEPSEEK_API_KEY=你的 DeepSeek API Key ^
  -e DEEPSEEK_MODEL=deepseek-v4-flash ^
  -e RAG_MIN_SCORE=0.35 ^
  rag-workbench-demo
```

启动后打开：

```text
http://localhost:8000
```

健康检查：

```text
http://localhost:8000/health
```

## Render 部署

1. 将项目推送到 GitHub。
2. 在 Render 新建 Web Service。
3. 选择 GitHub 仓库。
4. Runtime 选择 Docker。
5. 在 Environment 中添加：
   - `DEEPSEEK_API_KEY`
   - `DEEPSEEK_MODEL`
   - `RAG_MIN_SCORE`
6. Health Check Path 填 `/health`。
7. 创建服务并等待构建完成。

Render 会根据仓库根目录的 `Dockerfile` 构建镜像。构建过程中会自动运行 `python main.py --ingest`。

## 免费实例预热

Render 免费实例在空闲后可能休眠。重新唤醒时，服务需要重新加载 Python 应用、向量库和 embedding 模型，因此首次提问会比后续提问慢。

项目提供了运行时预热接口：

```text
https://你的域名/warmup
```

例如：

```text
https://rag-workbench-demo.onrender.com/warmup
```

该接口会提前加载 embedding 模型和向量库缓存。部署后演示前可以先访问一次 `/warmup`，再回到主页提问。

接口返回示例：

```json
{
  "status": "ok",
  "embedding_model": "ready",
  "vector_store": "ready",
  "vector_count": 83
}
```

这不能消除 Render 免费实例的冷启动，但可以避免用户第一次提问时才触发模型加载。

## Railway 部署

1. 在 Railway 新建 Project。
2. 选择 Deploy from GitHub repo。
3. 选择本项目仓库。
4. Railway 会识别根目录 `Dockerfile`。
5. 在 Variables 中添加：
   - `DEEPSEEK_API_KEY`
   - `DEEPSEEK_MODEL`
   - `RAG_MIN_SCORE`
6. 部署完成后访问 Railway 生成的公网域名。

## 固定 Demo 的限制

- 线上新增文档不会持久保存。
- 如果修改 `data/` 中的知识库，需要重新提交代码并触发重新部署。
- `vector_store.json` 不需要提交到 GitHub，镜像构建时会自动生成。
- `监控 / Admin` 中的重新建库按钮仍可用于演示，但固定 Demo 的正式更新应通过修改仓库后重新部署完成。

## 推荐上线前检查

```bash
python main.py --ingest
python -m pytest -q
```

确认本地测试通过后再推送到 GitHub。
