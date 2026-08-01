# 阿里云服务器部署问题复盘

本文档记录 RAG Workbench 项目从本地部署到阿里云服务器过程中遇到的问题、排查过程和最终解决方案。它既可以作为后续重新部署的参考，也可以作为项目工程化经历的总结材料。

## 1. 部署目标

本次部署目标是将本地 RAG Workbench 项目发布到公网，使其他用户可以通过浏览器访问 Web 问答页面。

最终采用的方案是：

- 本机使用 Docker 构建项目镜像
- 将镜像导出为 `rag-workbench-demo.tar`
- 通过阿里云 Workbench 上传到服务器
- 服务器执行 `docker load` 导入镜像
- 使用 Docker 容器运行 FastAPI Web 服务
- 通过公网 IP + 端口访问项目

最终公网访问形式：

```text
http://服务器公网IP:8001
```

## 2. 服务器选择问题

最开始使用的是 2 核 1GB 左右内存的轻量应用服务器。部署后虽然服务可以启动，但模型回答速度非常慢。

排查服务器资源后发现：

```text
Mem total:      896Mi
Mem available: 80Mi
Swap:          0B
```

这说明服务器内存几乎耗尽。RAG 项目运行时需要加载 FastAPI、sentence-transformers、BGE embedding 模型、numpy 检索逻辑和 Docker 容器运行环境，1GB 内存明显不足。

后续更换为 ECS 4 核 8GB 服务器后，资源占用情况明显改善：

```text
Mem available: 6.1Gi
Container Mem: 413.5MiB / 7.096GiB
CPU:           0.12%
```

结论：

- 1GB 内存不适合部署该 RAG 项目
- 2 核 4GB 可以作为公网 Demo 的起步配置
- 4 核 8GB 更适合稳定展示和调试

## 3. Docker 构建下载慢的问题

在服务器上直接执行：

```bash
sudo docker build -t rag-workbench-demo .
```

时，依赖安装非常慢，主要卡在 `pip install` 阶段。

主要原因有两个：

1. 服务器访问 `files.pythonhosted.org`、`download.pytorch.org` 等国外源不稳定
2. 项目依赖中包含 `torch`、`sentence-transformers`、`transformers` 等较大的包

日志中出现过：

```text
ReadTimeoutError: HTTPSConnectionPool(host='files.pythonhosted.org', port=443): Read timed out.
```

尝试过的优化方式：

- 使用阿里云 PyPI 镜像源
- 单独安装 PyTorch CPU 版本
- 增加 pip timeout 和 retries

这些方式可以缓解部分依赖下载问题，但仍然容易在大包下载时卡住。

最终采用的稳定方案是：

```text
本机 Docker 构建镜像 -> docker save 导出 -> 上传服务器 -> docker load 导入
```

这样服务器不再负责下载 Python 依赖，只负责运行镜像。

## 4. SSH 上传失败问题

本机使用 `scp` 上传镜像文件时，出现：

```text
Permission denied (publickey,gssapi-keyex,gssapi-with-mic)
```

原因是服务器 SSH 不允许密码登录，只接受密钥登录，而本机没有配置对应私钥。

解决方案：

- 没有继续修改 SSH 配置
- 改用阿里云 Workbench 文件管理上传

上传路径需要注意：

- 普通 `admin` 用户上传目录通常是 `/home/admin`
- `root` 用户上传目录是 `/root`
- Linux 中不存在默认的 `/home/root`

最终使用的上传路径：

```text
/root/rag-workbench-demo.tar
```

## 5. 端口和安全组问题

容器内部服务监听端口是：

```text
8000
```

服务器对外暴露端口使用：

```text
8001
```

Docker 启动命令中的端口映射：

```bash
-p 8001:8000
```

含义是：

```text
公网访问服务器 8001 端口 -> 转发到容器内部 8000 端口
```

因此阿里云安全组或防火墙中需要放行：

```text
协议：TCP
端口：8001/8001
来源：0.0.0.0/0
```

常见误区：

- 删除安全组中的 8001 规则，只是关闭公网访问，不等于删除项目
- 删除 Docker 容器，才是停止项目服务
- 释放服务器实例，才是删除整台服务器

## 6. 容器启动和健康检查

服务器导入镜像：

```bash
docker load -i /root/rag-workbench-demo.tar
```

启动容器：

```bash
docker rm -f rag-workbench-demo || true

docker run -d \
  --name rag-workbench-demo \
  --restart unless-stopped \
  --env-file /opt/rag-project/.env \
  -p 8001:8000 \
  rag-workbench-demo:aliyun
```

健康检查：

```bash
curl http://127.0.0.1:8001/health
```

成功返回：

```json
{"status":"ok","version":"phase4"}
```

公网访问：

```text
http://服务器公网IP:8001
```

## 7. 模型缓存缺失问题

项目部署成功后，访问页面可以打开，但提问时非常慢，并且流式接口出现异常：

```text
curl: (18) transfer closed with outstanding read data remaining
```

查看容器日志发现：

```text
[WARN] Local model not found, downloading from HuggingFace...
Network is unreachable
```

根因是：

- 本机已经有 `BAAI/bge-small-zh-v1.5` 的 HuggingFace 模型缓存
- 但 `.dockerignore` 中排除了 `.hf_cache/`
- 导致 Docker 镜像没有携带 embedding 模型
- 服务器运行时尝试访问 HuggingFace 下载模型
- 服务器网络无法访问 HuggingFace，最终请求卡住并断开

修复方案：

1. 修改 `.dockerignore`，允许模型缓存进入 Docker 构建上下文：

```text
# 移除
.hf_cache/
.model_cache/

# 添加
*.tar
```

2. 修改 Dockerfile，开启 HuggingFace 离线模式：

```dockerfile
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
```

3. 在本机重新构建镜像并导出：

```powershell
docker build -t rag-workbench-demo:aliyun .
docker save -o rag-workbench-demo.tar rag-workbench-demo:aliyun
```

4. 重新上传并在服务器导入运行。

修复后，容器会直接从镜像内加载本地模型缓存，不再访问 HuggingFace。

## 8. 性能瓶颈判断

RAG 服务的完整链路是：

```text
用户提问
-> 服务器容器内使用 BGE 模型计算 query embedding
-> 在 vector_store.json 中进行向量检索
-> 将召回片段拼接进 prompt
-> 调用 DeepSeek API 生成答案
-> 页面流式输出
```

其中：

- embedding 和检索在服务器执行
- 最终文本生成由 DeepSeek API 执行

如果服务器资源占用很低，但回答仍然慢，需要区分：

1. 是否是首次请求未预热
2. 是否是容器在尝试下载 embedding 模型
3. 是否是 DeepSeek API 首包响应慢
4. 是否使用了 Agent 模式导致额外决策或联网搜索
5. 知识库是否包含小说类长文本，导致 chunk 数过多

常用排查命令：

```bash
free -h
docker stats --no-stream rag-workbench-demo
docker logs --tail=200 rag-workbench-demo
curl http://127.0.0.1:8001/warmup
```

测试严格 RAG：

```bash
time curl -N \
  -H "Content-Type: application/json" \
  -d '{"question":"请介绍一下这个 RAG-Agent 项目的核心功能。"}' \
  http://127.0.0.1:8001/query/stream
```

测试 Agent：

```bash
time curl -N \
  -H "Content-Type: application/json" \
  -d '{"question":"请介绍一下这个 RAG-Agent 项目的核心功能。"}' \
  http://127.0.0.1:8001/chat/stream
```

## 9. 后续维护命令

查看容器：

```bash
docker ps -a
```

查看日志：

```bash
docker logs --tail=100 rag-workbench-demo
```

重启服务：

```bash
docker restart rag-workbench-demo
```

停止服务：

```bash
docker stop rag-workbench-demo
```

删除容器：

```bash
docker rm -f rag-workbench-demo
```

删除镜像：

```bash
docker rmi rag-workbench-demo:aliyun
```

删除部署目录：

```bash
rm -rf /opt/rag-project
```

删除上传的镜像包：

```bash
rm -f /root/rag-workbench-demo.tar
```

## 10. 本次部署经验总结

本次部署暴露出 RAG 项目上云时几个典型工程问题：

1. 服务器规格不能只看 CPU，embedding 模型对内存也有要求
2. 国内服务器直接从国外源构建 Docker 镜像不稳定
3. 大模型应用需要提前处理模型缓存和离线部署问题
4. Docker 端口映射和阿里云安全组需要同时配置
5. Workbench 文件上传路径要区分普通用户目录和 root 目录
6. 项目能打开页面不代表推理链路完整可用，还需要测试 `/warmup` 和真实问答接口

最终较稳定的部署路径是：

```text
本机完成 Docker 构建
-> 镜像内携带依赖、知识库索引和 embedding 模型缓存
-> docker save 导出 tar
-> Workbench 上传服务器
-> docker load 导入
-> docker run 暴露 8001 端口
-> 阿里云安全组放行 8001
-> 访问 /health 和 /warmup 验证
```

这条路径避免了服务器构建阶段的大量网络下载问题，也让公网 Demo 更可复现、更容易迁移。
