"""FastAPI web service for the RAG knowledge base."""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="RAG Knowledge Base API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str


_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>RAG 知识库</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system, system-ui, sans-serif;background:#f5f5f7;padding:20px;color:#333}
.container{max-width:700px;margin:0 auto}
h1{font-size:22px;margin-bottom:8px;color:#1d1d1f}
p.desc{font-size:14px;color:#6e6e73;margin-bottom:20px}
.card{background:#fff;border-radius:12px;padding:20px;box-shadow:0 2px 10px rgba(0,0,0,.05)}
label{font-size:13px;font-weight:600;color:#1d1d1f;display:block;margin-bottom:6px}
textarea{width:100%;padding:10px 12px;border:1px solid #d2d2d7;border-radius:8px;font-size:14px;resize:vertical;min-height:60px;font-family:inherit}
textarea:focus{outline:none;border-color:#0071e3;box-shadow:0 0 0 3px rgba(0,113,227,.15)}
.modes{display:flex;gap:16px;margin:14px 0}
.modes label{font-weight:400;font-size:14px;display:flex;align-items:center;gap:6px;cursor:pointer}
.btn{background:#0071e3;color:#fff;border:none;padding:8px 24px;border-radius:20px;font-size:14px;cursor:pointer}
.btn:hover{background:#0077ed}
.btn:disabled{opacity:.5;cursor:not-allowed}
#output{margin-top:16px;padding:14px;border-radius:8px;background:#f5f5f7;font-size:14px;line-height:1.6;white-space:pre-wrap;display:none;min-height:40px}
#output.show{display:block}
.loading{color:#6e6e73;font-size:13px;margin-top:10px;display:none}
.loading.show{display:block}
.err{color:#d32f2f}
</style>
</head>
<body>
<div class="container">
<h1>RAG 知识库</h1>
<p class="desc">基于笔记的问答 / Agent 聊天 / 联网搜索</p>
<div class="card">
<label for="q">问题</label>
<textarea id="q" rows="2" placeholder="输入你的问题..."></textarea>
<div class="modes">
<label><input type="radio" name="mode" value="query" checked> 严格 RAG</label>
<label><input type="radio" name="mode" value="chat"> Agent 模式</label>
</div>
<button class="btn" id="btn" onclick="run()">发送</button>
<div class="loading" id="loading">正在回答...</div>
<div id="output"></div>
</div>
</div>
<script>
function run(){
    const q=document.getElementById("q").value.trim();
    if(!q)return;
    const mode=document.querySelector('input[name="mode"]:checked').value;
    const btn=document.getElementById("btn");
    const loading=document.getElementById("loading");
    const output=document.getElementById("output");
    btn.disabled=true;
    loading.className="loading show";
    output.className="";
    fetch("/"+mode,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({question:q})})
    .then(r=>r.json())
    .then(d=>{output.textContent=d.answer||d.detail||"无返回";output.className="show"})
    .catch(e=>{output.textContent="错误: "+e;output.className="show err"})
    .finally(()=>{btn.disabled=false;loading.className="loading"});
}
</script>
</body>
</html>"""


@app.get("/")
async def index():
    return HTMLResponse(_HTML)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "phase4"}


@app.post("/query")
async def query(req: QueryRequest):
    from app.query import ask
    answer = ask(req.question)
    return {"answer": answer}


@app.post("/chat")
async def chat(req: QueryRequest):
    from app.agent import chat as agent_chat
    answer = agent_chat(req.question)
    return {"answer": answer}


@app.post("/ingest")
async def ingest():
    os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), ".hf_cache")
    from app.ingest import run_ingest
    n = run_ingest()
    return {"chunks": n, "status": "ok"}
