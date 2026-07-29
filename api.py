"""FastAPI web service for the RAG knowledge base."""

import json
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi import Query
from pydantic import BaseModel

from app.config import config

ROOT_DIR = Path(__file__).resolve().parent


def _background_warmup():
    try:
        from app.runtime import warmup_runtime
        warmup_runtime()
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("RAG_AUTO_WARMUP", "1") == "0":
        yield
        return
    threading.Thread(target=_background_warmup, daemon=True).start()
    yield


app = FastAPI(title="RAG Knowledge Base API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class QueryRequest(BaseModel):
    question: str


@app.get("/")
async def index():
    html = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html.read_text(encoding="utf-8"))


@app.get("/health")
async def health():
    return {"status": "ok", "version": "phase4"}


@app.get("/warmup")
@app.post("/warmup")
async def warmup():
    from app.runtime import warmup_runtime
    result = warmup_runtime()
    return {"status": "ok", **result}


@app.get("/reports")
async def reports():
    reports_dir = ROOT_DIR / "evaluation" / "reports"
    result = {}
    for name in ("latest.md", "answer_quality.md"):
        path = reports_dir / name
        result[name] = path.read_text(encoding="utf-8") if path.exists() else ""
    return {"reports": result}


@app.get("/kb/files")
async def kb_files():
    data_dir = Path(config.data_dir)
    files = []
    if data_dir.exists():
        for path in sorted(data_dir.glob("*.md")):
            stat = path.stat()
            files.append({
                "name": path.name,
                "size": stat.st_size,
                "modified": stat.st_mtime,
            })
    return {"files": files}


@app.get("/logs/recent")
async def recent_logs(limit: Annotated[int, Query(ge=1, le=500)] = 100):
    log_path = ROOT_DIR / "logs" / "rag.log"
    if not log_path.exists():
        return {"lines": []}
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return {"lines": lines[-limit:]}


@app.get("/config")
async def read_config():
    return {
        "deepseek_api_key": "set" if config.deepseek_api_key else "missing",
        "deepseek_api_base": config.deepseek_api_base,
        "deepseek_model": config.deepseek_model,
        "embedding_model_name": config.embedding_model_name,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "retrieval_top_k": config.retrieval_top_k,
        "rag_min_score": config.rag_min_score,
        "data_dir": config.data_dir,
    }


@app.post("/query")
async def query(req: QueryRequest):
    from app.query import ask_with_sources
    return ask_with_sources(req.question)


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    from app.query import ask_stream_events

    def generate():
        for event in ask_stream_events(req.question):
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.post("/chat")
async def chat(req: QueryRequest):
    from app.agent import chat as agent_chat
    return {"answer": agent_chat(req.question)}


@app.post("/chat/stream")
async def chat_stream(req: QueryRequest):
    from app.agent import chat_stream_events

    def generate():
        for event in chat_stream_events(req.question):
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.post("/ingest")
async def ingest():
    os.environ["HF_HOME"] = str(Path(__file__).parent / ".hf_cache")
    from app.ingest import run_ingest
    n = run_ingest()
    return {"chunks": n, "status": "ok"}
