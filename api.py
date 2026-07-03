"""FastAPI web service for the RAG knowledge base."""

import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="RAG Knowledge Base API")
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


@app.post("/query")
async def query(req: QueryRequest):
    from app.query import ask
    return {"answer": ask(req.question)}


@app.post("/chat")
async def chat(req: QueryRequest):
    from app.agent import chat as agent_chat
    return {"answer": agent_chat(req.question)}


@app.post("/ingest")
async def ingest():
    os.environ["HF_HOME"] = str(Path(__file__).parent / ".hf_cache")
    from app.ingest import run_ingest
    n = run_ingest()
    return {"chunks": n, "status": "ok"}
