"""FastAPI web service for the RAG knowledge base."""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
