"""FastAPI web service for the RAG knowledge base."""

from datetime import datetime
import json
import os
import re
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi import Query
from pydantic import BaseModel

from app.config import config

ROOT_DIR = Path(__file__).resolve().parent
_LAST_INTERACTION: dict[str, Any] | None = None


def _log_path() -> Path:
    return ROOT_DIR / "logs" / "rag.log"


def write_runtime_log(action: str, **fields: Any) -> None:
    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "action": action,
        **fields,
    }
    path = _log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def record_interaction(
    question: str,
    mode: str,
    top_score: float | None = None,
    sources_count: int = 0,
) -> dict[str, Any]:
    global _LAST_INTERACTION
    _LAST_INTERACTION = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "mode": mode,
        "top_score": top_score,
        "sources_count": sources_count,
    }
    write_runtime_log("query", **_LAST_INTERACTION)
    return _LAST_INTERACTION


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: str) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _parse_latest_retrieval_report() -> dict[str, Any]:
    path = ROOT_DIR / "evaluation" / "reports" / "latest.md"
    if not path.exists():
        return {}

    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("|") or "---" in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells or cells[0] == "Run":
            continue
        if len(cells) >= 8:
            rows.append(
                {
                    "run": cells[0],
                    "questions": _to_int(cells[1]),
                    "k": _to_int(cells[2]),
                    "recall": _to_float(cells[3]),
                    "mrr": _to_float(cells[4]),
                    "precision": _to_float(cells[5]),
                    "source_hit": _to_float(cells[6]),
                    "misses": _to_int(cells[7]),
                }
            )

    for row in rows:
        if re.search(r"optimized|优化", row["run"], re.IGNORECASE):
            return row
    return rows[-1] if rows else {}


def _record_from_response(question: str, response: dict[str, Any], action: str = "query") -> None:
    sources = response.get("sources") if isinstance(response, dict) else []
    top_score = response.get("top_score") if isinstance(response, dict) else None
    mode = response.get("mode", action) if isinstance(response, dict) else action
    record_interaction(
        question=question,
        mode=str(mode),
        top_score=top_score if isinstance(top_score, (int, float)) else None,
        sources_count=len(sources) if isinstance(sources, list) else 0,
    )


def _background_warmup():
    try:
        from app.runtime import warmup_runtime
        result = warmup_runtime()
        write_runtime_log("warmup", status="ok", **result)
    except Exception:
        write_runtime_log("warmup", status="failed")


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
    write_runtime_log("warmup", status="ok", **result)
    return {"status": "ok", **result}


@app.get("/reports")
async def reports():
    reports_dir = ROOT_DIR / "evaluation" / "reports"
    result = {}
    for name in ("latest.md", "answer_quality.md"):
        path = reports_dir / name
        result[name] = path.read_text(encoding="utf-8") if path.exists() else ""
    return {"reports": result}


@app.get("/metrics")
async def metrics():
    from app import store

    data_dir = Path(config.data_dir)
    md_files = list(data_dir.glob("*.md")) if data_dir.exists() else []
    return {
        "retrieval": _parse_latest_retrieval_report(),
        "knowledge_base": {
            "chunks": store.count(),
            "files": len(md_files),
        },
        "last_answer": _LAST_INTERACTION,
    }


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
    log_path = _log_path()
    if not log_path.exists():
        return {"lines": ["暂未产生日志。发送一次问题、刷新预热或重新建库后，这里会显示运行记录。"]}
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return {"lines": ["日志文件为空。发送一次问题后会自动写入新的运行记录。"]}
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
    result = ask_with_sources(req.question)
    _record_from_response(req.question, result, action="query")
    return result


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    from app.query import ask_stream_events

    def generate():
        meta: dict[str, Any] = {}
        for event in ask_stream_events(req.question):
            if event.get("event") == "meta":
                meta = event
            yield json.dumps(event, ensure_ascii=False) + "\n"
        _record_from_response(req.question, meta, action="query_stream")

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.post("/chat")
async def chat(req: QueryRequest):
    from app.agent import chat as agent_chat
    answer = agent_chat(req.question)
    record_interaction(req.question, mode="agent", top_score=None, sources_count=0)
    return {"answer": answer}


@app.post("/chat/stream")
async def chat_stream(req: QueryRequest):
    from app.agent import chat_stream_events

    def generate():
        meta: dict[str, Any] = {}
        for event in chat_stream_events(req.question):
            if event.get("event") == "meta":
                meta = event
            yield json.dumps(event, ensure_ascii=False) + "\n"
        _record_from_response(req.question, meta or {"mode": "agent"}, action="chat_stream")

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.post("/ingest")
async def ingest():
    os.environ["HF_HOME"] = str(Path(__file__).parent / ".hf_cache")
    from app.ingest import run_ingest
    n = run_ingest()
    write_runtime_log("ingest", status="ok", chunks=n)
    return {"chunks": n, "status": "ok"}
