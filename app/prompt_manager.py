"""Prompt version manager: load, version, and log prompt usage."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("rag")

_DATA_DIR = Path(__file__).resolve().parent.parent / ".prompts"
_DATA_DIR.mkdir(exist_ok=True)
_PROMPTS_FILE = _DATA_DIR / "prompts.json"
_LOG_FILE = _DATA_DIR / "prompt_usage.log"

_DEFAULT_PROMPTS = {
    "v1": {
        "query": "Based on the context below, answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:",
        "chat": "Current date: {current_date}\n\nConversation history:\n{history}\n\nQuestion: {question}\n\nAnswer:",
        "kb_search": "Based on the context below, answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer concisely based on the context.",
        "multi_query": "Generate {n} different versions of this question.\n\nOriginal: {question}\n\nNumbered list:",
    }
}


def _ensure_file():
    if not _PROMPTS_FILE.exists():
        with open(_PROMPTS_FILE, "w", encoding="utf-8") as f:
            json.dump(_DEFAULT_PROMPTS, f, ensure_ascii=False, indent=2)


def _log_usage(name: str, version: str):
    with open(_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] version={version} prompt={name}\n")


def get(name: str, version: str = "v1") -> str:
    _ensure_file()
    with open(_PROMPTS_FILE, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    prompt = prompts.get(version, {}).get(name, "")
    _log_usage(name, version)
    return prompt


def list_versions() -> list:
    _ensure_file()
    with open(_PROMPTS_FILE, "r", encoding="utf-8") as f:
        return list(json.load(f).keys())


def list_prompts(version: str = "v1") -> list:
    _ensure_file()
    with open(_PROMPTS_FILE, "r", encoding="utf-8") as f:
        return list(json.load(f).get(version, {}).keys())


def save(version: str, name: str, template: str):
    _ensure_file()
    with open(_PROMPTS_FILE, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    if version not in prompts:
        prompts[version] = {}
    prompts[version][name] = template
    with open(_PROMPTS_FILE, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)
    logger.info("Prompt saved: %s/%s", version, name)


def create_version(base: str = "v1", new_version: str = "") -> str:
    if not new_version:
        now = datetime.now()
        new_version = f"v{now.year}{now.month:02d}{now.day:02d}_{now.hour:02d}{now.minute:02d}"
    _ensure_file()
    with open(_PROMPTS_FILE, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    if base not in prompts:
        raise ValueError(f"Base version {base} not found")
    prompts[new_version] = json.loads(json.dumps(prompts[base]))
    with open(_PROMPTS_FILE, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)
    logger.info("Version created: %s (copied from %s)", new_version, base)
    return new_version
