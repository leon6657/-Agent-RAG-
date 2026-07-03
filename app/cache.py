"""Response cache using diskcache (Redis-ready)."""

import hashlib
import json
from pathlib import Path
from typing import Optional

_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
_CACHE_DIR.mkdir(exist_ok=True)

_cache = None


def _get_cache():
    global _cache
    if _cache is None:
        from diskcache import Cache
        _cache = Cache(str(_CACHE_DIR / "responses"))
    return _cache


def _make_key(question: str, mode: str = "query") -> str:
    key = f"{mode}:{question.strip().lower()}"
    return hashlib.md5(key.encode()).hexdigest()


def get(question: str, mode: str = "query") -> Optional[str]:
    cache = _get_cache()
    key = _make_key(question, mode)
    result = cache.get(key)
    return result


def set(question: str, response: str, mode: str = "query", expire: int = 3600):
    cache = _get_cache()
    key = _make_key(question, mode)
    cache.set(key, response, expire=expire)


def clear():
    cache = _get_cache()
    cache.clear()


# Redis-ready interface (swap import to use Redis)
# from redis import Redis
# _client = Redis(...)
# def get(q, m="query"):
#     return _client.get(_make_key(q, m))
# def set(q, r, m="query", expire=3600):
#     _client.setex(_make_key(q, m), expire, r)
