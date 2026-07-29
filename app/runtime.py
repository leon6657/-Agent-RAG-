"""Runtime caches and warmup helpers for low-resource deployments."""

from collections import OrderedDict
from threading import Lock
from typing import Any

from app import store
from app.ingest import build_embeddings

_MAX_QUERY_CACHE_SIZE = 128

_lock = Lock()
_embeddings = None
_query_vector_cache: OrderedDict[str, list[float]] = OrderedDict()


def get_embeddings():
    """Return a process-wide embedding model instance."""
    global _embeddings
    if _embeddings is None:
        with _lock:
            if _embeddings is None:
                _embeddings = build_embeddings()
    return _embeddings


def embed_query_cached(query: str) -> list[float]:
    """Embed a query once and reuse the vector for repeated questions."""
    normalized = " ".join(query.split())
    if not normalized:
        normalized = query

    with _lock:
        cached = _query_vector_cache.get(normalized)
        if cached is not None:
            _query_vector_cache.move_to_end(normalized)
            return cached

    vector = get_embeddings().embed_query(normalized)

    with _lock:
        _query_vector_cache[normalized] = vector
        _query_vector_cache.move_to_end(normalized)
        while len(_query_vector_cache) > _MAX_QUERY_CACHE_SIZE:
            _query_vector_cache.popitem(last=False)

    return vector


def warmup_runtime() -> dict[str, Any]:
    """Load the vector store and embedding model before the first user query."""
    vector_count = store.count()
    if vector_count:
        store._get_cached_vectors()
    get_embeddings()
    return {
        "embedding_model": "ready",
        "vector_store": "ready" if vector_count else "empty",
        "vector_count": vector_count,
    }


def reset_runtime_caches() -> None:
    """Reset runtime caches for tests and manual diagnostics."""
    global _embeddings
    with _lock:
        _embeddings = None
        _query_vector_cache.clear()
