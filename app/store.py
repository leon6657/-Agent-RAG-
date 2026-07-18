"""Simple vector store using numpy for similarity search.
Supports incremental updates with source-based deletion.
"""

import json
import os
import uuid
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from langchain_core.documents import Document

_DATA_PATH = Path(__file__).resolve().parent.parent / "vector_store.json"


def _normalize(v: np.ndarray) -> np.ndarray:
    """L2归一化，用于余弦相似度计算"""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def _load_items() -> List[Dict[str, Any]]:
    """加载所有条目"""
    if not _DATA_PATH.exists():
        return []
    try:
        with open(_DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def _save_items(items: List[Dict[str, Any]]) -> None:
    """保存所有条目"""
    with open(_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def add_documents(docs: List[Document], embeddings: List[List[float]]) -> None:
    """
    添加文档。如果文档已有chunk_id则更新（upsert），否则新增。
    支持按 source 去重：同一来源的旧chunk会被清理。

    接口签名完全不变，保持与原有调用兼容。
    """
    if not docs or not embeddings:
        return

    # 构建新的条目列表
    new_items = []
    sources_to_remove = set()

    for doc, emb in zip(docs, embeddings):
        # 使用已有的chunk_id或生成新的
        chunk_id = doc.metadata.get("chunk_id")
        if not chunk_id:
            chunk_id = str(uuid.uuid4())
            doc.metadata["chunk_id"] = chunk_id

        # 记录该文档来源，用于清理旧数据
        source = doc.metadata.get("source")
        if source:
            sources_to_remove.add(source)

        new_items.append({
            "id": chunk_id,
            "text": doc.page_content,
            "metadata": doc.metadata,
            "embedding": emb,
        })

    # 加载现有数据
    existing = _load_items()

    if sources_to_remove:
        # 删除这些来源的所有旧chunk
        existing = [
            item for item in existing
            if item.get("metadata", {}).get("source") not in sources_to_remove
        ]

    # 按ID去重：如果新条目ID已存在，用新条目替换
    existing_ids = {item["id"] for item in existing}
    for item in new_items:
        if item["id"] in existing_ids:
            # 替换旧的
            for i, existing_item in enumerate(existing):
                if existing_item["id"] == item["id"]:
                    existing[i] = item
                    break
        else:
            existing.append(item)

    _save_items(existing)


def search(query_embedding: List[float], k: int = 4, min_score: float = 0.0) -> List[Document]:
    """Search and return documents with similarity scores in metadata.

    接口签名完全不变，保持与原有调用兼容。
    """
    items = _load_items()
    if not items:
        return []

    q = _normalize(np.array(query_embedding, dtype=np.float32))
    vectors = np.array([item["embedding"] for item in items], dtype=np.float32)
    vectors = np.apply_along_axis(_normalize, 1, vectors)
    scores = vectors @ q
    top_idx = np.argsort(scores)[-k:][::-1]

    results = []
    for i in top_idx:
        sim = float(scores[i])
        if sim >= min_score:
            meta = items[i].get("metadata", {}).copy()
            meta["score"] = round(sim, 4)
            results.append(Document(page_content=items[i]["text"], metadata=meta))
    return results


def search_top_score(query_embedding: List[float]) -> float:
    """Return the highest similarity score for a query (0 if empty store).

    接口签名完全不变，保持与原有调用兼容。
    """
    items = _load_items()
    if not items:
        return 0.0
    q = _normalize(np.array(query_embedding, dtype=np.float32))
    vectors = np.array([item["embedding"] for item in items], dtype=np.float32)
    vectors = np.apply_along_axis(_normalize, 1, vectors)
    scores = vectors @ q
    return float(np.max(scores))


def count() -> int:
    """返回存储的总条目数。

    接口签名完全不变，保持与原有调用兼容。
    """
    return len(_load_items())


def clear() -> None:
    """清空整个向量存储。

    接口签名完全不变，保持与原有调用兼容。
    """
    if _DATA_PATH.exists():
        _DATA_PATH.unlink()


# ============ 以下是新增函数，不影响现有调用 ============

def delete_by_source(source_path: str) -> int:
    """
    删除指定来源的所有chunk。

    Args:
        source_path: 文件路径（字符串）

    Returns:
        int: 删除的条目数量
    """
    items = _load_items()
    original_count = len(items)
    items = [
        item for item in items
        if item.get("metadata", {}).get("source") != source_path
    ]
    removed = original_count - len(items)
    if removed > 0:
        _save_items(items)
    return removed


def delete_by_sources(source_paths: List[str]) -> int:
    """批量删除多个来源的所有chunk"""
    if not source_paths:
        return 0
    source_set = set(source_paths)
    items = _load_items()
    original_count = len(items)
    items = [
        item for item in items
        if item.get("metadata", {}).get("source") not in source_set
    ]
    removed = original_count - len(items)
    if removed > 0:
        _save_items(items)
    return removed


def get_all_sources() -> List[str]:
    """获取所有已索引文件的来源路径列表"""
    items = _load_items()
    sources = set()
    for item in items:
        source = item.get("metadata", {}).get("source")
        if source:
            sources.add(source)
    return list(sources)


def count_by_source(source_path: str) -> int:
    """统计某个来源的chunk数量"""
    items = _load_items()
    return sum(
        1 for item in items
        if item.get("metadata", {}).get("source") == source_path
    )


def rebuild_from_items(items: List[Dict[str, Any]]) -> None:
    """
    用新的条目列表完全替换存储（用于重建索引）
    """
    if not items:
        clear()
    else:
        _save_items(items)


# ============ 性能优化：缓存 ============

_cache = {
    "items": None,
    "vectors": None,
    "mtime": None,
}


def _get_cached_vectors():
    """带缓存的向量加载，提升搜索性能"""
    items = _load_items()
    if not items:
        return items, None

    # 简单缓存：检查数据是否变化（通过文件修改时间）
    if _DATA_PATH.exists():
        mtime = _DATA_PATH.stat().st_mtime
        if _cache.get("mtime") == mtime and _cache.get("items") is not None:
            return _cache["items"], _cache["vectors"]

    vectors = np.array([item["embedding"] for item in items], dtype=np.float32)
    vectors = np.apply_along_axis(_normalize, 1, vectors)

    _cache["items"] = items
    _cache["vectors"] = vectors
    _cache["mtime"] = _DATA_PATH.stat().st_mtime if _DATA_PATH.exists() else None

    return items, vectors


def search_cached(query_embedding: List[float], k: int = 4, min_score: float = 0.0) -> List[Document]:
    """使用缓存的搜索（性能更优），接口与 search 一致"""
    items, vectors = _get_cached_vectors()
    if items is None or vectors is None:
        return []

    q = _normalize(np.array(query_embedding, dtype=np.float32))
    scores = vectors @ q
    top_idx = np.argsort(scores)[-k:][::-1]

    results = []
    for i in top_idx:
        sim = float(scores[i])
        if sim >= min_score:
            meta = items[i].get("metadata", {}).copy()
            meta["score"] = round(sim, 4)
            results.append(Document(page_content=items[i]["text"], metadata=meta))
    return results


def invalidate_cache():
    """使缓存失效（在写入操作后调用）"""
    _cache["items"] = None
    _cache["vectors"] = None
    _cache["mtime"] = None