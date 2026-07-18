"""Document ingestion pipeline: load -> split -> embed -> store."""

import hashlib
import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import config
from app import store

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_BATCH_SIZE = 128
_STATE_FILE = _PROJECT_ROOT / ".ingest_state.json"


class MarkdownLoader(TextLoader):
    """支持Markdown文件的加载器，自动识别编码并提取元数据"""

    def __init__(self, file_path: str):
        super().__init__(file_path, encoding="utf-8")
        self.file_path = file_path

    def load(self) -> List[Document]:
        """加载文档并附加文件元数据"""
        docs = super().load()
        for doc in docs:
            path = Path(self.file_path)
            doc.metadata.update({
                "source": str(path),
                "filename": path.name,
                "file_stem": path.stem,
                "file_extension": path.suffix,
                "file_size": path.stat().st_size,
                "last_modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            })
        return docs


def load_markdown_files(data_dir: Optional[str] = None) -> List[Document]:
    """
    加载指定目录下的所有Markdown文件。

    接口签名完全不变，保持与原有调用兼容。
    """
    if data_dir is None:
        data_dir = config.data_dir

    data_path = Path(data_dir)
    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)
        return []

    loader = DirectoryLoader(
        str(data_path),
        glob="**/*.md",
        loader_cls=MarkdownLoader,
        show_progress=True,
        loader_kwargs={"autodetect_encoding": True},
    )
    return loader.load()


def split_documents(docs: List[Document]) -> List[Document]:
    """
    改进的分块策略，针对Markdown优化。

    接口签名完全不变，保持与原有调用兼容。
    """
    # 为Markdown优化的分隔符
    separators = [
        "\n\n# ",  # 一级标题
        "\n\n## ",  # 二级标题
        "\n\n### ",  # 三级标题
        "\n\n",  # 段落
        "\n",  # 行
        "。", "！", "？",  # 中文句子边界
        ". ", "! ", "? ",  # 英文句子边界
        " ",  # 词边界
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(docs)

    # 增强chunk元数据：添加chunk索引和父文档信息
    for idx, chunk in enumerate(chunks):
        chunk.metadata.update({
            "chunk_index": idx,
            "chunk_id": hashlib.md5(chunk.page_content.encode()).hexdigest()[:16],
            "chunk_size": len(chunk.page_content),
        })

    return chunks


def _get_model_path() -> str:
    """
    获取模型路径，优先使用本地缓存，fallback到HuggingFace。

    接口签名完全不变，保持与原有调用兼容。
    """
    # 方法1：检查项目本地缓存
    local_cache = (
            _PROJECT_ROOT
            / ".hf_cache"
            / "hub"
            / "models--BAAI--bge-small-zh-v1.5"
            / "snapshots"
    )
    if local_cache.exists():
        snapshots = [d for d in local_cache.iterdir() if d.is_dir()]
        if snapshots:
            # 按修改时间排序，取最新的
            latest = max(snapshots, key=lambda p: p.stat().st_mtime)
            return str(latest)

    # 方法2：检查系统HuggingFace缓存
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hf_cache = Path(hf_home) / "hub" / "models--BAAI--bge-small-zh-v1.5" / "snapshots"
    if hf_cache.exists():
        snapshots = [d for d in hf_cache.iterdir() if d.is_dir()]
        if snapshots:
            latest = max(snapshots, key=lambda p: p.stat().st_mtime)
            return str(latest)

    # 方法3：直接使用模型名称（会从HuggingFace下载）
    print("⚠️  Local model not found, downloading from HuggingFace...")
    return "BAAI/bge-small-zh-v1.5"


def build_embeddings():
    """构建embedding模型实例。

    接口签名完全不变，保持与原有调用兼容。
    """
    model_path = _get_model_path()

    return HuggingFaceBgeEmbeddings(
        model_name=model_path,
        model_kwargs={"device": "cpu"},  # 可改为 "cuda" 如果有GPU
        encode_kwargs={"normalize_embeddings": True},
    )


def compute_file_hash(file_path: Path) -> str:
    """计算文件的MD5哈希，用于检测文件变化"""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_ingest_state() -> Dict[str, Any]:
    """加载ingest状态文件"""
    if _STATE_FILE.exists():
        with open(_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"files": {}, "last_run": None}


def save_ingest_state(state: Dict[str, Any]):
    """保存ingest状态"""
    state["last_run"] = datetime.now().isoformat()
    with open(_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def get_files_to_process(data_dir: str) -> tuple[List[Path], List[Path]]:
    """
    检测哪些文件需要处理
    返回: (新增或修改的文件, 需要删除的文件)
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        return [], []

    state = load_ingest_state()
    current_files = {p: compute_file_hash(p) for p in data_path.rglob("*.md")}
    previous_files = state.get("files", {})

    # 新增或修改的文件
    changed = []
    for file_path, file_hash in current_files.items():
        # 使用相对路径作为key，更稳定
        rel_path = str(file_path.relative_to(data_path))
        if rel_path not in previous_files or previous_files[rel_path] != file_hash:
            changed.append(file_path)

    # 需要删除的文件（之前存在但现在不存在了）
    deleted = []
    for rel_path in previous_files:
        if not (data_path / rel_path).exists():
            deleted.append(Path(rel_path))

    return changed, deleted


def batch_embed_and_store(
        chunks: List[Document],
        embeddings,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        clear_first: bool = False,
):
    """
    分批嵌入并存储，避免OOM。
    """
    total = len(chunks)
    if total == 0:
        return

    if clear_first:
        store.clear()
        print("🗑️  Cleared existing vector store")

    print(f"📊 Processing {total} chunks in batches of {batch_size}...")

    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c.page_content for c in batch]

        try:
            vectors = embeddings.embed_documents(texts)
            store.add_documents(batch, vectors)
            print(f"  ✅ Processed {min(i + batch_size, total)}/{total} chunks")
        except Exception as e:
            print(f"  ❌ Failed at batch {i // batch_size + 1}: {e}")
            # 如果是第一批失败，清空操作需要回滚
            if i == 0 and clear_first:
                print("⚠️  First batch failed, vector store may be empty")
            raise


def run_ingest(force_rebuild: bool = False) -> int:
    """
    主入口：执行完整的ingest流程。

    接口签名完全不变，保持与原有调用兼容。

    Args:
        force_rebuild: 是否强制全量重建（忽略增量检测）

    Returns:
        int: 处理的chunk数量
    """
    print(f"📂 Loading documents from {config.data_dir}...")

    # 检测文件变化
    changed_files, deleted_files = get_files_to_process(config.data_dir)

    # 如果有删除的文件，检查 store 是否支持按源删除
    # 如果不支持，自动切换到全量重建
    supports_delete = hasattr(store, 'delete_by_source')

    if deleted_files and not supports_delete:
        print("⚠️  Store doesn't support delete_by_source, switching to full rebuild")
        force_rebuild = True
    elif deleted_files and supports_delete:
        # 支持增量删除，处理删除的文件
        print(f"🗑️  {len(deleted_files)} file(s) deleted, removing from index...")
        deleted_paths = [str(p) for p in deleted_files]
        removed = store.delete_by_sources(deleted_paths)
        print(f"  ✅ Removed {removed} chunks")

    # 如果没有变化且不强制重建，直接返回
    if not force_rebuild and not changed_files and not deleted_files:
        print("✅ No changes detected, skipping ingest")
        try:
            return store.count()
        except:
            return 0

    if force_rebuild:
        print("🔄 Force rebuild mode: processing all files")
        docs = load_markdown_files(config.data_dir)
    else:
        # 只加载发生变化的文件
        docs = []
        for file_path in changed_files:
            loader = MarkdownLoader(str(file_path))
            docs.extend(loader.load())
        print(f"📄 {len(changed_files)} file(s) changed, loading...")

    if not docs:
        print("📭 No documents to process")
        return 0

    print(f"📄 Loaded {len(docs)} document(s)")

    # 分块
    chunks = split_documents(docs)
    print(f"✂️  Split into {len(chunks)} chunk(s)")

    # 构建embeddings
    print("🧮 Computing embeddings...")
    emb = build_embeddings()

    # 分批嵌入并存储
    if force_rebuild:
        # 全量重建：清空后写入
        batch_embed_and_store(chunks, emb, clear_first=True)
    else:
        # 增量更新：不清除，由 add_documents 内部处理去重
        batch_embed_and_store(chunks, emb, clear_first=False)

    # 更新状态文件
    if not force_rebuild and changed_files:
        state = load_ingest_state()
        data_path = Path(config.data_dir)
        for file_path in changed_files:
            rel_path = str(file_path.relative_to(data_path))
            state["files"][rel_path] = compute_file_hash(file_path)
        # 删除已经不存在的文件（已在上面处理过，但状态文件也要同步）
        for deleted_path in deleted_files:
            rel_path = str(deleted_path)
            state["files"].pop(rel_path, None)
        save_ingest_state(state)
    elif force_rebuild:
        # 全量重建后重置状态
        data_path = Path(config.data_dir)
        state = {"files": {}}
        for file_path in data_path.rglob("*.md"):
            rel_path = str(file_path.relative_to(data_path))
            state["files"][rel_path] = compute_file_hash(file_path)
        save_ingest_state(state)

    print(f"✅ Ingest completed! Total chunks: {len(chunks)}")
    return len(chunks)


# 保持向后兼容的调用方式
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force full rebuild")
    args = parser.parse_args()
    run_ingest(force_rebuild=args.force)