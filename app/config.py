"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


def _positive_int_env(name: str, default: int, *, maximum: int) -> int:
    value = os.getenv(name)
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return parsed if 1 <= parsed <= maximum else default


@dataclass
class Config:
    deepseek_api_key: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", "")
    )
    deepseek_api_base: str = "https://api.deepseek.com"
    deepseek_model: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    )

    embedding_model_name: str = "BAAI/bge-small-zh-v1.5"

    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_top_k: int = 4
    rag_min_score: float = field(
        default_factory=lambda: float(os.getenv("RAG_MIN_SCORE", "0.35"))
    )

    tavily_api_key: str = field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY", "")
    )
    tavily_max_results: int = field(
        default_factory=lambda: _positive_int_env(
            "TAVILY_MAX_RESULTS", 5, maximum=20
        )
    )
    tavily_timeout_seconds: int = field(
        default_factory=lambda: _positive_int_env(
            "TAVILY_TIMEOUT_SECONDS", 10, maximum=60
        )
    )

    chroma_persist_dir: str = field(
        default_factory=lambda: os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "chroma_db"
        )
    )
    data_dir: str = field(
        default_factory=lambda: os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data"
        )
    )

    def __post_init__(self):
        if not self.deepseek_api_key:
            import warnings
            warnings.warn(
                "DEEPSEEK_API_KEY not set. Set it in .env or environment variables.",
                stacklevel=2,
            )


config = Config()
