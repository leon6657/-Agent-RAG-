"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    deepseek_api_key: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", "")
    )
    deepseek_api_base: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    embedding_model_name: str = "BAAI/bge-small-zh-v1.5"

    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_top_k: int = 4

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
