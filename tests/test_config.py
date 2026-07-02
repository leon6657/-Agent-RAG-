"""Tests for the config module."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import config


def test_config_has_required_fields():
    assert hasattr(config, "deepseek_api_key")
    assert hasattr(config, "embedding_model_name")
    assert hasattr(config, "chunk_size")
    assert hasattr(config, "chroma_persist_dir")
    assert hasattr(config, "data_dir")


def test_config_defaults():
    assert config.chunk_size == 500
    assert config.chunk_overlap == 50
    assert config.deepseek_api_base == "https://api.deepseek.com"
    assert config.retrieval_top_k == 4
