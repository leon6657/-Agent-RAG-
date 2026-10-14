"""Tests for the config module."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import Config, config


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


def test_deepseek_model_can_be_configured_from_environment(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

    cfg = Config()

    assert cfg.deepseek_model == "deepseek-v4-flash"


def test_rag_min_score_can_be_configured_from_environment(monkeypatch):
    monkeypatch.setenv("RAG_MIN_SCORE", "0.42")

    cfg = Config()

    assert cfg.rag_min_score == 0.42


def test_tavily_config_can_be_configured_from_environment(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setenv("TAVILY_MAX_RESULTS", "3")
    monkeypatch.setenv("TAVILY_TIMEOUT_SECONDS", "7")

    cfg = Config()

    assert cfg.tavily_api_key == "test-key"
    assert cfg.tavily_max_results == 3
    assert cfg.tavily_timeout_seconds == 7


@pytest.mark.parametrize(
    ("name", "value", "field", "default"),
    [
        ("TAVILY_MAX_RESULTS", "0", "tavily_max_results", 5),
        ("TAVILY_MAX_RESULTS", "-1", "tavily_max_results", 5),
        ("TAVILY_MAX_RESULTS", "21", "tavily_max_results", 5),
        ("TAVILY_TIMEOUT_SECONDS", "not-an-integer", "tavily_timeout_seconds", 10),
        ("TAVILY_TIMEOUT_SECONDS", "61", "tavily_timeout_seconds", 10),
    ],
)
def test_invalid_tavily_numeric_configuration_uses_controlled_defaults(
    monkeypatch, name, value, field, default
):
    monkeypatch.setenv(name, value)

    cfg = Config()

    assert getattr(cfg, field) == default
