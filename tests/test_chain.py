"""Tests for the RAG chain."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.chain import build_prompt


def test_build_prompt_is_template():
    prompt = build_prompt()
    rendered = prompt.format(context="test context", question="test question")
    assert "test context" in rendered
    assert "test question" in rendered


def test_build_prompt_contains_required_placeholders():
    prompt = build_prompt()
    rendered = prompt.format(context="X", question="Y")
    assert "X" in rendered
    assert "Y" in rendered
    assert "{context}" not in rendered
    assert "{question}" not in rendered
