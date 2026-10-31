from __future__ import annotations

import re
from pathlib import Path

import anthropic
import pytest

from vus_panel.llm import LLMConfigError, get_client, get_model

SRC = Path(__file__).resolve().parents[1] / "src"
MODEL_STRING_RE = re.compile(r"claude-|opus|sonnet|haiku|fable", re.IGNORECASE)


def test_get_model_requires_env(monkeypatch):
    with pytest.raises(LLMConfigError, match="ANTHROPIC_MODEL"):
        get_model()
    monkeypatch.setenv("ANTHROPIC_MODEL", "   ")
    with pytest.raises(LLMConfigError):
        get_model()


def test_get_model_reads_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "placeholder-model-name")
    assert get_model() == "placeholder-model-name"


def test_get_client_requires_key():
    with pytest.raises(LLMConfigError, match="ANTHROPIC_API_KEY"):
        get_client()


def test_get_client_builds_without_a_request(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder-key")
    assert isinstance(get_client(), anthropic.Anthropic)


def test_src_has_no_hardcoded_model_strings():
    hits = []
    for path in SRC.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc" or any(p.endswith(".egg-info") for p in path.parts):
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            if MODEL_STRING_RE.search(line):
                hits.append(f"{path.relative_to(SRC)}:{number}: {line.strip()}")
    assert not hits, "model strings found in src/:\n" + "\n".join(hits)
