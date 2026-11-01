"""The .env-loading entry point. Owner: orchestration-eval."""

from __future__ import annotations

import os

import pytest

from vus_panel import serve

SECRET = "placeholder-secret-value"


def test_reads_names_values_comments_export_and_quotes(tmp_path):
    path = tmp_path / ".env"
    path.write_text("\ufeff# comment\n\nANTHROPIC_MODEL=placeholder-model\nexport EVO2_SCORES_DIR='C:/x y/runs'\n"
                    f'ANTHROPIC_API_KEY="{SECRET}"\nEVO2_BACKEND_URL=\n', encoding="utf-8")
    assert serve.read_env_file(path) == {"ANTHROPIC_MODEL": "placeholder-model", "EVO2_SCORES_DIR": "C:/x y/runs",
                                         "ANTHROPIC_API_KEY": SECRET, "EVO2_BACKEND_URL": ""}


def test_a_malformed_line_names_the_line_number_but_never_its_text(tmp_path):
    path = tmp_path / ".env"
    path.write_text(f"ANTHROPIC_MODEL=placeholder\n{SECRET}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2") as info:
        serve.read_env_file(path)
    assert SECRET not in str(info.value)


def test_shell_values_win_and_blank_values_are_filled(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text(f"ANTHROPIC_API_KEY={SECRET}\nANTHROPIC_MODEL=from-file\nEVO2_SCORES_DIR=runs\nEVO2_BACKEND_URL=\n",
                    encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_MODEL", "from-shell")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")  # blank counts as unset
    assert serve.load_env_file(path) == ["ANTHROPIC_API_KEY", "EVO2_SCORES_DIR"]
    assert os.environ["ANTHROPIC_MODEL"] == "from-shell"
    assert os.environ["ANTHROPIC_API_KEY"] == SECRET
    assert "EVO2_BACKEND_URL" not in os.environ


def test_a_missing_file_loads_nothing(tmp_path):
    assert serve.load_env_file(tmp_path / ".env") == []


def test_main_prints_names_but_never_values(tmp_path, monkeypatch, capsys):
    path = tmp_path / ".env"
    path.write_text(f"ANTHROPIC_API_KEY={SECRET}\nEVO2_SCORES_DIR=runs\n", encoding="utf-8")
    monkeypatch.setattr(serve, "ENV_FILE", path)
    monkeypatch.setattr(serve.load_env_file, "__defaults__", (path,))
    started = {}
    import uvicorn
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: started.update(app=app, **kwargs))
    assert serve.main(["--port", "8123"]) == 0
    out = capsys.readouterr().out
    assert "ANTHROPIC_API_KEY" in out and "ANTHROPIC_MODEL not set" in out
    assert SECRET not in out
    assert started == {"app": "vus_panel.api:app", "host": "127.0.0.1", "port": 8123}
