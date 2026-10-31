"""Paths and environment settings shared across the package.

Paths are functions, not constants, so tests can point the data and run
directories at a temporary folder through environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = REPO_ROOT / "web"
FIXTURES_PATH = REPO_ROOT / "data" / "fixtures" / "panel_reports.json"

DATA_DIR_ENV = "VUS_PANEL_DATA_DIR"
RUNS_DIR_ENV = "VUS_PANEL_RUNS_DIR"


def get_env(name: str) -> str | None:
    """Return the variable's value, treating unset and blank the same way."""
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return value.strip()


def data_dir() -> Path:
    override = get_env(DATA_DIR_ENV)
    return Path(override) if override else REPO_ROOT / "data"


def raw_dir() -> Path:
    return data_dir() / "raw"


def cache_dir() -> Path:
    return data_dir() / "cache"


def splits_dir() -> Path:
    return data_dir() / "splits"


def runs_dir() -> Path:
    override = get_env(RUNS_DIR_ENV)
    return Path(override) if override else REPO_ROOT / "runs"


def evo2_backend_url() -> str | None:
    return get_env("EVO2_BACKEND_URL")


def display_path(path: Path) -> str:
    """Show a path relative to the repo when possible, so messages don't leak home directories."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()
