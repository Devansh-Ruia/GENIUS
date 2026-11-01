"""Start the API and page with settings from the repo's .env.

Usage: python -m vus_panel.serve [--host 127.0.0.1] [--port 8000]

Only this entry point reads .env; importing the app does not, so tests and
other imports never pick up real settings. A variable already set to a
non-blank value in the shell wins over .env. Only variable names are ever
printed, never their values.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from vus_panel import config

ENV_FILE = config.REPO_ROOT / ".env"
REQUIRED = ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL")


def read_env_file(path: Path) -> dict[str, str]:
    """NAME=value lines. Blank lines, # comments and a leading 'export ' are allowed; quotes are stripped."""
    values: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        name, sep, value = line.partition("=")
        name = name.strip()
        if not sep or not name.isidentifier():
            # The line number only: the line itself may hold a secret.
            raise ValueError(f"{path.name} line {number} is not NAME=value")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[name] = value
    return values


def load_env_file(path: Path = ENV_FILE) -> list[str]:
    """Put .env values into os.environ where the shell has none. Returns the names it set."""
    if not path.is_file():
        return []
    loaded = []
    for name, value in read_env_file(path).items():
        if config.get_env(name) is None and value.strip():
            os.environ[name] = value
            loaded.append(name)
    return loaded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the VUS panel API and page, with settings from .env.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    loaded = load_env_file()
    print(f"settings from .env: {', '.join(loaded) if loaded else 'none'}")
    missing = [name for name in REQUIRED if config.get_env(name) is None]
    if missing:
        print(f"warning: {', '.join(missing)} not set, so every agent will report unavailable")
    if config.evo2_scores_dir() is None and config.evo2_backend_url() is None:
        print("warning: neither EVO2_SCORES_DIR nor EVO2_BACKEND_URL is set, so evo2 will report unavailable")
    import uvicorn

    uvicorn.run("vus_panel.api:app", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
