"""The checker subagent's Bash guard allows tests and read-only git, and nothing else."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GUARD = REPO / ".claude" / "hooks" / "checker_bash_guard.py"
CHECKER_MD = REPO / ".claude" / "agents" / "checker.md"
_spec = importlib.util.spec_from_file_location("checker_bash_guard", GUARD)
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)

PY = ".venv/Scripts/python -m pytest"

ALLOWED = [
    f"{PY} -q",
    ".venv/bin/python -m pytest -q",
    "./.venv/Scripts/python -m pytest -q tests/test_schemas.py",
    ".venv/Scripts/pytest -q",
    f"{PY} -k 'chair and not slow'",
    f"{PY} -x --tb=short -rA tests/test_isolation.py::test_chair_signature_takes_findings_only",
    f"{PY} -vv tests tests/",
    f"{REPO.as_posix()}/.venv/Scripts/python -m pytest -q",
    "git diff main...HEAD",
    "git diff --stat main",
    "git --no-pager diff",
    "git log --oneline -5",
    "git show HEAD:README.md",
    "git status",
    "git branch",
    "git branch -a",
    "git rev-parse HEAD",
]

BLOCKED = [
    "",
    "rm -rf src",
    "cat README.md",
    "sed -i s/a/b/ README.md",
    "FOO=1 .venv/Scripts/python -m pytest",
    # interpreters other than this repo's .venv
    "pytest -q",
    "python -m pytest -q",
    "python3 -m pytest -q",
    "/tmp/evil/.venv/bin/python -m pytest -q",
    "other/.venv/bin/python -m pytest -q",
    ".venv/Scripts/python script.py",
    ".venv/Scripts/python -c 'import os'",
    # shell operators and expansions
    f"{PY}; rm -rf .",
    f"{PY} && echo hi",
    f"{PY} | tee out.txt",
    f"{PY} > out.txt",
    "echo $(whoami)",
    "echo `whoami`",
    f"{PY}\nrm -rf .",
    f"{PY} tests\\test_schemas.py",
    f"{PY} --base{{temp,temp}}=src",
    "git diff --out{put,put}=README.md",
    f"{PY} tests/test_*.py",
    f"{PY} tests/test_x.py?",
    "git show HEAD:[ab]",
    "git diff HEAD~1",
    "git branch --list 'feature*'",
    # pytest options that write files, load code or change config
    f"{PY} --log-file=README.md",
    f"{PY} --log-file README.md",
    f"{PY} --log-file-level=DEBUG",
    f"{PY} -o log_file=README.md",
    f"{PY} -o cache_dir=X",
    f"{PY} -qo cache_dir=X",
    f"{PY} --override-ini=cache_dir=X",
    f"{PY} -c other.toml",
    f"{PY} --config-file=other.toml",
    f"{PY} --rootdir=src",
    f"{PY} -p no:cacheprovider",
    f"{PY} @args.txt",
    f"{PY} -k @args.txt",
    f"{PY} -k",
    f"{PY} --basetemp=src",
    f"{PY} --pdb",
    f"{PY} --trace",
    f"{PY} --junitxml=out.xml",
    f"{PY} --pastebin=all",
    f"{PY} src",
    f"{PY} ../outside",
    f"{PY} tests/../src",
    # git commands and options that write, run programs or change config
    "git checkout main",
    "git push",
    "git commit -m x",
    "git branch new-feature",
    "git branch -D main",
    "git branch -m old new",
    "git diff --output=patch.txt",
    "git diff --out=README.md",
    "git diff --outp=README.md",
    "git log --output=x",
    "git diff --ext-diff",
    "git diff --ext=1",
    "git -c core.pager=less diff",
    "git -C .. log",
    "git",
]


@pytest.mark.parametrize("command", ALLOWED)
def test_allows_tests_and_read_only_git(command):
    assert guard.check(command) is None


@pytest.mark.parametrize("command", BLOCKED)
def test_blocks_everything_else(command):
    assert guard.check(command) is not None


def test_relative_interpreter_needs_the_repo_as_cwd(tmp_path):
    assert guard.check(f"{PY} -q", cwd=str(REPO)) is None
    assert guard.check(f"{PY} -q", cwd=str(tmp_path)) is not None


def _run_guard(stdin: bytes) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(GUARD)], input=stdin, capture_output=True, check=False)


def _payload(command: str) -> bytes:
    return json.dumps({"tool_name": "Bash", "tool_input": {"command": command}, "cwd": str(REPO)}).encode()


def test_guard_exit_codes_match_claude_code_contract():
    assert _run_guard(_payload(f"{PY} -q")).returncode == 0
    blocked = _run_guard(_payload("rm -rf ."))
    assert blocked.returncode == 2
    assert b"Blocked" in blocked.stderr


def test_guard_fails_closed_on_bad_input():
    assert _run_guard(b"not json").returncode == 2
    assert _run_guard(json.dumps({"tool_input": {}}).encode()).returncode == 2


# The hook command line in checker.md

def _hook_command() -> str:
    match = re.search(r"^\s*command:\s*'(.*)'\s*$", CHECKER_MD.read_text(encoding="utf-8"), re.MULTILINE)
    assert match, "checker.md has no single-quoted hook command"
    return match.group(1).replace("''", "'")


def test_hook_config_fails_closed():
    text = CHECKER_MD.read_text(encoding="utf-8")
    assert re.search(r"^\s*onFailure:\s*block\s*$", text, re.MULTILINE)
    assert re.search(r"^\s*shell:\s*bash\s*$", text, re.MULTILINE)
    assert _hook_command().rstrip().endswith("exit 2")


SH = shutil.which("sh")
needs_sh = pytest.mark.skipif(SH is None, reason="no POSIX sh on PATH")


def _run_hook_command(path_dirs: list[Path], stdin: bytes) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(str(d) for d in path_dirs)
    env["CLAUDE_PROJECT_DIR"] = str(REPO)
    return subprocess.run([SH, "-c", _hook_command()], input=stdin, env=env, capture_output=True, check=False, timeout=60)


def _fake_interpreter(directory: Path, name: str, probe_exit: int, run_exit: int) -> None:
    script = directory / name
    script.write_text(f'#!/bin/sh\nif [ "$1" = "-c" ]; then exit {probe_exit}; fi\nexit {run_exit}\n', newline="\n")
    script.chmod(0o755)


@needs_sh
def test_hook_command_blocks_when_no_interpreter_exists(tmp_path):
    assert _run_hook_command([tmp_path], _payload(f"{PY} -q")).returncode == 2


@needs_sh
def test_hook_command_blocks_when_interpreters_are_bogus(tmp_path):
    _fake_interpreter(tmp_path, "python3", probe_exit=49, run_exit=0)  # like a Store alias stub
    _fake_interpreter(tmp_path, "python", probe_exit=127, run_exit=0)
    assert _run_hook_command([tmp_path], _payload(f"{PY} -q")).returncode == 2


@needs_sh
def test_hook_command_blocks_when_the_guard_crashes(tmp_path):
    _fake_interpreter(tmp_path, "python3", probe_exit=0, run_exit=1)
    assert _run_hook_command([tmp_path], _payload(f"{PY} -q")).returncode == 2


@needs_sh
def test_hook_command_runs_the_real_guard():
    path = [Path(sys.executable).parent]
    assert _run_hook_command(path, _payload(f"{PY} -q")).returncode == 0
    assert _run_hook_command(path, _payload("rm -rf .")).returncode == 2
