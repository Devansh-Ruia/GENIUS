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
ROOT = str(REPO)
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
    "git diff main...HEAD -- src/vus_panel/schemas.py",
    "git --no-pager diff",
    "git log --oneline -5",
    "git log --oneline main..HEAD",
    "git show HEAD:README.md",
    "git show refs/heads/d:README.md",
    "git status",
    "git status --short",
    "git branch",
    "git branch -a",
    "git branch --list",
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
    # shell operators, comments, expansions and control characters
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
    "git branch new #x --list",
    "git status #x --short",
    "#",
    "x#y",
    f"{PY} tests/@(test_x).py",
    "git diff +(main)",
    f"{PY} -k '(chair)'",
    "(git status)",
    "git status\x00 --short",
    "git\tstatus",
    "git status\x7f",
    "git status\u00a0--short",
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
    # git commands and options that write, run programs or read outside the repo
    "git checkout main",
    "git push",
    "git commit -m x",
    "git branch new-feature",
    "git branch --list feature",
    "git branch --merged main",
    "git branch -D main",
    "git branch -m old new",
    "git diff --output=patch.txt",
    "git diff --out=README.md",
    "git diff --outp=README.md",
    "git log --output=x",
    "git diff --ext-diff",
    "git diff --ext=1",
    "git diff --no-index a b",
    "git diff --no-i a b",
    "git diff --no-ind a b",
    "git diff /etc/passwd README.md",
    "git diff /c/Users/other/file README.md",
    "git diff C:/Windows/win.ini README.md",
    "git diff C:file README.md",
    "git show d:README.md",
    "git diff ../outside README.md",
    "git diff -- ../outside",
    "git diff -- /etc/passwd",
    "git diff tests/../../outside",
    "git show HEAD:../outside",
    "git log -- ..",
    "git -c core.pager=less diff",
    "git -C .. log",
    "git",
]


@pytest.mark.parametrize("command", ALLOWED)
def test_allows_tests_and_read_only_git(command):
    assert guard.check(command, ROOT) is None


@pytest.mark.parametrize("command", BLOCKED)
def test_blocks_everything_else(command):
    assert guard.check(command, ROOT) is not None


def test_relative_interpreter_needs_the_repo_as_cwd(tmp_path):
    assert guard.check(f"{PY} -q", ROOT) is None
    assert guard.check(f"{PY} -q", str(tmp_path)) is not None


def _run_guard(stdin: bytes) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", str(GUARD)], input=stdin, capture_output=True, check=False)


def _payload(command: str, **extra) -> bytes:
    payload = {"tool_name": "Bash", "tool_input": {"command": command}, "cwd": ROOT}
    payload.update(extra)
    return json.dumps(payload).encode()


def test_guard_exit_codes_match_claude_code_contract():
    assert _run_guard(_payload(f"{PY} -q")).returncode == 0
    blocked = _run_guard(_payload("rm -rf ."))
    assert blocked.returncode == 2
    assert b"Blocked" in blocked.stderr


def test_guard_fails_closed_on_bad_input():
    assert _run_guard(b"not json").returncode == 2
    assert _run_guard(json.dumps({"tool_input": {}, "cwd": ROOT}).encode()).returncode == 2


def test_guard_fails_closed_without_cwd():
    no_cwd = json.dumps({"tool_name": "Bash", "tool_input": {"command": f"{PY} -q"}}).encode()
    assert _run_guard(no_cwd).returncode == 2
    assert _run_guard(_payload(f"{PY} -q", cwd="")).returncode == 2


# The hook command line in checker.md

def _hook_command() -> str:
    match = re.search(r"^\s*command:\s*'(.*)'\s*$", CHECKER_MD.read_text(encoding="utf-8"), re.MULTILINE)
    assert match, "checker.md has no single-quoted hook command"
    return match.group(1).replace("''", "'")


def test_hook_config_fails_closed_and_isolates_python():
    text = CHECKER_MD.read_text(encoding="utf-8")
    assert re.search(r"^\s*onFailure:\s*block\s*$", text, re.MULTILINE)
    assert re.search(r"^\s*shell:\s*bash\s*$", text, re.MULTILINE)
    command = _hook_command()
    assert command.rstrip().endswith("exit 2")
    assert '"$py" -I ' in command


SH = shutil.which("sh")
needs_sh = pytest.mark.skipif(SH is None, reason="no POSIX sh on PATH")


def _run_hook_command(path_dirs: list[Path], stdin: bytes, project_dir: Path = REPO) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(str(d) for d in path_dirs)
    env["CLAUDE_PROJECT_DIR"] = str(project_dir)
    return subprocess.run([SH, "-c", _hook_command()], input=stdin, env=env, capture_output=True, check=False, timeout=60)


def _fake_interpreter(directory: Path, name: str, probe_exit: int, run_exit: int) -> None:
    script = directory / name
    script.write_text(f'#!/bin/sh\nif [ "$1" = "-c" ]; then exit {probe_exit}; fi\nexit {run_exit}\n', newline="\n")
    script.chmod(0o755)


@needs_sh
def test_fake_interpreters_really_run():
    # Harness check: a fake that passes the probe and exits 0 is trusted, which proves
    # the fakes below execute rather than silently failing to start.
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        _fake_interpreter(Path(directory), "python3", probe_exit=0, run_exit=0)
        assert _run_hook_command([Path(directory)], _payload(f"{PY} -q")).returncode == 0


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
def test_hook_command_blocks_when_the_guard_script_is_missing(tmp_path):
    path = [Path(sys.executable).parent]
    assert _run_hook_command(path, _payload(f"{PY} -q"), project_dir=tmp_path).returncode == 2


@needs_sh
def test_hook_command_runs_the_real_guard():
    path = [Path(sys.executable).parent]
    assert _run_hook_command(path, _payload(f"{PY} -q")).returncode == 0
    assert _run_hook_command(path, _payload("rm -rf .")).returncode == 2
