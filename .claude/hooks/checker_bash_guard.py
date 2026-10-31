#!/usr/bin/env python3
"""PreToolUse guard for the checker subagent's Bash tool.

Claude Code passes the tool call as JSON on stdin, with the command at
tool_input.command and the working directory at cwd. Exit 0 lets the call
continue to the normal permission flow. Exit 2 blocks it, and stderr goes
back to the checker as the reason. Anything the guard can't parse, including
input without a cwd, is blocked.

Allowed, as one plain command with no shell operators or expansions:
- this repo's .venv python with -m pytest, or this repo's .venv pytest,
  with arguments from a short allowlist and test paths under tests/
- read-only git: diff, log, show, status, branch (listing only), rev-parse,
  with no path that points outside the repo

Standard library only, so it runs under any Python 3.8+.
"""

from __future__ import annotations

import json
import os
import posixpath
import re
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Operators, redirects, substitution, escapes, comments, subshells, and brace,
# glob, extglob, tilde or history expansion. Bash would act on these after this
# guard has checked the raw text, so any command containing one is blocked.
# Control characters and anything outside printable ASCII are blocked separately.
FORBIDDEN = (";", "&", "|", ">", "<", "`", "$", "\\", "{", "}", "*", "?", "[", "]", "~", "!", "#", "(", ")")

VENV_PYTHONS = (".venv/Scripts/python.exe", ".venv/Scripts/python", ".venv/bin/python", ".venv/bin/python3")
VENV_PYTESTS = (".venv/Scripts/pytest.exe", ".venv/Scripts/pytest", ".venv/bin/pytest")

PYTEST_FLAGS = {"-q", "-qq", "-v", "-vv", "-x", "--no-header"}
PYTEST_TB_STYLES = {"auto", "long", "short", "line", "native", "no"}
PYTEST_REPORT_RE = re.compile(r"^-r[fEsxXpPaAN]+$")
TEST_PATH_RE = re.compile(r"^tests(?:/[A-Za-z0-9_.-]+)*/?(?:::[A-Za-z0-9_]+)*$")

GIT_READ_ONLY = {"diff", "log", "show", "status", "branch", "rev-parse"}
# Git accepts unambiguous abbreviations of long options, so "--out=x" means
# "--output=x". Any long option that could abbreviate one of these is blocked.
GIT_DANGEROUS = ("--output", "--ext-diff", "--no-index")
BRANCH_FLAGS = {
    "-a", "--all", "-r", "--remotes", "-v", "-vv", "--verbose", "--show-current", "--no-color",
    "-l", "--list", "--merged", "--no-merged", "--contains", "--no-contains",
}

ALLOWED_SUMMARY = (
    "Allowed: .venv/Scripts/python -m pytest (or .venv/bin/python) with -q, -v, -x, -k EXPR, --tb=STYLE, "
    "-rCHARS and paths under tests/, and read-only git (diff, log, show, status, branch listing, rev-parse) "
    "with paths inside the repo. One command from the repo root, no shell operators, comments or expansions, "
    "forward slashes in paths."
)


def _norm(path_text: str) -> str:
    text = path_text.replace("\\", "/")
    drive = re.match(r"^/([A-Za-z])/(.*)$", text)
    if drive and os.name == "nt":
        text = f"{drive.group(1)}:/{drive.group(2)}"
    text = posixpath.normpath(text)
    return text.lower() if os.name == "nt" else text


def _is_repo_venv_tool(head: str, cwd: str, names: tuple[str, ...]) -> bool:
    """True only for this repo's own .venv executable, relative (from the repo root) or absolute."""
    if head.startswith("./"):
        head = head[2:]
    root = _norm(str(REPO_ROOT))
    if head in names:
        return _norm(cwd) == root
    return _norm(head) in {_norm(f"{root}/{name}") for name in names}


def _check_pytest(args: list[str]) -> str | None:
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in PYTEST_FLAGS or PYTEST_REPORT_RE.match(arg):
            i += 1
        elif arg == "-k":
            if i + 1 >= len(args) or args[i + 1].startswith(("-", "@")):
                return "-k needs an expression that doesn't start with - or @"
            i += 2
        elif arg.startswith("--tb=") and arg[len("--tb="):] in PYTEST_TB_STYLES:
            i += 1
        elif TEST_PATH_RE.match(arg) and ".." not in arg.split("::")[0].split("/"):
            i += 1
        else:
            return f"pytest argument {arg!r} is not on the allowlist"
    return None


def _git_option_blocked(arg: str) -> bool:
    if not arg.startswith("--") or arg == "--":
        return False
    name = arg.split("=", 1)[0]
    return any(danger.startswith(name) or name.startswith(danger) for danger in GIT_DANGEROUS)


def _git_path_problem(arg: str) -> str | None:
    """A positional that could name a file outside the repo. Revision ranges like main...HEAD pass."""
    if arg.startswith("/"):
        return "absolute paths are not allowed"
    if re.match(r"^[A-Za-z]:", arg):
        return "drive paths are not allowed (for a one-letter branch use refs/heads/NAME:PATH)"
    if ".." in re.split(r"[/:]", arg):
        return "'..' path segments are not allowed"
    return None


def _check_git(args: list[str]) -> str | None:
    if args[:1] == ["--no-pager"]:
        args = args[1:]
    if not args:
        return "git needs a subcommand"
    sub, rest = args[0], args[1:]
    if sub.startswith("-"):
        return "git global options are not allowed (only --no-pager)"
    if sub not in GIT_READ_ONLY:
        return f"git {sub} is not on the read-only list"
    after_double_dash = False
    for arg in rest:
        if not after_double_dash and arg == "--":
            after_double_dash = True
            continue
        if not after_double_dash and arg.startswith("-"):
            if _git_option_blocked(arg):
                return f"git option {arg.split('=')[0]} could write a file, run a program or read outside the repo"
            if sub == "branch" and arg not in BRANCH_FLAGS:
                return f"git branch {arg} is not a listing option"
            continue
        if sub == "branch":
            return "git branch takes no names or patterns here, since a name would create a branch"
        problem = _git_path_problem(arg)
        if problem is not None:
            return f"git argument {arg!r}: {problem}"
    return None


def check(command: str, cwd: str) -> str | None:
    """Return the reason to block, or None to allow."""
    if not command or not command.strip():
        return "empty command"
    bad = [c for c in command if not 0x20 <= ord(c) <= 0x7E]
    if bad:
        return f"character {bad[0]!r} is not allowed (printable ASCII and space only)"
    for char in FORBIDDEN:
        if char in command:
            return f"{char!r} is not allowed"
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        return f"could not parse the command ({exc})"
    if not tokens:
        return "empty command"
    head = tokens[0]
    if _is_repo_venv_tool(head, cwd, VENV_PYTHONS):
        if tokens[1:3] != ["-m", "pytest"]:
            return "python is allowed only as '-m pytest'"
        return _check_pytest(tokens[3:])
    if _is_repo_venv_tool(head, cwd, VENV_PYTESTS):
        return _check_pytest(tokens[1:])
    if head == "git":
        return _check_git(tokens[1:])
    return f"{head} is not allowed for the checker (use this repo's .venv python)"


def main() -> int:
    try:
        payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))
        command = payload["tool_input"]["command"]
        cwd = payload["cwd"]
        if not isinstance(command, str):
            raise TypeError("tool_input.command is not a string")
        if not isinstance(cwd, str) or not cwd:
            raise TypeError("cwd is missing or not a string")
    except Exception as exc:  # fail closed
        print(f"Blocked: the checker Bash guard could not read the tool call ({exc}).", file=sys.stderr)
        return 2
    reason = check(command, cwd)
    if reason is not None:
        print(f"Blocked: {reason}. {ALLOWED_SUMMARY}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
