"""Tool isolation: each agent reaches only its own tool, and the chair sees findings only.

Three layers: a static walk of each agent's vus_panel import closure, a ban on
the dynamic tricks that could get around that walk, and runtime spies on
every tool entry point.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import typing
from pathlib import Path

import pytest

from _factories import finding, placeholder_variant
from vus_panel import orchestrator
from vus_panel.agents import chair, single_agent
from vus_panel.schemas import EVIDENCE_AGENTS, AgentFinding, PanelReport, ToolResult, Variant

SRC = Path(__file__).resolve().parents[1] / "src"
PKG = SRC / "vus_panel"

# Owners of private helpers or subpackages under tools/, keyed by their name
# directly under tools/. Example: "_evo2_backend": {"evo2", "sae"}. A module or
# subpackage named after an evidence agent belongs to that agent. Anything
# else under tools/ needs an entry here, or test_every_module_under_tools_has_an_owner fails.
PRIVATE_TOOL_OWNERS: dict[str, set[str]] = {}


def tool_modules(root: Path = SRC) -> list[str]:
    """Every module and subpackage under vus_panel/tools/, private ones included."""
    names = set()
    for path in (root / "vus_panel" / "tools").rglob("*.py"):
        parts = list(path.relative_to(root).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        names.add(".".join(parts))
    names.discard("vus_panel.tools")
    return sorted(names)


def tool_owners(module: str) -> set[str]:
    top = module.split(".")[2]
    if top in EVIDENCE_AGENTS:
        return {top}
    return set(PRIVATE_TOOL_OWNERS.get(top, set()))


TOOL_MODULES = tool_modules()
AGENT_MODULES = sorted(f"vus_panel.agents.{p.stem}" for p in (PKG / "agents").glob("*.py") if not p.stem.startswith("_"))
AGENT_FILES = sorted((PKG / "agents").glob("*.py"))
ALLOWED_TOOLS = {
    **{f"vus_panel.agents.{name}": {m for m in TOOL_MODULES if name in tool_owners(m)} for name in EVIDENCE_AGENTS},
    "vus_panel.agents.chair": set(),
    "vus_panel.agents.single_agent": set(TOOL_MODULES),
}


def module_file(name: str, root: Path = SRC) -> Path | None:
    relative = Path(*name.split("."))
    if (root / relative).with_suffix(".py").is_file():
        return (root / relative).with_suffix(".py")
    if (root / relative / "__init__.py").is_file():
        return root / relative / "__init__.py"
    return None


def _resolve_from(node: ast.ImportFrom, package: list[str]) -> str:
    if node.level:
        parts = package[: len(package) - (node.level - 1)]
        return ".".join(parts + ([node.module] if node.module else []))
    return node.module or ""


def _package_of(name: str, path: Path) -> list[str]:
    return name.split(".") if path.name == "__init__.py" else name.split(".")[:-1]


def direct_imports(name: str, root: Path = SRC) -> set[str]:
    """vus_panel modules that `name` imports anywhere in its file, plus their parent packages."""
    path = module_file(name, root)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = _package_of(name, path)
    candidates: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            candidates.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_from(node, package)
            candidates.add(base)
            candidates.update(f"{base}.{alias.name}" for alias in node.names)
    found: set[str] = set()
    for candidate in candidates:
        if candidate.split(".")[0] != "vus_panel" or module_file(candidate, root) is None:
            continue
        parts = candidate.split(".")
        found.update(".".join(parts[:i]) for i in range(1, len(parts) + 1))
    return found


def closure(name: str, root: Path = SRC) -> set[str]:
    parts = name.split(".")
    stack = [".".join(parts[:i]) for i in range(1, len(parts) + 1)]
    seen: set[str] = set()
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(direct_imports(current, root) - seen)
    return seen


BANNED_MODULES = {"importlib", "runpy", "pkgutil", "builtins"}
BANNED_NAMES = {"__import__", "importlib", "exec", "eval", "compile", "__builtins__"}
# Any attribute named "modules" is flagged, whatever it hangs off, which catches
# sys.modules through an alias or another module (os.sys.modules). String-based
# access such as getattr(os.sys, "modules") is not caught here; the runtime spies
# are the backstop for that.
BANNED_ATTRIBUTES = {"__dict__", "__import__", "__builtins__", "modules"}


def agent_violations(source: str, package: list[str] | None = None) -> list[str]:
    """Ways a module could reach a tool without a plain from-import the walker sees.

    `package` is the module's package, for resolving relative imports.
    """
    tree = ast.parse(source)
    package = package if package is not None else ["vus_panel", "agents"]
    bound: set[str] = set()
    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                bound.add(alias.asname or root)
                if root in BANNED_MODULES:
                    problems.append(f"import {alias.name}")
                if root == "vus_panel":
                    problems.append(f"plain 'import {alias.name}' exposes the whole package")
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_from(node, package)
            if base.split(".")[0] in BANNED_MODULES:
                problems.append(f"from {base} import")
            for alias in node.names:
                bound.add(alias.asname or alias.name)
                if alias.name == "*":
                    problems.append(f"star import from {base}")
                if f"{base}.{alias.name}" in {"vus_panel.tools", "vus_panel.agents"}:
                    problems.append(f"imports the {alias.name} package itself")
                if base == "sys" and alias.name == "modules":
                    problems.append("sys.modules")
                if alias.name == "sys":
                    problems.append(f"from {base} import sys")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            problems.append(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in BANNED_ATTRIBUTES:
            problems.append(node.attr)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func = node.func.id
            if func in {"globals", "locals"} or (func == "vars" and not node.args):
                problems.append(f"{func}()")
            elif func in {"getattr", "hasattr", "setattr", "vars"} and node.args:
                first = node.args[0]
                if isinstance(first, ast.Name) and first.id in bound:
                    problems.append(f"{func}() on imported name {first.id}")
    return problems


# Ownership of everything under tools/

def test_every_agent_module_has_a_tool_allowlist():
    assert set(AGENT_MODULES) == set(ALLOWED_TOOLS)


def test_every_module_under_tools_has_an_owner():
    unowned = [m for m in TOOL_MODULES if not tool_owners(m)]
    assert not unowned, f"modules under tools/ with no owner (add them to PRIVATE_TOOL_OWNERS): {unowned}"


@pytest.mark.parametrize("name", EVIDENCE_AGENTS)
def test_every_evidence_agent_owns_its_tool(name):
    assert f"vus_panel.tools.{name}" in ALLOWED_TOOLS[f"vus_panel.agents.{name}"]


def test_tool_listing_includes_private_modules_and_subpackages(tmp_path):
    for relative in ["tools/__init__.py", "tools/evo2.py", "tools/_helper.py", "tools/sub/__init__.py", "tools/sub/x.py"]:
        path = tmp_path / "vus_panel" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    listed = tool_modules(tmp_path)
    assert listed == ["vus_panel.tools._helper", "vus_panel.tools.evo2", "vus_panel.tools.sub", "vus_panel.tools.sub.x"]
    assert tool_owners("vus_panel.tools._helper") == set()
    assert tool_owners("vus_panel.tools.sub.x") == set()


# Static import closure

@pytest.mark.parametrize("agent_module", sorted(ALLOWED_TOOLS))
def test_agent_reaches_only_its_allowed_tools(agent_module):
    reached = closure(agent_module) & set(TOOL_MODULES)
    extra = reached - ALLOWED_TOOLS[agent_module]
    assert not extra, f"{agent_module} imports another agent's tool: {sorted(extra)}"


@pytest.mark.parametrize("agent_module", AGENT_MODULES)
def test_agent_does_not_import_other_agents_or_the_orchestrator(agent_module):
    banned = (set(AGENT_MODULES) - {agent_module}) | {"vus_panel.orchestrator", "vus_panel.api"}
    reached = closure(agent_module) & banned
    assert not reached, f"{agent_module} imports {sorted(reached)}"


def test_import_walker_follows_shared_helpers(tmp_path):
    files = {
        "vus_panel/__init__.py": "",
        "vus_panel/tools/__init__.py": "",
        "vus_panel/tools/a.py": "",
        "vus_panel/tools/b.py": "",
        "vus_panel/agents/__init__.py": "",
        "vus_panel/agents/_shared.py": "from vus_panel.tools import b\n",
        "vus_panel/agents/a.py": "from vus_panel.tools import a\nfrom . import _shared\n",
    }
    for relative, text in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    assert {"vus_panel.tools.a", "vus_panel.tools.b"} <= closure("vus_panel.agents.a", tmp_path)


# Bans on the tricks the walker can't see

@pytest.mark.parametrize("path", AGENT_FILES, ids=lambda p: p.name)
def test_agent_files_use_no_dynamic_access(path):
    assert agent_violations(path.read_text(encoding="utf-8"), ["vus_panel", "agents"]) == []


@pytest.mark.parametrize(
    "source",
    [
        "from vus_panel import tools\ntools.sae.feature_activations(v)\n",
        "from .. import tools\n",
        "import vus_panel.tools.evo2\nvus_panel.tools.sae.feature_activations(v)\n",
        "import vus_panel\n",
        "import sys\nsys.modules['vus_panel.tools.sae']\n",
        "from sys import modules\n",
        "from vus_panel.tools import evo2\ngetattr(evo2, 'log_likelihood_delta')\n",
        "from vus_panel.tools import *\n",
        "import importlib\nimportlib.import_module('vus_panel.tools.sae')\n",
        "__import__('vus_panel.tools.sae')\n",
        "globals()['x']\n",
        "from vus_panel.tools import evo2\nevo2.__dict__\n",
        "import sys as s\ns.modules['vus_panel.tools.sae']\n",
        "from os import sys\nsys.modules['vus_panel.tools.sae']\n",
        "from os import sys as s\n",
        "import os\nos.sys.modules['vus_panel.tools.sae']\n",
        "exec('from vus_panel.tools import sae')\n",
        "eval('1')\n",
        "compile('x = 1', 'f', 'exec')\n",
        "__builtins__['__import__']('vus_panel.tools.sae')\n",
        "import runpy\n",
        "from pkgutil import resolve_name\n",
        "import builtins\n",
    ],
)
def test_ban_detector_catches_known_bypasses(source):
    assert agent_violations(source), source


@pytest.mark.parametrize("agent_module", [*AGENT_MODULES, "vus_panel.agents._common"])
def test_agent_import_closure_uses_no_dynamic_access(agent_module):
    for module in sorted(closure(agent_module)):
        path = module_file(module)
        problems = agent_violations(path.read_text(encoding="utf-8"), _package_of(module, path))
        assert not problems, f"{module} (reached from {agent_module}): {problems}"


def test_chair_signature_takes_findings_only():
    signature = inspect.signature(chair.decide)
    assert list(signature.parameters) == ["variant_id", "findings_initial", "findings_after_rebuttal"]
    hints = typing.get_type_hints(chair.decide)
    assert hints["variant_id"] is str
    assert hints["findings_initial"] == list[AgentFinding]
    assert hints["findings_after_rebuttal"] == list[AgentFinding]
    assert hints["return"] is PanelReport


# Runtime spies on every tool entry point

def _tool_entry_points():
    for name in TOOL_MODULES:
        module = importlib.import_module(name)
        tool = getattr(module, "TOOL", None)
        for attr, fn in vars(module).items():
            if inspect.isfunction(fn) and fn.__module__ == name and not attr.startswith("_"):
                if typing.get_type_hints(fn).get("return") is ToolResult:
                    yield module, attr, tool


@pytest.fixture
def tool_calls(monkeypatch):
    calls: list[str] = []
    entry_points = list(_tool_entry_points())
    assert len(entry_points) >= len(EVIDENCE_AGENTS)
    for module, attr, tool in entry_points:
        def recorder(*args, _name=module.__name__, _tool=tool, **kwargs):
            calls.append(_name)
            return ToolResult(tool=_tool, status="unavailable", message="recorded")

        monkeypatch.setattr(module, attr, recorder)
    return calls


@pytest.mark.parametrize("name", EVIDENCE_AGENTS)
def test_runtime_agent_calls_only_its_own_tool(name, tool_calls):
    agent = importlib.import_module(f"vus_panel.agents.{name}")
    own = agent.run(placeholder_variant())
    assert f"vus_panel.tools.{name}" in tool_calls
    assert set(tool_calls) <= ALLOWED_TOOLS[f"vus_panel.agents.{name}"]
    tool_calls.clear()
    others = [finding(a, variant_id=own.variant_id) for a in EVIDENCE_AGENTS if a != name]
    agent.rebut(own, others)
    assert tool_calls == [], "rebut must see findings only"


def test_runtime_rebut_and_chair_with_ok_findings_call_no_tool(tool_calls):
    ok = [finding(a, "ok", "damaging") for a in EVIDENCE_AGENTS]
    for name in EVIDENCE_AGENTS:
        agent = importlib.import_module(f"vus_panel.agents.{name}")
        own = next(f for f in ok if f.agent == name)
        agent.rebut(own, [f for f in ok if f.agent != name])
    chair.decide("variant-A", ok, ok)
    assert tool_calls == []


def test_runtime_chair_calls_no_tool(tool_calls):
    findings = [finding(a) for a in EVIDENCE_AGENTS]
    chair.decide("variant-A", findings, findings)
    assert tool_calls == []


def test_runtime_single_agent_calls_all_four_tools(tool_calls):
    single_agent.review(placeholder_variant())
    assert set(tool_calls) == {f"vus_panel.tools.{name}" for name in EVIDENCE_AGENTS}


class RecordingAgent:
    def __init__(self, name):
        self.name = name
        self.run_calls = []
        self.rebut_calls = []

    def run(self, *args, **kwargs):
        self.run_calls.append((args, kwargs))
        return AgentFinding(agent=self.name, variant_id=args[0].variant_id, status="unavailable", rationale="stub")

    def rebut(self, *args, **kwargs):
        self.rebut_calls.append((args, kwargs))
        return args[0]


def test_orchestrator_hands_the_chair_findings_only():
    calls = []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return chair.decide(*args, **kwargs)

    orchestrator.review(placeholder_variant(), chair=spy)
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert kwargs == {}
    assert len(args) == 3
    variant_id, initial, after = args
    assert type(variant_id) is str
    for findings in (initial, after):
        assert type(findings) is list
        assert len(findings) == 4
        assert all(type(f) is AgentFinding for f in findings)


def test_rebuttal_sees_only_the_other_three_findings():
    panel = {name: RecordingAgent(name) for name in EVIDENCE_AGENTS}
    orchestrator.review(placeholder_variant(), agents=panel)
    for name, agent in panel.items():
        assert len(agent.rebut_calls) == 1
        args, kwargs = agent.rebut_calls[0]
        assert kwargs == {}
        own, others = args
        assert type(own) is AgentFinding and own.agent == name
        assert type(others) is list and len(others) == 3
        assert all(type(f) is AgentFinding for f in others)
        assert {f.agent for f in others} == set(EVIDENCE_AGENTS) - {name}


def test_agents_never_receive_findlay_fields():
    panel = {name: RecordingAgent(name) for name in EVIDENCE_AGENTS}
    variant = placeholder_variant(findlay_function_class="LOF", findlay_score=0.0)
    orchestrator.review(variant, agents=panel)
    for agent in panel.values():
        (args, kwargs), = agent.run_calls
        assert kwargs == {}
        (seen,) = args
        assert type(seen) is Variant
        assert seen.findlay_function_class is None and seen.findlay_score is None


def _record_tool_variants(monkeypatch):
    seen: list[Variant] = []
    for module, attr, tool in list(_tool_entry_points()):
        def recorder(variant, _tool=tool):
            seen.append(variant)
            return ToolResult(tool=_tool, status="unavailable", message="stub")

        monkeypatch.setattr(module, attr, recorder)
    return seen


def test_real_tools_never_receive_findlay_fields(monkeypatch):
    seen = _record_tool_variants(monkeypatch)
    orchestrator.review(placeholder_variant(findlay_function_class="FUNC", findlay_score=0.0))
    assert len(seen) == 4
    assert all(v.findlay_function_class is None and v.findlay_score is None for v in seen)


def test_single_agent_strips_findlay_fields_and_skips_rebuttal(monkeypatch):
    seen = _record_tool_variants(monkeypatch)
    report = single_agent.review(placeholder_variant(findlay_function_class="INT", findlay_score=0.0))
    assert len(seen) == 4
    assert all(v.findlay_function_class is None and v.findlay_score is None for v in seen)
    assert report.call == "unresolved"
    assert [f.agent for f in report.findings_initial] == ["single_agent"]
    assert report.findings_after_rebuttal == report.findings_initial
