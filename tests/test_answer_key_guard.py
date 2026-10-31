"""The Findlay answer key can't reach an agent, a tool or the panel API.

The orchestrator and single_agent strip the Findlay fields before any agent
or tool sees the variant. These tests cover the second line of defense: every
agent run() and tool entry point refuses a labeled variant on its own, POST
/review rejects either field, and nothing turns that refusal into an error
finding.
"""

from __future__ import annotations

import importlib
import inspect
import typing
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from _factories import placeholder_variant
from vus_panel import orchestrator
from vus_panel.agents import single_agent
from vus_panel.api import app
from vus_panel.schemas import ANSWER_KEY_FIELDS, EVIDENCE_AGENTS, AgentFinding, AnswerKeyError, ToolResult, Variant

PKG = Path(__file__).resolve().parents[1] / "src" / "vus_panel"
# Placeholder labels, not Findlay data. A score of 0.0 is falsy, so it also
# catches a guard that tests truthiness instead of None.
LABELS = {"findlay_function_class": "LOF", "findlay_score": 0.0}
PLACEHOLDER = {"chrom": "17", "pos": 100, "ref": "A", "alt": "G", "genome_build": "GRCh37"}

client = TestClient(app)


def agents_with_run() -> list[str]:
    """Every module in vus_panel.agents with a public run(), so a new agent is covered automatically."""
    names = []
    for path in sorted((PKG / "agents").glob("*.py")):
        if path.stem.startswith("_"):
            continue
        module = importlib.import_module(f"vus_panel.agents.{path.stem}")
        if inspect.isfunction(vars(module).get("run")):
            names.append(path.stem)
    return names


def tool_entry_points() -> list[tuple[object, str]]:
    """Public functions under vus_panel.tools that return a ToolResult, private modules included."""
    found = []
    for path in sorted((PKG / "tools").rglob("*.py")):
        parts = list(path.relative_to(PKG.parent).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        name = ".".join(parts)
        if name == "vus_panel.tools":
            continue
        module = importlib.import_module(name)
        for attr, fn in vars(module).items():
            if inspect.isfunction(fn) and fn.__module__ == name and not attr.startswith("_"):
                if typing.get_type_hints(fn).get("return") is ToolResult:
                    found.append((module, attr))
    return found


RUN_AGENTS = agents_with_run()
# Both lists, so a discovery that finds nothing fails instead of skipping every case.
AGENT_CASES = list(dict.fromkeys([*EVIDENCE_AGENTS, *RUN_AGENTS]))
TOOL_ENTRY_POINTS = tool_entry_points()
TOOL_IDS = [f"{module.__name__.removeprefix('vus_panel.tools.')}.{attr}" for module, attr in TOOL_ENTRY_POINTS]


@pytest.fixture
def tool_calls(monkeypatch):
    assert len(TOOL_ENTRY_POINTS) >= len(EVIDENCE_AGENTS), f"too few tool entry points found: {TOOL_IDS}"
    calls: list[str] = []
    for (module, attr), tool_id in zip(TOOL_ENTRY_POINTS, TOOL_IDS):
        def recorder(*args, _id=tool_id, _tool=vars(module).get("TOOL"), **kwargs):
            calls.append(_id)
            return ToolResult(tool=_tool, status="unavailable", message="recorded")

        monkeypatch.setattr(module, attr, recorder)
    return calls


@pytest.mark.parametrize("field", ANSWER_KEY_FIELDS)
@pytest.mark.parametrize("name", AGENT_CASES)
def test_agent_run_rejects_answer_key(name, field, tool_calls):
    assert set(RUN_AGENTS) == set(EVIDENCE_AGENTS)
    agent = importlib.import_module(f"vus_panel.agents.{name}")
    with pytest.raises(AnswerKeyError) as raised:
        agent.run(placeholder_variant(**{field: LABELS[field]}))
    message = str(raised.value)
    assert field in message
    assert f"agents.{name}.run" in message
    unset = [other for other in ANSWER_KEY_FIELDS if other != field]
    assert not any(other in message for other in unset), "only the fields that are set are named"
    assert tool_calls == [], "the agent must refuse the variant before any tool sees it"


@pytest.mark.parametrize("field", ANSWER_KEY_FIELDS)
@pytest.mark.parametrize("module, attr", TOOL_ENTRY_POINTS, ids=TOOL_IDS)
def test_tool_rejects_answer_key(module, attr, field):
    assert len(TOOL_ENTRY_POINTS) >= len(EVIDENCE_AGENTS)
    with pytest.raises(AnswerKeyError) as raised:
        vars(module)[attr](placeholder_variant(**{field: LABELS[field]}))
    assert field in str(raised.value)
    assert attr in str(raised.value)


@pytest.mark.parametrize("field, value", [(f, v) for f in ANSWER_KEY_FIELDS for v in (LABELS[f], None)])
def test_review_rejects_answer_key_field(field, value):
    response = client.post("/review", json={**PLACEHOLDER, field: value})
    assert response.status_code == 422
    # FastAPI echoes the whole body under "input", so look for the field in the error message itself.
    messages = [error["msg"] for error in response.json()["detail"]]
    assert any(field in message for message in messages), messages


def test_answer_key_fields_cover_every_findlay_field():
    assert sorted(ANSWER_KEY_FIELDS) == sorted(name for name in Variant.model_fields if name.startswith("findlay_"))


def test_orchestrator_and_single_agent_fail_loudly_if_stripping_breaks(monkeypatch):
    monkeypatch.setattr(Variant, "without_answer_key", lambda self: self)
    labeled = placeholder_variant(**LABELS)
    with pytest.raises(AnswerKeyError):
        orchestrator.review(labeled)
    with pytest.raises(AnswerKeyError):
        single_agent.review(labeled)


class LeakyRebuttalAgent:
    """Runs cleanly, then raises AnswerKeyError from rebut."""

    def run(self, variant):
        return AgentFinding(agent="evo2", variant_id=variant.variant_id, status="unavailable", rationale="stub")

    def rebut(self, own, others):
        raise AnswerKeyError("placeholder answer key in rebuttal")


def test_orchestrator_lets_answer_key_error_out_of_rebuttal():
    panel = {**orchestrator.DEFAULT_AGENTS, "evo2": LeakyRebuttalAgent()}
    with pytest.raises(AnswerKeyError, match="placeholder answer key in rebuttal"):
        orchestrator.review(placeholder_variant(), agents=panel)
