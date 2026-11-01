"""The agent LLM layer: evidence agents, rebuttal, chair and single-agent baseline.

Every test uses a fake LLM client that records its calls. Nothing reaches the
network. Tool results are placeholder values, not evidence.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from _factories import placeholder_variant
from vus_panel import orchestrator
from vus_panel.agents import _interpret, chair, conservation, evo2, population, sae, single_agent
from vus_panel.schemas import AgentFinding, PanelReport, ToolResult
from vus_panel.tools import conservation as conservation_tool
from vus_panel.tools import evo2 as evo2_tool
from vus_panel.tools import population as population_tool
from vus_panel.tools import sae as sae_tool

AGENTS = {"evo2": evo2, "sae": sae, "conservation": conservation, "population": population}
TOOLS = {
    "evo2": (evo2_tool, "log_likelihood_delta"),
    "sae": (sae_tool, "feature_activations"),
    "conservation": (conservation_tool, "conservation_score"),
    "population": (population_tool, "allele_frequency"),
}
VALUES = {"placeholder_score": -1.25, "nested": {"n": 3}}
VID = placeholder_variant().variant_id


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


def install_llm(monkeypatch, *replies):
    fake = FakeClient(replies)
    monkeypatch.setenv("ANTHROPIC_MODEL", "placeholder-model")
    monkeypatch.setattr(_interpret, "get_client", lambda: fake)
    monkeypatch.setattr(chair, "get_client", lambda: fake)
    return fake


def stub_tool(monkeypatch, name, status="ok", values=None):
    module, attr = TOOLS[name]
    result = ToolResult(tool=name, status=status, values=(values if values is not None else VALUES) if status == "ok" else {},
                        message=f"placeholder {name} result", sources=[f"placeholder {name} source"])
    monkeypatch.setattr(module, attr, lambda variant: result)
    return result


def verdict_reply(verdict="damaging", **extra):
    return {"verdict": verdict, "rationale": "Placeholder rationale citing the value.",
            "what_would_change_my_mind": "A placeholder value on the other side.", **extra}


@pytest.mark.parametrize("name", AGENTS)
def test_llm_cannot_change_the_tools_numbers_or_sources(monkeypatch, name):
    tool = stub_tool(monkeypatch, name)
    fake = install_llm(monkeypatch, verdict_reply(raw_values={"placeholder_score": 999}, placeholder_score=999,
                                                  sources=["invented"]))
    finding = AGENTS[name].run(placeholder_variant())
    assert len(fake.calls) == 1
    assert finding.status == "ok" and finding.verdict == "damaging"
    assert finding.raw_values == tool.values and finding.sources == tool.sources
    assert finding.what_would_change_my_mind == "A placeholder value on the other side."


@pytest.mark.parametrize("name", AGENTS)
def test_rebuttal_changes_only_the_verdict_with_a_reason(monkeypatch, name):
    tool = stub_tool(monkeypatch, name)
    fake = install_llm(monkeypatch, verdict_reply(),
                       {"verdict": "tolerated", "changed": True, "reason": "Placeholder reason.",
                        "what_would_change_my_mind": "A placeholder value for damaging.",
                        "raw_values": {"placeholder_score": 999}, "sources": ["invented"]})
    own = AGENTS[name].run(placeholder_variant())
    after = AGENTS[name].rebut(own, other_findings(name))
    assert len(fake.calls) == 2
    assert after.verdict == "tolerated" and after.revision_reason == "Placeholder reason."
    assert after.raw_values == own.raw_values == tool.values and after.sources == own.sources
    # The old text argued against 'damaging'; a revised verdict carries its own.
    assert after.what_would_change_my_mind == "A placeholder value for damaging."
    assert orchestrator._rebuttal_problem(own, after) is None
    assert "What would change their mind: placeholder other" in fake.calls[1]["messages"][0]["content"]


def other_findings(name):
    return [AgentFinding(agent=a, variant_id=VID, status="ok", verdict="tolerated", rationale="placeholder",
                         what_would_change_my_mind="placeholder other") for a in AGENTS if a != name]


REVISION = {"verdict": "tolerated", "changed": True, "reason": "Placeholder reason.",
            "what_would_change_my_mind": "A placeholder value for damaging."}


@pytest.mark.parametrize("reply", [
    RuntimeError("placeholder API failure"),
    "not json",
    {**REVISION, "verdict": "damaging"},  # 'changed' but the same verdict
    {**REVISION, "reason": ""},
    {**REVISION, "what_would_change_my_mind": ""},
    {**REVISION, "changed": "true"},  # only the JSON true counts
    {**REVISION, "reason": "ClinVar lists it as benign."},
    {**REVISION, "what_would_change_my_mind": "A CLNSIG of Pathogenic."},
])
def test_rebuttal_keeps_the_finding_unless_the_revision_is_complete(monkeypatch, reply):
    stub_tool(monkeypatch, "evo2")
    fake = install_llm(monkeypatch, verdict_reply(), reply)
    own = evo2.run(placeholder_variant())
    assert evo2.rebut(own, other_findings("evo2")) == own
    assert len(fake.calls) == 2


@pytest.mark.parametrize("name", AGENTS)
@pytest.mark.parametrize("status", ["unavailable", "error"])
def test_a_tool_without_a_result_makes_no_llm_call(monkeypatch, name, status):
    stub_tool(monkeypatch, name, status=status)
    fake = install_llm(monkeypatch)
    finding = AGENTS[name].run(placeholder_variant())
    assert finding.status == status and finding.verdict is None and finding.raw_values == {}
    others = [AgentFinding(agent=a, variant_id=VID, status="unavailable", rationale="placeholder") for a in AGENTS if a != name]
    assert AGENTS[name].rebut(finding, others) == finding
    assert fake.calls == []


@pytest.mark.parametrize("reply", [
    "not json",
    "[1, 2]",
    {"verdict": "maybe", "rationale": "x", "what_would_change_my_mind": "y"},
    {"verdict": "damaging", "rationale": "x"},
    {"verdict": "damaging", "rationale": "", "what_would_change_my_mind": "y"},
    {"verdict": "damaging", "rationale": 5, "what_would_change_my_mind": "y"},
    RuntimeError("placeholder API failure"),
    verdict_reply(what_would_change_my_mind="A ClinVar pathogenic classification."),
    verdict_reply(rationale="It matches VCV000012345."),
])
def test_an_unusable_reply_becomes_an_error_finding(monkeypatch, reply):
    stub_tool(monkeypatch, "evo2")
    fake = install_llm(monkeypatch, reply)
    finding = evo2.run(placeholder_variant())
    assert len(fake.calls) == 1
    assert finding.status == "error" and finding.verdict is None and finding.raw_values == {}
    assert finding.what_would_change_my_mind is None


def test_missing_llm_settings_give_unavailable_without_a_call(monkeypatch):
    stub_tool(monkeypatch, "evo2")
    finding = evo2.run(placeholder_variant())
    assert finding.status == "unavailable" and "ANTHROPIC" in finding.rationale


def test_what_would_change_my_mind_is_optional_and_only_for_ok_findings():
    ok = AgentFinding(agent="evo2", variant_id=VID, status="ok", verdict="uncertain", rationale="x")
    assert ok.what_would_change_my_mind is None
    AgentFinding(agent="evo2", variant_id=VID, status="ok", verdict="uncertain", rationale="x", what_would_change_my_mind="y")
    with pytest.raises(ValidationError, match="what_would_change_my_mind"):
        AgentFinding(agent="evo2", variant_id=VID, status="unavailable", rationale="x", what_would_change_my_mind="y")
    with pytest.raises(ValidationError):
        AgentFinding(agent="evo2", variant_id=VID, status="ok", verdict="uncertain", rationale="x", what_would_change_my_mind="")
    with pytest.raises(ValidationError, match="ClinVar"):
        AgentFinding(agent="evo2", variant_id=VID, status="ok", verdict="uncertain", rationale="x",
                     what_would_change_my_mind="A ClinVar submission.")


def test_clinvar_patterns_match_the_eval_refusal():
    # run_eval does not scan what_would_change_my_mind, so the schema's copy of its patterns must not drift.
    from vus_panel import schemas
    from vus_panel.eval import run_eval

    assert (schemas._CLINVAR_RE.pattern, schemas._CLINVAR_RE.flags) == (run_eval.CLINVAR_RE.pattern, run_eval.CLINVAR_RE.flags)
    assert (schemas._CLINVAR_VCF_TAG_RE.pattern, schemas._CLINVAR_VCF_TAG_RE.flags) == (
        run_eval.CLINVAR_VCF_TAG_RE.pattern, run_eval.CLINVAR_VCF_TAG_RE.flags)
    for text in ("clinvar", "RCV000012345", "CLNREVSTAT", "no database named"):
        expected = bool(run_eval.CLINVAR_RE.search(text) or run_eval.CLINVAR_VCF_TAG_RE.search(text))
        assert schemas.mentions_clinvar(text) is expected


def chair_findings():
    return [
        AgentFinding(agent="evo2", variant_id=VID, status="ok", verdict="damaging", raw_values=VALUES,
                     rationale="placeholder", what_would_change_my_mind="placeholder"),
        AgentFinding(agent="sae", variant_id=VID, status="unavailable", rationale="placeholder"),
    ]


CHAIR_REPLY = {"call": "likely_damaging", "confidence": 0.7, "driving_agents": ["evo2"], "conflicts": [],
               "what_would_settle_it": "placeholder", "counselor_summary": "placeholder", "patient_summary": "placeholder"}


@pytest.mark.parametrize("reply,call", [
    (CHAIR_REPLY, "likely_damaging"),
    ({**CHAIR_REPLY, "driving_agents": ["sae"]}, "unresolved"),
    ({**CHAIR_REPLY, "confidence": "high"}, "unresolved"),
    ({**CHAIR_REPLY, "confidence": float("nan")}, "unresolved"),
    ({**CHAIR_REPLY, "confidence": float("inf")}, "unresolved"),
    ("not json", "unresolved"),
])
def test_chair_makes_one_call_over_findings_and_names_only_ok_agents(monkeypatch, reply, call):
    fake = install_llm(monkeypatch, reply)
    findings = chair_findings()
    report = chair.decide(VID, findings, findings)
    assert len(fake.calls) == 1 and isinstance(report, PanelReport)
    assert report.call == call and report.calibrated is False
    assert set(report.driving_agents) <= {"evo2"}
    assert "What would change its mind: placeholder" in fake.calls[0]["messages"][0]["content"]


def test_chair_without_ok_findings_makes_no_call(monkeypatch):
    fake = install_llm(monkeypatch)
    findings = [AgentFinding(agent="evo2", variant_id=VID, status="unavailable", rationale="placeholder")]
    assert chair.decide(VID, findings, findings).call == "unresolved"
    assert fake.calls == []


def test_panel_end_to_end_with_the_fake_llm(monkeypatch):
    stub_tool(monkeypatch, "evo2")
    stub_tool(monkeypatch, "sae", status="unavailable")
    stub_tool(monkeypatch, "conservation", values={"placeholder_phylop": 2.5})
    stub_tool(monkeypatch, "population", status="error")
    keep = {"verdict": "damaging", "changed": False, "reason": ""}
    fake = install_llm(monkeypatch, verdict_reply(), verdict_reply(), keep, keep, CHAIR_REPLY)
    report = orchestrator.review(placeholder_variant())
    # Two agent runs, two rebuttals, one chair call: no call for the agents without evidence.
    assert len(fake.calls) == 5
    assert report.call == "likely_damaging" and report.calibrated is False and report.driving_agents == ["evo2"]
    by_agent = {f.agent: f for f in report.findings_after_rebuttal}
    assert by_agent["evo2"].raw_values == VALUES and by_agent["conservation"].raw_values == {"placeholder_phylop": 2.5}
    assert by_agent["sae"].status == "unavailable" and by_agent["population"].status == "error"


SINGLE_REPLY = {**verdict_reply(), "call": "likely_damaging", "confidence": 0.6, "what_would_settle_it": "placeholder",
                "counselor_summary": "placeholder", "patient_summary": "placeholder"}


def test_single_agent_reads_all_tools_in_one_call_with_no_rebuttal(monkeypatch):
    stub_tool(monkeypatch, "evo2")
    stub_tool(monkeypatch, "conservation", values={"placeholder_phylop": 2.5})
    stub_tool(monkeypatch, "sae", status="unavailable")
    stub_tool(monkeypatch, "population", status="unavailable")
    fake = install_llm(monkeypatch, {**SINGLE_REPLY, "raw_values": {"evo2": {"placeholder_score": 999}}})
    report = single_agent.review(placeholder_variant())
    assert len(fake.calls) == 1
    (finding,) = report.findings_initial
    assert report.findings_after_rebuttal == report.findings_initial
    assert finding.agent == "single_agent" and finding.status == "ok"
    assert finding.raw_values == {"evo2": VALUES, "conservation": {"placeholder_phylop": 2.5}}
    assert report.call == "likely_damaging" and report.calibrated is False and report.driving_agents == ["single_agent"]


@pytest.mark.parametrize("reply", [
    {**SINGLE_REPLY, "confidence": None},
    {**SINGLE_REPLY, "confidence": float("nan")},
    {**SINGLE_REPLY, "confidence": float("inf")},
    {**SINGLE_REPLY, "call": "probably"},
    {**SINGLE_REPLY, "verdict": "tolerated"},  # a call that contradicts the agent's own verdict
    {**SINGLE_REPLY, "verdict": "uncertain"},  # 'uncertain' allows only 'unresolved'
    "not json",
])
def test_single_agent_falls_back_to_unresolved(monkeypatch, reply):
    stub_tool(monkeypatch, "evo2")
    for name in ("sae", "conservation", "population"):
        stub_tool(monkeypatch, name, status="unavailable")
    fake = install_llm(monkeypatch, reply)
    report = single_agent.review(placeholder_variant())
    assert len(fake.calls) == 1 and report.call == "unresolved" and report.confidence is None


def test_single_agent_without_ok_tools_makes_no_call(monkeypatch):
    for name in AGENTS:
        stub_tool(monkeypatch, name, status="unavailable")
    fake = install_llm(monkeypatch)
    report = single_agent.review(placeholder_variant())
    assert fake.calls == [] and report.call == "unresolved"
    assert report.findings_initial[0].status == "unavailable"


def test_single_agent_accepts_an_unresolved_call_and_names_missing_evidence(monkeypatch):
    stub_tool(monkeypatch, "evo2")
    stub_tool(monkeypatch, "sae", status="unavailable")
    stub_tool(monkeypatch, "conservation", status="error")
    stub_tool(monkeypatch, "population", status="unavailable")
    fake = install_llm(monkeypatch, {**SINGLE_REPLY, "verdict": "uncertain", "call": "unresolved", "confidence": None})
    report = single_agent.review(placeholder_variant())
    assert report.call == "unresolved" and report.confidence is None and report.driving_agents == []
    assert report.findings_initial[0].status == "ok" and report.findings_initial[0].verdict == "uncertain"
    prompt = fake.calls[0]["messages"][0]["content"]
    assert "sae is unavailable" in prompt and "conservation is error" in prompt


def test_single_agent_with_ok_tools_but_no_llm_settings_stays_unresolved(monkeypatch):
    stub_tool(monkeypatch, "evo2")
    for name in ("sae", "conservation", "population"):
        stub_tool(monkeypatch, name, status="unavailable")
    report = single_agent.review(placeholder_variant())
    assert report.call == "unresolved"
    assert report.findings_initial[0].status == "unavailable"
    assert "ANTHROPIC" in report.what_would_settle_it
