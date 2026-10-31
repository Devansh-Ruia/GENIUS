from __future__ import annotations

import pytest

from _factories import placeholder_variant
from vus_panel import orchestrator
from vus_panel.agents import chair, conservation, evo2, population, sae
from vus_panel.schemas import EVIDENCE_AGENTS, AgentFinding


class StubAgent:
    """Returns an unavailable finding, or misbehaves on request."""

    def __init__(self, name, run_raises=False, rebut_raises=False, run_returns=None):
        self.name = name
        self.run_raises = run_raises
        self.rebut_raises = rebut_raises
        self.run_returns = run_returns

    def run(self, variant):
        if self.run_raises:
            raise RuntimeError("placeholder failure")
        if self.run_returns is not None:
            return self.run_returns
        return AgentFinding(agent=self.name, variant_id=variant.variant_id, status="unavailable", rationale="stub")

    def rebut(self, own, others):
        if self.rebut_raises:
            raise RuntimeError("placeholder rebuttal failure")
        return own


def stub_panel(**overrides):
    panel = {name: StubAgent(name) for name in EVIDENCE_AGENTS}
    panel.update(overrides)
    return panel


def test_all_unavailable_findings_give_unresolved_report():
    report = orchestrator.review(placeholder_variant())
    assert report.variant_id == "GRCh37-17-100-A-G"
    assert report.call == "unresolved"
    assert report.confidence is None
    assert report.calibrated is False
    assert report.driving_agents == []
    assert report.mock is False
    for findings in (report.findings_initial, report.findings_after_rebuttal):
        assert [f.agent for f in findings] == list(EVIDENCE_AGENTS)
        assert all(f.status == "unavailable" and f.verdict is None and f.raw_values == {} for f in findings)
    assert "ACMG" in report.counselor_summary
    assert report.patient_summary


def test_default_panel_is_the_four_evidence_agents():
    assert orchestrator.DEFAULT_AGENTS == {"evo2": evo2, "sae": sae, "conservation": conservation, "population": population}


def test_agent_exception_becomes_error_finding():
    report = orchestrator.review(placeholder_variant(), agents=stub_panel(sae=StubAgent("sae", run_raises=True)))
    by_agent = {f.agent: f for f in report.findings_initial}
    assert by_agent["sae"].status == "error"
    assert "placeholder failure" in by_agent["sae"].rationale
    assert by_agent["evo2"].status == "unavailable"
    assert report.call == "unresolved"


def test_rebuttal_exception_becomes_error_finding():
    report = orchestrator.review(placeholder_variant(), agents=stub_panel(evo2=StubAgent("evo2", rebut_raises=True)))
    initial = {f.agent: f for f in report.findings_initial}
    after = {f.agent: f for f in report.findings_after_rebuttal}
    assert initial["evo2"].status == "unavailable"
    assert after["evo2"].status == "error"
    assert "rebuttal" in after["evo2"].rationale


def test_wrong_return_type_becomes_error_finding():
    report = orchestrator.review(placeholder_variant(), agents=stub_panel(sae=StubAgent("sae", run_returns="not a finding")))
    assert {f.agent: f for f in report.findings_initial}["sae"].status == "error"


def test_finding_for_another_agent_becomes_error_finding():
    imposter = AgentFinding(agent="evo2", variant_id="GRCh37-17-100-A-G", status="unavailable", rationale="stub")
    report = orchestrator.review(placeholder_variant(), agents=stub_panel(sae=StubAgent("sae", run_returns=imposter)))
    sae_finding = {f.agent: f for f in report.findings_initial}["sae"]
    assert sae_finding.status == "error"
    assert sae_finding.agent == "sae"


def test_report_is_never_marked_mock_by_the_orchestrator():
    def mock_chair(variant_id, initial, after):
        return chair.decide(variant_id, initial, after).model_copy(update={"mock": True})

    assert orchestrator.review(placeholder_variant(), chair=mock_chair).mock is False


def test_chair_must_return_a_report():
    with pytest.raises(TypeError):
        orchestrator.review(placeholder_variant(), chair=lambda *args: {"call": "unresolved"})


def test_unknown_agent_name_is_rejected():
    with pytest.raises(ValueError, match="unknown"):
        orchestrator.review(placeholder_variant(), agents={"clinvar": StubAgent("evo2")})


def test_cross_build_cache_error_passes_through(tmp_path):
    from vus_panel.tools import conservation as conservation_tool

    path = conservation_tool.cache_path("GRCh38")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("placeholder\n", encoding="utf-8")
    report = orchestrator.review(placeholder_variant(genome_build="GRCh37"))
    finding = {f.agent: f for f in report.findings_after_rebuttal}["conservation"]
    assert finding.status == "error"
    assert "liftover" in finding.rationale
    assert report.call == "unresolved"


def test_chair_reports_verdict_conflicts_and_makes_no_call_without_an_llm():
    findings = [
        AgentFinding(agent="evo2", variant_id="variant-A", status="ok", verdict="damaging", rationale="placeholder"),
        AgentFinding(agent="population", variant_id="variant-A", status="ok", verdict="tolerated", rationale="placeholder"),
    ]
    report = chair.decide("variant-A", findings, findings)
    assert report.call == "unresolved"
    assert report.conflicts == ["evo2 says damaging, population says tolerated"]
    assert report.confidence is None
    assert report.calibrated is False
    assert report.driving_agents == []
    assert "VUS" in report.counselor_summary


# Rebuttal rules: a break becomes a status "error" finding for that agent

PLACEHOLDER_ID = "GRCh37-17-100-A-G"


class OkAgent(StubAgent):
    """Starts with an ok 'damaging' finding and rebuts with `rebut_result`."""

    def __init__(self, name, rebut_result):
        super().__init__(name)
        self.rebut_result = rebut_result

    def run(self, variant):
        return AgentFinding(agent=self.name, variant_id=variant.variant_id, status="ok", verdict="damaging", rationale="placeholder")

    def rebut(self, own, others):
        return self.rebut_result(own)


def test_initial_finding_with_revision_reason_becomes_error():
    bad = AgentFinding(agent="sae", variant_id=PLACEHOLDER_ID, status="unavailable", rationale="stub", revision_reason="placeholder")
    report = orchestrator.review(placeholder_variant(), agents=stub_panel(sae=StubAgent("sae", run_returns=bad)))
    sae_initial = {f.agent: f for f in report.findings_initial}["sae"]
    assert sae_initial.status == "error"
    assert "revision_reason" in sae_initial.rationale


@pytest.mark.parametrize(
    "changed",
    [
        lambda own: own.model_copy(update={"status": "ok", "verdict": "damaging", "revision_reason": "placeholder"}),
        lambda own: own.model_copy(update={"rationale": "rewritten in rebuttal"}),
    ],
    ids=["gains_verdict", "rewrites_rationale"],
)
def test_rebuttal_cannot_change_a_finding_without_evidence(changed):
    class Changer(StubAgent):
        def rebut(self, own, others):
            return changed(own)

    report = orchestrator.review(placeholder_variant(), agents=stub_panel(evo2=Changer("evo2")))
    after = {f.agent: f for f in report.findings_after_rebuttal}["evo2"]
    assert after.status == "error"
    assert "no evidence" in after.rationale


def test_verdict_change_without_reason_becomes_error():
    flip = OkAgent("evo2", lambda own: own.model_copy(update={"verdict": "tolerated"}))
    report = orchestrator.review(placeholder_variant(), agents=stub_panel(evo2=flip))
    after = {f.agent: f for f in report.findings_after_rebuttal}["evo2"]
    assert after.status == "error"
    assert "revision_reason" in after.rationale


def test_verdict_change_with_reason_is_kept():
    flip = OkAgent("evo2", lambda own: own.model_copy(update={"verdict": "tolerated", "revision_reason": "placeholder"}))
    report = orchestrator.review(placeholder_variant(), agents=stub_panel(evo2=flip))
    after = {f.agent: f for f in report.findings_after_rebuttal}["evo2"]
    assert (after.status, after.verdict, after.revision_reason) == ("ok", "tolerated", "placeholder")


# An ok finding keeps its evidence through rebuttal, and only gives it up with a reason

@pytest.mark.parametrize(
    "change",
    [{"raw_values": {"placeholder": 0.0}}, {"sources": ["placeholder source"]}],
    ids=["raw_values", "sources"],
)
def test_rebuttal_cannot_add_evidence(change):
    agent = OkAgent("evo2", lambda own: own.model_copy(update=change))
    report = orchestrator.review(placeholder_variant(), agents=stub_panel(evo2=agent))
    after = {f.agent: f for f in report.findings_after_rebuttal}["evo2"]
    assert after.status == "error"
    assert "no tool access" in after.rationale


def _after(agent):
    report = orchestrator.review(placeholder_variant(), agents=stub_panel(evo2=agent))
    return {f.agent: f for f in report.findings_after_rebuttal}["evo2"]


def test_withdrawing_an_ok_finding_needs_a_reason():
    withdraw = {"status": "unavailable", "verdict": None, "raw_values": {}}
    silent = _after(OkAgent("evo2", lambda own: own.model_copy(update=withdraw)))
    assert silent.status == "error"
    assert "withdrawn" in silent.rationale
    explained = _after(OkAgent("evo2", lambda own: own.model_copy(update={**withdraw, "revision_reason": "placeholder"})))
    assert (explained.status, explained.revision_reason) == ("unavailable", "placeholder")


def test_orchestrator_rebuttal_error_on_an_ok_finding_records_why():
    class Crash(OkAgent):
        def rebut(self, own, others):
            raise RuntimeError("placeholder crash")

    after = _after(Crash("evo2", None))
    assert after.status == "error"
    assert after.revision_reason.startswith("set by the orchestrator")


def test_agent_finding_subclass_is_rejected():
    class SubFinding(AgentFinding):
        pass

    imposter = SubFinding(agent="sae", variant_id=PLACEHOLDER_ID, status="unavailable", rationale="stub")
    report = orchestrator.review(placeholder_variant(), agents=stub_panel(sae=StubAgent("sae", run_returns=imposter)))
    sae_initial = {f.agent: f for f in report.findings_initial}["sae"]
    assert type(sae_initial) is AgentFinding
    assert sae_initial.status == "error"


def test_chair_must_pass_findings_through_unchanged():
    def rewriting_chair(variant_id, initial, after):
        changed = [after[0].model_copy(update={"rationale": "rewritten by the chair"}), *after[1:]]
        return chair.decide(variant_id, initial, changed)

    with pytest.raises(ValueError, match="chair changed the findings"):
        orchestrator.review(placeholder_variant(), chair=rewriting_chair)


def test_chair_editing_findings_in_place_is_caught():
    # The chair edits the ok evo2 finding the same way in both lists, so the report
    # it builds still validates. Only the comparison against untouched originals
    # catches it, which needs the chair to have been handed deep copies.
    def mutating_chair(variant_id, initial, after):
        for findings in (initial, after):
            evo2_finding = next(f for f in findings if f.agent == "evo2")
            evo2_finding.raw_values["placeholder"] = 0.0  # frozen model, mutable dict
            evo2_finding.sources.append("placeholder source")
        return chair.decide(variant_id, initial, after)

    keep = OkAgent("evo2", lambda own: own)
    with pytest.raises(ValueError, match="chair changed the findings"):
        orchestrator.review(placeholder_variant(), agents=stub_panel(evo2=keep), chair=mutating_chair)
