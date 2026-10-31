from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from _factories import VID, finding, placeholder_variant, report, report_fields, resolved_report
from vus_panel.schemas import AgentFinding, PanelReport, RunRecord, ToolResult, Variant, parse_variant_id


# Variant

def test_variant_accepts_a_valid_snv():
    variant = placeholder_variant()
    assert variant.variant_id == "GRCh37-17-100-A-G"


@pytest.mark.parametrize("chrom, expected", [("17", "17"), ("chr17", "17"), (17, "17"), ("chrX", "X"), ("chrM", "MT")])
def test_variant_normalizes_chrom(chrom, expected):
    assert placeholder_variant(chrom=chrom).chrom == expected


@pytest.mark.parametrize("chrom", ["23", "chr0", "", "17p", True])
def test_variant_rejects_unknown_chrom(chrom):
    with pytest.raises(ValidationError):
        placeholder_variant(chrom=chrom)


@pytest.mark.parametrize("pos", [0, -1])
def test_variant_rejects_non_positive_position(pos):
    with pytest.raises(ValidationError):
        placeholder_variant(pos=pos)


@pytest.mark.parametrize("base", ["a", "N", "AC", "", "U"])
@pytest.mark.parametrize("field", ["ref", "alt"])
def test_variant_rejects_non_acgt_bases(field, base):
    with pytest.raises(ValidationError):
        placeholder_variant(**{field: base})


def test_variant_rejects_ref_equal_to_alt():
    with pytest.raises(ValidationError, match="differ"):
        placeholder_variant(ref="C", alt="C")


def test_variant_requires_genome_build():
    with pytest.raises(ValidationError):
        Variant(chrom="17", pos=100, ref="A", alt="G")


@pytest.mark.parametrize("build", ["hg19", "GRCh36", "grch37"])
def test_variant_rejects_other_builds(build):
    with pytest.raises(ValidationError):
        placeholder_variant(genome_build=build)


def test_variant_forbids_extra_fields():
    with pytest.raises(ValidationError):
        placeholder_variant(clinvar_class="placeholder")


def test_variant_rejects_unknown_findlay_class():
    with pytest.raises(ValidationError):
        placeholder_variant(findlay_function_class="BAD")


def test_variant_is_frozen():
    variant = placeholder_variant()
    with pytest.raises(ValidationError):
        variant.pos = 101


def test_without_answer_key_clears_findlay_fields():
    variant = placeholder_variant(findlay_function_class="LOF", findlay_score=0.0, hgvs_c="c.placeholder")
    blinded = variant.without_answer_key()
    assert blinded.findlay_function_class is None
    assert blinded.findlay_score is None
    assert blinded.hgvs_c == "c.placeholder"
    assert blinded.variant_id == variant.variant_id


def test_parse_variant_id_round_trips():
    variant = placeholder_variant(chrom="chrX", genome_build="GRCh38")
    assert parse_variant_id(variant.variant_id) == variant


@pytest.mark.parametrize("bad", ["variant-A", "GRCh37-17-100-A", "hg19-17-100-A-G", "GRCh37-17-100-A-A"])
def test_parse_variant_id_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_variant_id(bad)


# ToolResult

def test_tool_result_rejects_values_unless_ok():
    with pytest.raises(ValidationError, match="values must be empty"):
        ToolResult(tool="evo2", status="unavailable", values={"placeholder": 0.0}, message="m")
    assert ToolResult(tool="evo2", status="ok", values={"placeholder": 0.0}, message="m").values


# AgentFinding

@pytest.mark.parametrize("status", ["unavailable", "error"])
def test_finding_rejects_verdict_unless_ok(status):
    with pytest.raises(ValidationError, match="verdict must be null"):
        finding(status=status, verdict="damaging")


@pytest.mark.parametrize("status", ["unavailable", "error"])
def test_finding_rejects_raw_values_unless_ok(status):
    with pytest.raises(ValidationError, match="raw_values must be empty"):
        finding(status=status, raw_values={"placeholder": 0.0})


def test_finding_ok_needs_a_verdict():
    with pytest.raises(ValidationError, match="needs a verdict"):
        finding(status="ok")


def test_finding_ok_allows_verdict_and_values():
    ok = finding(status="ok", verdict="uncertain", raw_values={"placeholder": 0.0})
    assert ok.verdict == "uncertain"


def test_finding_rejects_unknown_agent_and_empty_rationale():
    with pytest.raises(ValidationError):
        finding(agent="clinvar")
    with pytest.raises(ValidationError):
        finding(rationale="")


# PanelReport

def test_report_defaults_validate():
    assert report().call == "unresolved"


@pytest.mark.parametrize("confidence", [1.5, -0.1, math.nan, math.inf])
def test_report_rejects_confidence_outside_unit_interval(confidence):
    with pytest.raises(ValidationError):
        resolved_report(VID, "likely_damaging", confidence)


def test_resolved_call_needs_a_confidence():
    findings = [finding("evo2", "ok", "damaging"), finding("sae"), finding("conservation"), finding("population")]
    with pytest.raises(ValidationError, match="needs a confidence"):
        report(call="likely_damaging", driving_agents=["evo2"], findings_initial=findings, findings_after_rebuttal=findings)


def test_unresolved_report_has_no_confidence_and_is_not_calibrated():
    with pytest.raises(ValidationError, match="unresolved"):
        report(confidence=0.5)
    with pytest.raises(ValidationError, match="unresolved"):
        report(confidence=0.5, calibrated=True)
    assert report().confidence is None


def test_report_finding_without_evidence_cannot_gain_a_verdict_in_rebuttal():
    with pytest.raises(ValidationError, match="no evidence before rebuttal"):
        report(
            findings_initial=[finding("evo2")],
            findings_after_rebuttal=[finding("evo2", "ok", "damaging", revision_reason="placeholder")],
        )


def test_report_rejects_calibrated_without_confidence():
    with pytest.raises(ValidationError, match="calibrated"):
        report(calibrated=True)


def test_report_requires_mock_flag():
    fields = report_fields()
    del fields["mock"]
    with pytest.raises(ValidationError):
        PanelReport(**fields)


def test_report_rejects_findings_for_another_variant():
    other = [finding(agent, variant_id="variant-B") for agent in ("evo2", "sae", "conservation", "population")]
    with pytest.raises(ValidationError, match="variant-B"):
        report(findings_initial=other, findings_after_rebuttal=other)


def test_report_rejects_driving_agent_that_is_not_ok():
    with pytest.raises(ValidationError, match="driving_agents"):
        report(driving_agents=["evo2"])


def test_report_call_needs_a_driving_agent():
    findings = [finding("evo2", "ok", "damaging"), finding("sae"), finding("conservation"), finding("population")]
    with pytest.raises(ValidationError, match="driving agent"):
        report(call="likely_damaging", confidence=0.5, findings_initial=findings, findings_after_rebuttal=findings)
    ok = report(
        call="likely_damaging",
        confidence=0.5,
        driving_agents=["evo2"],
        findings_initial=findings,
        findings_after_rebuttal=findings,
    )
    assert ok.driving_agents == ["evo2"]


def test_report_rejects_duplicate_agents():
    findings = [finding("evo2"), finding("evo2")]
    with pytest.raises(ValidationError, match="more than once"):
        report(findings_initial=findings, findings_after_rebuttal=findings)


def test_report_rejects_mismatched_agent_sets():
    with pytest.raises(ValidationError, match="same agents"):
        report(findings_initial=[finding("evo2")], findings_after_rebuttal=[finding("sae")])


def test_report_revision_needs_a_reason():
    before = [finding("evo2", "ok", "damaging")]
    after_no_reason = [finding("evo2", "ok", "tolerated")]
    with pytest.raises(ValidationError, match="revision_reason"):
        report(findings_initial=before, findings_after_rebuttal=after_no_reason)
    after = [finding("evo2", "ok", "tolerated", revision_reason="placeholder reason")]
    assert report(findings_initial=before, findings_after_rebuttal=after).findings_after_rebuttal[0].verdict == "tolerated"


def test_report_initial_findings_carry_no_revision_reason():
    findings = [finding("evo2", revision_reason="placeholder")]
    with pytest.raises(ValidationError, match="initial finding"):
        report(findings_initial=findings, findings_after_rebuttal=findings)


# RunRecord

def _run_record(**overrides):
    fields = {
        "run_id": "panel-test",
        "git_commit": "abc1234",
        "config": {"system": "panel"},
        "seed": 0,
        "started_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "variant_ids": [VID],
        "outputs_path": "runs/panel-test",
    }
    fields.update(overrides)
    return RunRecord(**fields)


def test_run_record_accepts_valid_fields():
    assert _run_record().git_commit == "abc1234"


def test_run_record_requires_timezone_aware_start():
    with pytest.raises(ValidationError):
        _run_record(started_at=datetime(2026, 1, 1))


@pytest.mark.parametrize("commit", ["xyz1234", "abc12", "ABC1234", "a" * 41])
def test_run_record_rejects_bad_commit(commit):
    with pytest.raises(ValidationError):
        _run_record(git_commit=commit)


@pytest.mark.parametrize("run_id", ["../escape", "", ".hidden", "a/b"])
def test_run_record_rejects_unsafe_run_id(run_id):
    with pytest.raises(ValidationError):
        _run_record(run_id=run_id)


def test_agent_finding_is_frozen():
    item = finding()
    with pytest.raises(ValidationError):
        item.status = "ok"
    assert isinstance(item, AgentFinding)


@pytest.mark.parametrize("change", [{"raw_values": {"placeholder": 0.0}}, {"sources": ["placeholder"]}], ids=["raw_values", "sources"])
def test_report_ok_finding_keeps_its_evidence_through_rebuttal(change):
    before = [finding("evo2", "ok", "damaging")]
    after = [finding("evo2", "ok", "damaging", **change)]
    with pytest.raises(ValidationError, match="raw_values or sources"):
        report(findings_initial=before, findings_after_rebuttal=after)


def test_report_withdrawn_ok_finding_needs_a_reason():
    before = [finding("evo2", "ok", "damaging")]
    with pytest.raises(ValidationError, match="withdrew"):
        report(findings_initial=before, findings_after_rebuttal=[finding("evo2", "error")])
    kept = report(findings_initial=before, findings_after_rebuttal=[finding("evo2", "error", revision_reason="placeholder")])
    assert kept.findings_after_rebuttal[0].status == "error"
