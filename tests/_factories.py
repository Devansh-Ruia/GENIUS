"""Builders for test records.

Everything here uses abstract IDs or placeholder coordinates (chromosome 17,
positions in the low hundreds or below) and placeholder values. None of it
is real evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vus_panel.schemas import EVIDENCE_AGENTS, AgentFinding, PanelReport, Variant

VID = "variant-A"
FINDLAY_HEADER = ["func.class", "chromosome", "unused column", "position (hg19)", "reference", "alt", "function.score.mean"]


def placeholder_variant(**overrides: Any) -> Variant:
    fields = {"chrom": "17", "pos": 100, "ref": "A", "alt": "G", "genome_build": "GRCh37"}
    fields.update(overrides)
    return Variant(**fields)


def finding(agent: str = "evo2", status: str = "unavailable", verdict: str | None = None, **overrides: Any) -> AgentFinding:
    fields: dict[str, Any] = {
        "agent": agent,
        "variant_id": VID,
        "status": status,
        "verdict": verdict,
        "rationale": "placeholder rationale",
    }
    fields.update(overrides)
    return AgentFinding(**fields)


def panel_findings(variant_id: str = VID, ok: dict[str, str] | None = None) -> list[AgentFinding]:
    """Four findings. Agents named in `ok` get status ok with that verdict."""
    ok = ok or {}
    return [
        finding(agent, "ok", ok[agent], variant_id=variant_id) if agent in ok else finding(agent, variant_id=variant_id)
        for agent in EVIDENCE_AGENTS
    ]


def report_fields(**overrides: Any) -> dict[str, Any]:
    variant_id = overrides.get("variant_id", VID)
    findings = panel_findings(variant_id)
    fields: dict[str, Any] = {
        "variant_id": variant_id,
        "call": "unresolved",
        "confidence": None,
        "calibrated": False,
        "driving_agents": [],
        "conflicts": [],
        "what_would_settle_it": "placeholder",
        "counselor_summary": "placeholder",
        "patient_summary": "placeholder",
        "findings_initial": findings,
        "findings_after_rebuttal": findings,
        "mock": False,
    }
    fields.update(overrides)
    return fields


def report(**overrides: Any) -> PanelReport:
    return PanelReport(**report_fields(**overrides))


def resolved_report(variant_id: str, call: str, confidence: float, **overrides: Any) -> PanelReport:
    verdict = "damaging" if call == "likely_damaging" else "tolerated"
    findings = panel_findings(variant_id, ok={"evo2": verdict})
    return report(
        variant_id=variant_id,
        call=call,
        confidence=confidence,
        calibrated=True,
        driving_agents=["evo2"],
        findings_initial=findings,
        findings_after_rebuttal=findings,
        **overrides,
    )


def write_findlay_xlsx(path: Path, rows: list[list[Any]], header: list[str] | None = None) -> Path:
    """Write a Findlay-shaped workbook: two preamble rows, the header on row 3, then rows."""
    from openpyxl import Workbook

    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["placeholder preamble"])
    sheet.append([None])
    sheet.append(header or FINDLAY_HEADER)
    for row in rows:
        sheet.append(row)
    workbook.save(path)
    return path


def findlay_row(func_class: str, pos: int, ref: str = "A", alt: str = "G", chrom: Any = 17) -> list[Any]:
    """A row in FINDLAY_HEADER order, with a placeholder score of 0.0."""
    return [func_class, chrom, None, pos, ref, alt, 0.0]
