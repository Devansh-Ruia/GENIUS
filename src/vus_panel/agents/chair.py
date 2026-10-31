"""Chair agent. It sees findings only, never tools or the variant record.

Owner: orchestration-eval. The reasoning step that turns ok findings into a
call is not written yet, so every report is 'unresolved' for now.
"""

from __future__ import annotations

from vus_panel.agents._common import ACMG_VUS_RULE, PATIENT_UNRESOLVED_TEXT
from vus_panel.schemas import AgentFinding, PanelReport

SETTLING_TEST = "A functional assay result for this exact variant would settle it."


def _verdict_conflicts(findings: list[AgentFinding]) -> list[str]:
    conflicts = []
    for i, first in enumerate(findings):
        for second in findings[i + 1 :]:
            if {first.verdict, second.verdict} == {"damaging", "tolerated"}:
                conflicts.append(f"{first.agent} says {first.verdict}, {second.agent} says {second.verdict}")
    return conflicts


def decide(
    variant_id: str,
    findings_initial: list[AgentFinding],
    findings_after_rebuttal: list[AgentFinding],
) -> PanelReport:
    for finding in [*findings_initial, *findings_after_rebuttal]:
        if not isinstance(finding, AgentFinding):
            raise TypeError("the chair takes findings only")

    ok = [f for f in findings_after_rebuttal if f.status == "ok"]
    gaps = "; ".join(f"{f.agent} is {f.status}" for f in findings_after_rebuttal if f.status != "ok")

    if not ok:
        what_would_settle_it = (
            f"No evidence source returned a result ({gaps or 'no findings'}). Bringing those sources "
            f"online would let the panel review it. {SETTLING_TEST}"
        )
        counselor_summary = (
            f"No evidence source returned a usable result for {variant_id}, so the panel makes no call. "
            f"Treat it as a VUS. {ACMG_VUS_RULE}"
        )
    else:
        what_would_settle_it = (
            f"{len(ok)} of {len(findings_after_rebuttal)} evidence sources returned a result, but chair "
            f"reasoning is not implemented yet. {SETTLING_TEST}"
        )
        counselor_summary = (
            f"Some evidence came back for {variant_id}, but chair reasoning is not implemented yet, so "
            f"the panel makes no call. Treat it as a VUS. {ACMG_VUS_RULE}"
        )

    return PanelReport(
        variant_id=variant_id,
        call="unresolved",
        confidence=None,
        calibrated=False,
        driving_agents=[],
        conflicts=_verdict_conflicts(ok),
        what_would_settle_it=what_would_settle_it,
        counselor_summary=counselor_summary,
        patient_summary=PATIENT_UNRESOLVED_TEXT,
        findings_initial=list(findings_initial),
        findings_after_rebuttal=list(findings_after_rebuttal),
        mock=False,
    )
