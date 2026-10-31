"""Single-agent baseline: one agent with all four tools and no rebuttal round.

Owner: orchestration-eval. Used to compare multi-agent against single-agent.
Interpretation is not written yet, so every report is 'unresolved'.
"""

from __future__ import annotations

from vus_panel.agents._common import ACMG_VUS_RULE, PATIENT_UNRESOLVED_TEXT
from vus_panel.schemas import AgentFinding, PanelReport, ToolResult, Variant
from vus_panel.tools import conservation, evo2, population, sae

AGENT = "single_agent"


def _run_tools(variant: Variant) -> list[ToolResult]:
    calls = (
        ("evo2", evo2.log_likelihood_delta),
        ("sae", sae.feature_activations),
        ("conservation", conservation.conservation_score),
        ("population", population.allele_frequency),
    )
    results = []
    for name, call in calls:
        try:
            results.append(call(variant))
        except Exception as exc:  # a broken tool becomes an error result, never a value
            results.append(ToolResult(tool=name, status="error", message=f"{name} tool raised {type(exc).__name__}: {exc}"))
    return results


def review(variant: Variant) -> PanelReport:
    blinded = variant.without_answer_key()
    variant_id = blinded.variant_id
    results = _run_tools(blinded)

    if any(r.status == "ok" for r in results):
        status = "unavailable"
        lead = "Tools returned data, but single-agent interpretation is not implemented, so no verdict is given."
    elif any(r.status == "error" for r in results):
        status = "error"
        lead = "No tool returned a usable result, and at least one failed."
    else:
        status = "unavailable"
        lead = "No tool returned a usable result."
    rationale = " ".join([lead, *(f"{r.tool}: {r.message}" for r in results)])
    sources = list(dict.fromkeys(s for r in results for s in r.sources))

    finding = AgentFinding(agent=AGENT, variant_id=variant_id, status=status, rationale=rationale, sources=sources)
    return PanelReport(
        variant_id=variant_id,
        call="unresolved",
        confidence=None,
        calibrated=False,
        driving_agents=[],
        conflicts=[],
        what_would_settle_it=(
            "The single-agent baseline has no usable evidence yet. "
            "A functional assay result for this exact variant would settle it."
        ),
        counselor_summary=(
            f"The single-agent baseline makes no call for {variant_id}. Treat it as a VUS. {ACMG_VUS_RULE}"
        ),
        patient_summary=PATIENT_UNRESOLVED_TEXT,
        findings_initial=[finding],
        # There is no rebuttal round for the baseline, so both lists hold the same finding.
        findings_after_rebuttal=[finding],
        mock=False,
    )
