"""Single-agent baseline: one agent with all four tools and no rebuttal round.

Owner: orchestration-eval. Used to compare multi-agent against single-agent.
One LLM call reads every ok tool result and returns both the agent's finding
and the report's call, checked against the same rules as the chair's. Agents
may not import each other, so this module does not use the chair.
"""

from __future__ import annotations

import math
from typing import Any

from vus_panel.agents import _interpret
from vus_panel.agents._common import ACMG_VUS_RULE, PATIENT_UNRESOLVED_TEXT
from vus_panel.schemas import AgentFinding, AnswerKeyError, PanelReport, ToolResult, Variant
from vus_panel.tools import conservation, evo2, population, sae

AGENT = "single_agent"
CALLS: tuple[str, ...] = ("likely_damaging", "likely_tolerated", "unresolved")
# The only verdict each resolved call may follow; 'uncertain' allows only 'unresolved'.
CALL_FOR_VERDICT = {"damaging": "likely_damaging", "tolerated": "likely_tolerated"}
SETTLING_TEST = "A functional assay result for this exact variant would settle it."


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
        except AnswerKeyError:  # a leaked answer key fails the review, never becomes an error result
            raise
        except Exception as exc:  # a broken tool becomes an error result, never a value
            results.append(ToolResult(tool=name, status="error", message=f"{name} tool raised {type(exc).__name__}: {exc}"))
    return results


def _unresolved(variant_id: str, finding: AgentFinding, settle: str, counselor: str) -> PanelReport:
    return PanelReport(
        variant_id=variant_id,
        call="unresolved",
        confidence=None,
        calibrated=False,
        driving_agents=[],
        conflicts=[],
        what_would_settle_it=settle,
        counselor_summary=counselor,
        patient_summary=PATIENT_UNRESOLVED_TEXT,
        findings_initial=[finding],
        # There is no rebuttal round for the baseline, so both lists hold the same finding.
        findings_after_rebuttal=[finding],
        mock=False,
    )


def _from_reply(variant_id: str, finding: AgentFinding, reply: dict[str, Any]) -> PanelReport | None:
    """The report the reply asks for, or None if any part breaks the rules the chair applies."""
    call = reply.get("call")
    settle = _interpret.reply_text(reply, "what_would_settle_it") or SETTLING_TEST
    counselor = _interpret.reply_text(reply, "counselor_summary")
    patient = _interpret.reply_text(reply, "patient_summary")
    if call not in CALLS or not counselor or not patient:
        return None
    if call == "unresolved":
        return _unresolved(variant_id, finding, settle, f"{counselor} {ACMG_VUS_RULE}")
    if CALL_FOR_VERDICT.get(finding.verdict) != call:
        return None  # a resolved call must agree with the agent's own verdict
    confidence = reply.get("confidence")
    # json.loads accepts NaN and Infinity, which would otherwise clamp to full confidence.
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not math.isfinite(confidence):
        return None
    return PanelReport(
        variant_id=variant_id,
        call=call,
        confidence=max(0.0, min(1.0, float(confidence))),
        calibrated=False,  # stated by a model, not fitted on the calibration split
        driving_agents=[AGENT],
        conflicts=[],
        what_would_settle_it=settle,
        counselor_summary=counselor,
        patient_summary=patient,
        findings_initial=[finding],
        findings_after_rebuttal=[finding],
        mock=False,
    )


def review(variant: Variant) -> PanelReport:
    blinded = variant.without_answer_key()
    variant_id = blinded.variant_id
    results = _run_tools(blinded)

    if any(r.status == "ok" for r in results):
        finding, reply = _interpret.interpret_combined(AGENT, variant_id, results)
        report = _from_reply(variant_id, finding, reply) if reply is not None else None
        if report is not None:
            return report
        reason = "the reply was not a usable call" if reply is not None else finding.rationale
        return _unresolved(
            variant_id, finding,
            settle=f"The single-agent baseline could not reach a call ({reason}). {SETTLING_TEST}",
            counselor=(f"The single-agent baseline makes no call for {variant_id}. Treat it as a VUS. "
                       f"{ACMG_VUS_RULE}"),
        )

    status = "error" if any(r.status == "error" for r in results) else "unavailable"
    lead = ("No tool returned a usable result, and at least one failed." if status == "error"
            else "No tool returned a usable result.")
    rationale = " ".join([lead, *(f"{r.tool}: {r.message}" for r in results)])
    sources = list(dict.fromkeys(s for r in results for s in r.sources))
    finding = AgentFinding(agent=AGENT, variant_id=variant_id, status=status, rationale=rationale, sources=sources)
    return _unresolved(
        variant_id, finding,
        settle=f"The single-agent baseline has no usable evidence yet. {SETTLING_TEST}",
        counselor=f"The single-agent baseline makes no call for {variant_id}. Treat it as a VUS. {ACMG_VUS_RULE}",
    )
