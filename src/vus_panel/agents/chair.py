"""Chair agent. It sees findings only, never tools or the variant record.

Owner: orchestration-eval.

The chair reads the findings after rebuttal and returns one call. It holds
no tool, so it can only weigh what the evidence agents reported. Three rules
shape it:

- No ok findings means unresolved. There is nothing to weigh.
- Real disagreement between an ok 'damaging' and an ok 'tolerated' finding is
  reported as a conflict and, on its own, is a reason to stay unresolved
  rather than to pick a side.
- Confidence is a number the model states, not a fitted probability, so
  `calibrated` stays false until a calibration is fitted on the calibration
  split. The report says so.

If the LLM is unavailable or answers unusably, the chair falls back to
unresolved with the reason named. It never invents a call.
"""

from __future__ import annotations

import json

from vus_panel.agents._common import ACMG_VUS_RULE, PATIENT_UNRESOLVED_TEXT
from vus_panel.llm import LLMConfigError, get_client, get_model
from vus_panel.schemas import AgentFinding, PanelReport

SETTLING_TEST = "A functional assay result for this exact variant would settle it."
CALLS: tuple[str, ...] = ("likely_damaging", "likely_tolerated", "unresolved")
MAX_TOKENS = 1200

SYSTEM = (
    "You chair a panel reviewing one DNA variant. Four specialists each examined one kind of "
    "evidence. You see their conclusions and nothing else: no tools, no raw data beyond what they "
    "reported, and no variant databases. If you recognise the variant, set that aside.\n"
    "Weigh what they found and return one call.\n"
    "  likely_damaging  — the evidence points to the variant disrupting the gene's function\n"
    "  likely_tolerated — the evidence points to it being compatible with normal function\n"
    "  unresolved       — the evidence is thin, weak, or genuinely conflicting\n"
    "Unresolved is a real answer and often the right one. Two specialists agreeing on weak "
    "evidence is not a strong case. One specialist with strong evidence and three with none is "
    "not a strong case either. A direct conflict between well-supported findings is a reason to "
    "stay unresolved, not to pick the louder one.\n"
    "Name as driving agents only those whose findings actually carried your call.\n"
    "Your confidence is your own judgement, not a measured probability, and it will be reported "
    "as uncalibrated.\n"
    "Write for a genetic counselor in the counselor summary, and for the patient they will speak "
    "to in the patient summary: plain words, no jargon, no numbers, and no instruction to act.\n"
    "Reply with JSON only, no markdown:\n"
    '{"call": "likely_damaging" | "likely_tolerated" | "unresolved",\n'
    ' "confidence": 0.0 to 1.0,\n'
    ' "driving_agents": ["evo2", ...],\n'
    ' "conflicts": ["..."],\n'
    ' "what_would_settle_it": "...",\n'
    ' "counselor_summary": "...",\n'
    ' "patient_summary": "..."}\n'
    "For an unresolved call set confidence to null and driving_agents to an empty list.\n"
    "conflicts holds only direct disagreements between two specialists, such as \"evo2 says "
    "damaging, conservation says tolerated\". Weak evidence, missing sources, and your own "
    "uncertainty are not conflicts: those belong in what_would_settle_it and the counselor "
    "summary. If no two specialists disagree, conflicts is an empty list."
)


def _verdict_conflicts(findings: list[AgentFinding]) -> list[str]:
    conflicts = []
    for i, first in enumerate(findings):
        for second in findings[i + 1 :]:
            if {first.verdict, second.verdict} == {"damaging", "tolerated"}:
                conflicts.append(f"{first.agent} says {first.verdict}, {second.agent} says {second.verdict}")
    return conflicts


def _findings_text(findings: list[AgentFinding]) -> str:
    lines = []
    for f in findings:
        if f.status != "ok":
            lines.append(f"- {f.agent}: no result ({f.status}). {f.rationale}")
            continue
        revised = f" Revised in rebuttal: {f.revision_reason}" if f.revision_reason else ""
        lines.append(
            f"- {f.agent}: {f.verdict}. {f.rationale}"
            f" Values: {json.dumps(f.raw_values, default=str)}.{revised}"
        )
    return "\n".join(lines)


def _unresolved(
    variant_id: str,
    findings_initial: list[AgentFinding],
    findings_after_rebuttal: list[AgentFinding],
    what_would_settle_it: str,
    counselor_summary: str,
    conflicts: list[str],
) -> PanelReport:
    return PanelReport(
        variant_id=variant_id,
        call="unresolved",
        confidence=None,
        calibrated=False,
        driving_agents=[],
        conflicts=conflicts,
        what_would_settle_it=what_would_settle_it,
        counselor_summary=counselor_summary,
        patient_summary=PATIENT_UNRESOLVED_TEXT,
        findings_initial=list(findings_initial),
        findings_after_rebuttal=list(findings_after_rebuttal),
        mock=False,
    )


def _ask(prompt: str) -> dict:
    client = get_client()
    response = client.messages.create(
        model=get_model(),
        max_tokens=MAX_TOKENS,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text").strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0]
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("the model replied with JSON that is not an object")
    return parsed


def decide(
    variant_id: str,
    findings_initial: list[AgentFinding],
    findings_after_rebuttal: list[AgentFinding],
) -> PanelReport:
    for finding in [*findings_initial, *findings_after_rebuttal]:
        if not isinstance(finding, AgentFinding):
            raise TypeError("the chair takes findings only")

    ok = [f for f in findings_after_rebuttal if f.status == "ok"]
    conflicts = _verdict_conflicts(ok)
    gaps = "; ".join(f"{f.agent} is {f.status}" for f in findings_after_rebuttal if f.status != "ok")

    if not ok:
        return _unresolved(
            variant_id, findings_initial, findings_after_rebuttal,
            what_would_settle_it=(
                f"No evidence source returned a result ({gaps or 'no findings'}). Bringing those "
                f"sources online would let the panel review it. {SETTLING_TEST}"
            ),
            counselor_summary=(
                f"No evidence source returned a usable result for {variant_id}, so the panel makes "
                f"no call. Treat it as a VUS. {ACMG_VUS_RULE}"
            ),
            conflicts=conflicts,
        )

    prompt = (
        f"Variant: {variant_id}\n\n"
        f"What the specialists found, after the rebuttal round:\n{_findings_text(findings_after_rebuttal)}\n\n"
        + (f"Direct disagreements: {'; '.join(conflicts)}\n\n" if conflicts else "")
        + (f"Evidence sources with no result: {gaps}\n\n" if gaps else "")
        + "Give your call."
    )
    try:
        parsed = _ask(prompt)
    except LLMConfigError as exc:
        reason = str(exc)
    except Exception as exc:  # noqa: BLE001 - a failed chair means no call, never a crash
        reason = f"{type(exc).__name__}: {exc}"
    else:
        report = _from_reply(parsed, variant_id, findings_initial, findings_after_rebuttal, ok, conflicts)
        if report is not None:
            return report
        reason = "the reply was not a usable call"

    return _unresolved(
        variant_id, findings_initial, findings_after_rebuttal,
        what_would_settle_it=(
            f"{len(ok)} of {len(findings_after_rebuttal)} evidence sources returned a result, but the "
            f"chair could not reach a call ({reason}). {SETTLING_TEST}"
        ),
        counselor_summary=(
            f"Evidence came back for {variant_id}, but the panel could not complete its review "
            f"({reason}), so it makes no call. Treat it as a VUS. {ACMG_VUS_RULE}"
        ),
        conflicts=conflicts,
    )


def _from_reply(
    parsed: dict,
    variant_id: str,
    findings_initial: list[AgentFinding],
    findings_after_rebuttal: list[AgentFinding],
    ok: list[AgentFinding],
    detected_conflicts: list[str],
) -> PanelReport | None:
    """Build a report from the model's reply, or None if it isn't usable.

    Every field is checked against the contract here rather than trusted,
    because a chair that returns a call it cannot support is worse than one
    that says it could not decide.
    """
    call = str(parsed.get("call", "")).strip()
    if call not in CALLS:
        return None

    settle = str(parsed.get("what_would_settle_it", "")).strip() or SETTLING_TEST
    counselor = str(parsed.get("counselor_summary", "")).strip()
    patient = str(parsed.get("patient_summary", "")).strip()
    if not counselor or not patient:
        return None

    stated = parsed.get("conflicts") or []
    conflicts = list(detected_conflicts)
    for item in stated:
        text = str(item).strip()
        if text and text not in conflicts:
            conflicts.append(text)

    if call == "unresolved":
        return _unresolved(variant_id, findings_initial, findings_after_rebuttal,
                           what_would_settle_it=settle,
                           counselor_summary=f"{counselor} {ACMG_VUS_RULE}".strip(),
                           conflicts=conflicts)

    ok_names = {f.agent for f in ok}
    driving = []
    for name in parsed.get("driving_agents") or []:
        name = str(name).strip()
        if name in ok_names and name not in driving:
            driving.append(name)
    if not driving:
        return None  # the contract needs at least one, and the chair must not invent one

    confidence = parsed.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        return None
    confidence = max(0.0, min(1.0, float(confidence)))

    return PanelReport(
        variant_id=variant_id,
        call=call,
        confidence=confidence,
        calibrated=False,  # stated by a model, not fitted on the calibration split
        driving_agents=driving,
        conflicts=conflicts,
        what_would_settle_it=settle,
        counselor_summary=counselor,
        patient_summary=patient,
        findings_initial=list(findings_initial),
        findings_after_rebuttal=list(findings_after_rebuttal),
        mock=False,
    )
