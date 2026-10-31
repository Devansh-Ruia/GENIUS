"""Turning one tool's numbers into a verdict, with the LLM.

Shared by the evidence agents. Imports schemas, config and the LLM client,
never a tool and never another agent, so the isolation tests still hold.

Two rules this module exists to enforce:

1. The model is never asked to invent a threshold. It is handed reference
   points measured on the CALIBRATION SPLIT ONLY (data/cache/calibration_refs.json,
   written by scripts/calibration_refs.py) and asked to read its number
   against them. Without that file an agent can still answer, but it says
   plainly that it had no reference.
2. Anything that goes wrong — no API key, no model name, a malformed reply —
   yields an 'unavailable' finding with a message naming the cause. An agent
   never guesses a verdict, and never crashes the panel.
"""

from __future__ import annotations

import json
from typing import Any

from vus_panel import config
from vus_panel.llm import LLMConfigError, get_client, get_model
from vus_panel.schemas import AgentFinding, ToolResult

CALIBRATION_FILE = "calibration_refs.json"
VERDICTS: tuple[str, ...] = ("damaging", "tolerated", "uncertain")
MAX_TOKENS = 700

SYSTEM = (
    "You are one evidence specialist on a panel reviewing a single BRCA1 DNA variant.\n"
    "You see one kind of evidence and nothing else. Other specialists hold the rest, so do not "
    "speculate about evidence you were not given, and do not recall what you may know about this "
    "variant from elsewhere. ClinVar and other variant databases are not inputs here; if you "
    "recognise the variant, set that aside and reason only from the numbers below.\n"
    "Reference points come from a held-out calibration set. Read your number against them rather "
    "than against a threshold you invent.\n"
    "Weak evidence deserves the verdict 'uncertain'. Saying so is a useful answer, not a failure.\n"
    'Reply with JSON only: {"verdict": "damaging" | "tolerated" | "uncertain", "rationale": "..."}\n'
    "The rationale is two or three sentences, cites your actual number, and says how strong the "
    "evidence is. No preamble, no markdown."
)

REBUTTAL_SYSTEM = (
    "You are one evidence specialist on a BRCA1 variant panel. You have already given a verdict. "
    "You now see what the other specialists found. You see their conclusions only, never their "
    "tools or raw data, and your own evidence has not changed.\n"
    "Change your verdict only if their findings genuinely bear on how yours should be read. "
    "Agreement alone is not a reason to change, and neither is disagreement: your evidence is "
    "still your evidence. Holding firm is the common and correct outcome.\n"
    'Reply with JSON only: {"verdict": "damaging" | "tolerated" | "uncertain", "changed": true | false, '
    '"reason": "..."}\n'
    "If changed is false, reason may be an empty string. If changed is true, reason is one sentence "
    "naming what moved you."
)


def calibration_reference(section: str) -> dict[str, Any] | None:
    """Reference points for one evidence type, or None if they were never built."""
    path = config.cache_dir() / CALIBRATION_FILE
    if not path.is_file():
        return None
    try:
        refs = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    block = refs.get(section)
    if not isinstance(block, dict):
        return None
    return {"base_rate_lof": refs.get("base_rate_lof"), "n_calibration": refs.get("n_calibration"), **block}


def _reference_text(reference: dict[str, Any] | None) -> str:
    if reference is None:
        return (
            "No calibration reference points are available, so you have no measured sense of what "
            "counts as high or low here. Be correspondingly cautious, and say so."
        )
    lines = [
        f"Measured on {reference.get('n_calibration')} held-out calibration variants. "
        f"{_percent(reference.get('base_rate_lof'))} of them were loss of function, which is the base rate "
        "to compare against."
    ]
    for key, label in (("bins_low_to_high", "bins, low to high"),
                       ("bins_low_to_high_af_when_observed", "frequency bins, low to high")):
        for b in reference.get(key) or []:
            lo, hi = b.get("range", [None, None])
            lines.append(
                f"  {label} {b.get('bin')}: {lo} to {hi}, n={b.get('n')}, "
                f"loss of function {_percent(b.get('fraction_lof'))}"
            )
    for key, label in (("fraction_lof_when_observed", "when the variant is seen in gnomAD"),
                       ("fraction_lof_when_not_observed", "when it is absent from gnomAD")):
        if reference.get(key) is not None:
            lines.append(f"  {label}: loss of function {_percent(reference[key])}")
    if reference.get("note"):
        lines.append(f"  Note: {reference['note']}")
    return "\n".join(lines)


def _percent(value: Any) -> str:
    return f"{float(value) * 100:.1f}%" if isinstance(value, (int, float)) else "unknown"


def _ask(system: str, prompt: str) -> dict[str, Any]:
    client = get_client()
    response = client.messages.create(
        model=get_model(),
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text").strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0]
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("the model replied with JSON that is not an object")
    return parsed


def _unavailable(agent: str, variant_id: str, message: str, sources: list[str]) -> AgentFinding:
    return AgentFinding(agent=agent, variant_id=variant_id, status="unavailable",
                        rationale=message, sources=sources)


def interpret(agent: str, variant_id: str, result: ToolResult, section: str, question: str) -> AgentFinding:
    """Read one tool result against the calibration reference and return a finding."""
    sources = list(result.sources)
    if result.status != "ok":
        return AgentFinding(agent=agent, variant_id=variant_id, status=result.status,
                            rationale=result.message, sources=sources)

    reference = calibration_reference(section)
    prompt = (
        f"Variant: {variant_id}\n"
        f"Your question: {question}\n\n"
        f"What your tool returned:\n{result.message}\n"
        f"Values: {json.dumps(result.values, default=str)}\n\n"
        f"Calibration reference:\n{_reference_text(reference)}\n\n"
        "Give your verdict on whether this variant disrupts the gene's function, judging from your "
        "evidence alone."
    )
    try:
        parsed = _ask(SYSTEM, prompt)
    except LLMConfigError as exc:
        return _unavailable(agent, variant_id, f"{agent} could not interpret its evidence: {exc}", sources)
    except Exception as exc:  # noqa: BLE001 - one agent failing must not take down the panel
        return _unavailable(agent, variant_id,
                            f"{agent} reached the model but could not read a verdict from the reply "
                            f"({type(exc).__name__}: {exc}).", sources)

    verdict = str(parsed.get("verdict", "")).strip().lower()
    rationale = str(parsed.get("rationale", "")).strip()
    if verdict not in VERDICTS or not rationale:
        return _unavailable(agent, variant_id,
                            f"{agent} got a reply that was not a usable verdict and rationale, so no "
                            "verdict is recorded.", sources)
    if reference is None:
        rationale = f"{rationale} No calibration reference was available for this run."
    return AgentFinding(agent=agent, variant_id=variant_id, status="ok",
                        verdict=verdict, raw_values=dict(result.values),
                        rationale=rationale, sources=sources)


def rebut(agent: str, own: AgentFinding, others: list[AgentFinding]) -> AgentFinding:
    """Revise once after seeing the other findings, or come back unchanged."""
    if own.status != "ok":
        return own.model_copy()

    summary = "\n".join(
        f"- {o.agent}: {o.verdict or o.status} — {o.rationale}" for o in others
    ) or "- no other findings were available"
    prompt = (
        f"Variant: {own.variant_id}\n\n"
        f"Your verdict: {own.verdict}\n"
        f"Your reasoning: {own.rationale}\n"
        f"Your numbers: {json.dumps(own.raw_values, default=str)}\n\n"
        f"What the other specialists concluded:\n{summary}\n\n"
        "Keep or change your verdict."
    )
    try:
        parsed = _ask(REBUTTAL_SYSTEM, prompt)
    except Exception:  # noqa: BLE001 - a failed rebuttal leaves the first verdict standing
        return own.model_copy()

    verdict = str(parsed.get("verdict", "")).strip().lower()
    reason = str(parsed.get("reason", "")).strip()
    changed = bool(parsed.get("changed")) and verdict in VERDICTS and verdict != own.verdict
    if not changed or not reason:
        return own.model_copy()
    return own.model_copy(update={"verdict": verdict, "revision_reason": reason})
