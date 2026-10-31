"""Runs the panel: four evidence agents, one rebuttal round, then the chair.

Owner: orchestration-eval.
- The answer key (Findlay class and score) is removed before any agent sees the variant.
- Each rebuttal sees only the other three findings.
- The chair gets the variant ID and two lists of findings, nothing else, and
  must hand both lists back unchanged in its report.
- An exception in any agent, or a finding that breaks the rebuttal rules,
  becomes a status 'error' finding for that agent.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol

from vus_panel.agents import chair as chair_agent
from vus_panel.agents import conservation, evo2, population, sae
from vus_panel.schemas import EVIDENCE_AGENTS, AgentFinding, PanelReport, Variant


class EvidenceAgent(Protocol):
    def run(self, variant: Variant) -> AgentFinding: ...

    def rebut(self, own: AgentFinding, others: list[AgentFinding]) -> AgentFinding: ...


ChairFn = Callable[[str, list[AgentFinding], list[AgentFinding]], PanelReport]

DEFAULT_AGENTS: dict[str, EvidenceAgent] = {
    "evo2": evo2,
    "sae": sae,
    "conservation": conservation,
    "population": population,
}


def _error_finding(agent: str, variant_id: str, stage: str, detail: str) -> AgentFinding:
    rationale = f"{agent} agent failed during {stage}: {detail}"
    # A rebuttal-stage error replaces the agent's finding, so it records why.
    revision_reason = f"set by the orchestrator: {detail}" if stage == "rebuttal" else None
    return AgentFinding(
        agent=agent,
        variant_id=variant_id,
        status="error",
        rationale=rationale,
        revision_reason=revision_reason,
    )


def _identity_problem(agent: str, variant_id: str, result: object) -> str | None:
    if type(result) is not AgentFinding:
        return f"returned {type(result).__name__}, not AgentFinding"
    if result.agent != agent or result.variant_id != variant_id:
        return "returned a finding for another agent or variant"
    return None


def _validate_run(agent: str, variant_id: str, result: object) -> AgentFinding:
    problem = _identity_problem(agent, variant_id, result)
    if problem is None and result.revision_reason is not None:
        problem = "an initial finding carried a revision_reason"
    return result if problem is None else _error_finding(agent, variant_id, "run", problem)


def _rebuttal_problem(own: AgentFinding, result: AgentFinding) -> str | None:
    has_reason = bool((result.revision_reason or "").strip())
    if own.status != "ok":
        return "rebuttal changed a finding that had no evidence" if result != own else None
    if result.status == "ok":
        if result.raw_values != own.raw_values or result.sources != own.sources:
            return "rebuttal changed raw_values or sources, but rebut has no tool access"
        if result.verdict != own.verdict and not has_reason:
            return "verdict changed in rebuttal without a revision_reason"
        return None
    if not has_reason:
        return "an ok finding was withdrawn in rebuttal without a revision_reason"
    return None


def _validate_rebuttal(agent: str, variant_id: str, own: AgentFinding, result: object) -> AgentFinding:
    problem = _identity_problem(agent, variant_id, result)
    if problem is None:
        problem = _rebuttal_problem(own, result)
    return result if problem is None else _error_finding(agent, variant_id, "rebuttal", problem)


def review(
    variant: Variant,
    *,
    agents: Mapping[str, EvidenceAgent] | None = None,
    chair: ChairFn | None = None,
) -> PanelReport:
    panel = dict(DEFAULT_AGENTS if agents is None else agents)
    unknown = [name for name in panel if name not in EVIDENCE_AGENTS]
    if unknown:
        raise ValueError(f"unknown evidence agents: {unknown}")
    decide = chair_agent.decide if chair is None else chair

    blinded = variant.without_answer_key()
    variant_id = blinded.variant_id

    initial: list[AgentFinding] = []
    for name, agent in panel.items():
        try:
            result = agent.run(blinded)
        except Exception as exc:
            initial.append(_error_finding(name, variant_id, "run", f"{type(exc).__name__}: {exc}"))
            continue
        initial.append(_validate_run(name, variant_id, result))

    after: list[AgentFinding] = []
    for index, (name, agent) in enumerate(panel.items()):
        own = initial[index]
        others = [f.model_copy(deep=True) for i, f in enumerate(initial) if i != index]
        try:
            result = agent.rebut(own.model_copy(deep=True), others)
        except Exception as exc:
            after.append(_error_finding(name, variant_id, "rebuttal", f"{type(exc).__name__}: {exc}"))
            continue
        after.append(_validate_rebuttal(name, variant_id, own, result))

    # The chair gets deep copies, because raw_values and sources stay mutable inside
    # frozen models. Its report is then compared against the untouched originals.
    report = decide(
        variant_id,
        [f.model_copy(deep=True) for f in initial],
        [f.model_copy(deep=True) for f in after],
    )
    if type(report) is not PanelReport:
        raise TypeError(f"chair returned {type(report).__name__}, not PanelReport")
    if report.variant_id != variant_id:
        raise ValueError("chair returned a report for a different variant")
    if report.findings_initial != initial or report.findings_after_rebuttal != after:
        raise ValueError("chair changed the findings it was given; it must pass both lists through unchanged")
    return report.model_copy(update={"mock": False})
