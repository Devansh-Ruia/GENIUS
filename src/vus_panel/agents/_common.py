"""Helpers shared by the agents. Imports schemas only, never a tool."""

from __future__ import annotations

from vus_panel.schemas import AgentFinding, ToolResult

ACMG_VUS_RULE = (
    "ACMG guidance is that a variant of uncertain significance should not be used in "
    "clinical decision making."
)

PATIENT_UNRESOLVED_TEXT = (
    "This change in the BRCA1 gene is not understood yet. It does not show a higher or a lower "
    "cancer risk, and care decisions should not be based on it. A genetic counselor can explain "
    "what this means and will share any update."
)


def finding_from_tool(agent: str, variant_id: str, result: ToolResult) -> AgentFinding:
    """Turn a tool result into a finding without inventing a verdict.

    A tool that isn't ok passes through as an unavailable or error finding.
    A tool that is ok still yields 'unavailable', because interpretation is
    not written yet.
    """
    if result.status != "ok":
        return AgentFinding(
            agent=agent,
            variant_id=variant_id,
            status=result.status,
            rationale=result.message,
            sources=list(result.sources),
        )
    return AgentFinding(
        agent=agent,
        variant_id=variant_id,
        status="unavailable",
        rationale=f"The {agent} tool returned data, but interpretation is not implemented, so no verdict is given.",
        sources=list(result.sources),
    )


def check_rebuttal_inputs(agent: str, own: AgentFinding, others: list[AgentFinding]) -> None:
    if own.agent != agent:
        raise ValueError(f"{agent}.rebut was handed a finding from {own.agent}")
    for other in others:
        if not isinstance(other, AgentFinding):
            raise TypeError("rebut takes findings only")
        if other.agent == agent:
            raise ValueError(f"{agent}.rebut must see the other agents' findings, not its own")
        if other.variant_id != own.variant_id:
            raise ValueError("rebut was handed findings for a different variant")


def rebut_unchanged(agent: str, own: AgentFinding, others: list[AgentFinding]) -> AgentFinding:
    """Rebuttal placeholder. A non-ok finding always comes back unchanged.

    An ok finding also comes back unchanged until the LLM revision step is
    written. A revision must set revision_reason.
    """
    check_rebuttal_inputs(agent, own, others)
    return own.model_copy()
