"""conservation evidence agent. It sees only the conservation score tool.

Owner: evidence-data. Imports its own tool and nothing from another agent.

The score is phyloP 100-way vertebrate: how little a base has changed across
species. It is per position, so every alt at one position scores alike. On
the calibration split this separates well, which is why this agent can give
a confident verdict at the extremes and should not at the middle.
"""

from __future__ import annotations

from vus_panel.agents import _interpret
from vus_panel.agents._common import check_rebuttal_inputs
from vus_panel.schemas import AgentFinding, Variant, reject_answer_key
from vus_panel.tools import conservation as conservation_tool

AGENT = "conservation"
SECTION = "conservation"
QUESTION = (
    "Has evolution kept this exact base unchanged across species? A base held still for millions "
    "of years is more likely to matter; one that drifts freely is less likely to."
)


def run(variant: Variant) -> AgentFinding:
    reject_answer_key(variant, f"agents.{AGENT}.run")
    result = conservation_tool.conservation_score(variant)
    return _interpret.interpret(AGENT, variant.variant_id, result, SECTION, QUESTION)


def rebut(own: AgentFinding, others: list[AgentFinding]) -> AgentFinding:
    """Revise once after seeing the other three findings. Sees findings, never tools."""
    check_rebuttal_inputs(AGENT, own, others)
    return _interpret.rebut(AGENT, own, others)
