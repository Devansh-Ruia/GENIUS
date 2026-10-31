"""evo2 evidence agent. It sees only the Evo 2 log-likelihood delta tool.

Owner: evidence-models. Imports its own tool and nothing from another agent.
"""

from __future__ import annotations

from vus_panel.agents._common import finding_from_tool, rebut_unchanged
from vus_panel.schemas import AgentFinding, Variant
from vus_panel.tools import evo2 as evo2_tool

AGENT = "evo2"


def run(variant: Variant) -> AgentFinding:
    result = evo2_tool.log_likelihood_delta(variant)
    return finding_from_tool(AGENT, variant.variant_id, result)


def rebut(own: AgentFinding, others: list[AgentFinding]) -> AgentFinding:
    """Revise once after seeing the other three findings. Sees findings, never tools."""
    return rebut_unchanged(AGENT, own, others)
