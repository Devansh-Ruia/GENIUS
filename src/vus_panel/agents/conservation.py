"""conservation evidence agent. It sees only the conservation score tool.

Owner: evidence-data. Imports its own tool and nothing from another agent.
"""

from __future__ import annotations

from vus_panel.agents._common import finding_from_tool, rebut_unchanged
from vus_panel.schemas import AgentFinding, Variant, reject_answer_key
from vus_panel.tools import conservation as conservation_tool

AGENT = "conservation"


def run(variant: Variant) -> AgentFinding:
    reject_answer_key(variant, f"agents.{AGENT}.run")
    result = conservation_tool.conservation_score(variant)
    return finding_from_tool(AGENT, variant.variant_id, result)


def rebut(own: AgentFinding, others: list[AgentFinding]) -> AgentFinding:
    """Revise once after seeing the other three findings. Sees findings, never tools."""
    return rebut_unchanged(AGENT, own, others)
