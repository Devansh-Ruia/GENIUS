"""sae evidence agent. It sees only the SAE feature activations tool.

Owner: evidence-models. Imports its own tool and nothing from another agent.
"""

from __future__ import annotations

from vus_panel.agents._common import finding_from_tool, rebut_unchanged
from vus_panel.schemas import AgentFinding, Variant
from vus_panel.tools import sae as sae_tool

AGENT = "sae"


def run(variant: Variant) -> AgentFinding:
    result = sae_tool.feature_activations(variant)
    return finding_from_tool(AGENT, variant.variant_id, result)


def rebut(own: AgentFinding, others: list[AgentFinding]) -> AgentFinding:
    """Revise once after seeing the other three findings. Sees findings, never tools."""
    return rebut_unchanged(AGENT, own, others)
