"""sae evidence agent. It sees only the SAE feature activations tool.

Owner: evidence-models. Imports its own tool and nothing from another agent.
"""

from __future__ import annotations

from vus_panel.agents import _interpret
from vus_panel.agents._common import check_rebuttal_inputs
from vus_panel.schemas import AgentFinding, Variant, reject_answer_key
from vus_panel.tools import sae as sae_tool

AGENT = "sae"
SECTION = "sae"
QUESTION = (
    "These are sparse autoencoder feature activations from Evo 2 at the variant position. Do the "
    "features that change suggest the variant disrupts a functional element of the gene?"
)


def run(variant: Variant) -> AgentFinding:
    reject_answer_key(variant, f"agents.{AGENT}.run")
    result = sae_tool.feature_activations(variant)
    return _interpret.interpret(AGENT, variant.variant_id, result, SECTION, QUESTION)


def rebut(own: AgentFinding, others: list[AgentFinding]) -> AgentFinding:
    """Revise once after seeing the other three findings. Sees findings, never tools."""
    check_rebuttal_inputs(AGENT, own, others)
    return _interpret.rebut(AGENT, own, others)
