"""evo2 evidence agent. It sees only the Evo 2 log-likelihood delta tool.

Owner: evidence-models. Imports its own tool and nothing from another agent.
"""

from __future__ import annotations

from vus_panel.agents import _interpret
from vus_panel.agents._common import check_rebuttal_inputs
from vus_panel.schemas import AgentFinding, Variant, reject_answer_key
from vus_panel.tools import evo2 as evo2_tool

AGENT = "evo2"
SECTION = "evo2"
QUESTION = (
    "Evo 2 is a genome language model. delta is how much this one-base change shifts its summed "
    "log-likelihood of the surrounding 8,192 bases (negative means the change looks less like "
    "natural sequence); delta_mean is the same per position. calibration_percentile, when present, "
    "places this delta among every calibration-split variant, with lower meaning more negative. "
    "Does the model find this change surprising enough to suggest it disrupts the gene?"
)


def run(variant: Variant) -> AgentFinding:
    reject_answer_key(variant, f"agents.{AGENT}.run")
    result = evo2_tool.log_likelihood_delta(variant)
    return _interpret.interpret(AGENT, variant.variant_id, result, SECTION, QUESTION)


def rebut(own: AgentFinding, others: list[AgentFinding]) -> AgentFinding:
    """Revise once after seeing the other three findings. Sees findings, never tools."""
    check_rebuttal_inputs(AGENT, own, others)
    return _interpret.rebut(AGENT, own, others)
