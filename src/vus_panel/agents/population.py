"""population evidence agent. It sees only the gnomAD allele-frequency tool.

Owner: evidence-data. Imports its own tool and nothing from another agent.

The evidence is asymmetric, and the agent is told so. A variant carried by
many people who were not selected for disease is unlikely to be strongly
damaging. A variant nobody has been seen carrying is barely informative,
because almost every possible variant is absent from gnomAD.
"""

from __future__ import annotations

from vus_panel.agents import _interpret
from vus_panel.agents._common import check_rebuttal_inputs
from vus_panel.schemas import AgentFinding, Variant, reject_answer_key
from vus_panel.tools import population as population_tool

AGENT = "population"
SECTION = "population"
QUESTION = (
    "How common is this variant among people not selected for disease? Being carried at an "
    "appreciable frequency argues against it being strongly damaging. Being absent argues for "
    "very little, since most possible variants have never been seen in anyone; and a site gnomAD "
    "could not call tells you nothing at all."
)


def run(variant: Variant) -> AgentFinding:
    reject_answer_key(variant, f"agents.{AGENT}.run")
    result = population_tool.allele_frequency(variant)
    return _interpret.interpret(AGENT, variant.variant_id, result, SECTION, QUESTION)


def rebut(own: AgentFinding, others: list[AgentFinding]) -> AgentFinding:
    """Revise once after seeing the other three findings. Sees findings, never tools."""
    check_rebuttal_inputs(AGENT, own, others)
    return _interpret.rebut(AGENT, own, others)
