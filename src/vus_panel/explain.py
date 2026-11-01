"""Answers questions about the panel's inputs, outputs and method.

Owner: frontend. This is a reading aid, not a fifth agent. It explains what a
field means, where a number came from and what the panel concluded. It never
re-interprets a variant, never adds evidence, and never gives clinical advice.

Everything it is allowed to assert about the data is in GROUNDING below. If a
question falls outside that and outside the report it was handed, the honest
answer is that it does not know.
"""

from __future__ import annotations

import json
from typing import Any

from vus_panel.llm import LLMConfigError, get_client, get_model
from vus_panel.schemas import PanelReport

MAX_TOKENS = 900
MAX_QUESTION_CHARS = 1000


class ExplainError(RuntimeError):
    """The question could not be answered."""


GROUNDING = """
WHAT GENIUS IS
A panel of four agents reviews one single-letter DNA change in BRCA1. Each
agent sees one kind of evidence and nothing else. They report independently,
then each sees the other three conclusions (never their underlying data) and
may revise once with a written reason. A chair then reads the findings only,
never any tool, and returns one call.

BRCA1 repairs broken DNA. A change that stops it working raises the risk of
breast and ovarian cancer. Most changes found in testing are of uncertain
significance, which is the problem this panel addresses.

THE FOUR AGENTS AND THEIR DATA
- conservation: phyloP 100-way vertebrate scores from UCSC, genome build
  GRCh37 (also called hg19). One score per position, roughly -6.9 to 6.5 in
  the BRCA1 region. High means the base stayed unchanged across 100 species,
  which suggests it matters. The score is per position, not per substitution,
  so every possible change at one position shares a score.
- population: gnomAD v2.1.1, about 141,000 people, GRCh37. Allele count (AC)
  is how many copies were seen, allele number (AN) how many were looked at,
  and frequency (AF) is AC divided by AN. Three distinct outcomes: observed
  (in gnomAD with a frequency), not_observed (the region is cached but this
  change is not in it), and no_call (gnomAD lists the site but reported no
  called alleles, so frequency is unknown rather than low). Being common in
  people not selected for disease argues against serious harm. Being absent
  argues for very little, because most possible changes have never been seen
  in anyone.
- evo2: Evo 2 7B, a DNA language model, scoring how natural the sequence
  looks before and after the change. The value is a log-likelihood delta,
  alternate minus reference. Needs a GPU backend.
- sae: a Goodfire sparse autoencoder reading Evo 2's internal activations, to
  see which interpretable features change at the position. Same GPU
  dependency as evo2.

HOW A NUMBER BECOMES A VERDICT
The tool returns a raw number. The agent is then given reference points
measured on a held-out calibration split of 779 variants, and reads its
number against them. On conservation, the lowest calibration band is 3.8%
loss of function and the highest is 49%, against a base rate of 21.2%. On
population, variants seen in gnomAD were loss of function 12.3% of the time
versus 22.0% for those absent. The thresholds come from the data, not from
the model's intuition.

VERDICTS AND CALLS
An agent's verdict is damaging, tolerated or uncertain. The chair's call is
likely_damaging, likely_tolerated or unresolved. Unresolved is a real answer:
it means the evidence is thin or genuinely divided, and the report then names
the test that would settle it.

CONFIDENCE
The number the chair states is its own judgement. Nothing has checked whether
its 0.6 calls are right 60 percent of the time, so calibrated is false and
the figure must not be read as a probability. Calibrating it would mean
running the panel over the calibration split, comparing stated confidence to
the known answers, and fitting a mapping.

THE ANSWER KEY
Findlay et al. 2018 measured 3,893 BRCA1 single-letter changes in the lab by
saturation genome editing, classing each FUNC, INT or LOF. It is used only to
split calibration from test, to build the reference points, and to score the
evaluation. The orchestrator strips those fields before any agent sees a
variant, and a leak fails the whole review. ClinVar is deliberately not an
input, because it is built from the same kind of evidence being tested.

WHAT IS NOT BUILT YET
The GPU backend, so evo2 and sae usually return no evidence. Confidence
calibration. The accuracy comparison against a single agent, a statistical
baseline and Evo 2 alone. The treatment-reasoning step.

LIMITS
Single-letter substitutions only, because that is where the lab answer key
is. Insertions and deletions are out of scope. GRCh37 only, with no liftover,
so a GRCh38 position returns an error rather than a wrong answer. A language
model may recall well-known BRCA1 variants from its training even though
ClinVar is never supplied, and a source check cannot catch that.
"""

SYSTEM = f"""You explain a genomic variant-review tool called GENIUS to someone
reading its output. Many readers have no biology background; some are genetic
counselors or lab curators. Judge from the question which you are talking to,
and if you cannot tell, write for the non-specialist.

Answer in two parts, always in this order:
1. Plain language first: two to four sentences a non-specialist can follow,
   with no jargon, or jargon explained the moment it appears.
2. Then a short paragraph headed "In technical terms" with the precise
   version: the actual field names, data sources, numbers and caveats.

Keep the whole answer under about 200 words unless the question genuinely
needs more.

What you may assert is limited to the reference below and the report you are
shown. If a question falls outside both, say plainly that you do not know
rather than guessing. Never invent a number.

You are a reading aid, not a fifth agent. Do not give your own opinion on
whether a variant is damaging, do not re-interpret the evidence, and do not
give clinical or treatment advice. If asked for any of those, say that the
panel's call and the counselor are where that belongs, and explain what the
panel did conclude and why.

Be straight about what the system cannot do yet. A reader who asks why two
agents returned nothing, or why confidence is not calibrated, deserves the
real answer.

REFERENCE
{GROUNDING}"""


def _report_context(report: PanelReport | None) -> str:
    if report is None:
        return "No report is open. Answer about the method and the data in general."
    findings = []
    for finding in report.findings_after_rebuttal:
        entry: dict[str, Any] = {"agent": finding.agent, "status": finding.status}
        if finding.status == "ok":
            entry.update(verdict=finding.verdict, values=finding.raw_values, reasoning=finding.rationale)
        else:
            entry["why_not"] = finding.rationale
        if finding.revision_reason:
            entry["changed_because"] = finding.revision_reason
        findings.append(entry)
    context = {
        "variant_id": report.variant_id,
        "call": report.call,
        "confidence": report.confidence,
        "calibrated": report.calibrated,
        "driving_agents": report.driving_agents,
        "conflicts": report.conflicts,
        "what_would_settle_it": report.what_would_settle_it,
        "counselor_summary": report.counselor_summary,
        "findings_after_rebuttal": findings,
        "is_sample_data": report.mock,
    }
    header = "The reader has this report open:"
    if report.mock:
        header += " it is sample data for interface work, so say so if the question is about its content."
    return f"{header}\n{json.dumps(context, indent=2, default=str)}"


def answer(question: str, report: PanelReport | None = None) -> str:
    """Answer one question about the panel, grounded in the reference and the open report."""
    question = (question or "").strip()
    if not question:
        raise ExplainError("Ask a question about the data, the method or the report on screen.")
    if len(question) > MAX_QUESTION_CHARS:
        raise ExplainError(f"That question is longer than {MAX_QUESTION_CHARS} characters. Try a shorter one.")

    prompt = f"{_report_context(report)}\n\nTheir question: {question}"
    try:
        client = get_client()
        response = client.messages.create(
            model=get_model(),
            max_tokens=MAX_TOKENS,
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
    except LLMConfigError as exc:
        raise ExplainError(f"The explainer is not configured: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - the page stays usable without the explainer
        raise ExplainError(f"The explainer could not be reached ({type(exc).__name__}).") from exc

    text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text").strip()
    if not text:
        raise ExplainError("The explainer returned nothing. Try rephrasing the question.")
    return text
