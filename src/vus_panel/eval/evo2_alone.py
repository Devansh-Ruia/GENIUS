"""Evo 2-alone baseline on the test split, scored through run_eval.

Owner: orchestration-eval.

Usage: python -m vus_panel.eval.evo2_alone [--run-id ID]

Each test ID goes through the evo2 tool, which reads stored score_split results
under EVO2_SCORES_DIR (or the live backend at EVO2_BACKEND_URL). No label is
read here; run_eval joins the labels. The score is minus the ALT-minus-REF
delta, as in the NVIDIA Evo 2 BRCA1 notebook.

run_eval needs a probability, so the score becomes p = (mid-rank - 0.5) / n over
the test variants with a delta. That keeps order and ties exactly, so run_eval's
AUROC is the AUROC of minus delta. p is a rank, not a probability: the reports
say calibrated: false, and run_eval's reliability bins and ECE mean nothing for
this system.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

from vus_panel.eval._baseline import baseline_report, evaluate, evo2_provenance, split_ids, tool_finding
from vus_panel.eval.metrics import _average_ranks
from vus_panel.schemas import PanelReport, parse_variant_id
from vus_panel.tools import evo2

SYSTEM = "evo2_alone"
RATIONALE = ("Evo 2-alone baseline. The score is minus the ALT-minus-REF summed log-likelihood delta, ranked "
             "across the test split. No LLM and no fitted threshold.")
SUMMARY = "Evo 2-alone baseline output, made for evaluation only. It is not a panel review."


def rank_probabilities(scores: Sequence[float]) -> np.ndarray:
    """p = (mid-rank - 0.5) / n: same order and ties as the scores, all strictly inside (0, 1)."""
    s = np.asarray(scores, dtype=float)
    if s.ndim != 1 or s.size == 0:
        raise ValueError("need a non-empty one-dimensional list of scores")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    return (_average_ranks(s) - 0.5) / s.size


def build_reports(ids: Sequence[str]) -> tuple[list[PanelReport], dict[str, Any]]:
    """One report per ID. A variant with no delta is unresolved. Reads no labels."""
    results = [evo2.log_likelihood_delta(parse_variant_id(vid)) for vid in ids]
    deltas = [result.values["delta"] for result in results if result.status == "ok"]
    if not deltas:
        raise ValueError("no test variant has an Evo 2 delta; set EVO2_SCORES_DIR or EVO2_BACKEND_URL")
    probabilities = iter(rank_probabilities([-delta for delta in deltas]))
    reports = []
    for vid, result in zip(ids, results):
        p = float(next(probabilities)) if result.status == "ok" else None
        reports.append(baseline_report(vid, p, [tool_finding("evo2", vid, result, RATIONALE)],
                                       calibrated=False, summary=SUMMARY))
    details = {
        "system": SYSTEM,
        "score": "minus the evo2 delta (ALT minus REF, summed natural-log likelihood)",
        "probability_map": "p = (mid-rank - 0.5) / n over the test variants with a delta; order and ties kept",
        "calibrated": False,
        "reliability_meaningful": False,
        "n_variants": len(ids),
        "n_scored": len(deltas),
        "tool_status": dict(Counter(result.status for result in results)),
        **evo2_provenance(results),
    }
    return reports, details


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score the Evo 2-alone baseline on the test split through run_eval.")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)
    try:
        reports, details = build_reports(split_ids("test"))
        record = evaluate(SYSTEM, reports, details, run_id=args.run_id)
    except (ValueError, OSError, RuntimeError) as exc:  # ValueError covers EvalRefused and AnswerKeyError
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {record.outputs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
