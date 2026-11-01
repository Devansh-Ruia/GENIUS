"""Logistic-regression baseline: fit on the calibration split, scored on the test split through run_eval.

Owner: orchestration-eval.

Usage: python -m vus_panel.eval.logistic_baseline [--run-id ID]

Features, each read through the repo's own tool, the same way for both splits:
    evo2_delta        evo2 tool: ALT-minus-REF summed log-likelihood delta
    phylop            conservation tool: phyloP 100-way score at the position
    gnomad_present    population tool: 1 if gnomAD observed the variant with AC > 0, else 0.
                      Absent, no-call and AC = 0 all count as not present.
    gnomad_log10_af   log10 of the gnomAD allele frequency when present, else 0

Labels come from the Findlay workbook for calibration-split IDs only: LOF is 1,
FUNC and INT are 0. Test labels are never loaded here; run_eval joins them.

The fit is plain maximum-likelihood logistic regression by Newton-IRLS on
features standardized with the calibration mean and sample SD (ddof=1). It
refuses rather than guesses: on separation, non-convergence, or any
calibration variant without every feature. The output is a probability fit
on the calibration split, so reports say calibrated: true. A test variant
missing a feature is unresolved.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vus_panel import config
from vus_panel.eval import run_eval
from vus_panel.eval._baseline import baseline_report, evaluate, evo2_provenance, split_ids, tool_finding
from vus_panel.eval.metrics import auroc
from vus_panel.eval.split import default_findlay_path, load_findlay, sha256_file
from vus_panel.schemas import AgentFinding, PanelReport, ToolResult, parse_variant_id
from vus_panel.tools import conservation, evo2, population

SYSTEM = "logistic_baseline"
FEATURES: tuple[str, ...] = ("evo2_delta", "phylop", "gnomad_present", "gnomad_log10_af")
TOOLS = (
    ("evo2", evo2.log_likelihood_delta),
    ("conservation", conservation.conservation_score),
    ("population", population.allele_frequency),
)
MAX_ITERATIONS = 100
TOLERANCE = 1e-10
# A standardized coefficient this large means the likelihood has no finite maximum (separation).
MAX_ABS_COEFFICIENT = 30.0
RATIONALE = "Input to the logistic baseline, fit on the calibration split. No per-tool verdict and no LLM."
SUMMARY = ("Logistic-regression baseline output (Evo 2 delta, phyloP, gnomAD presence and frequency, fit on the "
           "calibration split), made for evaluation only. It is not a panel review.")


class LogisticFitError(ValueError):
    """The calibration data can't give a finite, converged fit."""


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    ez = np.exp(z[~positive])
    out[~positive] = ez / (1.0 + ez)
    return out


@dataclass(frozen=True)
class LogisticModel:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray  # on standardized features
    intercept: float  # on standardized features
    iterations: int

    def predict(self, x: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        z = ((np.asarray(x, dtype=float) - self.mean) / self.scale) @ self.coef + self.intercept
        return _sigmoid(z)

    def raw_coefficients(self) -> tuple[np.ndarray, float]:
        """The same model on unstandardized features."""
        coef = self.coef / self.scale
        return coef, float(self.intercept - (coef * self.mean).sum())


def fit_logistic(x: Sequence[Sequence[float]] | np.ndarray, y: Sequence[int] | np.ndarray,
                 max_iterations: int = MAX_ITERATIONS, tolerance: float = TOLERANCE) -> LogisticModel:
    """Maximum-likelihood logistic regression by Newton-IRLS, with no penalty."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0] or x.shape[0] == 0:
        raise ValueError("need a 2-D feature array with one label per row")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0 or 1")
    if not np.isfinite(x).all():
        raise ValueError("features must be finite")
    if y.min() == y.max():
        raise LogisticFitError("the labels hold only one class")
    y = y.astype(float)
    mean = x.mean(axis=0)
    scale = x.std(axis=0, ddof=1)
    if not (np.isfinite(scale).all() and (scale > 0).all()):
        raise LogisticFitError("a feature is constant on the fitting data")
    design = np.column_stack([np.ones(x.shape[0]), (x - mean) / scale])
    w = np.zeros(design.shape[1])
    for iteration in range(1, max_iterations + 1):
        p = _sigmoid(design @ w)
        gradient = design.T @ (y - p)
        hessian = design.T @ (design * (p * (1.0 - p))[:, None])
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as exc:
            raise LogisticFitError(f"the Hessian is singular at iteration {iteration}") from exc
        w = w + step
        if not np.isfinite(w).all() or np.abs(w[1:]).max() > MAX_ABS_COEFFICIENT:
            raise LogisticFitError("the coefficients diverge, so the classes are separable on these features")
        if np.abs(step).max() < tolerance:
            return LogisticModel(mean=mean, scale=scale, coef=w[1:].copy(), intercept=float(w[0]),
                                 iterations=iteration)
    raise LogisticFitError(f"no convergence in {max_iterations} iterations")


def encode(results: Mapping[str, ToolResult]) -> list[float] | None:
    """The feature vector, or None if any tool has no result."""
    if any(result.status != "ok" for result in results.values()):
        return None
    pop = results["population"].values
    af = pop.get("af")
    present = pop.get("observation") == "observed" and (pop.get("ac") or 0) > 0 and af is not None and af > 0
    return [
        float(results["evo2"].values["delta"]),
        float(results["conservation"].values["score"]),
        1.0 if present else 0.0,
        math.log10(af) if present else 0.0,
    ]


def gather(variant_id: str) -> tuple[list[AgentFinding], list[float] | None, dict[str, ToolResult]]:
    """Run the three tools on one variant: its findings, its features (or None) and the raw results."""
    variant = parse_variant_id(variant_id)
    results = {name: call(variant) for name, call in TOOLS}
    findings = [tool_finding(name, variant_id, result, RATIONALE) for name, result in results.items()]
    return findings, encode(results), results


def calibration_labels(split_dir: Path | None = None, findlay_path: Path | None = None) -> dict[str, int]:
    """LOF = 1, FUNC and INT = 0, for calibration-split IDs only, in calibration.txt order."""
    split_dir = split_dir if split_dir is not None else config.splits_dir()
    calibration = split_ids("calibration", split_dir)
    overlap = set(calibration) & set(split_ids("test", split_dir))
    if overlap:
        raise ValueError(f"{len(overlap)} IDs are in both the calibration and test splits")
    meta = run_eval.read_split_meta(split_dir)
    findlay_path = Path(findlay_path) if findlay_path is not None else default_findlay_path()
    if sha256_file(findlay_path) != meta["source_sha256"]:
        raise ValueError("the Findlay file differs from the one the split was made from (sha256 mismatch)")
    wanted = set(calibration)
    classes = {v.variant_id: v.findlay_function_class for v in load_findlay(findlay_path) if v.variant_id in wanted}
    missing = [vid for vid in calibration if vid not in classes]
    if missing:
        raise ValueError(f"{len(missing)} calibration IDs have no Findlay row, for example {missing[0]}")
    return {vid: int(classes[vid] == "LOF") for vid in calibration}


def _cache_sha256(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def _named(values: Sequence[float]) -> dict[str, float]:
    return {name: float(value) for name, value in zip(FEATURES, values)}


def build_reports(split_dir: Path | None = None,
                  findlay_path: Path | None = None) -> tuple[list[PanelReport], dict[str, Any]]:
    """Fit on every calibration variant, then one report per test variant."""
    labels = calibration_labels(split_dir, findlay_path)
    rows, y, missing, all_results = [], [], [], []
    for vid, label in labels.items():
        _, features, results = gather(vid)
        all_results.append(results["evo2"])
        if features is None:
            missing.append(vid)
            continue
        rows.append(features)
        y.append(label)
    if missing:
        raise LogisticFitError(f"{len(missing)} calibration variants lack a feature, for example {missing[0]}; "
                               "the fit uses the whole calibration split or nothing")
    model = fit_logistic(rows, y)

    reports, status = [], Counter()
    for vid in split_ids("test", split_dir):
        findings, features, results = gather(vid)
        all_results.append(results["evo2"])
        status["scored" if features is not None else "unresolved"] += 1
        p = None if features is None else float(model.predict([features])[0])
        reports.append(baseline_report(vid, p, findings, calibrated=True, summary=SUMMARY))

    raw_coef, raw_intercept = model.raw_coefficients()
    x = np.asarray(rows)
    details = {
        "system": SYSTEM,
        "label_rule": "LOF = 1; FUNC and INT = 0; calibration split only",
        "features": {
            "evo2_delta": "evo2 tool delta (ALT minus REF, summed natural-log likelihood)",
            "phylop": "conservation tool phyloP 100-way score",
            "gnomad_present": "1 if the population tool observed the variant with AC > 0, else 0",
            "gnomad_log10_af": "log10 allele frequency when gnomad_present, else 0",
        },
        "fit": {
            "method": "maximum-likelihood logistic regression, Newton-IRLS, no penalty",
            "standardization": "calibration mean and sample SD (ddof=1)",
            "iterations": model.iterations,
            "n_calibration": len(y),
            "n_lof": int(sum(y)),
            "n_not_lof": len(y) - int(sum(y)),
            "n_gnomad_present": int(x[:, 2].sum()),
            "feature_mean": _named(model.mean),
            "feature_sd": _named(model.scale),
            "coef_standardized": _named(model.coef),
            "intercept_standardized": model.intercept,
            "coef_raw": _named(raw_coef),
            "intercept_raw": raw_intercept,
            "calibration_in_sample_auroc": auroc(y, model.predict(x)),
        },
        "test": {"n_variants": len(reports), **status},
        "calibrated": True,
        "calibrated_basis": "probabilities from a model fit on the calibration split only",
        "inputs": {
            **evo2_provenance(all_results),
            "conservation_cache_sha256": _cache_sha256(conservation.cache_path("GRCh37")),
            "gnomad_cache_sha256": _cache_sha256(population.cache_path("GRCh37")),
        },
    }
    return reports, details


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit the logistic baseline on calibration and score the test split.")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)
    try:
        reports, details = build_reports()
        record = evaluate(SYSTEM, reports, details, run_id=args.run_id)
    except (ValueError, OSError, RuntimeError) as exc:  # covers LogisticFitError, EvalRefused and AnswerKeyError
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {record.outputs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
