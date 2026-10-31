"""Evaluation metrics.

AUROC is only ever reported on resolved variants, together with coverage
(the fraction not unresolved), so a system can't look good by abstaining.
Any standard deviation uses ddof=1.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def _labels(y_true: Sequence[int] | np.ndarray) -> np.ndarray:
    y = np.asarray(y_true)
    if y.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0 or 1")
    return y.astype(int)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks, with tied values sharing the mean of their ranks."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    last_rank = np.cumsum(counts)
    mean_rank = last_rank - (counts - 1) / 2.0
    return mean_rank[inverse]


def auroc(y_true: Sequence[int] | np.ndarray, score: Sequence[float] | np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney U), ties counted as half."""
    y = _labels(y_true)
    s = np.asarray(score, dtype=float)
    if s.shape != y.shape:
        raise ValueError("labels and scores must have the same length")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC needs at least one positive and one negative")
    ranks = _average_ranks(s)
    u = ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


@dataclass(frozen=True)
class SelectiveAUROC:
    auroc: float | None  # None when the resolved set lacks a positive or a negative
    coverage: float
    n_resolved: int
    n_total: int


def selective_auroc(
    y_true: Sequence[int] | np.ndarray,
    score: Sequence[float] | np.ndarray,
    resolved: Sequence[bool] | np.ndarray,
) -> SelectiveAUROC:
    """AUROC on resolved items, always returned with coverage and counts."""
    y = _labels(y_true)
    s = np.asarray(score, dtype=float)
    mask = np.asarray(resolved, dtype=bool)
    if not (y.shape == s.shape == mask.shape):
        raise ValueError("labels, scores and resolved flags must have the same length")
    n_total = int(y.size)
    if n_total == 0:
        raise ValueError("no items to score")
    n_resolved = int(mask.sum())
    y_resolved = y[mask]
    value = None
    if n_resolved and 0 < int(y_resolved.sum()) < n_resolved:
        value = auroc(y_resolved, s[mask])
    return SelectiveAUROC(auroc=value, coverage=n_resolved / n_total, n_resolved=n_resolved, n_total=n_total)


@dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_predicted: float | None
    observed_rate: float | None


@dataclass(frozen=True)
class Reliability:
    bins: list[ReliabilityBin]
    ece: float
    n: int


def reliability_bins(
    prob: Sequence[float] | np.ndarray,
    outcome: Sequence[int] | np.ndarray,
    n_bins: int = 10,
) -> Reliability:
    """Equal-width reliability bins over [0, 1] plus expected calibration error.

    Bin i holds probabilities in [i/n, (i+1)/n), and the last bin also holds 1.0.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    p = np.asarray(prob, dtype=float)
    y = _labels(outcome)
    if p.shape != y.shape:
        raise ValueError("probabilities and outcomes must have the same length")
    if p.size == 0:
        raise ValueError("no items to bin")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("probabilities must be finite and in [0, 1]")

    index = np.minimum((p * n_bins).astype(int), n_bins - 1)
    bins: list[ReliabilityBin] = []
    ece = 0.0
    for i in range(n_bins):
        in_bin = index == i
        count = int(in_bin.sum())
        if count:
            mean_predicted = float(p[in_bin].mean())
            observed_rate = float(y[in_bin].mean())
            ece += count / p.size * abs(mean_predicted - observed_rate)
        else:
            mean_predicted = observed_rate = None
        bins.append(ReliabilityBin(i / n_bins, (i + 1) / n_bins, count, mean_predicted, observed_rate))
    return Reliability(bins=bins, ece=float(ece), n=int(p.size))


def mean_sd(values: Sequence[float] | np.ndarray) -> tuple[float, float]:
    """Mean and sample standard deviation (ddof=1), for example across seeds."""
    v = np.asarray(values, dtype=float)
    if v.ndim != 1 or v.size < 2:
        raise ValueError("need at least two values for a sample standard deviation")
    return float(v.mean()), float(v.std(ddof=1))
