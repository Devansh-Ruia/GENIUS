"""Metric tests on small hand-built arrays with answers worked out by hand."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vus_panel.eval.metrics import auroc, mean_sd, reliability_bins, selective_auroc


def test_auroc_known_answer():
    # Positive scores .35 and .8 against negative .1 and .4: 3 of 4 pairs ranked right.
    assert auroc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)


def test_auroc_all_ties_is_half():
    assert auroc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)


def test_auroc_partial_tie_counts_half():
    # Pairs: (.5 vs .5) tie = 0.5, (.5 vs .2) = 1, (.9 vs .5) = 1, (.9 vs .2) = 1, so 3.5 / 4.
    assert auroc([0, 1, 0, 1], [0.5, 0.5, 0.2, 0.9]) == pytest.approx(0.875)


def test_auroc_perfect_and_inverted():
    assert auroc([0, 0, 1, 1], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert auroc([0, 0, 1, 1], [4, 3, 2, 1]) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "y, s",
    [
        ([1, 1, 1], [0.1, 0.2, 0.3]),
        ([0, 0], [0.1, 0.2]),
        ([0, 1], [0.1]),
        ([0, 2], [0.1, 0.2]),
        ([0, 1], [0.1, math.nan]),
    ],
)
def test_auroc_rejects_bad_input(y, s):
    with pytest.raises(ValueError):
        auroc(y, s)


def test_selective_auroc_reports_coverage_with_auroc():
    result = selective_auroc(
        [0, 0, 1, 1, 1, 0],
        [0.1, 0.4, 0.35, 0.8, math.nan, math.nan],
        [True, True, True, True, False, False],
    )
    assert result.auroc == pytest.approx(0.75)
    assert result.coverage == pytest.approx(4 / 6)
    assert result.n_resolved == 4
    assert result.n_total == 6


def test_selective_auroc_without_both_classes_returns_none_but_keeps_coverage():
    nothing = selective_auroc([0, 1], [math.nan, math.nan], [False, False])
    assert nothing.auroc is None and nothing.coverage == 0.0 and nothing.n_resolved == 0
    one_class = selective_auroc([1, 1, 0], [0.9, 0.8, math.nan], [True, True, False])
    assert one_class.auroc is None and one_class.coverage == pytest.approx(2 / 3)


def test_reliability_bins_known_ece():
    # Bin [0, .5): mean .1, observed .5. Bin [.5, 1]: mean .9, observed 1. ECE = .5*.4 + .5*.1.
    result = reliability_bins([0.1, 0.1, 0.9, 0.9], [0, 1, 1, 1], n_bins=2)
    assert result.ece == pytest.approx(0.25)
    assert result.n == 4
    low, high = result.bins
    assert (low.count, low.mean_predicted, low.observed_rate) == (2, pytest.approx(0.1), pytest.approx(0.5))
    assert (high.count, high.mean_predicted, high.observed_rate) == (2, pytest.approx(0.9), pytest.approx(1.0))


def test_reliability_perfect_calibration_and_edges():
    result = reliability_bins([0.0, 0.0, 1.0, 1.0], [0, 0, 1, 1], n_bins=10)
    assert result.ece == pytest.approx(0.0)
    assert result.bins[0].count == 2
    assert result.bins[-1].count == 2  # 1.0 lands in the last bin
    assert all(b.count == 0 and b.mean_predicted is None and b.observed_rate is None for b in result.bins[1:-1])


@pytest.mark.parametrize("prob", [[1.2, 0.5], [-0.1, 0.5], [math.nan, 0.5]])
def test_reliability_rejects_bad_probabilities(prob):
    with pytest.raises(ValueError):
        reliability_bins(prob, [0, 1])


def test_mean_sd_uses_ddof_one():
    mean, sd = mean_sd([2, 4, 4, 4, 5, 5, 7, 9])
    assert mean == pytest.approx(5.0)
    assert sd == pytest.approx(math.sqrt(32 / 7))  # 2.138, not the population value 2.0
    assert sd == pytest.approx(np.std([2, 4, 4, 4, 5, 5, 7, 9], ddof=1))
    assert sd != pytest.approx(2.0)


def test_mean_sd_needs_two_values():
    with pytest.raises(ValueError):
        mean_sd([1.0])
