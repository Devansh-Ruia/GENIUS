"""Logistic baseline. Owner: orchestration-eval."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from test_evo2_alone import CAL_RUN, TEST_RUN, Site, vid, write_splits_and_workbook, write_world
from vus_panel import config
from vus_panel.eval import logistic_baseline as lb
from vus_panel.eval import run_eval
from vus_panel.eval._baseline import DETAILS_FILE, REPORTS_FILE, evaluate
from vus_panel.eval.metrics import auroc
from vus_panel.eval.run_eval import report_probability
from vus_panel.schemas import ToolResult


def sites(start, groups):
    """Each group is (delta, phylop, gnomad, n_lof, n_not_lof); FUNC and INT alternate among the non-LOF."""
    out, pos = [], start
    for delta, phylop, gnomad, n_lof, n_not in groups:
        for cls in ["LOF"] * n_lof + [("FUNC", "INT")[i % 2] for i in range(n_not)]:
            out.append(Site(pos, cls, delta, phylop, gnomad))
            pos += 1
    return out


# Every calibration feature vector carries both labels, so no hyperplane separates the
# classes and the maximum-likelihood fit is finite. Two gnomAD frequencies keep
# gnomad_log10_af from being a multiple of gnomad_present.
CAL = sites(5000, [
    (-40.0, 5.0, None, 3, 1),
    (-10.0, 3.0, None, 2, 2),
    (-2.0, 1.0, None, 1, 3),
    (2.0, -1.0, None, 1, 3),
    (-5.0, 2.0, ("obs", 3, 1000, 0.003), 1, 2),
    (1.0, 0.5, ("obs", 1, 10000, 0.0001), 1, 2),
    (-1.0, 0.0, ("ac0",), 1, 1),
])
TEST = sites(6000, [
    (-35.0, 4.5, None, 2, 1),
    (-4.0, 1.5, None, 1, 2),
    (3.0, -2.0, None, 0, 2),
    (0.5, 0.2, ("obs", 18, 1000, 0.018), 0, 1),
    (-8.0, 2.5, ("nocall",), 1, 0),
])


def ok(tool, **values):
    return ToolResult(tool=tool, status="ok", values=values, message="placeholder")


def results(population_values, evo2_status="ok"):
    evo = ok("evo2", delta=-3.0) if evo2_status == "ok" else ToolResult(tool="evo2", status=evo2_status, message="none")
    return {"evo2": evo, "conservation": ok("conservation", score=2.5), "population": ok("population", **population_values)}


# Fitting

def test_one_binary_feature_fits_each_groups_observed_rate():
    x = [[0.0]] * 10 + [[1.0]] * 10
    y = [1] * 3 + [0] * 7 + [1] * 8 + [0] * 2
    model = lb.fit_logistic(x, y)
    assert model.predict([[0.0], [1.0]]) == pytest.approx([0.3, 0.8], abs=1e-9)


def test_the_gradient_is_zero_at_the_fit_and_raw_coefficients_agree():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 3)) * [1.0, 5.0, 0.2] + [0.0, 10.0, -1.0]
    y = (rng.random(500) < 1 / (1 + np.exp(-(0.5 + x @ [1.0, -0.3, 2.0] - (-3.0))))).astype(int)
    model = lb.fit_logistic(x, y)
    design = np.column_stack([np.ones(len(x)), (x - model.mean) / model.scale])
    gradient = design.T @ (y - model.predict(x))
    assert np.abs(gradient).max() < 1e-8
    coef, intercept = model.raw_coefficients()
    assert 1 / (1 + np.exp(-(x @ coef + intercept))) == pytest.approx(model.predict(x), abs=1e-12)
    assert model.scale == pytest.approx(x.std(axis=0, ddof=1))


def test_separable_classes_are_refused():
    with pytest.raises(lb.LogisticFitError, match="separable"):
        lb.fit_logistic([[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1])


@pytest.mark.parametrize("x, y, message", [
    ([[1.0], [1.0], [1.0]], [0, 1, 0], "constant"),
    ([[0.0], [1.0], [2.0]], [1, 1, 1], "one class"),
])
def test_degenerate_fits_are_refused(x, y, message):
    with pytest.raises(lb.LogisticFitError, match=message):
        lb.fit_logistic(x, y)


# Features

@pytest.mark.parametrize("population_values, present, log_af", [
    ({"observation": "observed", "ac": 12, "an": 1000, "af": 0.012}, 1.0, math.log10(0.012)),
    ({"observation": "observed", "ac": 0, "an": 1000, "af": 0.0}, 0.0, 0.0),
    ({"observation": "no_call", "ac": None, "an": 0, "af": None}, 0.0, 0.0),
    ({"observation": "not_observed", "ac": 0, "an": None, "af": None}, 0.0, 0.0),
])
def test_gnomad_presence_and_frequency_encoding(population_values, present, log_af):
    assert lb.encode(results(population_values)) == [-3.0, 2.5, present, log_af]


@pytest.mark.parametrize("status", ["unavailable", "error"])
def test_a_tool_without_a_result_gives_no_features(status):
    assert lb.encode(results({"observation": "not_observed", "ac": 0, "an": None, "af": None}, evo2_status=status)) is None


# Labels: calibration only

def test_test_split_labels_cannot_change_the_reports(tmp_path, monkeypatch):
    write_world(tmp_path, monkeypatch, CAL, TEST)
    first, first_details = lb.build_reports()
    flipped = {s.pos: ("FUNC" if s.cls == "LOF" else "LOF") for s in TEST}
    write_splits_and_workbook(CAL, TEST, flipped)
    second, second_details = lb.build_reports()
    assert first == second
    assert first_details["fit"] == second_details["fit"]


def test_calibration_labels_drive_the_fit(tmp_path, monkeypatch):
    write_world(tmp_path, monkeypatch, CAL, TEST)
    _, before = lb.build_reports()
    # Swap the classes of the first calibration group: the most negative delta becomes mostly non-LOF.
    swapped = {s.pos: ("FUNC" if s.cls == "LOF" else "LOF") for s in CAL[:4]}
    write_splits_and_workbook(CAL, TEST, swapped)
    _, after = lb.build_reports()
    assert after["fit"]["coef_raw"] != before["fit"]["coef_raw"]


def test_labels_are_read_for_calibration_ids_only(tmp_path, monkeypatch):
    write_world(tmp_path, monkeypatch, CAL, TEST)
    labels = lb.calibration_labels()
    assert list(labels) == [vid(s.pos) for s in CAL]
    assert labels == {vid(s.pos): int(s.cls == "LOF") for s in CAL}


def test_overlapping_splits_are_refused(tmp_path, monkeypatch):
    write_world(tmp_path, monkeypatch, CAL, TEST)
    path = config.splits_dir() / "test.txt"
    path.write_text(path.read_text(encoding="ascii") + f"{vid(CAL[0].pos)}\n", encoding="ascii")
    with pytest.raises(ValueError, match="both the calibration and test splits"):
        lb.build_reports()


def test_a_different_workbook_is_refused(tmp_path, monkeypatch):
    write_world(tmp_path, monkeypatch, CAL, TEST)
    meta_path = config.splits_dir() / "split_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta_path.write_text(json.dumps({**meta, "source_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        lb.build_reports()


# Missing features

def test_a_calibration_variant_without_a_feature_stops_the_fit(tmp_path, monkeypatch):
    cal = [Site(s.pos, s.cls, s.delta, None if i == 0 else s.phylop, s.gnomad) for i, s in enumerate(CAL)]
    write_world(tmp_path, monkeypatch, cal, TEST)
    with pytest.raises(lb.LogisticFitError, match="1 calibration variants lack a feature"):
        lb.build_reports()


def test_a_test_variant_without_a_feature_is_unresolved(tmp_path, monkeypatch):
    test = [Site(s.pos, s.cls, s.delta, None if i == 0 else s.phylop, s.gnomad) for i, s in enumerate(TEST)]
    write_world(tmp_path, monkeypatch, CAL, test)
    reports, details = lb.build_reports()
    first = reports[0]
    assert first.call == "unresolved" and first.confidence is None and first.calibrated is False
    assert {f.agent: f.status for f in first.findings_initial}["conservation"] == "unavailable"
    assert details["test"] == {"n_variants": len(TEST), "scored": len(TEST) - 1, "unresolved": 1}


# End to end through run_eval

def test_end_to_end_scores_the_fitted_probability_through_run_eval(tmp_path, monkeypatch):
    write_world(tmp_path, monkeypatch, CAL, TEST)
    reports, details = lb.build_reports()
    record = evaluate(lb.SYSTEM, reports, details, run_id="logit-test", runs_root=tmp_path / "runs",
                      git_commit="abc1234", git_dirty=False)
    out = tmp_path / "runs" / "logit-test"
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))

    p = [report_probability(r) for r in reports]
    y = [int(s.cls == "LOF") for s in TEST]
    assert metrics["auroc_resolved"] == auroc(y, p)
    assert metrics["coverage"] == 1.0 and metrics["n_resolved"] == metrics["n_total"] == len(TEST)
    assert metrics["n_calibrated"] == len(TEST)
    assert record.config["system"] == "logistic_baseline"
    assert all(0.0 < q < 1.0 for q in p)
    assert all(r.calibrated and r.driving_agents == ["evo2", "conservation", "population"] for r in reports)
    assert all(f.verdict == "uncertain" and f.raw_values for r in reports for f in r.findings_initial)

    stored = json.loads((out / DETAILS_FILE).read_text(encoding="utf-8"))
    assert stored["fit"]["n_calibration"] == len(CAL)
    assert stored["fit"]["n_lof"] == sum(s.cls == "LOF" for s in CAL)
    # Only the two ("obs", ...) groups count as present; the ("ac0",) group does not.
    assert stored["fit"]["n_gnomad_present"] == sum(s.gnomad is not None and s.gnomad[0] == "obs" for s in CAL) == 6
    assert set(stored["fit"]["coef_raw"]) == set(lb.FEATURES)
    assert stored["inputs"]["evo2_stored_runs"] == [CAL_RUN, TEST_RUN]
    assert stored["inputs"]["conservation_cache_sha256"] and stored["inputs"]["gnomad_cache_sha256"]
    assert (out / REPORTS_FILE).is_file()


def test_the_fitted_probabilities_follow_the_model(tmp_path, monkeypatch):
    write_world(tmp_path, monkeypatch, CAL, TEST)
    reports, details = lb.build_reports()
    fit = details["fit"]
    for report, site in zip(reports, TEST):
        present = site.gnomad is not None and site.gnomad[0] == "obs"
        x = {"evo2_delta": site.delta, "phylop": site.phylop, "gnomad_present": float(present),
             "gnomad_log10_af": math.log10(site.gnomad[3]) if present else 0.0}
        z = fit["intercept_raw"] + sum(fit["coef_raw"][k] * x[k] for k in lb.FEATURES)
        assert report_probability(report) == pytest.approx(1 / (1 + math.exp(-z)), abs=1e-9)


def test_cli_writes_one_run_folder_and_exits_2_when_the_fit_is_impossible(tmp_path, monkeypatch, capsys):
    write_world(tmp_path, monkeypatch, CAL, TEST)
    monkeypatch.setattr(run_eval, "git_commit_hash", lambda: "abc1234")
    monkeypatch.setattr(run_eval, "git_is_dirty", lambda: False)
    assert lb.main(["--run-id", "logit-cli"]) == 0
    assert sorted(p.name for p in (config.runs_dir() / "logit-cli").iterdir()) == sorted(
        ["metrics.json", "run_record.json", REPORTS_FILE, DETAILS_FILE])

    separable = [Site(s.pos, s.cls, -50.0 if s.cls == "LOF" else 5.0, s.phylop, s.gnomad) for s in CAL]
    write_world(tmp_path / "second", monkeypatch, separable, TEST)
    assert lb.main(["--run-id", "logit-cli-2"]) == 2
    assert "separable" in capsys.readouterr().err
