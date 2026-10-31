from __future__ import annotations

import hashlib
import json

import pytest

from _factories import finding, findlay_row, panel_findings, report, report_fields, resolved_report, write_findlay_xlsx
from vus_panel.eval import run_eval
from vus_panel.eval.run_eval import EvalRefused, check_reports, cites_clinvar, report_probability

# Placeholder coordinates only. Classes per position: LOF, FUNC, LOF, FUNC, INT.
CLASSES = ["LOF", "FUNC", "LOF", "FUNC", "INT"]
IDS = [f"GRCh37-17-{pos}-A-G" for pos in range(1, 6)]


def _write_reports(path, reports):
    payload = [r if isinstance(r, dict) else r.model_dump(mode="json") for r in reports]
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_meta(split_dir, findlay, **overrides):
    meta = {
        "seed": 0,
        "calibration_fraction": 0.2,
        "source_sha256": hashlib.sha256(findlay.read_bytes()).hexdigest(),
    }
    meta.update(overrides)
    (split_dir / "split_meta.json").write_text(json.dumps(meta), encoding="utf-8")


@pytest.fixture
def eval_inputs(tmp_path):
    findlay = write_findlay_xlsx(tmp_path / "findlay.xlsx", [findlay_row(c, p) for p, c in enumerate(CLASSES, start=1)])
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    (split_dir / "test.txt").write_text("".join(f"{v}\n" for v in IDS), encoding="utf-8")
    _write_meta(split_dir, findlay)
    return findlay, split_dir


@pytest.fixture
def scored_reports(tmp_path):
    return _write_reports(
        tmp_path / "r.json",
        [
            resolved_report(IDS[0], "likely_damaging", 0.9),   # LOF, p = .9
            resolved_report(IDS[1], "likely_tolerated", 0.8),  # FUNC, p = .2
            resolved_report(IDS[2], "likely_tolerated", 0.6),  # LOF, p = .4
            resolved_report(IDS[3], "likely_damaging", 0.7),   # FUNC, p = .7
            report(variant_id=IDS[4]),                         # INT, unresolved
        ],
    )


def _run(reports, eval_inputs, tmp_path, **overrides):
    findlay, split_dir = eval_inputs
    kwargs = {
        "system": "panel",
        "run_id": "panel-test",
        "split_dir": split_dir,
        "findlay_path": findlay,
        "runs_root": tmp_path / "runs",
        "git_commit": "abc1234",
        "git_dirty": True,
    }
    kwargs.update(overrides)
    return run_eval.run(reports, **kwargs)


def test_refuses_mock_records_before_reading_anything_else(tmp_path):
    fields = report_fields(variant_id=IDS[0], mock=True)
    reports = _write_reports(tmp_path / "r.json", [report(**fields)])
    with pytest.raises(EvalRefused, match=IDS[0]):
        run_eval.run(reports, system="panel", git_commit="abc1234")  # no split or Findlay file exists


def test_refuses_records_without_an_explicit_mock_false(tmp_path):
    fields = report(variant_id=IDS[0]).model_dump(mode="json")
    del fields["mock"]
    reports = _write_reports(tmp_path / "r.json", [fields])
    with pytest.raises(EvalRefused, match="mock"):
        run_eval.run(reports, system="panel", git_commit="abc1234")


def test_refuses_variants_outside_the_test_split():
    with pytest.raises(EvalRefused, match="outside the test split"):
        check_reports([report(variant_id="GRCh37-17-99-A-G")], set(IDS))


def test_refuses_duplicate_reports():
    with pytest.raises(EvalRefused, match="more than one"):
        check_reports([report(variant_id=IDS[0]), report(variant_id=IDS[0])], set(IDS))


def _with_finding(**fields):
    """A report whose evo2 finding (both lists) carries the given fields."""
    findings = [finding("evo2", "ok", "damaging", variant_id=IDS[0], **fields)] + panel_findings(IDS[0])[1:]
    return report(variant_id=IDS[0], findings_initial=findings, findings_after_rebuttal=findings)


def _with_revision(reason):
    before = [finding("evo2", "ok", "damaging", variant_id=IDS[0])] + panel_findings(IDS[0])[1:]
    after = [finding("evo2", "ok", "tolerated", variant_id=IDS[0], revision_reason=reason)] + panel_findings(IDS[0])[1:]
    return report(variant_id=IDS[0], findings_initial=before, findings_after_rebuttal=after)


@pytest.mark.parametrize(
    "build",
    [
        lambda: _with_finding(sources=["ClinVar"]),
        lambda: _with_finding(rationale="placeholder citing clinvar"),
        lambda: _with_finding(raw_values={"clinvar_class": "placeholder"}),
        lambda: _with_finding(raw_values={"nested": {"note": "from ClinVar"}}),
        lambda: _with_finding(raw_values={"ids": ["VCV000000001"]}),
        lambda: _with_finding(rationale="see RCV000000001"),
        lambda: _with_revision("matches ClinVar"),
        lambda: report(variant_id=IDS[0], counselor_summary="ClinVar says so"),
        lambda: report(variant_id=IDS[0], patient_summary="clinvar placeholder"),
        lambda: report(variant_id=IDS[0], what_would_settle_it="a ClinVar submission"),
        lambda: report(variant_id=IDS[0], conflicts=["evo2 disagrees with CLINVAR"]),
    ],
    ids=["sources", "rationale", "raw_key", "raw_nested_value", "vcv", "rcv", "revision_reason",
         "counselor", "patient", "settle", "conflicts"],
)
def test_refuses_clinvar_anywhere_in_a_test_report(build):
    item = build()
    assert cites_clinvar(item)
    with pytest.raises(EvalRefused, match="ClinVar"):
        check_reports([item], set(IDS))


def test_clinvar_scan_has_no_false_positive_on_plain_reports():
    assert not cites_clinvar(report(variant_id=IDS[0]))
    assert not cites_clinvar(_with_finding(rationale="placeholder VCV12 too short"))


def test_report_probability_maps_call_direction():
    assert report_probability(resolved_report(IDS[0], "likely_damaging", 0.8)) == pytest.approx(0.8)
    assert report_probability(resolved_report(IDS[0], "likely_tolerated", 0.8)) == pytest.approx(0.2)
    assert report_probability(report(variant_id=IDS[0])) is None


def test_end_to_end_writes_run_record_and_metrics(eval_inputs, scored_reports, tmp_path):
    record = _run(scored_reports, eval_inputs, tmp_path)
    out_dir = tmp_path / "runs" / "panel-test"
    saved = json.loads((out_dir / "run_record.json").read_text(encoding="utf-8"))
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    findlay, _ = eval_inputs
    assert saved["run_id"] == record.run_id == "panel-test"
    assert saved["git_commit"] == "abc1234"
    assert saved["seed"] == 0
    assert saved["variant_ids"] == IDS
    assert saved["config"]["system"] == "panel"
    assert saved["config"]["calibration_fraction"] == 0.2
    assert saved["config"]["split_source_sha256"] == hashlib.sha256(findlay.read_bytes()).hexdigest()
    assert saved["config"]["git_dirty"] is True
    # Positives .9 and .4 against negatives .2 and .7: 3 of 4 pairs ranked right.
    assert metrics["auroc_resolved"] == pytest.approx(0.75)
    assert metrics["coverage"] == pytest.approx(0.8)
    assert metrics["n_resolved"] == 4 and metrics["n_total"] == 5
    assert metrics["reliability"]["n"] == 4

    with pytest.raises(FileExistsError):
        _run(scored_reports, eval_inputs, tmp_path)


def test_seed_is_read_from_split_meta(eval_inputs, scored_reports, tmp_path):
    findlay, split_dir = eval_inputs
    _write_meta(split_dir, findlay, seed=3)
    assert _run(scored_reports, eval_inputs, tmp_path, seed=3).seed == 3
    with pytest.raises(EvalRefused, match="disagrees"):
        _run(scored_reports, eval_inputs, tmp_path, run_id="other", seed=0)


def test_refuses_without_split_meta(eval_inputs, scored_reports, tmp_path):
    _, split_dir = eval_inputs
    (split_dir / "split_meta.json").unlink()
    with pytest.raises(FileNotFoundError, match="split metadata"):
        _run(scored_reports, eval_inputs, tmp_path)


def test_refuses_a_different_findlay_file(eval_inputs, scored_reports, tmp_path):
    findlay, split_dir = eval_inputs
    _write_meta(split_dir, findlay, source_sha256="0" * 64)
    with pytest.raises(EvalRefused, match="sha256"):
        _run(scored_reports, eval_inputs, tmp_path)


def test_unknown_system_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        run_eval.run(tmp_path / "r.json", system="ensemble", git_commit="abc1234")


def test_cli_exits_2_on_mock_input(tmp_path, capsys):
    reports = _write_reports(tmp_path / "r.json", [report(**report_fields(variant_id=IDS[0], mock=True))])
    assert run_eval.main(["--reports", str(reports), "--system", "panel"]) == 2
    assert "mock" in capsys.readouterr().err


def test_cli_exits_2_on_malformed_json(tmp_path, capsys):
    path = tmp_path / "r.json"
    path.write_text("{not json", encoding="utf-8")
    assert run_eval.main(["--reports", str(path), "--system", "panel"]) == 2
    assert "error" in capsys.readouterr().err
