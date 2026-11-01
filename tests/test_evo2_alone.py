"""Evo 2-alone baseline. Owner: orchestration-eval.

Also holds write_world, a small offline world for the baselines: frozen split
files, a Findlay-shaped workbook, stored Evo 2 runs and the two tool caches.
Positions are placeholders well away from BRCA1.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import pytest

from _factories import findlay_row, write_findlay_xlsx
from test_tools_models import stored_response, write_run
from vus_panel import config
from vus_panel.eval import evo2_alone, run_eval
from vus_panel.eval._baseline import DETAILS_FILE, REPORTS_FILE, evaluate, evo2_provenance, split_ids
from vus_panel.eval.metrics import auroc
from vus_panel.eval.run_eval import report_probability
from vus_panel.eval.split import default_findlay_path
from vus_panel.schemas import PanelReport, ToolResult
from vus_panel.tools import conservation, population

CAL_RUN = "20260101T000000Z-calibration-aaaaaaaaaaaa"
TEST_RUN = "20260101T000100Z-test-bbbbbbbbbbbb"


@dataclass(frozen=True)
class Site:
    pos: int
    cls: str
    delta: float
    phylop: float | None = 0.0  # None: no conservation row, so that tool is unavailable here
    gnomad: tuple | None = None  # None: absent; ("obs", ac, an, af); ("ac0",); ("nocall",)


def vid(pos: int) -> str:
    return f"GRCh37-17-{pos}-A-G"


def write_splits_and_workbook(cal, test, workbook_classes=None):
    splits = config.splits_dir()
    splits.mkdir(parents=True, exist_ok=True)
    (splits / "calibration.txt").write_text("".join(f"{vid(s.pos)}\n" for s in cal), encoding="ascii")
    (splits / "test.txt").write_text("".join(f"{vid(s.pos)}\n" for s in test), encoding="ascii")
    classes = workbook_classes or {}
    path = default_findlay_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_findlay_xlsx(path, [findlay_row(classes.get(s.pos, s.cls), s.pos) for s in [*cal, *test]])
    meta = {"seed": 0, "calibration_fraction": 0.2, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (splits / "split_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return path


def write_world(tmp_path, monkeypatch, cal, test, *, errors=(), workbook_classes=None):
    """Write every input both baselines read, and point EVO2_SCORES_DIR at the stored runs."""
    write_splits_and_workbook(cal, test, workbook_classes)
    scores = tmp_path / "scores"
    write_run(scores, CAL_RUN, [stored_response(s.pos, s.delta) for s in cal])
    write_run(scores, TEST_RUN, [stored_response(s.pos, error_code="inference") if s.pos in errors
                                 else stored_response(s.pos, s.delta) for s in test])
    monkeypatch.setenv("EVO2_SCORES_DIR", str(scores))

    sites = [*cal, *test]
    cons = conservation.cache_path("GRCh37")
    cons.parent.mkdir(parents=True, exist_ok=True)
    cons.write_text("chrom\tpos\tscore\n" + "".join(f"17\t{s.pos}\t{s.phylop}\n" for s in sites if s.phylop is not None),
                    encoding="utf-8")
    low, high = min(s.pos for s in sites) - 1, max(s.pos for s in sites) + 1
    rows = [f"17\t{low}\tA\tT\t1\t1000\t0.001\n", f"17\t{high}\tA\tT\t1\t1000\t0.001\n"]  # region bounds only
    for s in sites:
        if s.gnomad is None:
            continue
        if s.gnomad[0] == "obs":
            _, ac, an, af = s.gnomad
            rows.append(f"17\t{s.pos}\tA\tG\t{ac}\t{an}\t{af}\n")
        elif s.gnomad[0] == "ac0":
            rows.append(f"17\t{s.pos}\tA\tG\t0\t1000\t0\n")
        else:
            rows.append(f"17\t{s.pos}\tA\tG\t\t0\t\n")
    population.cache_path("GRCh37").write_text("chrom\tpos\tref\talt\tac\tan\taf\n" + "".join(rows), encoding="utf-8")
    return scores


CAL = [Site(5000 + i, cls, d) for i, (cls, d) in enumerate(
    [("LOF", -40.0), ("FUNC", -1.5), ("FUNC", 2.0), ("LOF", -12.5), ("INT", -20.0), ("FUNC", -3.0)])]
# Two tied deltas (-8.0), an INT, and an LOF that Evo 2 ranks badly, so the AUROC is not trivially 1.
TEST = [Site(6000 + i, cls, d) for i, (cls, d) in enumerate(
    [("LOF", -30.0), ("FUNC", -8.0), ("LOF", -8.0), ("FUNC", 1.0), ("INT", -25.0), ("LOF", 0.5), ("FUNC", -2.0)])]


def run_baseline(reports, details, tmp_path, run_id="evo2-test"):
    record = evaluate(evo2_alone.SYSTEM, reports, details, run_id=run_id, runs_root=tmp_path / "runs",
                      git_commit="abc1234", git_dirty=False)
    out = tmp_path / "runs" / run_id
    return record, json.loads((out / "metrics.json").read_text(encoding="utf-8")), out


def test_rank_probabilities_keep_order_and_ties():
    p = evo2_alone.rank_probabilities([3.0, -1.0, 3.0, 0.5])
    assert p.tolist() == [0.75, 0.125, 0.75, 0.375]
    assert ((p > 0) & (p < 1)).all()


@pytest.mark.parametrize("scores", [[], [1.0, float("nan")], [[1.0, 2.0]]])
def test_rank_probabilities_refuse_bad_input(scores):
    with pytest.raises(ValueError):
        evo2_alone.rank_probabilities(scores)


def test_end_to_end_auroc_is_the_auroc_of_minus_delta(tmp_path, monkeypatch):
    write_world(tmp_path, monkeypatch, CAL, TEST)
    reports, details = evo2_alone.build_reports(split_ids("test"))
    record, metrics, out = run_baseline(reports, details, tmp_path)

    y = [int(s.cls == "LOF") for s in TEST]
    expected = auroc(y, [-s.delta for s in TEST])
    assert 0.5 < expected < 1.0
    assert metrics["auroc_resolved"] == expected
    assert metrics["coverage"] == 1.0 and metrics["n_resolved"] == metrics["n_total"] == len(TEST)
    assert metrics["n_calibrated"] == 0
    assert record.config["system"] == "evo2_alone" and record.config["reports_file"] == REPORTS_FILE

    saved = [PanelReport.model_validate(r) for r in json.loads((out / REPORTS_FILE).read_text(encoding="utf-8"))]
    assert saved == reports
    assert all(r.mock is False and r.calibrated is False and r.driving_agents == ["evo2"] for r in reports)
    # The probability run_eval reads ties exactly where the deltas tie.
    p = {r.variant_id: report_probability(r) for r in reports}
    assert p[vid(6001)] == p[vid(6002)]
    assert sorted(p, key=p.get) == [vid(s.pos) for s in sorted(TEST, key=lambda s: -s.delta)]

    stored = json.loads((out / DETAILS_FILE).read_text(encoding="utf-8"))
    assert stored["n_scored"] == len(TEST) and stored["evo2_stored_runs"] == [TEST_RUN]
    assert stored["reliability_meaningful"] is False


def test_findings_carry_the_tools_numbers_and_sources(tmp_path, monkeypatch):
    write_world(tmp_path, monkeypatch, CAL, TEST)
    reports, _ = evo2_alone.build_reports(split_ids("test"))
    finding = reports[0].findings_initial[0]
    assert finding.status == "ok" and finding.verdict == "uncertain"
    assert finding.raw_values["delta"] == TEST[0].delta and finding.raw_values["score_source"] == "stored"
    assert f"run {TEST_RUN};" in finding.sources[0]


def test_provenance_reads_score_source_and_refuses_an_unreadable_stored_source():
    stored = ToolResult(tool="evo2", status="ok", values={"score_source": "stored"}, message="m",
                        sources=[f"Evo 2 stored result, run {TEST_RUN}; service session s"])
    backend = ToolResult(tool="evo2", status="ok", values={"score_source": "backend"}, message="m",
                         sources=["Evo 2 service session s"])
    missing = ToolResult(tool="evo2", status="unavailable", message="m")
    assert evo2_provenance([stored, backend, missing]) == {
        "evo2_scores_dir": None, "evo2_stored_runs": [TEST_RUN], "evo2_backend_scored": 1}
    reworded = stored.model_copy(update={"sources": ["a stored result from somewhere"]})
    with pytest.raises(ValueError, match="stored run name"):
        evo2_provenance([reworded])


def test_a_stored_backend_error_is_unresolved_and_lowers_coverage(tmp_path, monkeypatch):
    write_world(tmp_path, monkeypatch, CAL, TEST, errors={6003})
    reports, details = evo2_alone.build_reports(split_ids("test"))
    failed = next(r for r in reports if r.variant_id == vid(6003))
    assert failed.call == "unresolved" and failed.confidence is None
    assert failed.findings_initial[0].status == "error"
    _, metrics, _ = run_baseline(reports, details, tmp_path)
    assert metrics["n_resolved"] == len(TEST) - 1
    assert metrics["coverage"] == pytest.approx((len(TEST) - 1) / len(TEST))
    assert details["tool_status"] == {"ok": len(TEST) - 1, "error": 1}


def test_building_reports_needs_no_workbook_or_labels(tmp_path, monkeypatch):
    write_world(tmp_path, monkeypatch, CAL, TEST)
    default_findlay_path().unlink()
    (config.splits_dir() / "split_meta.json").unlink()
    reports, _ = evo2_alone.build_reports(split_ids("test"))
    assert len(reports) == len(TEST)


def test_no_delta_for_any_variant_is_refused(tmp_path, monkeypatch):
    write_world(tmp_path, monkeypatch, CAL, TEST)
    monkeypatch.delenv("EVO2_SCORES_DIR")
    with pytest.raises(ValueError, match="no test variant has an Evo 2 delta"):
        evo2_alone.build_reports(split_ids("test"))


def test_cli_writes_one_run_folder_and_exits_2_when_refused(tmp_path, monkeypatch, capsys):
    write_world(tmp_path, monkeypatch, CAL, TEST)
    monkeypatch.setattr(run_eval, "git_commit_hash", lambda: "abc1234")
    monkeypatch.setattr(run_eval, "git_is_dirty", lambda: False)
    assert evo2_alone.main(["--run-id", "evo2-cli"]) == 0
    out = config.runs_dir() / "evo2-cli"
    assert sorted(p.name for p in out.iterdir()) == sorted(["metrics.json", "run_record.json", REPORTS_FILE, DETAILS_FILE])
    assert evo2_alone.main(["--run-id", "evo2-cli"]) == 2  # run_eval refuses an existing run directory
    assert "already exists" in capsys.readouterr().err
