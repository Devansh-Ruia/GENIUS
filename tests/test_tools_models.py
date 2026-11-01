"""evo2 and sae tools. Owner: evidence-models."""

from __future__ import annotations

import builtins
import json
import pathlib

import pytest

from _factories import placeholder_variant
from vus_panel import config
from vus_panel.scoring.contracts import ContextRecord, ScoreRequest, ScoreResponse
from vus_panel.tools import evo2, sae

BACKEND_TOOLS = [(evo2, evo2.log_likelihood_delta), (sae, sae.feature_activations)]


@pytest.mark.parametrize("module, tool", BACKEND_TOOLS)
def test_backend_tool_without_url_names_the_gpu_requirement(module, tool):
    result = tool(placeholder_variant())
    assert result.tool == module.TOOL
    assert result.status == "unavailable"
    assert "EVO2_BACKEND_URL" in result.message
    assert "FP8" in result.message
    assert result.values == {}


def test_sae_with_url_sends_nothing_and_stays_unavailable(monkeypatch):
    monkeypatch.setenv("EVO2_BACKEND_URL", "http://placeholder.invalid")
    result = sae.feature_activations(placeholder_variant())
    assert result.status == "unavailable"
    assert "not implemented" in result.message
    assert "evidence-models" in result.message
    assert result.values == {}


def test_evo2_uses_typed_client_and_returns_only_successful_values(monkeypatch):
    from test_scoring_service import service

    instance = service()
    monkeypatch.setenv("EVO2_BACKEND_URL", "http://placeholder.invalid")
    monkeypatch.setattr(evo2.ScoringClient, "score", lambda self, variant: instance.score(variant))
    variant = placeholder_variant(pos=4097, alt="C")
    result = evo2.log_likelihood_delta(variant)
    assert result.status == "ok" and result.values["delta"] == -2
    instance.backend.fail_at = len(instance.backend.calls) + 2
    result = evo2.log_likelihood_delta(variant)
    assert result.status == "error" and result.values == {}


def test_evo2_backend_failure_does_not_invent_values(monkeypatch):
    from vus_panel.scoring.client import ScoringClientError

    def fail(self, variant):
        raise ScoringClientError("offline")

    monkeypatch.setenv("EVO2_BACKEND_URL", "http://placeholder.invalid")
    monkeypatch.setattr(evo2.ScoringClient, "score", fail)
    result = evo2.log_likelihood_delta(placeholder_variant())
    assert result.status == "error" and result.values == {}


# Stored results under EVO2_SCORES_DIR, written in the pilot and score_split
# outcomes.jsonl format. Positions are placeholders well away from BRCA1, and
# totals are exact binary values so ALT minus REF equals delta exactly.

IDENTITY = {"session_id": "session-1", "created_at": "2026-01-01T00:00:00+00:00", "gpu": [{"name": "placeholder"}],
            "load_seconds": 1.0, "warmup": {}, "checkpoint_sha256": "c" * 64, "precision": {"projection_mode": "fp8"},
            "dependencies": {"torch": "placeholder"}}
CALIBRATION = {5000: -8.0, 5001: -2.5, 5002: -0.25, 5003: 1.0}


def stored_response(pos, delta=-2.5, error_code=None):
    request = ScoreRequest(chrom="17", pos=pos, ref="A", alt="G", genome_build="GRCh37")
    common = {"variant_id": request.variant_id, "request": request, "session_id": "session-1", "metadata_sha256": "d" * 64}
    if error_code:
        return ScoreResponse(**common, status="error", error="backend failed", error_code=error_code, backend_seconds=1)
    context = ContextRecord(start=pos - 4096, end=pos + 4095, ref_sha256="a" * 64, alt_sha256="b" * 64)
    return ScoreResponse(**common, status="ok", context=context, ref_total=-100.0, alt_total=-100.0 + delta,
                         delta=delta, ref_seconds=1, alt_seconds=1, backend_seconds=2)


def row(response, **overrides):
    return {"variant_id": response.variant_id, "status": response.status,
            "response": response.model_dump(mode="json"), **overrides}


def write_run(root, name, rows, *, kind="split", identity=IDENTITY, record=True):
    run = root / name
    run.mkdir(parents=True)
    if record:
        meta = {"backend_metadata": identity} if kind == "pilot" else {"sessions": [{"metadata": None}, {"metadata": identity}]}
        (run / "run_record.json").write_text(json.dumps(meta), encoding="utf-8")
    rows = [r if isinstance(r, dict) else row(r) for r in rows]
    (run / "outcomes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return run


def write_calibration(positions):
    path = config.splits_dir() / "calibration.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"GRCh37-17-{pos}-A-G\n" for pos in positions), encoding="ascii")


def lookup(tmp_path, monkeypatch, pos, directory=None):
    monkeypatch.setenv("EVO2_SCORES_DIR", str(directory or tmp_path))
    return evo2.log_likelihood_delta(placeholder_variant(pos=pos))


@pytest.mark.parametrize("point_at_run", [False, True])
def test_evo2_reads_stored_sum_mean_and_calibration_percentile(tmp_path, monkeypatch, point_at_run):
    run = write_run(tmp_path, "20260101T000000Z-calibration-aaaaaaaaaaaa",
                    [stored_response(pos, delta) for pos, delta in CALIBRATION.items()])
    with (run / "outcomes.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"variant_id": "torn')  # an interrupted writer's partial line is ignored
    write_calibration(CALIBRATION)
    result = lookup(tmp_path, monkeypatch, 5001, run if point_at_run else None)
    assert result.status == "ok"
    values = result.values
    assert values["delta"] == -2.5 and values["delta_mean"] == -2.5 / 8191
    assert values["ref_total"] == -100.0 and values["alt_total"] == -102.5
    assert values["next_token_terms"] == 8191 and values["context_length"] == 8192
    assert values["score_source"] == "stored"
    assert values["calibration_percentile"] == 37.5 and values["n_calibration_reference"] == 4
    assert run.name in result.sources[0]


def test_evo2_reference_uses_all_calibration_results_and_nothing_else(tmp_path, monkeypatch):
    write_run(tmp_path, "20260101T000000Z-calibration-aaaaaaaaaaaa",
              [stored_response(5000, -8.0), stored_response(5001, -2.5), stored_response(5002, error_code="context")])
    write_run(tmp_path, "20260101T000001Z-test-bbbbbbbbbbbb", [stored_response(6000, -100.0)])
    write_calibration([5000, 5001, 5002])
    result = lookup(tmp_path, monkeypatch, 5001)
    # The final error at 5002 counts as covered but has no delta; the test-split 6000 is left out.
    assert result.values["n_calibration_reference"] == 2 and result.values["calibration_percentile"] == 75.0


def test_evo2_gives_no_percentile_from_a_partial_calibration_set(tmp_path, monkeypatch):
    write_run(tmp_path, "20260101T000000Z-pilot20-aaaaaaaaaaaa",
              [stored_response(5000, -8.0), stored_response(5001, -2.5)], kind="pilot")
    write_calibration(CALIBRATION)
    result = lookup(tmp_path, monkeypatch, 5001)
    assert result.status == "ok" and result.values["delta"] == -2.5
    assert result.values["calibration_percentile"] is None and result.values["n_calibration_reference"] == 0
    assert "only 2 of 4" in result.message


NOT_FINAL = [
    lambda: row(stored_response(5001), status="error", failure="client_fatal"),  # score_split: session changed
    lambda: row(stored_response(5001), status="error"),  # pilot: a fatal client error that kept the response
    lambda: row(stored_response(5001, error_code="context").model_copy(update={"error_code": "metadata_drift"})),
    lambda: {"variant_id": "GRCh37-17-5001-A-G", "status": "interrupted", "failure": "interrupted", "response": None},
    lambda: {"variant_id": "GRCh37-17-5001-A-G", "status": "unattempted", "response": None},
]


@pytest.mark.parametrize("stored", NOT_FINAL)
def test_evo2_ignores_attempts_that_are_not_final(tmp_path, monkeypatch, stored):
    write_run(tmp_path, "20260101T000000Z-calibration-aaaaaaaaaaaa", [stored()])
    result = lookup(tmp_path, monkeypatch, 5001)
    assert result.status == "unavailable" and result.values == {}


def test_evo2_newest_stored_run_wins_and_a_superseded_setup_is_ignored(tmp_path, monkeypatch):
    other = {**IDENTITY, "precision": {"projection_mode": "bf16"}}
    write_run(tmp_path, "20260101T000000Z-pilot20-aaaaaaaaaaaa", [stored_response(5001, -8.0)], kind="pilot", identity=other)
    write_run(tmp_path, "20260102T000000Z-calibration-bbbbbbbbbbbb", [stored_response(5001, -2.5)])
    write_run(tmp_path, "zz-hand-copied", [stored_response(5001, -50.0)])  # not a runner's name, so not scanned
    result = lookup(tmp_path, monkeypatch, 5001)
    assert result.status == "ok" and result.values["delta"] == -2.5 and "bbbbbbbbbbbb" in result.sources[0]


def test_evo2_refuses_stored_runs_from_different_model_setups(tmp_path, monkeypatch):
    write_run(tmp_path, "20260101T000000Z-calibration-aaaaaaaaaaaa", [stored_response(5000)])
    other = {**IDENTITY, "precision": {"projection_mode": "bf16"}}
    write_run(tmp_path, "20260102T000000Z-test-bbbbbbbbbbbb", [stored_response(5001)], identity=other)
    result = lookup(tmp_path, monkeypatch, 5000)
    assert result.status == "error" and result.values == {} and "different model setups" in result.message


def test_evo2_stored_backend_error_is_an_error(tmp_path, monkeypatch):
    write_run(tmp_path, "20260101T000000Z-calibration-aaaaaaaaaaaa", [stored_response(5000, error_code="context")])
    result = lookup(tmp_path, monkeypatch, 5000)
    assert result.status == "error" and result.values == {} and "context" in result.message


@pytest.mark.parametrize("damage,message", [
    ("no_record", "no backend identity"),
    ("bad_delta", "not a valid Evo 2 result"),
    ("bad_protocol", "not a valid Evo 2 result"),
    ("corrupt_line", "not a valid Evo 2 result"),
])
def test_evo2_damaged_stored_results_give_an_error(tmp_path, monkeypatch, damage, message):
    good = row(stored_response(5000))
    rows = [good]
    if damage == "bad_delta":
        rows = [{**good, "response": {**good["response"], "delta": -3.0}}]
    elif damage == "bad_protocol":
        rows = [{**good, "response": {**good["response"], "protocol": {**good["response"]["protocol"], "next_token_terms": 1}}}]
    run = write_run(tmp_path, "20260101T000000Z-calibration-aaaaaaaaaaaa", rows, record=damage != "no_record")
    if damage == "corrupt_line":
        (run / "outcomes.jsonl").write_text("not json\n" + json.dumps(good) + "\n", encoding="utf-8")
    result = lookup(tmp_path, monkeypatch, 5000)
    assert result.status == "error" and result.values == {} and message in result.message


@pytest.mark.parametrize("same_setup", [True, False])
def test_evo2_falls_back_to_backend_and_ranks_only_against_the_same_setup(tmp_path, monkeypatch, same_setup):
    from test_scoring_service import service

    instance = service()
    write_run(tmp_path, "20260101T000000Z-calibration-aaaaaaaaaaaa",
              [stored_response(pos, delta) for pos, delta in CALIBRATION.items()],
              identity=instance.metadata.model_dump(mode="json") if same_setup else IDENTITY)
    write_calibration(CALIBRATION)
    missing = placeholder_variant(pos=4097, alt="C")
    monkeypatch.setenv("EVO2_SCORES_DIR", str(tmp_path))
    result = evo2.log_likelihood_delta(missing)
    assert result.status == "unavailable"
    assert "EVO2_SCORES_DIR" in result.message and "EVO2_BACKEND_URL" in result.message and "FP8" in result.message
    monkeypatch.setenv("EVO2_BACKEND_URL", "http://placeholder.invalid")
    monkeypatch.setattr(evo2.ScoringClient, "metadata", lambda self: instance.metadata)
    monkeypatch.setattr(evo2.ScoringClient, "score", lambda self, request: instance.score(request))
    result = evo2.log_likelihood_delta(missing)
    assert result.status == "ok" and result.values["score_source"] == "backend" and result.values["delta"] == -2
    if same_setup:
        assert result.values["n_calibration_reference"] == 4 and result.values["calibration_percentile"] is not None
    else:
        assert result.values["calibration_percentile"] is None and "differs" in result.message


def test_evo2_reads_no_label_files(tmp_path, monkeypatch):
    run = write_run(tmp_path, "20260101T000000Z-pilot20-aaaaaaaaaaaa", [stored_response(5001, -2.5)], kind="pilot")
    poisoned = {"variant_id": "GRCh37-17-5001-A-G", "findlay_function_class": "LOF", "findlay_score": -2.0,
                "status": "ok", "response": {"delta": 999.0}}
    for name in ("evaluation.jsonl", "metrics.json", "selection.json", "split_meta.json", "report.md"):
        (run / name).write_text(json.dumps(poisoned) + "\n", encoding="utf-8")
    write_calibration([5001])
    monkeypatch.setenv("EVO2_SCORES_DIR", str(tmp_path))
    evo2._cached.clear()
    read = []

    def recorder(original):
        def record(self, *args, **kwargs):
            read.append(pathlib.Path(self).name)
            return original(self, *args, **kwargs)
        return record

    real_open = builtins.open

    def recording_open(file, *args, **kwargs):
        read.append(pathlib.Path(file).name if isinstance(file, (str, pathlib.Path)) else str(file))
        return real_open(file, *args, **kwargs)

    with monkeypatch.context() as patch:
        for name in ("read_text", "read_bytes", "open"):
            patch.setattr(pathlib.Path, name, recorder(getattr(pathlib.Path, name)))
        patch.setattr(builtins, "open", recording_open)
        result = evo2.log_likelihood_delta(placeholder_variant(pos=5001))
    assert result.values["delta"] == -2.5
    assert set(read) <= {"outcomes.jsonl", "run_record.json", "calibration.txt"}
    assert "findlay" not in json.dumps(result.model_dump())
