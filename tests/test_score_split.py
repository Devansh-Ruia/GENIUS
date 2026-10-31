from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from test_pilot_data import pilot_inputs
from test_pilot_run import metadata
from vus_panel import config
from vus_panel.eval import pilot_data as data
from vus_panel.eval import pilot_run
from vus_panel.eval import score_split
from vus_panel.scoring.client import ScoringClientError
from vus_panel.scoring.contracts import ContextRecord, ScoreResponse

BACKEND = "http://offline.test"


@pytest.fixture
def prepared(pilot_inputs, monkeypatch):
    data.prepare(download=True)
    monkeypatch.setattr(pilot_run, "_git", lambda *args: "offline-test")


def split_ids(split):
    return (config.splits_dir() / f"{split}.txt").read_text(encoding="ascii").splitlines()


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def install_client(monkeypatch, *, events=None, session_id="session-1", update=None, on_call=None,
                   metadata_error=False):
    calls = []
    frozen = metadata().model_copy(update={"session_id": session_id, **(update or {})})

    class Client:
        def __init__(self, endpoint, timeout):
            self.endpoint = endpoint

        def metadata(self):
            if metadata_error:
                raise ScoringClientError("scoring backend request failed: URLError")
            return frozen

        def score(self, request):
            payload = request.model_dump()
            assert set(payload) == {"chrom", "pos", "ref", "alt", "genome_build"}
            calls.append(request.variant_id)
            index = len(calls) - 1
            if on_call:
                on_call(index, request.variant_id)
            context = ContextRecord(start=request.pos - 4096, end=request.pos + 4095,
                                    ref_sha256="a" * 64, alt_sha256="b" * 64)
            response = ScoreResponse(variant_id=request.variant_id, request=request,
                                     session_id=frozen.session_id, metadata_sha256=frozen.sha256,
                                     status="ok", context=context, ref_total=-100, alt_total=-101,
                                     delta=-1, ref_seconds=1, alt_seconds=2, backend_seconds=3)
            failed = {"status": "error", "context": None, "ref_total": None, "alt_total": None, "delta": None,
                      "ref_seconds": None, "alt_seconds": None, "error": "backend failed"}
            event = (events or {}).get(index)
            if event == "interrupt":
                raise KeyboardInterrupt()
            if event == "transport":
                raise ScoringClientError("connection failed")
            if event == "fatal":
                raise ScoringClientError("backend metadata or service session changed", fatal=True)
            if event in ("context", "inference"):
                return response.model_copy(update={**failed, "error_code": event})
            if event == "nocode":
                return response.model_copy(update=failed)
            return response

    monkeypatch.setattr(score_split, "ScoringClient", Client)
    return calls


def only_run(split):
    runs = sorted(config.runs_dir().glob(f"*-{split}-*"))
    assert len(runs) == 1
    return runs[0]


def no_labels(text):
    return "findlay_function_class" not in text and "findlay_score" not in text


def test_calibration_appends_each_result_and_joins_labels_only_after_reread(prepared, monkeypatch):
    ids = split_ids("calibration")

    def check(index, vid):
        run = only_run("calibration")
        assert len(read_lines(run / "outcomes.jsonl")) == index
        assert data.read_json(run / "in_flight.json")["variant_id"] == vid

    original_evaluate = pilot_run.evaluate_outcomes

    def evaluate(outcomes, labels, **kwargs):
        run = only_run("calibration")
        assert not (run / "evaluation.jsonl").exists()
        saved = read_lines(run / "outcomes.jsonl")
        assert [row for row in outcomes if row["status"] == "ok"] == [row for row in saved if row["status"] == "ok"]
        assert no_labels((run / "outcomes.jsonl").read_text(encoding="utf-8"))
        return original_evaluate(outcomes, labels, **kwargs)

    monkeypatch.setattr(pilot_run, "evaluate_outcomes", evaluate)
    calls = install_client(monkeypatch, on_call=check)
    directory, complete = score_split.run_split("calibration", backend_url=BACKEND)
    assert complete and calls == ids
    assert [row["variant_id"] for row in read_lines(directory / "outcomes.jsonl")] == ids
    assert all("findlay_score" in row for row in read_lines(directory / "evaluation.jsonl"))
    record = data.read_json(directory / "run_record.json")
    assert record["state"] == "completed" and len(record["sessions"]) == 1 and record["seed"] == 0
    assert record["sessions"][0]["backend_url"] == BACKEND
    assert record["artifact_sha256"]["outcomes.jsonl"] == data.sha256_file(directory / "outcomes.jsonl")
    assert record["artifact_sha256"]["metrics.json"] == data.sha256_file(directory / "metrics.json")
    metrics = data.read_json(directory / "metrics.json")
    assert metrics["coverage"] == 1 and metrics["partial"] is False
    assert metrics["timing"]["remaining_calibration_variants"] == 0
    assert not (directory / "runner.lock").exists() and not (directory / "in_flight.json").exists()
    assert (directory / "code" / "src" / "vus_panel" / "eval" / "score_split.py").is_file()


def test_interrupted_run_resumes_in_a_new_session(prepared, monkeypatch):
    ids = split_ids("calibration")
    install_client(monkeypatch, events={5: "interrupt"})
    first, complete = score_split.run_split("calibration", backend_url=BACKEND)
    assert not complete
    assert data.read_json(first / "run_record.json")["state"] == "interrupted"
    timing = data.read_json(first / "metrics.json")["timing"]
    pending = len(ids) - 5
    assert timing["remaining_calibration_variants"] == pending
    assert timing["estimated_remaining_backend_seconds"] == 3 * pending
    calls = install_client(monkeypatch, session_id="session-2")
    second, complete = score_split.run_split("calibration", backend_url=BACKEND)
    assert second == first and complete
    assert calls == ids[5:]
    record = data.read_json(first / "run_record.json")
    assert [s["metadata"]["session_id"] for s in record["sessions"]] == ["session-1", "session-2"]
    retried = [row for row in read_lines(first / "outcomes.jsonl") if row["variant_id"] == ids[5]]
    assert [(row["attempt"], row["status"]) for row in retried] == [(1, "interrupted"), (2, "ok")]
    assert record["state"] == "completed"


def test_resume_accepts_a_session_that_differs_only_in_session_fields(prepared, monkeypatch):
    install_client(monkeypatch, events={2: "interrupt"})
    directory, _ = score_split.run_split("calibration", backend_url=BACKEND)
    warmup = metadata().warmup.model_copy(update={"ref_total": -20.0, "backend_seconds": 9.0})
    install_client(monkeypatch, session_id="session-2", update={
        "gpu": [{"name": "another-gpu"}], "load_seconds": 99.0, "warmup": warmup,
        "created_at": datetime(2027, 1, 1, tzinfo=timezone.utc)})
    _, complete = score_split.run_split("calibration", backend_url=BACKEND)
    assert complete
    assert len(data.read_json(directory / "run_record.json")["sessions"]) == 2


@pytest.mark.parametrize("update,message", [
    ({"precision": {**metadata().precision, "projection_mode": "bf16", "use_fp8_input_projections": False}}, "precision"),
    ({"dependencies": {"torch": "a-different-build"}}, "dependencies"),
    ({"checkpoint_sha256": "0" * 64}, "checkpoint_sha256"),
    ({"reference_sequence_sha256": "f" * 64}, "reference checksum"),
])
def test_resume_refuses_a_backend_with_different_identity(prepared, monkeypatch, update, message):
    install_client(monkeypatch, events={2: "interrupt"})
    directory, _ = score_split.run_split("calibration", backend_url=BACKEND)
    before = (directory / "outcomes.jsonl").read_bytes()
    calls = install_client(monkeypatch, session_id="session-2", update=update)
    with pytest.raises(ValueError, match=message):
        score_split.run_split("calibration", backend_url=BACKEND)
    assert calls == []
    assert (directory / "outcomes.jsonl").read_bytes() == before
    assert len(data.read_json(directory / "run_record.json")["sessions"]) == 1
    assert not (directory / "runner.lock").exists()


def refreeze():
    for name in data.FREEZE_FILES:
        (config.splits_dir() / name).unlink()
    data.prepare()


@pytest.mark.parametrize("change,message", [
    ("ids", "incompatible or corrupt freeze"),
    ("refreeze", "split_meta.json changed"),
    ("code", "runtime code differs"),
    ("commit", "runtime code differs"),
])
def test_resume_refuses_changed_selection_or_code(prepared, monkeypatch, change, message):
    install_client(monkeypatch, events={2: "interrupt"})
    score_split.run_split("calibration", backend_url=BACKEND)
    if change == "ids":
        path = config.splits_dir() / "calibration.txt"
        path.write_bytes(path.read_bytes() + b"GRCh37-17-9999-A-C\n")
    elif change == "refreeze":
        refreeze()
        # Same IDs, new split_meta.json: only the resume check can catch this.
        assert (config.splits_dir() / "calibration.txt").read_bytes() == (only_run("calibration") / "calibration.txt").read_bytes()
    elif change == "code":
        monkeypatch.setattr(score_split, "_runtime_hashes", lambda: {"src/vus_panel/x.py": "0" * 64})
    else:
        monkeypatch.setattr(pilot_run, "_git", lambda *args: "another-commit")
    calls = install_client(monkeypatch, session_id="session-2")
    with pytest.raises(ValueError, match=message):
        score_split.run_split("calibration", backend_url=BACKEND)
    assert calls == []


def test_torn_line_and_in_flight_attempt_are_recovered(prepared, monkeypatch):
    ids = split_ids("calibration")
    install_client(monkeypatch, events={3: "interrupt"})
    directory, _ = score_split.run_split("calibration", backend_url=BACKEND)
    torn = b'{"variant_id": "GRCh37'
    with (directory / "outcomes.jsonl").open("ab") as handle:
        handle.write(torn)
    data.write_json(directory / "in_flight.json", {"variant_id": ids[4], "attempt": 1, "session_id": "session-1",
                                                    "started_at": "2026-01-01T00:00:00+00:00"})
    calls = install_client(monkeypatch, session_id="session-2")
    _, complete = score_split.run_split("calibration", backend_url=BACKEND)
    assert complete and calls == ids[3:]
    repair = data.read_json(directory / "run_record.json")["repairs"][0]
    assert repair["dropped_torn_bytes"] == len(torn)
    assert (directory / repair["saved_to"]).read_bytes() == torn
    rows = read_lines(directory / "outcomes.jsonl")
    assert [(r["attempt"], r["status"]) for r in rows if r["variant_id"] == ids[4]] == [(1, "interrupted"), (2, "ok")]


def test_unreadable_in_flight_marker_is_recorded_and_resume_continues(prepared, monkeypatch):
    ids = split_ids("calibration")
    install_client(monkeypatch, events={3: "interrupt"})
    directory, _ = score_split.run_split("calibration", backend_url=BACKEND)
    (directory / "in_flight.json").write_bytes(b"")
    calls = install_client(monkeypatch, session_id="session-2")
    _, complete = score_split.run_split("calibration", backend_url=BACKEND)
    assert complete and calls == ids[3:]
    repairs = data.read_json(directory / "run_record.json")["repairs"]
    assert len(repairs) == 1 and repairs[0]["unreadable_in_flight_bytes"] == 0


@pytest.mark.parametrize("corruption,message", [
    ("unknown_session", "does not record"),
    ("duplicate_attempt", "repeats attempt"),
    ("outside_split", "not in the calibration split"),
])
def test_resume_refuses_outcomes_that_break_invariants(prepared, monkeypatch, corruption, message):
    install_client(monkeypatch, events={3: "interrupt"})
    directory, _ = score_split.run_split("calibration", backend_url=BACKEND)
    row = read_lines(directory / "outcomes.jsonl")[0]
    if corruption == "unknown_session":
        row = {**row, "attempt": 2, "session_id": "never-recorded"}
    elif corruption == "outside_split":
        row = {**row, "variant_id": split_ids("test")[0]}
    score_split._append(directory / "outcomes.jsonl", row)
    calls = install_client(monkeypatch, session_id="session-2")
    with pytest.raises(ValueError, match=message):
        score_split.run_split("calibration", backend_url=BACKEND)
    assert calls == []


@pytest.mark.parametrize("code", ["context", "inference"])
def test_backend_variant_error_is_final_and_transport_failure_is_retried(prepared, monkeypatch, code):
    ids = split_ids("calibration")
    calls = install_client(monkeypatch, events={1: code, 3: "transport"})
    directory, complete = score_split.run_split("calibration", backend_url=BACKEND)
    assert not complete and calls == ids[:4]
    assert data.read_json(directory / "run_record.json")["state"] == "backend_lost"
    calls = install_client(monkeypatch, session_id="session-2")
    _, complete = score_split.run_split("calibration", backend_url=BACKEND)
    assert complete and calls == ids[3:]
    assert data.read_json(directory / "run_record.json")["state"] == "completed_with_errors"
    metrics = data.read_json(directory / "metrics.json")
    assert metrics["n_successful"] == len(ids) - 1 and metrics["progress"]["n_final_error"] == 1


@pytest.mark.parametrize("event", ["nocode", "fatal"])
def test_non_variant_failures_abort_and_are_retried(prepared, monkeypatch, event):
    ids = split_ids("calibration")
    calls = install_client(monkeypatch, events={2: event})
    directory, complete = score_split.run_split("calibration", backend_url=BACKEND)
    assert not complete and calls == ids[:3]
    assert data.read_json(directory / "run_record.json")["state"] == "aborted"
    calls = install_client(monkeypatch, session_id="session-2")
    _, complete = score_split.run_split("calibration", backend_url=BACKEND)
    assert complete and calls == ids[2:]
    assert data.read_json(directory / "run_record.json")["state"] == "completed"


def test_unreachable_backend_blocks_then_resumes(prepared, monkeypatch):
    ids = split_ids("calibration")
    install_client(monkeypatch, metadata_error=True)
    directory, complete = score_split.run_split("calibration", backend_url=BACKEND)
    assert not complete
    record = data.read_json(directory / "run_record.json")
    assert record["state"] == "blocked" and record["sessions"][0]["metadata"] is None
    assert read_lines(directory / "outcomes.jsonl") == []
    calls = install_client(monkeypatch)
    _, complete = score_split.run_split("calibration", backend_url=BACKEND)
    assert complete and calls == ids
    assert len(data.read_json(directory / "run_record.json")["sessions"]) == 2


def test_test_split_writes_no_labels_or_discrimination(prepared, monkeypatch):
    ids = split_ids("test")
    calls = install_client(monkeypatch)
    directory, complete = score_split.run_split("test", backend_url=BACKEND)
    assert complete and calls == ids
    assert not (directory / "evaluation.jsonl").exists() and not (directory / "metrics.json").exists()
    assert data.read_json(directory / "progress.json")["n_ok"] == len(ids)
    for path in directory.rglob("*"):
        if path.is_file() and "code" not in path.relative_to(directory).parts:
            text = path.read_text(encoding="utf-8").lower()
            assert no_labels(text) and "auroc" not in text


def test_lock_blocks_a_second_runner_until_forced(prepared, monkeypatch):
    install_client(monkeypatch, events={2: "interrupt"})
    directory, _ = score_split.run_split("calibration", backend_url=BACKEND)
    (directory / "runner.lock").write_text('{"pid": 1}', encoding="utf-8")
    calls = install_client(monkeypatch, session_id="session-2")
    with pytest.raises(ValueError, match="force-unlock"):
        score_split.run_split("calibration", backend_url=BACKEND)
    assert calls == []
    _, complete = score_split.run_split("calibration", backend_url=BACKEND, force_unlock=True)
    assert complete
    assert len(data.read_json(directory / "run_record.json")["lock_overrides"]) == 1


def test_finished_run_needs_new_flag_and_status_reports_counts(prepared, monkeypatch, capsys):
    ids = split_ids("calibration")
    install_client(monkeypatch, events={3: "interrupt"})
    directory, _ = score_split.run_split("calibration", backend_url=BACKEND)
    line = score_split.status_lines("calibration")[0]
    assert f"ok=3/{len(ids)}" in line and f"pending={len(ids) - 3}" in line
    assert "state=interrupted" in line and "lock=free" in line
    assert score_split.main(["status"]) == 0
    assert "test: no run yet" in capsys.readouterr().out
    install_client(monkeypatch, session_id="session-2")
    assert score_split.main(["run", "--split", "calibration", "--backend", BACKEND]) == 0
    with pytest.raises(ValueError, match="--new"):
        score_split.run_split("calibration", backend_url=BACKEND)
    with pytest.raises(ValueError, match="do not combine"):
        score_split.run_split("calibration", backend_url=BACKEND, new=True, run_dir=directory)
    fresh, complete = score_split.run_split("calibration", backend_url=BACKEND, new=True)
    assert complete and fresh != directory


def test_refusals_before_any_run_is_created(prepared, monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="EVO2_BACKEND_URL"):
        score_split.run_split("calibration")
    assert score_split.main(["run", "--split", "test"]) == 2
    monkeypatch.setattr(pilot_run, "_git", lambda *args: None)
    with pytest.raises(ValueError, match="safe.directory"):
        score_split.run_split("calibration", backend_url=BACKEND)
    assert not config.runs_dir().exists()
    other = tmp_path / "not-a-split-run"
    other.mkdir()
    data.write_json(other / "run_record.json", {"run_id": "pilot"})
    assert score_split.main(["status", "--run-dir", str(other)]) == 2
