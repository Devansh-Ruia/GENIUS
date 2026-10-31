from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from test_pilot_data import pilot_inputs
from vus_panel import config
from vus_panel.eval import pilot_data as data
from vus_panel.eval import pilot_run as runner
from vus_panel.scoring.client import ScoringClientError
from vus_panel.scoring.contracts import ContextRecord, ScoreResponse, ServiceMetadata, WarmupRecord


def metadata():
    reference = data.read_json(config.splits_dir() / "split_meta.json")["reference"]
    warmup_ref = "ACGT" * 2048
    warmup_alt = warmup_ref[:4096] + "C" + warmup_ref[4097:]
    return ServiceMetadata(
        session_id="offline-test-session", created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        reference_sequence_sha256=reference["sequence_sha256"],
        reference_file_sha256=reference["decompressed_file_sha256"],
        precision={"parameter_dtypes": ["torch.bfloat16"], "logit_dtype": "torch.bfloat16",
                   "use_fp8_input_projections": True, "projection_mode": "fp8",
                   "log_softmax_dtype": "torch.bfloat16", "reduction_dtype": "float32",
                   "effective_config_sha256": "a" * 64},
        dependencies={"torch": "offline-test"}, gpu=[{"name": "offline-test"}], load_seconds=10.0,
        warmup=WarmupRecord(ref_sha256=data.digest_bytes(warmup_ref.encode()),
                            alt_sha256=data.digest_bytes(warmup_alt.encode()), ref_total=-10,
                            alt_total=-11, ref_seconds=1, alt_seconds=1, backend_seconds=2),
    )


@pytest.fixture
def prepared(pilot_inputs, monkeypatch):
    selection, variants = data.prepare(download=True)
    monkeypatch.setattr(runner, "_git", lambda *args: "offline-test")
    return selection, variants


def install_client(monkeypatch, *, events=None, wrong_reference=False):
    calls = []
    frozen = metadata()
    if wrong_reference:
        frozen = frozen.model_copy(update={"reference_sequence_sha256": "f" * 64})

    class Client:
        def __init__(self, endpoint, timeout):
            self.endpoint = endpoint

        def metadata(self):
            return frozen

        def score(self, request):
            payload = request.model_dump()
            assert set(payload) == {"chrom", "pos", "ref", "alt", "genome_build"}
            assert "findlay" not in json.dumps(payload)
            calls.append(request.variant_id)
            index = len(calls) - 1
            context = ContextRecord(start=request.pos - 4096, end=request.pos + 4095,
                                    ref_sha256="a" * 64, alt_sha256="b" * 64)
            response = ScoreResponse(variant_id=request.variant_id, request=request,
                                     session_id=frozen.session_id, metadata_sha256=frozen.sha256,
                                     status="ok", context=context, ref_total=-100, alt_total=-101,
                                     delta=-1, ref_seconds=1, alt_seconds=2, backend_seconds=3)
            event = events.get(index) if events else None
            if event == "partial":
                return response.model_copy(update={"status": "error", "alt_total": None, "delta": None,
                                                   "alt_seconds": None, "error": "ALT inference failed"})
            if event == "interrupt":
                raise KeyboardInterrupt()
            if event == "fatal":
                raise ScoringClientError("metadata changed", fatal=True, response=response)
            if event == "transport":
                raise ScoringClientError("connection failed")
            return response

    monkeypatch.setattr(runner, "ScoringClient", Client)
    return calls


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_success_saves_raw_before_labels_and_reports_real_observed_times(prepared, monkeypatch):
    selection, _ = prepared
    calls = install_client(monkeypatch)
    original_evaluate = runner.evaluate_outcomes

    def evaluate(outcomes, labels, **kwargs):
        run = next(config.runs_dir().iterdir())
        assert read_lines(run / "outcomes.jsonl") == outcomes
        assert not (run / "evaluation.jsonl").exists()
        assert "findlay_function_class" not in (run / "outcomes.jsonl").read_text()
        return original_evaluate(outcomes, labels, **kwargs)

    monkeypatch.setattr(runner, "evaluate_outcomes", evaluate)
    directory, complete = runner.run_pilot(config.splits_dir() / "pilot20.json", backend_url="http://offline.test")
    assert complete
    assert calls == selection["variant_ids"]
    assert len(read_lines(directory / "outcomes.jsonl")) == 20
    assert all("findlay_score" in row for row in read_lines(directory / "evaluation.jsonl"))
    metrics = data.read_json(directory / "metrics.json")
    assert metrics["coverage"] == 1 and metrics["lof_positive_auroc"] == 0.5
    assert metrics["timing"]["successful_backend_seconds"]["mean"] == 3
    assert metrics["timing"]["remaining_calibration_variants"] == 1
    assert metrics["timing"]["estimated_remaining_backend_seconds"] == 3
    assert metrics["timing"]["startup"] == {"model_load_seconds": 10, "warmup_seconds": 2}
    assert metrics["billing"] is None
    assert "Weak separation" in (directory / "report.md").read_text()
    record = data.read_json(directory / "run_record.json")
    assert record["outcomes_sha256"] == data.sha256_file(directory / "outcomes.jsonl")
    assert record["metrics_sha256"] == data.sha256_file(directory / "metrics.json")
    assert (directory / "code" / "src" / "vus_panel" / "eval" / "pilot_run.py").is_file()


def test_partial_failure_and_transport_are_retained_without_retry(prepared, monkeypatch):
    selection, _ = prepared
    calls = install_client(monkeypatch, events={0: "partial", 1: "transport"})
    directory, complete = runner.run_pilot(config.splits_dir() / "pilot20.json", backend_url="http://offline.test")
    assert not complete and calls == selection["variant_ids"]
    outcomes = read_lines(directory / "outcomes.jsonl")
    assert outcomes[0]["response"]["ref_total"] == -100
    assert outcomes[0]["response"]["alt_total"] is None
    assert outcomes[1]["response"] is None
    metrics = data.read_json(directory / "metrics.json")
    assert metrics["n_partial"] == 1 and metrics["n_successful"] == 18
    assert metrics["coverage"] == 0.9


@pytest.mark.parametrize("event,status", [("interrupt", "interrupted"), ("fatal", "aborted")])
def test_interruption_or_metadata_drift_retains_every_id_and_stops(prepared, monkeypatch, event, status):
    selection, _ = prepared
    calls = install_client(monkeypatch, events={1: event})
    directory, complete = runner.run_pilot(config.splits_dir() / "pilot20.json", backend_url="http://offline.test")
    assert not complete and calls == selection["variant_ids"][:2]
    outcomes = read_lines(directory / "outcomes.jsonl")
    assert [row["variant_id"] for row in outcomes] == selection["variant_ids"]
    assert outcomes[0]["status"] == "ok"
    assert all(row["status"] == "unattempted" for row in outcomes[2:])
    if event == "fatal":
        assert outcomes[1]["response"]["ref_total"] == -100
        assert outcomes[1]["status"] == "error"
    assert data.read_json(directory / "run_record.json")["state"] == status


def test_missing_backend_writes_no_mock_scores_and_keeps_all_ids(prepared):
    directory, complete = runner.run_pilot(config.splits_dir() / "pilot20.json")
    assert not complete
    outcomes = read_lines(directory / "outcomes.jsonl")
    assert len(outcomes) == 20 and all(row["status"] == "unattempted" for row in outcomes)
    assert all(row["response"] is None for row in outcomes)
    metrics = data.read_json(directory / "metrics.json")
    assert metrics["lof_positive_auroc"] is None
    assert metrics["timing"]["estimated_remaining_backend_seconds"] is None
    assert "EVO2_BACKEND_URL" in data.read_json(directory / "run_record.json")["error"]


def test_startup_interruption_retains_all_placeholders_and_finalizes_report(prepared, monkeypatch):
    selection, _ = prepared

    def interrupt(directory):
        assert [row["variant_id"] for row in read_lines(directory / "outcomes.jsonl")] == selection["variant_ids"]
        raise KeyboardInterrupt()

    monkeypatch.setattr(runner, "code_state", interrupt)
    directory, complete = runner.run_pilot(config.splits_dir() / "pilot20.json")
    assert not complete
    assert len(read_lines(directory / "outcomes.jsonl")) == 20
    assert all(row["status"] == "unattempted" for row in read_lines(directory / "outcomes.jsonl"))
    assert data.read_json(directory / "run_record.json")["state"] == "interrupted"
    assert (directory / "report.md").is_file()


def test_reference_metadata_mismatch_prevents_scoring(prepared, monkeypatch):
    calls = install_client(monkeypatch, wrong_reference=True)
    directory, complete = runner.run_pilot(config.splits_dir() / "pilot20.json", backend_url="http://offline.test")
    assert not complete and calls == []
    assert data.read_json(directory / "run_record.json")["state"] == "blocked"


def test_only_actual_provided_rate_produces_cost_estimate(prepared, monkeypatch):
    install_client(monkeypatch)
    directory, _ = runner.run_pilot(config.splits_dir() / "pilot20.json", backend_url="http://offline.test", billing_rate=3.6)
    billing = data.read_json(directory / "metrics.json")["billing"]
    assert billing["estimated_remaining_scoring_usd"] == pytest.approx(0.003)
    assert billing["estimated_startup_usd"] == pytest.approx(0.012)


@pytest.mark.parametrize("kwargs", [{"timeout": float("nan")}, {"timeout": 0}, {"billing_rate": float("inf")},
                                  {"backend_url": "https://user:secret@example.test"}])
def test_invalid_runtime_options_do_not_create_run(prepared, kwargs):
    with pytest.raises(ValueError):
        runner.run_pilot(config.splits_dir() / "pilot20.json", **kwargs)
    assert not config.runs_dir().exists()


def test_repeated_runs_create_distinct_directories(prepared):
    first, _ = runner.run_pilot(config.splits_dir() / "pilot20.json")
    second, _ = runner.run_pilot(config.splits_dir() / "pilot20.json")
    assert first != second and first.is_dir() and second.is_dir()
