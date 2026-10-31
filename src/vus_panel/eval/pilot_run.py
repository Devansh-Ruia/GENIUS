"""Run a frozen selection once and persist label-free outcomes before evaluation."""

from __future__ import annotations

import json
import math
import os
import shlex
import statistics
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from vus_panel import config
from vus_panel.eval.metrics import auroc
from vus_panel.eval.pilot_data import (
    FUNCTION_CLASSES, digest_bytes, json_bytes, prepare, read_json, sha256_file,
    utc_now, write_json,
)
from vus_panel.schemas import parse_variant_id
from vus_panel.scoring.client import ScoringClient, ScoringClientError
from vus_panel.scoring.contracts import Protocol, ScoreRequest


def _git(*args: str) -> str | None:
    try:
        completed = subprocess.run(["git", *args], cwd=config.REPO_ROOT, capture_output=True, text=True, check=True)
        return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def code_state(directory: Path) -> dict[str, Any]:
    """Save the exact runtime Python files, including uncommitted implementation."""
    files = sorted((config.REPO_ROOT / "src").rglob("*.py")) + [config.REPO_ROOT / "pyproject.toml"]
    hashes = {}
    for source in files:
        relative = source.relative_to(config.REPO_ROOT)
        contents = source.read_bytes()
        target = directory / "code" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents)
        hashes[relative.as_posix()] = digest_bytes(contents)
    status = _git("status", "--porcelain")
    return {"git_commit": _git("rev-parse", "HEAD"), "git_branch": _git("branch", "--show-current"),
            "working_tree_status": status, "dirty": bool(status) if status is not None else None,
            "runtime_file_sha256": hashes, "snapshot_directory": "code"}


def _save_outcomes(path: Path, outcomes: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    contents = "".join(json.dumps(outcome, sort_keys=True, allow_nan=False) + "\n" for outcome in outcomes)
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(contents)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _summary(values: list[float]) -> dict[str, Any]:
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            "median": statistics.median(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def evaluate_outcomes(outcomes: list[dict[str, Any]], labels: dict[str, Any], *,
                      calibration_count: int, metadata: dict[str, Any] | None,
                      elapsed_seconds: float, billing_rate: float | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join only locally, after outcomes.jsonl has been written and reread."""
    joined = []
    for outcome in outcomes:
        variant = labels[outcome["variant_id"]]
        joined.append({**outcome, "findlay_function_class": variant.findlay_function_class,
                       "findlay_score": variant.findlay_score})
    successes = [row for row in joined if row["status"] == "ok"]
    y = [int(row["findlay_function_class"] == "LOF") for row in successes]
    scores = [-row["response"]["delta"] for row in successes]
    auc = auroc(y, scores) if y and 0 < sum(y) < len(y) else None
    backend = [row["response"]["backend_seconds"] for row in successes]
    wall = [row["client_wall_seconds"] for row in successes]
    remaining = calibration_count - len(outcomes)
    backend_estimate = statistics.mean(backend) * remaining if backend else None
    wall_estimate = statistics.mean(wall) * remaining if wall else None
    startup = {"model_load_seconds": metadata["load_seconds"] if metadata else None,
               "warmup_seconds": metadata["warmup"]["backend_seconds"] if metadata else None}
    metrics = {
        "schema_version": 1, "n_selected": len(joined), "n_successful": len(successes),
        "coverage": len(successes) / len(joined) if joined else 0.0,
        "outcome_counts": dict(sorted(Counter(row["status"] for row in joined).items())),
        "n_partial": sum(row["status"] != "ok" and row["response"] is not None and row["response"].get("ref_total") is not None for row in joined),
        "lof_positive_auroc": auc, "score_direction": "negative ALT-minus-REF delta", "descriptive_only": True,
        "per_class": {
            function_class: {"n_selected": sum(row["findlay_function_class"] == function_class for row in joined),
                             "delta": _summary([row["response"]["delta"] for row in successes if row["findlay_function_class"] == function_class])}
            for function_class in FUNCTION_CLASSES
        },
        "timing": {"client_run_elapsed_seconds": elapsed_seconds, "startup": startup,
                   "successful_backend_seconds": _summary(backend), "successful_client_wall_seconds": _summary(wall),
                   "remaining_calibration_variants": remaining,
                   "estimated_remaining_backend_seconds": backend_estimate,
                   "estimated_remaining_client_seconds": wall_estimate,
                   "estimate_basis": "mean successful per-variant time times remaining calibration count; excludes startup, failures and retries"},
        "billing": None,
    }
    if billing_rate is not None:
        metrics["billing"] = {"actual_rate_usd_per_hour": billing_rate,
                              "estimated_remaining_scoring_usd": backend_estimate * billing_rate / 3600 if backend_estimate is not None else None,
                              "estimated_startup_usd": sum(startup.values()) * billing_rate / 3600 if metadata else None}
    return joined, metrics


def render_report(metrics: dict[str, Any], record: dict[str, Any]) -> str:
    auc = metrics["lof_positive_auroc"]
    if auc is None:
        separation = "Separation cannot be estimated because successful scores do not include both LOF and non-LOF variants."
    elif auc <= 0.6:
        separation = f"Weak separation in this pilot: descriptive LOF-positive AUROC is {auc:.3f} (at or below the stated 0.60 descriptive threshold)."
    else:
        separation = f"Descriptive LOF-positive AUROC is {auc:.3f}. This selected calibration sample is too small for a generalization claim."
    timing = metrics["timing"]
    load_time = timing["startup"]["model_load_seconds"]
    warmup_time = timing["startup"]["warmup_seconds"]
    load_text = f"{load_time:.3f} seconds" if load_time is not None else "unavailable"
    warmup_text = f"{warmup_time:.3f} seconds" if warmup_time is not None else "unavailable"
    lines = ["# Findlay/Evo 2 timing pilot", "",
             f"Run: `{record['run_id']}`. State: {record['state']}.", "",
             f"Successful scores: {metrics['n_successful']}/{metrics['n_selected']}; coverage {metrics['coverage']:.1%}. Partial responses: {metrics['n_partial']}.", "",
             separation, "", "LOF is positive; FUNC and INT are negative. The discrimination score is negative ALT-minus-REF delta.", "",
             "| Class | Selected | Scored | Mean delta | Sample SD |", "| --- | ---: | ---: | ---: | ---: |"]
    for function_class, item in metrics["per_class"].items():
        summary = item["delta"]
        mean = f"{summary['mean']:.6f}" if summary["mean"] is not None else "unavailable"
        sd = f"{summary['sample_sd']:.6f}" if summary["sample_sd"] is not None else "unavailable"
        lines.append(f"| {function_class} | {item['n_selected']} | {summary['n']} | {mean} | {sd} |")
    lines.extend(["", f"Measured client run elapsed time: {timing['client_run_elapsed_seconds']:.3f} seconds.",
                  f"Model load: {load_text}. Fixed synthetic warm-up: {warmup_text}.", ""])
    if timing["estimated_remaining_backend_seconds"] is None:
        lines.append(f"No successful timing observations are available to estimate the remaining {timing['remaining_calibration_variants']} calibration variants.")
    else:
        lines.append(f"Estimated remaining {timing['remaining_calibration_variants']} calibration variants: {timing['estimated_remaining_backend_seconds']:.1f} backend seconds; {timing['estimated_remaining_client_seconds']:.1f} client wall seconds. Startup is separate. Estimates use only successful observations and exclude failed attempts.")
    if metrics["billing"] is None:
        lines.append("Dollar estimates are unavailable because no actual billing rate was supplied.")
    else:
        lines.append(f"Actual billing rate supplied: ${metrics['billing']['actual_rate_usd_per_hour']}/hour. Estimated remaining scoring cost: {metrics['billing']['estimated_remaining_scoring_usd']} USD; startup: {metrics['billing']['estimated_startup_usd']} USD.")
    lines.extend(["", "Selection was frozen before scoring. Every selected ID is retained; there are no retries or replacement variants. Synthetic warm-up is excluded from pilot outcomes.",
                  "This is a descriptive timing and discrimination pilot on selected calibration data. It supplies no calibrated confidence, drug-response prediction, or panel-superiority result.", "",
                  f"Source SHA256: `{record['source']['sha256']}`.",
                  f"Reference sequence SHA256: `{record['reference']['sequence_sha256']}`.",
                  f"Selection SHA256: `{record['selection_sha256']}`.",
                  "Trace: `outcomes.jsonl`, `evaluation.jsonl`, `metrics.json`, `run_record.json`, and the saved `code/` snapshot.", "",
                  "Reproduction command:", "```", record["reproduction_command"], "```", ""])
    if record.get("error"):
        lines.extend([f"Run error: {record['error']}", ""])
    return "\n".join(lines)


def run_pilot(selection_path: Path, *, backend_url: str | None = None, timeout: float = 600.0,
              billing_rate: float | None = None) -> tuple[Path, bool]:
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    if billing_rate is not None and (not math.isfinite(billing_rate) or billing_rate < 0):
        raise ValueError("billing rate must be finite and nonnegative")
    endpoint = backend_url or config.evo2_backend_url()
    if endpoint:
        parsed = urlsplit(endpoint)
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("backend URL must not contain credentials, query parameters, or fragments")
    selection_path = Path(selection_path)
    selection, variants = prepare(require_existing=True, selection_path=selection_path)
    split_meta_path = selection_path.parent / "split_meta.json"
    split_meta = read_json(split_meta_path)
    selection_bytes = selection_path.read_bytes()
    split_meta_bytes = split_meta_path.read_bytes()
    validation_bytes = (config.cache_dir() / "pilot" / "validation.json").read_bytes()
    run_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-pilot20-{uuid.uuid4().hex[:12]}"
    directory = config.runs_dir() / run_id
    started = time.perf_counter()
    command = [sys.executable, "-m", "vus_panel.eval.pilot", "run", "--selection", str(selection_path), "--timeout", str(timeout)]
    if endpoint:
        command.extend(["--backend", endpoint])
    if billing_rate is not None:
        command.extend(["--billing-rate-usd-per-hour", str(billing_rate)])
    record = {
        "schema_version": 1, "run_id": run_id, "started_at": utc_now(), "finished_at": None,
        "state": "initializing", "error": None, "seed": 0, "protocol": Protocol().model_dump(mode="json"),
        "source": split_meta["source"], "reference": split_meta["reference"],
        "selection_sha256": digest_bytes(selection_bytes), "split_meta_sha256": digest_bytes(split_meta_bytes),
        "variant_ids": selection["variant_ids"], "code": None,
        "reproduction_argv": command,
        "reproduction_shell": "PowerShell" if os.name == "nt" else "POSIX sh",
        "reproduction_working_directory": str(config.REPO_ROOT),
        "reproduction_command": ("& " + " ".join("'" + value.replace("'", "''") + "'" for value in command)
                                 if os.name == "nt" else shlex.join(command)),
        "environment": {"VUS_PANEL_DATA_DIR": str(config.data_dir()), "VUS_PANEL_RUNS_DIR": str(config.runs_dir()),
                        "EVO2_BACKEND_URL": endpoint},
        "backend_metadata": None, "metadata_client_wall_seconds": None,
        "no_automatic_retries": True, "no_replacement_variants": True,
        "billing_rate_usd_per_hour": billing_rate,
    }
    outcomes = [{"variant_id": vid, "status": "unattempted", "started_at": None, "finished_at": None,
                 "client_wall_seconds": None, "response": None, "error": None} for vid in selection["variant_ids"]]
    output_path = directory / "outcomes.jsonl"
    metadata = None
    directory.mkdir(parents=True, exist_ok=False)
    try:
        # Initialize every outcome before potentially slow source snapshots or
        # backend startup. Startup interruptions finalize the same placeholders.
        _save_outcomes(output_path, outcomes)
        write_json(directory / "run_record.json", record)
        (directory / "selection.json").write_bytes(selection_bytes)
        (directory / "split_meta.json").write_bytes(split_meta_bytes)
        (directory / "validation.json").write_bytes(validation_bytes)
        record["code"] = code_state(directory)
        write_json(directory / "run_record.json", record)
        if not endpoint:
            raise ValueError("EVO2_BACKEND_URL is not set; no authenticated scoring backend is configured")
        client = ScoringClient(endpoint, timeout=timeout)
        metadata_started = time.perf_counter()
        metadata = client.metadata()
        record["metadata_client_wall_seconds"] = time.perf_counter() - metadata_started
        if (metadata.reference_sequence_sha256 != split_meta["reference"]["sequence_sha256"]
                or metadata.reference_file_sha256 != split_meta["reference"]["decompressed_file_sha256"]):
            raise ValueError("backend reference checksum does not match the frozen reference")
        record["backend_metadata"] = metadata.model_dump(mode="json")
        record["state"] = "running"
        write_json(directory / "run_record.json", record)
        for outcome in outcomes:
            request = ScoreRequest.from_variant(parse_variant_id(outcome["variant_id"]))
            outcome.update(status="interrupted", started_at=utc_now(), error="attempt began but no response was persisted")
            _save_outcomes(output_path, outcomes)
            call_started = time.perf_counter()
            fatal = False
            try:
                response = client.score(request)
                outcome.update(status=response.status, response=response.model_dump(mode="json"), error=response.error)
            except KeyboardInterrupt:
                outcome.update(status="interrupted", error="KeyboardInterrupt during scoring; server completion is unknown")
                record.update(state="interrupted", error=outcome["error"])
                fatal = True
            except ScoringClientError as exc:
                outcome.update(status="error", error=f"{type(exc).__name__}: {exc}")
                if exc.response is not None:
                    outcome["response"] = exc.response.model_dump(mode="json")
                fatal = exc.fatal
                if fatal:
                    record.update(state="aborted", error=outcome["error"])
            except Exception as exc:
                outcome.update(status="error", error=f"{type(exc).__name__}: {exc}")
                record.update(state="aborted", error=outcome["error"])
                fatal = True
            finally:
                outcome.update(finished_at=utc_now(), client_wall_seconds=time.perf_counter() - call_started)
                _save_outcomes(output_path, outcomes)
            if fatal:
                break
        if record["state"] == "running":
            record["state"] = "completed" if all(row["status"] == "ok" for row in outcomes) else "completed_with_errors"
    except KeyboardInterrupt:
        record.update(state="interrupted", error="KeyboardInterrupt before scoring")
    except Exception as exc:
        record.update(state="blocked", error=f"{type(exc).__name__}: {exc}")
    finally:
        for outcome in outcomes:
            if outcome["status"] == "unattempted":
                outcome["error"] = record["error"] or "run stopped before this selected variant"
        _save_outcomes(output_path, outcomes)
        # Read the durable label-free artifact before any answer-key join.
        saved = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
        joined, metrics = evaluate_outcomes(saved, {v.variant_id: v for v in variants},
                                           calibration_count=split_meta["n_calibration"],
                                           metadata=record["backend_metadata"],
                                           elapsed_seconds=time.perf_counter() - started, billing_rate=billing_rate)
        _save_outcomes(directory / "evaluation.jsonl", joined)
        write_json(directory / "metrics.json", metrics)
        record.update(finished_at=utc_now(), outcomes_sha256=sha256_file(output_path),
                      evaluation_sha256=sha256_file(directory / "evaluation.jsonl"),
                      metrics_sha256=sha256_file(directory / "metrics.json"))
        write_json(directory / "run_record.json", record)
        (directory / "report.md").write_text(render_report(metrics, record), encoding="utf-8")
    return directory, record["state"] == "completed"
