"""Score every variant in the frozen calibration or test split once, resumably.

Usage: python -m vus_panel.eval.score_split run --split calibration
       python -m vus_panel.eval.score_split run --split test
       python -m vus_panel.eval.score_split status
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import statistics
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from vus_panel import config
from vus_panel.eval import pilot_run
from vus_panel.eval.pilot_data import digest_bytes, json_bytes, prepare, read_json, sha256_file, utc_now
from vus_panel.schemas import parse_variant_id
from vus_panel.scoring.client import ScoringClient, ScoringClientError
from vus_panel.scoring.contracts import Protocol, ScoreRequest

RUN_KIND = "evo2-split-scoring"
SPLITS = ("calibration", "test")
FINISHED_STATES = ("completed", "completed_with_errors")
# A backend verdict about the variant itself. Anything else is retried on resume.
FINAL_ERROR_CODES = ("context", "inference")
# Service metadata that may differ between sessions of one run. All other fields must match.
SESSION_FIELDS = ("session_id", "created_at", "gpu", "load_seconds", "warmup")


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":  # Windows cannot open a directory for fsync.
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _save_json(path: Path, value: Any) -> None:
    """Atomic and durable, so the run record survives a hard restart."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _write_lines(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _append(path: Path, row: dict[str, Any]) -> None:
    line = (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with path.open("ab") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _read_outcomes(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    """Return complete lines and the bytes of a torn final line, if any."""
    if not path.exists():
        return [], b""
    lines = path.read_bytes().split(b"\n")
    tail = lines.pop()
    rows = []
    for number, line in enumerate(lines, start=1):
        try:
            rows.append(json.loads(line))
        except ValueError as exc:
            raise ValueError(f"outcomes.jsonl line {number} is corrupt; refusing to resume") from exc
    return rows, tail


def _runtime_hashes() -> dict[str, str]:
    # Same file set as pilot_run.code_state, without writing a snapshot.
    files = sorted((config.REPO_ROOT / "src").rglob("*.py")) + [config.REPO_ROOT / "pyproject.toml"]
    return {source.relative_to(config.REPO_ROOT).as_posix(): digest_bytes(source.read_bytes()) for source in files}


def _identity(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metadata.items() if key not in SESSION_FIELDS}


def _selection(split: str) -> tuple[list[str], bytes, bytes, list]:
    """Verify the whole frozen freeze, then read one split's ID list."""
    _, variants = prepare(require_existing=True)
    split_meta_path = config.splits_dir() / "split_meta.json"
    split_meta_bytes = split_meta_path.read_bytes()
    split_meta = read_json(split_meta_path)
    ids_bytes = (config.splits_dir() / f"{split}.txt").read_bytes()
    if digest_bytes(ids_bytes) != split_meta["id_file_sha256"][f"{split}.txt"]:
        raise ValueError(f"{split}.txt does not match split_meta.json")
    ids = ids_bytes.decode("ascii").splitlines()
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f"{split}.txt must list unique variant IDs")
    return ids, ids_bytes, split_meta_bytes, variants


def is_final(row: dict[str, Any]) -> bool:
    if row.get("failure") is not None:
        return False
    if row["status"] == "ok":
        return True
    response = row.get("response")
    return row["status"] == "error" and response is not None and response.get("error_code") in FINAL_ERROR_CODES


def latest_rows(ids: list[str], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per ID: its final row, else its latest attempt, else a pending placeholder."""
    final, latest = {}, {}
    for row in rows:
        latest[row["variant_id"]] = row
        if is_final(row) and row["variant_id"] not in final:
            final[row["variant_id"]] = row
    placeholder = {"status": "pending", "response": None, "client_wall_seconds": None, "error": None}
    return [final.get(vid) or latest.get(vid) or {"variant_id": vid, **placeholder} for vid in ids]


def progress(ids: list[str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    current = latest_rows(ids, rows)
    ok = [row for row in current if row["status"] == "ok"]
    final_errors = sum(is_final(row) and row["status"] != "ok" for row in current)
    wall = [row["client_wall_seconds"] for row in ok]
    pending = len(ids) - len(ok) - final_errors
    mean_wall = statistics.mean(wall) if wall else None
    return {"n_variants": len(ids), "n_ok": len(ok), "n_final_error": final_errors, "n_pending": pending,
            "n_attempts": len(rows), "coverage": len(ok) / len(ids),
            "last_finished_at": rows[-1]["finished_at"] if rows else None,
            "ok_client_wall_seconds": pilot_run._summary(wall),
            "estimated_remaining_seconds": mean_wall * pending if mean_wall is not None else None}


def _load_record(directory: Path) -> dict[str, Any]:
    try:
        record = read_json(directory / "run_record.json")
    except ValueError as exc:
        raise ValueError(f"{directory}: {exc}") from exc
    if record.get("kind") != RUN_KIND or record.get("split") not in SPLITS:
        raise ValueError(f"{directory} is not a split scoring run")
    return record


def _find_run(split: str, run_dir: Path | None) -> Path | None:
    if run_dir is not None:
        if _load_record(run_dir)["split"] != split:
            raise ValueError(f"{run_dir} is not a {split} run")
        return run_dir
    if not config.runs_dir().is_dir():
        return None
    candidates = [path for path in sorted(config.runs_dir().glob(f"*-{split}-*"))
                  if (path / "run_record.json").is_file() and _load_record(path)["split"] == split]
    return candidates[-1] if candidates else None


def _acquire_lock(directory: Path, force_unlock: bool) -> tuple[Path, dict[str, Any] | None]:
    lock = directory / "runner.lock"
    override = None
    if lock.exists():
        if not force_unlock:
            raise ValueError(f"{lock} exists. If `tmux ls` shows no live runner, pass --force-unlock.")
        override = {"at": utc_now(), "previous": lock.read_text(encoding="utf-8", errors="replace")}
        lock.unlink()
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "started_at": utc_now()}))
    return lock, override


def _check_invariants(record: dict[str, Any], ids: list[str], rows: list[dict[str, Any]]) -> None:
    allowed = set(ids)
    sessions = {s["metadata"]["session_id"] for s in record["sessions"] if s["metadata"]}
    attempts, finals = set(), Counter()
    for row in rows:
        vid = row["variant_id"]
        if vid not in allowed:
            raise ValueError(f"outcomes.jsonl has {vid}, which is not in the {record['split']} split")
        if (vid, row["attempt"]) in attempts:
            raise ValueError(f"outcomes.jsonl repeats attempt {row['attempt']} of {vid}")
        attempts.add((vid, row["attempt"]))
        if row["session_id"] not in sessions:
            raise ValueError(f"outcomes.jsonl names session {row['session_id']}, which run_record.json does not record")
        finals[vid] += is_final(row)
    repeated = sorted(vid for vid, count in finals.items() if count > 1)
    if repeated:
        raise ValueError(f"outcomes.jsonl has more than one final result for {', '.join(repeated[:3])}")


def _recover(directory: Path, record: dict[str, Any], ids: list[str]) -> None:
    """Repair a torn final line, record an attempt cut off mid-call, then check invariants."""
    output = directory / "outcomes.jsonl"
    record_path = directory / "run_record.json"
    rows, torn = _read_outcomes(output)
    if torn:
        sidecar = directory / f"outcomes.torn.{len(record['repairs'])}"
        sidecar.write_bytes(torn)
        record["repairs"].append({"at": utc_now(), "dropped_torn_bytes": len(torn), "saved_to": sidecar.name})
        _save_json(record_path, record)
        _write_lines(output, rows)
    in_flight_path = directory / "in_flight.json"
    if in_flight_path.exists():
        try:
            in_flight = read_json(in_flight_path)
            in_flight = {key: in_flight[key] for key in ("variant_id", "attempt", "session_id", "started_at")}
        except (ValueError, KeyError):
            in_flight = None
            record["repairs"].append({"at": utc_now(), "unreadable_in_flight_bytes": in_flight_path.stat().st_size})
            _save_json(record_path, record)
        if in_flight and not any(row["variant_id"] == in_flight["variant_id"] and row["attempt"] == in_flight["attempt"]
                                 for row in rows):
            row = {**in_flight, "status": "interrupted", "failure": "interrupted", "finished_at": None,
                   "client_wall_seconds": None, "response": None,
                   "error": "runner stopped during this attempt; server completion is unknown"}
            _append(output, row)
            rows.append(row)
        in_flight_path.unlink()
    _check_invariants(record, ids, rows)


def _create_run(split: str, ids: list[str], ids_bytes: bytes, split_meta_bytes: bytes) -> Path:
    split_meta = json.loads(split_meta_bytes)
    run_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{split}-{uuid.uuid4().hex[:12]}"
    directory = config.runs_dir() / run_id
    directory.mkdir(parents=True, exist_ok=False)
    (directory / f"{split}.txt").write_bytes(ids_bytes)
    (directory / "split_meta.json").write_bytes(split_meta_bytes)
    (directory / "outcomes.jsonl").touch()
    record = {
        "schema_version": 1, "kind": RUN_KIND, "run_id": run_id, "split": split,
        "created_at": utc_now(), "state": "initializing", "error": None, "seed": split_meta["seed"],
        "protocol": Protocol().model_dump(mode="json"),
        "source": split_meta["source"], "reference": split_meta["reference"],
        "selection": {"ids_file": f"{split}.txt", "ids_file_sha256": digest_bytes(ids_bytes),
                      "split_meta_sha256": digest_bytes(split_meta_bytes), "n_variants": len(ids)},
        "code": None, "sessions": [], "repairs": [], "lock_overrides": [],
        "no_automatic_retries": True, "label_free_outcomes": True,
    }
    _save_json(directory / "run_record.json", record)
    record["code"] = pilot_run.code_state(directory)
    _save_json(directory / "run_record.json", record)
    return directory


def _check_resume(record: dict[str, Any], split: str, ids_bytes: bytes, split_meta_bytes: bytes) -> None:
    selection = record["selection"]
    fresh = "Start a fresh run with --new if that is intended."
    if record["split"] != split or selection["ids_file_sha256"] != digest_bytes(ids_bytes):
        raise ValueError(f"frozen {split}.txt changed since this run started; refusing to resume. {fresh}")
    if selection["split_meta_sha256"] != digest_bytes(split_meta_bytes):
        raise ValueError(f"split_meta.json changed since this run started; refusing to resume. {fresh}")
    if record["code"] is None:
        raise ValueError(f"run has no code snapshot. {fresh}")
    if (record["code"]["git_commit"] != pilot_run._git("rev-parse", "HEAD")
            or record["code"]["runtime_file_sha256"] != _runtime_hashes()):
        raise ValueError(f"runtime code differs from this run's snapshot; refusing to resume. {fresh}")


def _report(record: dict[str, Any], status: dict[str, Any], metrics: dict[str, Any] | None) -> str:
    lines = [f"# Evo 2 {record['split']} split scoring", "",
             f"Run: `{record['run_id']}`. State: {record['state']}.", "",
             f"Scored: {status['n_ok']}/{status['n_variants']}; coverage {status['coverage']:.1%}. "
             f"Final errors: {status['n_final_error']}. Pending: {status['n_pending']}. Attempts: {status['n_attempts']}.", ""]
    if metrics is not None:
        auc = metrics["lof_positive_auroc"]
        partial = " Partial: not every ID is final yet." if status["n_pending"] else ""
        lines.extend([f"Descriptive LOF-positive AUROC on the calibration split: "
                      f"{'unavailable' if auc is None else f'{auc:.3f}'}.{partial} "
                      "This is calibration data, not a test result.", "",
                      "| Class | Selected | Scored | Mean delta | Sample SD |", "| --- | ---: | ---: | ---: | ---: |"])
        for function_class, item in metrics["per_class"].items():
            summary = item["delta"]
            mean = f"{summary['mean']:.6f}" if summary["mean"] is not None else "unavailable"
            sd = f"{summary['sample_sd']:.6f}" if summary["sample_sd"] is not None else "unavailable"
            lines.append(f"| {function_class} | {item['n_selected']} | {summary['n']} | {mean} | {sd} |")
        lines.append("")
    else:
        lines.extend(["Test-split outcomes are label-free. Evaluate them through run_eval, not here.", ""])
    lines.append("Sessions:")
    for session in record["sessions"]:
        metadata = session["metadata"] or {}
        lines.append(f"- {session['index']}: {session['state']}, scored {session['scored']}, "
                     f"model load {metadata.get('load_seconds')} s, elapsed {session['elapsed_seconds']:.1f} s")
    lines.extend(["", "Trace: `outcomes.jsonl`, `run_record.json`, the saved `code/` snapshot"
                  + (", `evaluation.jsonl` and `metrics.json`." if metrics is not None else " and `progress.json`."), ""])
    return "\n".join(lines)


def _finalize(directory: Path, record: dict[str, Any], ids: list[str], variants: list) -> bool:
    output = directory / "outcomes.jsonl"
    rows, _ = _read_outcomes(output)
    status = progress(ids, rows)
    metrics = None
    artifacts = ["outcomes.jsonl"]
    if record["split"] == "calibration":
        # Labels join only after the label-free outcomes are written and reread.
        first = next((session["metadata"] for session in record["sessions"] if session["metadata"]), None)
        elapsed = sum(session["elapsed_seconds"] for session in record["sessions"])
        joined, metrics = pilot_run.evaluate_outcomes(latest_rows(ids, rows), {v.variant_id: v for v in variants},
                                                      calibration_count=len(ids), metadata=first, elapsed_seconds=elapsed)
        # evaluate_outcomes estimates the rest of a pilot; here the remainder is the pending IDs.
        timing, pending = metrics["timing"], status["n_pending"]
        backend_mean = timing["successful_backend_seconds"]["mean"]
        wall_mean = timing["successful_client_wall_seconds"]["mean"]
        timing.update(remaining_calibration_variants=pending,
                      estimated_remaining_backend_seconds=backend_mean * pending if backend_mean is not None else None,
                      estimated_remaining_client_seconds=wall_mean * pending if wall_mean is not None else None,
                      estimate_basis="mean successful per-variant time times pending IDs in this run; excludes startup")
        metrics.update(split="calibration", partial=bool(pending), progress=status)
        _write_lines(directory / "evaluation.jsonl", joined)
        _save_json(directory / "metrics.json", metrics)
        artifacts += ["evaluation.jsonl", "metrics.json"]
    else:
        _save_json(directory / "progress.json", {"split": record["split"], **status})
        artifacts.append("progress.json")
    record["artifact_sha256"] = {name: sha256_file(directory / name) for name in artifacts}
    _save_json(directory / "run_record.json", record)
    (directory / "report.md").write_text(_report(record, status, metrics), encoding="utf-8")
    return status["n_pending"] == 0


def run_split(split: str, *, backend_url: str | None = None, timeout: float = 600.0, run_dir: Path | None = None,
              new: bool = False, force_unlock: bool = False) -> tuple[Path, bool]:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {', '.join(SPLITS)}")
    if new and run_dir is not None:
        raise ValueError("--new starts a fresh run; do not combine it with --run-dir")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    endpoint = backend_url or config.evo2_backend_url()
    if not endpoint:
        raise ValueError("EVO2_BACKEND_URL is not set; no scoring backend is configured")
    parsed = urlsplit(endpoint)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("backend URL must not contain credentials, query parameters, or fragments")
    if pilot_run._git("rev-parse", "HEAD") is None:
        raise ValueError("git commit is unreadable; in a root container, add the repo as a git safe.directory")
    ids, ids_bytes, split_meta_bytes, variants = _selection(split)
    existing = None if new else _find_run(split, run_dir)
    if new:
        latest = _find_run(split, None)
        if latest is not None and _load_record(latest)["state"] not in FINISHED_STATES:
            print(f"Leaving unfinished {split} run {latest.name} as it is", flush=True)
    directory = existing or _create_run(split, ids, ids_bytes, split_meta_bytes)
    lock, override = _acquire_lock(directory, force_unlock)
    try:
        # Read and check the record only while holding the lock.
        record = _load_record(directory)
        if record["state"] in FINISHED_STATES:
            raise ValueError(f"latest {split} run {directory.name} is finished; pass --new to start another")
        if override is not None:
            record["lock_overrides"].append(override)
        if existing is not None:
            _check_resume(record, split, ids_bytes, split_meta_bytes)
        print(f"{'Resuming' if existing is not None else 'Started'} {split} run: {directory}", flush=True)
        _recover(directory, record, ids)
        _save_json(directory / "run_record.json", record)
        argv = [sys.executable, "-m", "vus_panel.eval.score_split", "run", "--split", split,
                "--timeout", str(timeout), "--backend", endpoint]
        session = {"index": len(record["sessions"]), "started_at": utc_now(), "finished_at": None, "state": "connecting",
                   "error": None, "backend_url": endpoint, "argv": argv, "metadata": None,
                   "metadata_client_wall_seconds": None, "git_status": pilot_run._git("status", "--porcelain"),
                   "scored": 0, "elapsed_seconds": 0.0}
        started = time.perf_counter()
        client = ScoringClient(endpoint, timeout=timeout)
        try:
            metadata_started = time.perf_counter()
            metadata = client.metadata().model_dump(mode="json")
        except ScoringClientError as exc:
            record["state"] = session["state"] = "blocked"
            record["error"] = session["error"] = f"{type(exc).__name__}: {exc}"
            session["elapsed_seconds"] = time.perf_counter() - started
            record["sessions"].append(session)
            return directory, _finalize(directory, record, ids, variants)
        reference = record["reference"]
        if (metadata["reference_sequence_sha256"] != reference["sequence_sha256"]
                or metadata["reference_file_sha256"] != reference["decompressed_file_sha256"]):
            raise ValueError("backend reference checksum does not match the frozen reference")
        first = next((s["metadata"] for s in record["sessions"] if s["metadata"]), None)
        if first is not None and _identity(metadata) != _identity(first):
            differing = sorted(key for key in set(_identity(metadata)) | set(_identity(first))
                               if metadata.get(key) != first.get(key))
            raise ValueError(f"backend identity differs from this run's first session: {', '.join(differing)}")
        session.update(metadata=metadata, metadata_client_wall_seconds=time.perf_counter() - metadata_started, state="running")
        record["sessions"].append(session)
        record.update(state="running", error=None)
        _save_json(directory / "run_record.json", record)
        output = directory / "outcomes.jsonl"
        try:
            session["state"] = _score(directory, output, client, ids, session)
        except KeyboardInterrupt:
            session["state"], session["error"] = "interrupted", "KeyboardInterrupt between attempts"
        session.update(finished_at=utc_now(), elapsed_seconds=time.perf_counter() - started)
        rows = latest_rows(ids, _read_outcomes(output)[0])
        if all(is_final(row) for row in rows):
            record["state"] = "completed" if all(row["status"] == "ok" for row in rows) else "completed_with_errors"
            record["error"] = None
        else:
            record["state"] = session["state"] if session["state"] != "finished" else "interrupted"
            record["error"] = session["error"]
        return directory, _finalize(directory, record, ids, variants)
    finally:
        lock.unlink(missing_ok=True)


def _score(directory: Path, output: Path, client: ScoringClient, ids: list[str], session: dict[str, Any]) -> str:
    """Score pending IDs in split order. Any non-final failure ends the session."""
    rows, _ = _read_outcomes(output)
    attempts = Counter(row["variant_id"] for row in rows)
    done = {row["variant_id"] for row in rows if is_final(row)}
    in_flight_path = directory / "in_flight.json"
    for vid in ids:
        if vid in done:
            continue
        request = ScoreRequest.from_variant(parse_variant_id(vid))
        row = {"variant_id": vid, "attempt": attempts[vid] + 1, "session_id": session["metadata"]["session_id"],
               "status": "interrupted", "failure": "interrupted", "started_at": utc_now(), "finished_at": None,
               "client_wall_seconds": None, "response": None, "error": "attempt began but no response was saved"}
        _save_json(in_flight_path, {key: row[key] for key in ("variant_id", "attempt", "session_id", "started_at")})
        call_started = time.perf_counter()
        stop = None
        try:
            response = client.score(request)
            row.update(status=response.status, failure=None, response=response.model_dump(mode="json"), error=response.error)
            if not is_final(row):
                stop = "aborted"
        except KeyboardInterrupt:
            row["error"] = "KeyboardInterrupt during scoring; server completion is unknown"
            stop = "interrupted"
        except ScoringClientError as exc:
            row.update(status="error", failure="client_fatal" if exc.fatal else "transport", error=f"{type(exc).__name__}: {exc}")
            if exc.response is not None:
                row["response"] = exc.response.model_dump(mode="json")
            stop = "aborted" if exc.fatal else "backend_lost"
        except Exception as exc:
            row.update(status="error", failure="unexpected", error=f"{type(exc).__name__}: {exc}")
            stop = "aborted"
        finally:
            row.update(finished_at=utc_now(), client_wall_seconds=time.perf_counter() - call_started)
            _append(output, row)
            in_flight_path.unlink(missing_ok=True)
            attempts[vid] += 1
        if stop is not None:
            session["error"] = row["error"]
            return stop
        session["scored"] += 1
    return "finished"


def status_lines(split: str | None = None, run_dir: Path | None = None) -> list[str]:
    """Read-only progress summary for the latest run of each split."""
    lines = []
    if run_dir is not None:
        names = [_load_record(run_dir)["split"]]
    else:
        names = [split] if split else list(SPLITS)
    for name in names:
        directory = _find_run(name, run_dir)
        if directory is None:
            lines.append(f"{name}: no run yet")
            continue
        record = _load_record(directory)
        ids = (directory / record["selection"]["ids_file"]).read_text(encoding="ascii").splitlines()
        rows, torn = _read_outcomes(directory / "outcomes.jsonl")
        status = progress(ids, rows)
        eta = status["estimated_remaining_seconds"]
        mean_wall = status["ok_client_wall_seconds"]["mean"]
        lines.append(
            f"{name} {directory.name}: state={record['state']} lock={'held' if (directory / 'runner.lock').exists() else 'free'} "
            f"ok={status['n_ok']}/{status['n_variants']} final_errors={status['n_final_error']} pending={status['n_pending']} "
            f"attempts={status['n_attempts']} last={status['last_finished_at']} "
            f"mean_wall={'n/a' if mean_wall is None else f'{mean_wall:.3f}s'} "
            f"eta={'n/a' if eta is None else f'{eta / 60:.1f}min'}" + (" torn_tail=yes" if torn else ""))
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resumable Evo 2 scoring of a frozen split")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="start, or resume the newest unfinished run for the split")
    run.add_argument("--split", choices=SPLITS, required=True)
    run.add_argument("--backend", help="defaults to EVO2_BACKEND_URL")
    run.add_argument("--timeout", type=float, default=600.0)
    run.add_argument("--run-dir", type=Path)
    run.add_argument("--new", action="store_true", help="start a new run even if a finished one exists")
    run.add_argument("--force-unlock", action="store_true", help="take over a lock left by a dead runner")
    show = subparsers.add_parser("status", help="print progress without changing anything")
    show.add_argument("--split", choices=SPLITS)
    show.add_argument("--run-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "status":
            print("\n".join(status_lines(args.split, args.run_dir)))
            return 0
        directory, complete = run_split(args.split, backend_url=args.backend, timeout=args.timeout,
                                        run_dir=args.run_dir, new=args.new, force_unlock=args.force_unlock)
        print(f"{args.split} artifacts: {directory} ({'complete' if complete else 'incomplete; run again to resume'})")
        return 0 if complete else 1
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
