"""Evo 2 tool: log-likelihood delta (alt minus ref) from Evo 2 7B.

Owner: evidence-models. A stored result under EVO2_SCORES_DIR wins; otherwise
the model runs on a separate GPU backend reached through EVO2_BACKEND_URL.
Stored results are read from outcomes.jsonl and run_record.json only, plus
the calibration ID list, never an evaluation file, so no Findlay label is read.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vus_panel import config
from vus_panel.schemas import ToolResult, Variant, reject_answer_key
from vus_panel.scoring.client import ScoringClient, ScoringClientError
from vus_panel.scoring.contracts import Protocol, ScoreRequest, ScoreResponse

TOOL = "evo2"
OWNER = "evidence-models"
NEXT_TOKEN_TERMS = Protocol().next_token_terms
CONTEXT_LENGTH = Protocol().context_length
# Service metadata that may differ between sessions of one model setup.
SESSION_FIELDS = ("session_id", "created_at", "gpu", "load_seconds", "warmup")
FINAL_ERROR_CODES = ("context", "inference")
# Run directories written by the pilot runner and by eval.score_split.
RUN_NAME = re.compile(r"^\d{8}T\d{6}Z-(pilot20|calibration|test)-[0-9a-f]{12}$")


@dataclass
class _Store:
    results: dict[str, tuple[dict[str, Any], str]] = field(default_factory=dict)
    identity: dict[str, Any] | None = None
    calibration_ids: set[str] = field(default_factory=set)
    problem: str | None = None

    def calibration_reference(self) -> tuple[list[float], str | None]:
        """Every calibration delta, or none with the reason, so a partial set is never used."""
        if not self.calibration_ids:
            return [], "the calibration ID list is missing"
        missing = [vid for vid in self.calibration_ids if vid not in self.results]
        if missing:
            return [], (f"only {len(self.calibration_ids) - len(missing)} of {len(self.calibration_ids)} "
                        "calibration-split variants have a stored result")
        return [row["response"]["delta"] for vid, (row, _) in self.results.items()
                if vid in self.calibration_ids and row["status"] == "ok"], None


_cached: dict[tuple, _Store] = {}


def _outcome_files(root: Path) -> list[Path]:
    """outcomes.jsonl in root, then in each runner-named directory under it, oldest name first."""
    files = [root / "outcomes.jsonl"] if (root / "outcomes.jsonl").is_file() else []
    if root.is_dir():
        files += sorted(path / "outcomes.jsonl" for path in root.iterdir()
                        if path.is_dir() and RUN_NAME.match(path.name) and (path / "outcomes.jsonl").is_file())
    return files


def _without_session(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metadata.items() if key not in SESSION_FIELDS}


def _identity(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "run_record.json"
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    metadata = record.get("backend_metadata")
    if metadata is None:
        metadata = next((s["metadata"] for s in record.get("sessions", []) if s.get("metadata")), None)
    return None if metadata is None else _without_session(metadata)


def _final_response(row: dict[str, Any]) -> ScoreResponse | None:
    """The validated response if this attempt is final, else None."""
    if row.get("response") is None:
        return None
    response = ScoreResponse.model_validate(row["response"])
    if response.protocol != Protocol():
        raise ValueError("a stored result used a different scoring protocol")
    # A failed attempt can carry a valid response (a session change, say); it is never final.
    if row.get("failure") is not None or row.get("status") != response.status:
        return None
    if response.status == "ok" or response.error_code in FINAL_ERROR_CODES:
        return response
    return None


def _load(root: Path) -> _Store:
    files = _outcome_files(root)
    if not files:
        raise ValueError(f"no outcomes.jsonl in {config.display_path(root)} or the run directories under it")
    calibration = config.splits_dir() / "calibration.txt"
    watched = [*files, *(f.parent / "run_record.json" for f in files), calibration]
    key = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in watched if p.is_file())
    if key in _cached:
        return _cached[key]
    store = _Store()
    run_dirs = {path.parent.name: path.parent for path in files}
    for path in files:
        # Only complete lines; a torn final line from an interrupted writer is skipped.
        lines = path.read_text(encoding="utf-8").split("\n")[:-1]
        for number, line in enumerate(lines, start=1):
            try:
                row = json.loads(line)
                response = _final_response(row)
            except ValueError as exc:
                raise ValueError(f"{path.parent.name}/outcomes.jsonl line {number} is not a valid Evo 2 result: "
                                 f"{type(exc).__name__}") from exc
            if response is not None:
                # Later (newer) runs replace earlier ones.
                store.results[response.variant_id] = (row, path.parent.name)
    # Only runs that still supply a result must agree on the model setup.
    identities = {run: _identity(run_dirs[run]) for run in sorted({run for _, run in store.results.values()})}
    distinct = {json.dumps(identity, sort_keys=True) for identity in identities.values()}
    missing = sorted(run for run, identity in identities.items() if identity is None)
    if missing:
        store.problem = f"stored Evo 2 runs have no backend identity in run_record.json: {', '.join(missing)}"
    elif len(distinct) > 1:
        store.problem = f"stored Evo 2 runs come from different model setups: {', '.join(identities)}"
    elif identities:
        store.identity = next(iter(identities.values()))
    if calibration.is_file():
        store.calibration_ids = set(calibration.read_text(encoding="ascii").split())
    _cached.clear()
    _cached[key] = store
    return store


def _percentile(value: float, reference: list[float]) -> float | None:
    if not reference:
        return None
    less = sum(delta < value for delta in reference)
    equal = sum(delta == value for delta in reference)
    return 100.0 * (less + 0.5 * equal) / len(reference)


def _result(response: dict[str, Any], source: str, reference: list[float], why_not: str | None,
            sources: list[str]) -> ToolResult:
    delta = response["delta"]
    values = {"delta": delta, "delta_mean": delta / NEXT_TOKEN_TERMS,
              "ref_total": response["ref_total"], "alt_total": response["alt_total"],
              "next_token_terms": NEXT_TOKEN_TERMS, "context_length": CONTEXT_LENGTH, "score_source": source,
              "calibration_percentile": _percentile(delta, reference), "n_calibration_reference": len(reference)}
    message = (f"Evo 2 genomic-forward log-likelihood, ALT minus REF, summed over {NEXT_TOKEN_TERMS} next-token "
               f"terms of a {CONTEXT_LENGTH}-base window; delta_mean is delta / {NEXT_TOKEN_TERMS}.")
    if values["calibration_percentile"] is None:
        message += f" No calibration percentile: {why_not or 'no calibration-split deltas are stored'}."
    else:
        message += (f" calibration_percentile is this delta's mid-rank percentile among all {len(reference)} stored "
                    "calibration-split deltas from the same model setup; lower means more negative.")
    return ToolResult(tool=TOOL, status="ok", values=values, message=message, sources=sources)


def log_likelihood_delta(variant: Variant) -> ToolResult:
    reject_answer_key(variant, f"tools.{TOOL}.log_likelihood_delta")
    store = None
    scores_dir = config.evo2_scores_dir()
    if scores_dir is not None:
        try:
            store = _load(Path(scores_dir))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            return ToolResult(tool=TOOL, status="error", message=f"Could not read stored Evo 2 scores in EVO2_SCORES_DIR: {exc}")
        if store.problem:
            return ToolResult(tool=TOOL, status="error", message=f"Stored Evo 2 scores can't be used: {store.problem}.")
        found = store.results.get(variant.variant_id)
        if found is not None:
            row, run = found
            response = row["response"]
            source = f"Evo 2 stored result, run {run}; service session {response['session_id']}; metadata SHA256 {response['metadata_sha256']}"
            if row["status"] != "ok":
                return ToolResult(tool=TOOL, status="error", sources=[source],
                                  message=f"The stored Evo 2 result for this variant is a backend error "
                                          f"({response['error_code']}): {response['error']}")
            reference, why_not = store.calibration_reference()
            return _result(response, "stored", reference, why_not, [source])
    if config.evo2_backend_url() is None:
        lead = ("EVO2_SCORES_DIR holds no stored result for this variant, and EVO2_BACKEND_URL is not set."
                if store is not None else "Neither EVO2_SCORES_DIR nor EVO2_BACKEND_URL is set.")
        return ToolResult(tool=TOOL, status="unavailable",
                          message=f"{lead} Evo 2 7B needs an FP8-capable GPU backend, and none is configured.")
    reference, why_not = [], None
    try:
        request = ScoreRequest.from_variant(variant)
        client = ScoringClient(config.evo2_backend_url())
        if store is not None:
            reference, why_not = store.calibration_reference()
        # A live delta is only ranked against stored deltas from the same model setup.
        if reference and _without_session(client.metadata().model_dump(mode="json")) != store.identity:
            reference, why_not = [], "the live backend's model setup differs from the stored runs"
        response = client.score(request)
    except (ValueError, ScoringClientError) as exc:
        return ToolResult(tool=TOOL, status="error", message=f"Evo 2 scoring failed: {exc}")
    if response.status != "ok":
        return ToolResult(tool=TOOL, status="error", message=f"Evo 2 scoring failed: {response.error}")
    return _result(response.model_dump(mode="json"), "backend", reference, why_not,
                   [f"Evo 2 service session {response.session_id}; metadata SHA256 {response.metadata_sha256}"])
