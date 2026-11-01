"""Shared plumbing for the test-split baselines (evo2_alone, logistic_baseline).

Owner: orchestration-eval. A baseline turns tool results into PanelReports and
scores them through run_eval, unchanged, so it meets the same guards as the
panel. Each report's probability of damage is p: run_eval reads confidence for
a likely_damaging call and 1 - confidence for a likely_tolerated one, so the
call follows p >= 0.5 and report_probability(report) returns p.
"""

from __future__ import annotations

import json
import math
import re
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from vus_panel import config
from vus_panel.eval import run_eval
from vus_panel.schemas import AgentFinding, PanelReport, RunRecord, ToolResult

REPORTS_FILE = "reports.json"
DETAILS_FILE = "baseline.json"
SETTLING_TEST = "A functional assay result for this exact variant would settle it."
PATIENT_TEXT = "Evaluation baseline only. Not for sharing."
_REPORTS = TypeAdapter(list[PanelReport])
_STORED_RUN = re.compile(r"^Evo 2 stored result, run (\S+);")


def split_ids(split: str, split_dir: Path | None = None) -> list[str]:
    """IDs of one frozen split, in file order. read_ids returns a set, which loses the order."""
    path = (split_dir if split_dir is not None else config.splits_dir()) / f"{split}.txt"
    ids = path.read_text(encoding="ascii").split()
    if len(set(ids)) != len(ids):
        raise ValueError(f"{config.display_path(path)} lists an ID more than once")
    if not ids:
        raise ValueError(f"{config.display_path(path)} lists no IDs")
    return ids


def tool_finding(agent: str, variant_id: str, result: ToolResult, rationale: str) -> AgentFinding:
    """An ok tool result as an ok finding with no per-tool verdict ('uncertain'); anything else passes through."""
    if result.status == "ok":
        return AgentFinding(agent=agent, variant_id=variant_id, status="ok", verdict="uncertain",
                            raw_values=result.values, rationale=rationale, sources=list(result.sources))
    return AgentFinding(agent=agent, variant_id=variant_id, status=result.status, rationale=result.message,
                        sources=list(result.sources))


def baseline_report(variant_id: str, p: float | None, findings: list[AgentFinding], *, calibrated: bool,
                    summary: str) -> PanelReport:
    """A report whose probability of damage is p, or an unresolved one when p is None."""
    common: dict[str, Any] = {
        "variant_id": variant_id,
        "conflicts": [],
        "what_would_settle_it": SETTLING_TEST,
        "counselor_summary": summary,
        "patient_summary": PATIENT_TEXT,
        "findings_initial": findings,
        "findings_after_rebuttal": findings,
        "mock": False,
    }
    if p is None:
        return PanelReport(call="unresolved", confidence=None, calibrated=False, driving_agents=[], **common)
    if not (math.isfinite(p) and 0.0 <= p <= 1.0):
        raise ValueError(f"probability for {variant_id} must be in [0, 1], got {p}")
    damaging = p >= 0.5
    return PanelReport(
        call="likely_damaging" if damaging else "likely_tolerated",
        confidence=p if damaging else 1.0 - p,
        calibrated=calibrated,
        driving_agents=[f.agent for f in findings if f.status == "ok"],
        **common,
    )


def evo2_provenance(results: Sequence[ToolResult]) -> dict[str, Any]:
    """Where the evo2 deltas came from: the stored runs named in their sources, or the live backend."""
    runs, backend = set(), 0
    for result in results:
        if result.status != "ok":
            continue
        match = _STORED_RUN.match(result.sources[0]) if result.sources else None
        if match:
            runs.add(match.group(1))
        else:
            backend += 1
    scores_dir = config.evo2_scores_dir()
    return {
        "evo2_scores_dir": config.display_path(Path(scores_dir)) if scores_dir else None,
        "evo2_stored_runs": sorted(runs),
        "evo2_backend_scored": backend,
    }


def evaluate(system: str, reports: list[PanelReport], details: dict[str, Any], *, run_id: str | None = None,
             runs_root: Path | None = None, split_dir: Path | None = None, findlay_path: Path | None = None,
             git_commit: str | None = None, git_dirty: bool | None = None) -> RunRecord:
    """Score the reports with run_eval, then keep them and the baseline's details in the run directory."""
    run_id = run_id or f"{system}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    payload = _REPORTS.dump_json(reports, indent=1) + b"\n"
    with tempfile.TemporaryDirectory() as staging:
        path = Path(staging) / REPORTS_FILE
        path.write_bytes(payload)
        record = run_eval.run(path, system=system, run_id=run_id, runs_root=runs_root, split_dir=split_dir,
                              findlay_path=findlay_path, git_commit=git_commit, git_dirty=git_dirty)
    out_dir = (runs_root if runs_root is not None else config.runs_dir()) / run_id
    (out_dir / REPORTS_FILE).write_bytes(payload)
    (out_dir / DETAILS_FILE).write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    return record
