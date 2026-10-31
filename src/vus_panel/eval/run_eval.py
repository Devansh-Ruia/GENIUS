"""Score a set of PanelReports against the Findlay labels on the test split.

Refuses to run when:
- any record has mock: true (or is not explicitly mock: false)
- a report is for a variant outside the test split
- anything in a test-split report mentions ClinVar or a ClinVar accession
- the split metadata is missing, or disagrees with --seed or with the Findlay file

Labels: LOF is positive, FUNC and INT are negative. The score is confidence
for a likely_damaging call and 1 - confidence for a likely_tolerated call.
Unresolved reports are left out of AUROC and counted against coverage.

Writes runs/<run_id>/run_record.json and runs/<run_id>/metrics.json.

Usage: python -m vus_panel.eval.run_eval --reports FILE --system panel [--run-id ID] [--seed N]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Iterator, Mapping
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import TypeAdapter

from vus_panel import config
from vus_panel.eval.metrics import reliability_bins, selective_auroc
from vus_panel.eval.split import default_findlay_path, load_findlay, read_ids, sha256_file
from vus_panel.schemas import PanelReport, RunRecord

SYSTEMS = ("panel", "single_agent", "evo2_alone")
LABEL_RULE = "LOF positive; FUNC and INT negative"
CLINVAR_RE = re.compile(r"clinvar|\b(?:VCV|RCV)\d{6,}", re.IGNORECASE)
SPLIT_META_KEYS = ("seed", "calibration_fraction", "source_sha256")
_REPORTS = TypeAdapter(list[PanelReport])


class EvalRefused(ValueError):
    """The input breaks an eval rule, so no metrics are produced."""


def refuse_mock(raw_records: list[Any]) -> None:
    flagged = []
    for i, record in enumerate(raw_records):
        if not isinstance(record, dict):
            raise EvalRefused(f"record {i} is not a JSON object")
        if record.get("mock") is not False:
            flagged.append(str(record.get("variant_id", f"<record {i}>")))
    if flagged:
        raise EvalRefused(
            f"refusing records that are mock or not marked mock: false: {', '.join(flagged)}. "
            "Eval runs on real panel output only."
        )


def _strings(value: Any) -> Iterator[str]:
    """Every string inside a nested value, dict keys included."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _strings(key)
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def cites_clinvar(report: PanelReport) -> bool:
    texts = [report.what_would_settle_it, report.counselor_summary, report.patient_summary, *report.conflicts]
    for finding in [*report.findings_initial, *report.findings_after_rebuttal]:
        texts.extend([finding.rationale, finding.revision_reason or "", *finding.sources])
        texts.extend(_strings(finding.raw_values))
    return any(CLINVAR_RE.search(text) for text in texts)


def check_reports(reports: list[PanelReport], test_ids: set[str]) -> None:
    ids = [r.variant_id for r in reports]
    duplicates = sorted({v for v in ids if ids.count(v) > 1})
    if duplicates:
        raise EvalRefused(f"more than one report for: {', '.join(duplicates)}")
    outside = [v for v in ids if v not in test_ids]
    if outside:
        raise EvalRefused(f"reports for variants outside the test split: {', '.join(outside)}")
    clinvar = [r.variant_id for r in reports if cites_clinvar(r)]
    if clinvar:
        raise EvalRefused(f"test-split reports mention ClinVar, which is not allowed as input: {', '.join(clinvar)}")


def report_probability(report: PanelReport) -> float | None:
    """Probability that the variant is damaging, or None when the report is not scored."""
    if report.call == "unresolved" or report.confidence is None:
        return None
    if report.call == "likely_damaging":
        return report.confidence
    return 1.0 - report.confidence


def evaluate(reports: list[PanelReport], labels: Mapping[str, int], n_bins: int = 10) -> dict[str, Any]:
    if not reports:
        raise EvalRefused("no reports to evaluate")
    unlabeled = [r.variant_id for r in reports if r.variant_id not in labels]
    if unlabeled:
        raise EvalRefused(f"no Findlay label for: {', '.join(unlabeled)}")

    y = np.array([labels[r.variant_id] for r in reports], dtype=int)
    probs = [report_probability(r) for r in reports]
    resolved = np.array([p is not None for p in probs], dtype=bool)
    score = np.array([np.nan if p is None else p for p in probs], dtype=float)

    selective = selective_auroc(y, score, resolved)
    reliability = None
    if selective.n_resolved:
        reliability = asdict(reliability_bins(score[resolved], y[resolved], n_bins=n_bins))
    return {
        "label_rule": LABEL_RULE,
        "auroc_resolved": selective.auroc,
        "coverage": selective.coverage,
        "n_resolved": selective.n_resolved,
        "n_total": selective.n_total,
        "n_calibrated": sum(1 for r in reports if r.calibrated),
        "reliability": reliability,
    }


def read_split_meta(split_dir: Path) -> dict[str, Any]:
    path = split_dir / "split_meta.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"split metadata not found at {config.display_path(path)}. Run python -m vus_panel.eval.split first."
        )
    meta = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in SPLIT_META_KEYS if key not in meta]
    if missing:
        raise EvalRefused(f"split_meta.json is missing {missing}")
    return meta


def _git(*args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=config.REPO_ROOT, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout


def git_commit_hash() -> str:
    return _git("rev-parse", "HEAD").strip()


def git_is_dirty() -> bool:
    return bool(_git("status", "--porcelain").strip())


def run(
    reports_path: Path,
    *,
    system: str,
    run_id: str | None = None,
    split_dir: Path | None = None,
    findlay_path: Path | None = None,
    seed: int | None = None,
    runs_root: Path | None = None,
    git_commit: str | None = None,
    git_dirty: bool | None = None,
    n_bins: int = 10,
) -> RunRecord:
    """Evaluate one system's reports. `seed`, when given, must match the split's seed."""
    if system not in SYSTEMS:
        raise ValueError(f"system must be one of {SYSTEMS}")
    started_at = datetime.now(timezone.utc)
    run_id = run_id or f"{system}-{started_at:%Y%m%dT%H%M%SZ}"

    raw = json.loads(Path(reports_path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise EvalRefused("the reports file must hold a JSON array of PanelReports")
    refuse_mock(raw)
    reports = _REPORTS.validate_python(raw)

    split_dir = split_dir if split_dir is not None else config.splits_dir()
    check_reports(reports, read_ids(split_dir / "test.txt"))
    meta = read_split_meta(split_dir)
    if seed is not None and seed != meta["seed"]:
        raise EvalRefused(f"--seed {seed} disagrees with the split's seed {meta['seed']}")

    findlay_path = Path(findlay_path) if findlay_path is not None else default_findlay_path()
    variants = load_findlay(findlay_path)
    if sha256_file(findlay_path) != meta["source_sha256"]:
        raise EvalRefused("the Findlay file differs from the one the split was made from (sha256 mismatch)")
    labels = {v.variant_id: int(v.findlay_function_class == "LOF") for v in variants}
    metrics = evaluate(reports, labels, n_bins=n_bins)

    out_dir = (runs_root if runs_root is not None else config.runs_dir()) / run_id
    record = RunRecord(
        run_id=run_id,
        git_commit=git_commit if git_commit is not None else git_commit_hash(),
        config={
            "system": system,
            "reports_file": Path(reports_path).name,
            "split_dir": config.display_path(split_dir),
            "calibration_fraction": meta["calibration_fraction"],
            "split_source_sha256": meta["source_sha256"],
            "git_dirty": git_dirty if git_dirty is not None else git_is_dirty(),
            "n_bins": n_bins,
            "label_rule": LABEL_RULE,
        },
        seed=meta["seed"],
        started_at=started_at,
        variant_ids=[r.variant_id for r in reports],
        outputs_path=config.display_path(out_dir),
    )
    if out_dir.exists():
        raise FileExistsError(f"run directory already exists: {config.display_path(out_dir)}")
    out_dir.mkdir(parents=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (out_dir / "run_record.json").write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate PanelReports on the Findlay test split.")
    parser.add_argument("--reports", type=Path, required=True, help="JSON array of PanelReports")
    parser.add_argument("--system", choices=SYSTEMS, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--split-dir", type=Path, default=None, help="default: data/splits/")
    parser.add_argument("--findlay", type=Path, default=None, help="default: data/raw/ Findlay file")
    parser.add_argument("--seed", type=int, default=None, help="optional; must match the seed in split_meta.json")
    args = parser.parse_args(argv)
    try:
        record = run(
            args.reports,
            system=args.system,
            run_id=args.run_id,
            split_dir=args.split_dir,
            findlay_path=args.findlay,
            seed=args.seed,
        )
    except (ValueError, OSError, RuntimeError) as exc:  # ValueError covers EvalRefused, JSON and schema errors
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {record.outputs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
