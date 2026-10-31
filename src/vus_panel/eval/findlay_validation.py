"""Audit all Findlay rows before any selection is frozen.

Only genomic coordinates, alleles and the Findlay answer key are read. Other
workbook annotations are ignored. Reports contain issues and aggregate counts.
"""

from __future__ import annotations

import math
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from vus_panel.schemas import Variant


def inspect_findlay(path: Path, reference_sequence: str | None = None) -> tuple[list[Variant], dict[str, Any]]:
    from openpyxl import load_workbook
    from openpyxl.utils.exceptions import InvalidFileException

    from vus_panel.eval.split import (
        COLUMNS, FINDLAY_BUILD, FUNCTION_CLASSES, HEADER_ROW, FindlayFormatError,
        _position, _text,
    )

    categories = (
        "missing_fields", "invalid_coordinates", "invalid_alleles", "invalid_classes",
        "nonfinite_scores", "duplicate_ids", "reference_mismatches", "transcript_mismatches",
    )
    report: dict[str, Any] = {
        "schema_version": 1, "rows": 0, "blank_rows": 0,
        "issues": {category: [] for category in categories},
        "reference_checked": reference_sequence is not None,
        "coordinate_convention": "one-based genomic forward; sequence[position - 1]",
        "transcript_complement_checked": False,
    }
    try:
        workbook = load_workbook(path, read_only=True, data_only=True)
    except (zipfile.BadZipFile, InvalidFileException, KeyError, OSError) as exc:
        raise FindlayFormatError(f"{path.name} is not a readable .xlsx file ({type(exc).__name__}: {exc})") from exc
    variants: list[Variant] = []
    seen: set[str] = set()
    try:
        rows = workbook.worksheets[0].iter_rows(values_only=True)
        header = next((row for number, row in enumerate(rows, start=1) if number == HEADER_ROW), None)
        if header is None:
            raise FindlayFormatError(f"{path.name} has fewer than {HEADER_ROW} rows")
        names = [_text(cell) or "" for cell in header]
        missing = [name for name in COLUMNS.values() if name not in names]
        if missing:
            raise FindlayFormatError(f"{path.name} header row {HEADER_ROW} is missing columns: {missing}")
        if any(names.count(name) != 1 for name in COLUMNS.values()):
            raise FindlayFormatError("duplicate required column names")
        indices = {key: names.index(name) for key, name in COLUMNS.items()}
        transcript_columns = next((pair for pair in (
            ("transcript_ref", "transcript_alt"),
            ("reference (transcript)", "alt (transcript)"),
            ("transcript.reference", "transcript.alt"),
            ("ref (transcript)", "alt (transcript)"),
        ) if all(name in names for name in pair)), None)
        report["transcript_complement_checked"] = transcript_columns is not None
        complement = str.maketrans("ACGT", "TGCA")
        for number, row in enumerate(rows, start=HEADER_ROW + 1):
            if all(_text(value) in (None, "") for value in row):
                report["blank_rows"] += 1
                continue
            report["rows"] += 1
            values = {key: row[index] if index < len(row) else None for key, index in indices.items()}
            valid = True

            def issue(category: str, message: str) -> None:
                nonlocal valid
                report["issues"][category].append({"row": number, "message": message})
                valid = False

            for key, value in values.items():
                if _text(value) in (None, ""):
                    issue("missing_fields", f"missing {COLUMNS[key]}")
            try:
                pos = _position(values["pos"])
                chrom = str(values["chrom"]).strip().removeprefix("chr")
                if chrom != "17" or pos < 1 or (reference_sequence is not None and pos > len(reference_sequence)):
                    raise ValueError("requires a position within chromosome 17")
            except (ValueError, TypeError) as exc:
                issue("invalid_coordinates", str(exc))
                pos = None
            ref, alt = _text(values["ref"]), _text(values["alt"])
            alleles_valid = ref in ("A", "C", "G", "T") and alt in ("A", "C", "G", "T") and ref != alt
            if not alleles_valid:
                issue("invalid_alleles", "reference and alt must be distinct uppercase ACGT bases")
            func_class = _text(values["func_class"])
            if func_class not in FUNCTION_CLASSES:
                issue("invalid_classes", f"unknown func.class {func_class!r}")
            try:
                score = float(values["score"])
                if not math.isfinite(score):
                    raise ValueError("function.score.mean must be finite")
            except (ValueError, TypeError):
                issue("nonfinite_scores", "function.score.mean must be a finite number")
                score = None
            if pos is not None and alleles_valid:
                variant_id = f"{FINDLAY_BUILD}-17-{pos}-{ref}-{alt}"
                if variant_id in seen:
                    issue("duplicate_ids", f"duplicate variant {variant_id}")
                seen.add(variant_id)
                if reference_sequence is not None and reference_sequence[pos - 1] != ref:
                    issue("reference_mismatches", f"{variant_id}: genomic reference does not match sequence[position - 1]")
            if transcript_columns is not None and alleles_valid:
                transcript = [_text(row[names.index(name)]) for name in transcript_columns]
                if transcript != [ref.translate(complement), alt.translate(complement)]:
                    issue("transcript_mismatches", "transcript alleles are not complements of genomic alleles")
            if valid:
                variants.append(Variant(chrom="17", pos=pos, ref=ref, alt=alt,
                                        genome_build=FINDLAY_BUILD,
                                        findlay_function_class=func_class, findlay_score=score))
    finally:
        workbook.close()
    report["issue_counts"] = {key: len(value) for key, value in report["issues"].items()}
    report["valid_rows"] = len(variants)
    report["class_counts"] = dict(sorted(Counter(v.findlay_function_class for v in variants).items()))
    report["ok"] = bool(variants) and not any(report["issue_counts"].values())
    return variants, report
