"""Pinned source downloads, validation and immutable pilot selection."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vus_panel import config
from vus_panel.eval.split import (
    FINDLAY_FILENAME, FUNCTION_CLASSES, default_findlay_path, load_findlay,
    sha256_file, stratified_split,
)
from vus_panel.schemas import Variant

FINDLAY_URL = "https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-018-0461-z/MediaObjects/41586_2018_461_MOESM3_ESM.xlsx"
FINDLAY_SHA256 = "e9aa4186b8a5de91d61059d03f8dac1e5573d1e2802f50a6581c3053d22b923a"
REFERENCE_ACCESSION = "NC_000017.10"
REFERENCE_ASSEMBLY = "GRCh37.p13"
REFERENCE_LENGTH = 81195210
REFERENCE_SEQUENCE_SHA256 = "2463d54fb95c959a3fe545545fd11313986278881c7ded54721e172e7df1a41a"
REFERENCE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_000017.10&rettype=fasta&retmode=text"
EXPECTED_COUNTS = {"FUNC": 2821, "INT": 249, "LOF": 823}
PILOT_COUNTS = {"FUNC": 7, "INT": 6, "LOF": 7}
FREEZE_FILES = ("calibration.txt", "test.txt", "split_meta.json", "pilot20.json")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (OSError, ValueError) as exc:
        raise ValueError(f"unreadable artifact {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    """Replace a local run/report artifact atomically, never a frozen selection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(json_bytes(value))
    os.replace(temporary, path)


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".download")
    if temporary.exists():
        raise ValueError(f"incomplete download at {temporary}; inspect it before removing it")
    request = urllib.request.Request(url, headers={"User-Agent": "GENIUS-Findlay-pilot/1", "Accept-Encoding": "identity"})
    with urllib.request.urlopen(request, timeout=180) as response, temporary.open("xb") as handle:
        while chunk := response.read(1 << 20):
            handle.write(chunk)
    os.replace(temporary, destination)


def ensure_findlay(download: bool = False) -> tuple[Path, dict[str, Any]]:
    path = default_findlay_path()
    receipt = path.with_suffix(".source.json")
    downloaded_at = None
    if not path.exists():
        if receipt.exists():
            raise ValueError("incomplete Findlay download: receipt exists without workbook")
        if not download:
            raise FileNotFoundError(f"{path} is missing; run prepare --download")
        _download(FINDLAY_URL, path)
        downloaded_at = utc_now()
    actual = sha256_file(path)
    if actual != FINDLAY_SHA256:
        raise ValueError(f"Findlay checksum mismatch: {actual}; refusing to replace existing file")
    expected = {"url": FINDLAY_URL, "sha256": FINDLAY_SHA256, "size_bytes": path.stat().st_size,
                "filename": FINDLAY_FILENAME}
    if receipt.exists():
        saved = read_json(receipt)
        if {key: saved.get(key) for key in expected} != expected or "verified_at" not in saved or "downloaded_at" not in saved:
            raise ValueError("incompatible Findlay source receipt")
    else:
        saved = {**expected, "downloaded_at": downloaded_at, "verified_at": utc_now(),
                 "download_note": "downloaded by prepare" if downloaded_at else "pre-existing bytes matched the pinned publisher SHA256; download time unknown"}
        receipt.write_bytes(json_bytes(saved))
    return path, saved


def read_reference(path: Path) -> str:
    with path.open("r", encoding="ascii") as handle:
        header = handle.readline().strip()
        if not header.startswith(f">{REFERENCE_ACCESSION} "):
            raise ValueError(f"reference FASTA must identify {REFERENCE_ACCESSION}")
        parts = []
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                raise ValueError("reference FASTA must contain exactly one chromosome")
            parts.append(line)
    sequence = "".join(parts)
    if len(sequence) != REFERENCE_LENGTH or set(sequence) - set("ACGTN"):
        raise ValueError(f"reference must contain {REFERENCE_LENGTH} uppercase ACGTN bases")
    if digest_bytes(sequence.encode("ascii")) != REFERENCE_SEQUENCE_SHA256:
        raise ValueError("reference sequence does not match the pinned NC_000017.10 checksum")
    return sequence


def ensure_reference(download: bool = False) -> tuple[str, dict[str, Any], Path]:
    directory = config.cache_dir() / "reference"
    path = directory / f"{REFERENCE_ACCESSION}.fasta"
    receipt = directory / "reference.json"
    if path.exists() != receipt.exists():
        raise ValueError("incomplete reference download: FASTA and reference.json must both exist")
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"{path} is missing; run prepare --download")
        _download(REFERENCE_URL, path)
        sequence = read_reference(path)
        file_sha = sha256_file(path)
        saved = {
            "url": REFERENCE_URL, "accession": REFERENCE_ACCESSION, "assembly": REFERENCE_ASSEMBLY,
            "genome_build": "GRCh37", "chrom": "17", "length": len(sequence),
            "downloaded_at": utc_now(), "encoding": "identity",
            "download_sha256": file_sha, "decompressed_file_sha256": file_sha,
            "sequence_sha256": digest_bytes(sequence.encode("ascii")),
        }
        receipt.write_bytes(json_bytes(saved))
    else:
        saved = read_json(receipt)
        sequence = read_reference(path)
        file_sha = sha256_file(path)
        expected = {"url": REFERENCE_URL, "accession": REFERENCE_ACCESSION, "assembly": REFERENCE_ASSEMBLY,
                    "genome_build": "GRCh37", "chrom": "17", "length": len(sequence),
                    "encoding": "identity", "download_sha256": file_sha,
                    "decompressed_file_sha256": file_sha,
                    "sequence_sha256": digest_bytes(sequence.encode("ascii"))}
        if {key: saved.get(key) for key in expected} != expected or not saved.get("downloaded_at"):
            raise ValueError("reference provenance/checksum mismatch; refusing to replace existing reference")
    return sequence, saved, path


def pilot_order(variant_id: str) -> tuple[str, str]:
    return digest_bytes(f"pilot-v1:0:{variant_id}".encode("ascii")), variant_id


def select_pilot(variants: list[Variant], calibration: list[str]) -> list[str]:
    allowed = set(calibration)
    selected = []
    for function_class, count in PILOT_COUNTS.items():
        pool = [v.variant_id for v in variants if v.variant_id in allowed and v.findlay_function_class == function_class]
        if len(pool) < count:
            raise ValueError(f"calibration has too few {function_class} variants for the frozen pilot")
        selected.extend(sorted(pool, key=pilot_order)[:count])
    return sorted(selected, key=pilot_order)


def _freeze_payloads(variants: list[Variant], source: dict[str, Any], reference: dict[str, Any],
                     validation_sha256: str, created_at: str) -> dict[str, bytes]:
    split = stratified_split((v.variant_id, v.findlay_function_class) for v in variants)
    calibration = "".join(f"{vid}\n" for vid in split.calibration).encode("ascii")
    test = "".join(f"{vid}\n" for vid in split.test).encode("ascii")
    meta = {
        "schema_version": 1, "created_at": created_at,
        "seed": 0, "calibration_fraction": 0.2, "genome_build": "GRCh37",
        "algorithm": "sha256('{seed}:{variant_id}') within class, tie by ID; int(n*fraction+0.5); sorted output IDs",
        "source_file": source["filename"], "source_sha256": source["sha256"],
        "source": source, "reference": reference, "validation_report_sha256": validation_sha256,
        "counts": split.counts, "class_counts": dict(sorted(Counter(v.findlay_function_class for v in variants).items())),
        "n_calibration": len(split.calibration), "n_test": len(split.test),
        "id_file_sha256": {"calibration.txt": digest_bytes(calibration), "test.txt": digest_bytes(test)},
    }
    meta_bytes = json_bytes(meta)
    selection = {
        "schema_version": 1, "created_at": created_at, "seed": 0,
        "algorithm": "sha256('pilot-v1:0:{variant_id}'), tie by ID; class quotas then combined hash order",
        "class_counts": PILOT_COUNTS, "n_selected": sum(PILOT_COUNTS.values()),
        "variant_ids": select_pilot(variants, split.calibration),
        "split_meta_sha256": digest_bytes(meta_bytes),
        "calibration_sha256": digest_bytes(calibration), "source_sha256": source["sha256"],
        "reference_sequence_sha256": reference["sequence_sha256"],
    }
    return {"calibration.txt": calibration, "test.txt": test, "split_meta.json": meta_bytes,
            "pilot20.json": json_bytes(selection)}


def freeze_selection(variants: list[Variant], source: dict[str, Any], reference: dict[str, Any],
                     validation_sha256: str, directory: Path, *, require_existing: bool = False) -> dict[str, Any]:
    existing = [name for name in FREEZE_FILES if (directory / name).exists()]
    if existing and len(existing) != len(FREEZE_FILES):
        raise ValueError(f"incomplete freeze ({', '.join(existing)}); refusing to overwrite")
    if require_existing and not existing:
        raise ValueError("selection is not frozen; run prepare first")
    saved_meta = read_json(directory / "split_meta.json") if existing else None
    created_at = saved_meta.get("created_at") if saved_meta else utc_now()
    if not isinstance(created_at, str) or not created_at:
        raise ValueError("freeze is missing created_at")
    if existing:
        # The frozen receipt records the original download, not the timestamp of
        # a later machine's byte-identical download. Compare immutable identity.
        for key, current, variable_keys in (
            ("source", source, {"downloaded_at", "verified_at", "download_note"}),
            ("reference", reference, {"downloaded_at"}),
        ):
            saved = saved_meta.get(key)
            if not isinstance(saved, dict) or saved.keys() != current.keys():
                raise ValueError(f"incompatible frozen {key} provenance schema")
            if any(saved[field] != value for field, value in current.items() if field not in variable_keys):
                raise ValueError(f"incompatible frozen {key} identity/checksums")
            for field in variable_keys - {"download_note"}:
                value = saved[field]
                if value is None and key == "source" and field == "downloaded_at":
                    continue
                try:
                    stamp = datetime.fromisoformat(value)
                    if stamp.tzinfo is None:
                        raise ValueError("timezone required")
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"invalid frozen {key} {field}") from exc
        source, reference = saved_meta["source"], saved_meta["reference"]
    payloads = _freeze_payloads(variants, source, reference, validation_sha256, created_at)
    if existing:
        for name, expected in payloads.items():
            if (directory / name).read_bytes() != expected:
                raise ValueError(f"incompatible or corrupt freeze: {name}; refusing to overwrite")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        # Exclusive creates deliberately leave interrupted freezes detectable.
        for name, payload in payloads.items():
            with (directory / name).open("xb") as handle:
                handle.write(payload)
    return json.loads(payloads["pilot20.json"])


def prepare(download: bool = False, *, require_existing: bool = False,
            selection_path: Path | None = None) -> tuple[dict[str, Any], list[Variant]]:
    path, source = ensure_findlay(download)
    sequence, reference, _ = ensure_reference(download)
    report: dict[str, Any] = {}
    report_path = config.cache_dir() / "pilot" / "validation.json"
    try:
        variants = load_findlay(path, reference_sequence=sequence, validation_report=report)
    finally:
        if report:
            write_json(report_path, report)
    if report["class_counts"] != EXPECTED_COUNTS or len(variants) != sum(EXPECTED_COUNTS.values()):
        raise ValueError("publisher workbook class/row counts do not match the pinned source")
    if not report["transcript_complement_checked"]:
        raise ValueError("publisher transcript allele columns were not validated")
    selection_path = selection_path or config.splits_dir() / "pilot20.json"
    if selection_path.name != "pilot20.json":
        raise ValueError("selection must be the frozen pilot20.json")
    selection = freeze_selection(variants, source, reference, sha256_file(report_path), selection_path.parent,
                                 require_existing=require_existing)
    return selection, variants
