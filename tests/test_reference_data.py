"""Tracked reference data in data/reference/. Owner: orchestration-eval.

Every file there needs a MANIFEST.json entry whose sha256 and size match it.
The tmp-dir cases below use a few placeholder bytes, not real data.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from vus_panel import config

REFERENCE = Path("data") / "reference"
MANIFEST_NAME = "MANIFEST.json"
MAX_BYTES = 50 * 1024 * 1024
REQUIRED_KEYS = (
    "path",
    "source_url",
    "release",
    "download_date",
    "genome_build",
    "license",
    "sha256",
    "size_bytes",
    "owner_branch",
)
NON_EMPTY_KEYS = ("source_url", "license")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(root: Path) -> dict[str, Any]:
    return json.loads((root / REFERENCE / MANIFEST_NAME).read_text(encoding="utf-8"))


def reference_problems(root: Path, manifest: dict[str, Any], max_bytes: int = MAX_BYTES) -> list[str]:
    """Everything wrong with root/data/reference/ against its manifest. Empty means in sync."""
    entries = manifest.get("files")
    if not isinstance(entries, list):
        return ["MANIFEST.json needs a 'files' list"]

    problems: list[str] = []
    listed: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(entries):
        missing = [key for key in REQUIRED_KEYS if key not in entry]
        if missing:
            problems.append(f"entry {index} is missing {missing}")
            continue
        path = str(entry["path"])
        parts = Path(path).parts
        if parts[:2] != REFERENCE.parts or ".." in parts:
            problems.append(f"{path}: outside data/reference/")
            continue
        for key in NON_EMPTY_KEYS:
            if not str(entry[key]).strip():
                problems.append(f"{path}: empty {key}")
        if path in listed:
            problems.append(f"{path}: listed twice")
        listed[path] = entry

    ref_dir = root / REFERENCE
    on_disk = {
        p.relative_to(root).as_posix(): p
        for p in ref_dir.rglob("*")
        if p.is_file()
        and p != ref_dir / MANIFEST_NAME
        and not any(part.startswith(".") for part in p.relative_to(ref_dir).parts)
    }
    for rel, file in sorted(on_disk.items()):
        if rel not in listed:
            problems.append(f"{rel}: no manifest entry")
        if file.stat().st_size > max_bytes:
            problems.append(f"{rel}: {file.stat().st_size} bytes is over the {max_bytes}-byte limit")

    for path, entry in listed.items():
        file = root / path
        if not file.is_file():
            problems.append(f"{path}: listed but the file is missing")
            continue
        if file.stat().st_size != entry["size_bytes"]:
            problems.append(f"{path}: size is {file.stat().st_size}, manifest says {entry['size_bytes']}")
        if _sha256(file) != str(entry["sha256"]).lower():
            problems.append(f"{path}: sha256 does not match the manifest")
    return problems


def test_repo_reference_data_matches_manifest():
    # config.REPO_ROOT, not config.data_dir(): conftest points the data dir at tmp.
    manifest = load_manifest(config.REPO_ROOT)
    assert manifest.get("schema_version") == 1
    assert reference_problems(config.REPO_ROOT, manifest) == []


PLACEHOLDER = b"placeholder bytes, not real data\n"
DROP = object()


def _entry(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "path": "data/reference/placeholder.tsv",
        "source_url": "https://example.invalid/placeholder",
        "release": "placeholder",
        "download_date": "2026-01-01",
        "genome_build": "GRCh37",
        "license": "placeholder license",
        "sha256": hashlib.sha256(PLACEHOLDER).hexdigest(),
        "size_bytes": len(PLACEHOLDER),
        "owner_branch": "a",
    }
    fields.update(overrides)
    return {key: value for key, value in fields.items() if value is not DROP}


CASES = [
    # id, write the placeholder file?, entries, byte limit, expected problem (None means no problems)
    ("in_sync", True, [_entry()], MAX_BYTES, None),
    ("unlisted_file", True, [], MAX_BYTES, "no manifest entry"),
    ("missing_file", False, [_entry()], MAX_BYTES, "file is missing"),
    ("wrong_sha256", True, [_entry(sha256="0" * 64)], MAX_BYTES, "sha256 does not match"),
    ("wrong_size", True, [_entry(size_bytes=1)], MAX_BYTES, "size is"),
    ("over_limit", True, [_entry()], 4, "over the 4-byte limit"),
    ("empty_source", True, [_entry(source_url="  ")], MAX_BYTES, "empty source_url"),
    ("empty_license", True, [_entry(license="")], MAX_BYTES, "empty license"),
    ("missing_key", True, [_entry(license=DROP)], MAX_BYTES, "is missing ['license']"),
    ("outside_reference", True, [_entry(path="data/raw/placeholder.tsv")], MAX_BYTES, "outside data/reference/"),
    ("listed_twice", True, [_entry(), _entry()], MAX_BYTES, "listed twice"),
]


@pytest.mark.parametrize("write_file, entries, max_bytes, expected", [c[1:] for c in CASES], ids=[c[0] for c in CASES])
def test_reference_problems_catches_each_failure(tmp_path, write_file, entries, max_bytes, expected):
    ref_dir = tmp_path / REFERENCE
    ref_dir.mkdir(parents=True)
    if write_file:
        (ref_dir / "placeholder.tsv").write_bytes(PLACEHOLDER)
    problems = reference_problems(tmp_path, {"schema_version": 1, "files": entries}, max_bytes)
    if expected is None:
        assert problems == []
    else:
        assert any(expected in p for p in problems), problems


def test_dotfiles_and_the_manifest_need_no_entry(tmp_path):
    ref_dir = tmp_path / REFERENCE
    ref_dir.mkdir(parents=True)
    (ref_dir / ".DS_Store").write_bytes(b"x")
    (ref_dir / MANIFEST_NAME).write_text('{"schema_version": 1, "files": []}', encoding="utf-8")
    assert reference_problems(tmp_path, load_manifest(tmp_path)) == []
