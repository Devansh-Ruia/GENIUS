from __future__ import annotations

import json
from collections import Counter

import pytest

from _factories import FINDLAY_HEADER, findlay_row, write_findlay_xlsx
from vus_panel import config
from vus_panel.eval import pilot_data as data
from vus_panel.eval.split import FindlayFormatError, load_findlay


@pytest.fixture
def pilot_inputs(tmp_path, monkeypatch):
    # Synthetic chromosome positions, never biological evidence.
    classes = ["FUNC"] * 40 + ["INT"] * 30 + ["LOF"] * 35
    rows = [findlay_row(label, pos) + ["T", "C"] for pos, label in enumerate(classes, start=4100)]
    source = write_findlay_xlsx(tmp_path / "input.xlsx", rows,
                              header=FINDLAY_HEADER + ["transcript_ref", "transcript_alt"])
    sequence = "A" * 9000
    fasta = f">{data.REFERENCE_ACCESSION} synthetic chromosome\n{sequence}\n".encode()
    monkeypatch.setattr(data, "FINDLAY_SHA256", data.sha256_file(source))
    monkeypatch.setattr(data, "REFERENCE_LENGTH", len(sequence))
    monkeypatch.setattr(data, "REFERENCE_SEQUENCE_SHA256", data.digest_bytes(sequence.encode()))
    monkeypatch.setattr(data, "EXPECTED_COUNTS", dict(Counter(classes)))
    downloads = []

    def download(url, path):
        downloads.append(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.read_bytes() if url == data.FINDLAY_URL else fasta)

    monkeypatch.setattr(data, "_download", download)
    return downloads, sequence


def test_full_validation_collects_errors_instead_of_stopping(tmp_path):
    rows = [findlay_row("FUNC", 1), findlay_row("LOF", 1),
            findlay_row("INT", 0), findlay_row("FUNC", 2, ref="N"),
            findlay_row("LOF", 3), findlay_row("INT", 4)]
    rows[2][6] = "NaN"
    rows[3][1] = None
    rows[4][4] = "C"
    rows[5][6] = None
    source = write_findlay_xlsx(tmp_path / "bad.xlsx", rows)
    report = {}
    with pytest.raises(FindlayFormatError):
        load_findlay(source, reference_sequence="A" * 5, validation_report=report)
    assert report["rows"] == 6
    assert not report["ok"]
    assert report["issue_counts"] == {
        "missing_fields": 2, "invalid_coordinates": 2, "invalid_alleles": 1,
        "invalid_classes": 0, "nonfinite_scores": 2, "duplicate_ids": 1,
        "reference_mismatches": 1, "transcript_mismatches": 0,
    }


def test_one_based_forward_alleles_are_preserved_and_transcripts_checked(tmp_path):
    source = write_findlay_xlsx(tmp_path / "strand.xlsx", [findlay_row("FUNC", 2, ref="C", alt="T") + ["G", "A"]],
                               header=FINDLAY_HEADER + ["transcript_ref", "transcript_alt"])
    report = {}
    variants = load_findlay(source, reference_sequence="ACA", validation_report=report)
    assert variants[0].variant_id == "GRCh37-17-2-C-T"
    assert report["transcript_complement_checked"]
    with pytest.raises(FindlayFormatError, match="genomic reference"):
        load_findlay(source, reference_sequence="GGA")


def test_transcript_mismatch_is_not_silently_complemented(tmp_path):
    source = write_findlay_xlsx(tmp_path / "strand.xlsx", [findlay_row("FUNC", 1) + ["A", "G"]],
                               header=FINDLAY_HEADER + ["transcript_ref", "transcript_alt"])
    report = {}
    with pytest.raises(FindlayFormatError, match="transcript alleles"):
        load_findlay(source, reference_sequence="A", validation_report=report)
    assert report["issue_counts"]["transcript_mismatches"] == 1


def test_prepare_freezes_deterministic_disjoint_ids_and_reuses_bytes(pilot_inputs):
    downloads, _ = pilot_inputs
    selection, variants = data.prepare(download=True)
    directory = config.splits_dir()
    before = {name: (directory / name).read_bytes() for name in data.FREEZE_FILES}
    second, _ = data.prepare()
    assert second == selection
    assert {name: (directory / name).read_bytes() for name in data.FREEZE_FILES} == before
    assert downloads == [data.FINDLAY_URL, data.REFERENCE_URL]
    calibration = (directory / "calibration.txt").read_text().splitlines()
    test = (directory / "test.txt").read_text().splitlines()
    assert len(calibration) == 21 and len(test) == 84
    assert not set(calibration) & set(test)
    assert set(selection["variant_ids"]) <= set(calibration)
    labels = {v.variant_id: v.findlay_function_class for v in variants}
    assert Counter(labels[vid] for vid in selection["variant_ids"]) == data.PILOT_COUNTS
    assert data.select_pilot(list(reversed(variants)), calibration) == selection["variant_ids"]
    assert selection["variant_ids"] == [f"GRCh37-17-{pos}-A-G" for pos in (
        4121, 4187, 4125, 4141, 4163, 4198, 4147, 4160, 4110, 4190,
        4102, 4170, 4148, 4108, 4200, 4164, 4139, 4183, 4124, 4201,
    )]
    assert "findlay_function_class" not in before["pilot20.json"].decode()
    assert "findlay_score" not in before["pilot20.json"].decode()
    meta = data.read_json(directory / "split_meta.json")
    for name in ("calibration.txt", "test.txt"):
        assert meta["id_file_sha256"][name] == data.sha256_file(directory / name)
    assert selection["split_meta_sha256"] == data.sha256_file(directory / "split_meta.json")


@pytest.mark.parametrize("name", data.FREEZE_FILES)
def test_corrupt_freezes_are_rejected_without_replacement(pilot_inputs, name):
    data.prepare(download=True)
    path = config.splits_dir() / name
    corrupted = path.read_bytes() + b" "
    path.write_bytes(corrupted)
    with pytest.raises(ValueError, match="incompatible|corrupt"):
        data.prepare()
    assert path.read_bytes() == corrupted


def test_incomplete_freeze_is_rejected_without_overwrite(pilot_inputs):
    data.prepare(download=True)
    directory = config.splits_dir()
    (directory / "pilot20.json").unlink()
    before = (directory / "calibration.txt").read_bytes()
    with pytest.raises(ValueError, match="incomplete freeze"):
        data.prepare()
    assert (directory / "calibration.txt").read_bytes() == before
    assert not (directory / "pilot20.json").exists()


def test_byte_identical_download_on_another_date_reuses_original_provenance(pilot_inputs):
    data.prepare(download=True)
    before = {name: (config.splits_dir() / name).read_bytes() for name in data.FREEZE_FILES}
    source_receipt = data.default_findlay_path().with_suffix(".source.json")
    source = data.read_json(source_receipt)
    source.update(downloaded_at="2027-01-01T00:00:00+00:00", verified_at="2027-01-01T00:00:01+00:00",
                  download_note="downloaded on another machine")
    data.write_json(source_receipt, source)
    reference_receipt = config.cache_dir() / "reference" / "reference.json"
    reference = data.read_json(reference_receipt)
    reference["downloaded_at"] = "2027-01-01T00:00:00+00:00"
    data.write_json(reference_receipt, reference)
    data.prepare()
    assert {name: (config.splits_dir() / name).read_bytes() for name in data.FREEZE_FILES} == before


def test_pinned_download_and_reference_sequence_corruption_rejected(pilot_inputs):
    data.prepare(download=True)
    path = config.cache_dir() / "reference" / f"{data.REFERENCE_ACCESSION}.fasta"
    contents = path.read_text()
    path.write_text(contents[:-2] + "C\n")
    with pytest.raises(ValueError, match="pinned"):
        data.prepare()


def test_wrong_reference_accession_or_bounds_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "REFERENCE_LENGTH", 4)
    monkeypatch.setattr(data, "REFERENCE_SEQUENCE_SHA256", data.digest_bytes(b"ACGT"))
    path = tmp_path / "reference.fasta"
    path.write_text(">NC_000017.11 wrong version\nACGT\n")
    with pytest.raises(ValueError, match="NC_000017.10"):
        data.read_reference(path)
    path.write_text(">NC_000017.10 correct version\nACGT\n>second\nACGT\n")
    with pytest.raises(ValueError, match="exactly one"):
        data.read_reference(path)


def test_run_requires_freeze_before_any_scoring(pilot_inputs):
    with pytest.raises(ValueError, match="not frozen"):
        data.prepare(download=True, require_existing=True)


def test_missing_workbook_download_is_opt_in(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare --download"):
        data.ensure_findlay()
