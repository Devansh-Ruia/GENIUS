from __future__ import annotations

import hashlib
import json
import random
import re

import pytest

from _factories import FINDLAY_HEADER, findlay_row, write_findlay_xlsx
from vus_panel.eval import split
from vus_panel.eval.split import (
    FindlayFileMissing,
    FindlayFormatError,
    load_findlay,
    main,
    stratified_split,
)

ID_RE = re.compile(r"^GRCh37-17-[0-9]+-[ACGT]-[ACGT]$")
CLASS_PLAN = ["FUNC"] * 10 + ["INT"] * 5 + ["LOF"] * 10


def placeholder_rows():
    # Positions 1..25 on chromosome 17 are placeholders, not BRCA1 coordinates.
    return [findlay_row(func_class, pos) for pos, func_class in enumerate(CLASS_PLAN, start=1)]


@pytest.fixture
def findlay_file(tmp_path):
    return write_findlay_xlsx(tmp_path / "findlay.xlsx", placeholder_rows())


def test_missing_file_raises_clear_error_without_downloading(tmp_path):
    with pytest.raises(FindlayFileMissing) as excinfo:
        load_findlay(tmp_path / "absent.xlsx")
    message = str(excinfo.value)
    assert split.FINDLAY_FILENAME in message
    assert "never downloads" in message
    assert isinstance(excinfo.value, FileNotFoundError)


def test_default_path_is_under_data_raw(tmp_path):
    assert split.default_findlay_path() == tmp_path / "data" / "raw" / split.FINDLAY_FILENAME
    with pytest.raises(FindlayFileMissing):
        load_findlay()


def test_loads_rows_by_column_name_from_header_row_three(findlay_file):
    variants = load_findlay(findlay_file)
    assert len(variants) == 25
    assert all(v.genome_build == "GRCh37" for v in variants)
    assert all(ID_RE.match(v.variant_id) for v in variants)
    counts = {c: sum(v.findlay_function_class == c for v in variants) for c in ("FUNC", "INT", "LOF")}
    assert counts == {"FUNC": 10, "INT": 5, "LOF": 10}


def test_accepts_chr_prefix_float_positions_and_blank_rows(tmp_path):
    rows = [findlay_row("FUNC", 1.0, chrom="chr17"), [None] * 7, findlay_row("LOF", 2)]
    variants = load_findlay(write_findlay_xlsx(tmp_path / "f.xlsx", rows))
    assert [v.variant_id for v in variants] == ["GRCh37-17-1-A-G", "GRCh37-17-2-A-G"]


def test_missing_column_raises(tmp_path):
    header = [name for name in FINDLAY_HEADER if name != "func.class"] + ["something else"]
    path = write_findlay_xlsx(tmp_path / "f.xlsx", placeholder_rows(), header=header)
    with pytest.raises(FindlayFormatError, match="func.class"):
        load_findlay(path)


def test_unknown_class_raises(tmp_path):
    path = write_findlay_xlsx(tmp_path / "f.xlsx", [findlay_row("FUNC", 1), findlay_row("BENIGN", 2)])
    with pytest.raises(FindlayFormatError, match="row 5"):
        load_findlay(path)


def test_duplicate_variant_raises(tmp_path):
    path = write_findlay_xlsx(tmp_path / "f.xlsx", [findlay_row("FUNC", 1), findlay_row("LOF", 1)])
    with pytest.raises(FindlayFormatError, match="duplicate"):
        load_findlay(path)


def test_invalid_base_raises(tmp_path):
    path = write_findlay_xlsx(tmp_path / "f.xlsx", [findlay_row("FUNC", 1, ref="A", alt="A")])
    with pytest.raises(FindlayFormatError, match="row 4"):
        load_findlay(path)


def _pairs():
    return [(f"GRCh37-17-{pos}-A-G", func_class) for pos, func_class in enumerate(CLASS_PLAN, start=1)]


def test_split_is_deterministic_and_order_independent():
    pairs = _pairs()
    shuffled = pairs[:]
    random.Random(1).shuffle(shuffled)
    first = stratified_split(pairs)
    assert stratified_split(pairs) == first
    assert stratified_split(shuffled) == first


def test_split_is_stratified_and_disjoint():
    result = stratified_split(_pairs(), seed=0, calibration_fraction=0.2)
    assert result.counts == {
        "FUNC": {"calibration": 2, "test": 8},
        "INT": {"calibration": 1, "test": 4},
        "LOF": {"calibration": 2, "test": 8},
    }
    assert not set(result.calibration) & set(result.test)
    assert set(result.calibration) | set(result.test) == {vid for vid, _ in _pairs()}


def test_calibration_takes_lowest_sha256_within_each_class():
    result = stratified_split(_pairs(), seed=0)
    for func_class, n in (("FUNC", 2), ("INT", 1), ("LOF", 2)):
        ids = [vid for vid, c in _pairs() if c == func_class]
        expected = sorted(ids, key=lambda vid: hashlib.sha256(f"0:{vid}".encode()).hexdigest())[:n]
        assert set(expected) <= set(result.calibration)


def test_seed_changes_the_assignment():
    assert stratified_split(_pairs(), seed=0).calibration != stratified_split(_pairs(), seed=1).calibration


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.2, 1.5])
def test_split_rejects_bad_fraction(fraction):
    with pytest.raises(ValueError):
        stratified_split(_pairs(), calibration_fraction=fraction)


def test_split_rejects_duplicates_and_unknown_classes():
    with pytest.raises(ValueError, match="duplicate"):
        stratified_split([("v1", "FUNC"), ("v1", "LOF")])
    with pytest.raises(ValueError, match="unknown"):
        stratified_split([("v1", "BENIGN")])


def test_cli_writes_id_lists_and_metadata(findlay_file, tmp_path):
    out_dir = tmp_path / "splits"
    assert main(["--findlay", str(findlay_file), "--out", str(out_dir)]) == 0
    calibration = (out_dir / "calibration.txt").read_text(encoding="utf-8").splitlines()
    test = (out_dir / "test.txt").read_text(encoding="utf-8").splitlines()
    assert len(calibration) == 5 and len(test) == 20
    assert all(ID_RE.match(line) for line in calibration + test)  # IDs only, no scores or classes
    meta = json.loads((out_dir / "split_meta.json").read_text(encoding="utf-8"))
    assert meta["seed"] == 0
    assert meta["calibration_fraction"] == 0.2
    assert meta["genome_build"] == "GRCh37"
    assert meta["source_sha256"] == hashlib.sha256(findlay_file.read_bytes()).hexdigest()


def test_cli_reports_missing_file(tmp_path, capsys):
    assert main(["--findlay", str(tmp_path / "absent.xlsx"), "--out", str(tmp_path / "splits")]) == 2
    assert "Findlay file not found" in capsys.readouterr().err
    assert not (tmp_path / "splits").exists()


def test_cli_refuses_to_overwrite_without_force(findlay_file, tmp_path, capsys):
    out_dir = tmp_path / "splits"
    assert main(["--findlay", str(findlay_file), "--out", str(out_dir)]) == 0
    before = (out_dir / "test.txt").read_text(encoding="utf-8")
    assert main(["--findlay", str(findlay_file), "--out", str(out_dir), "--seed", "1"]) == 2
    assert "--force" in capsys.readouterr().err
    assert (out_dir / "test.txt").read_text(encoding="utf-8") == before
    assert main(["--findlay", str(findlay_file), "--out", str(out_dir), "--seed", "1", "--force"]) == 0
    assert (out_dir / "test.txt").read_text(encoding="utf-8") != before


def test_cli_reports_format_errors_without_a_traceback(tmp_path, capsys):
    path = write_findlay_xlsx(tmp_path / "f.xlsx", [findlay_row("BENIGN", 1)])
    assert main(["--findlay", str(path), "--out", str(tmp_path / "splits")]) == 2
    assert "unknown func.class" in capsys.readouterr().err


@pytest.mark.parametrize("name, content", [("f.xlsx", b"not a zip file"), ("f.csv", b"a,b\n")], ids=["bad_zip", "wrong_type"])
def test_corrupt_findlay_file_raises_format_error(tmp_path, capsys, name, content):
    path = tmp_path / name
    path.write_bytes(content)
    with pytest.raises(FindlayFormatError, match="not a readable .xlsx"):
        load_findlay(path)
    assert main(["--findlay", str(path), "--out", str(tmp_path / "splits")]) == 2
    assert "not a readable .xlsx" in capsys.readouterr().err
