"""Shared helpers for the data-cache scripts.

Coordinate convention for every cache file: GRCh37, chrom without "chr",
pos is 1-based (same as Findlay, gnomAD and the browsers' displayed numbers).
UCSC files and APIs are 0-based half-open internally; convert only at the edge.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"
CACHE = REPO / "data" / "cache"


def _find_findlay() -> Path:
    """Any .xlsx in data/raw/ — the file is named differently on Nature and PMC.

    Nature: 41586_2018_461_MOESM3_ESM.xlsx   PMC: NIHMS1501643-supplement-2.xlsx
    """
    hits = sorted(RAW.glob("*.xlsx"))
    if len(hits) > 1:
        sys.exit(f"More than one .xlsx in {RAW}: {[h.name for h in hits]}. Leave only the Findlay file.")
    return hits[0] if hits else RAW / "findlay-not-found.xlsx"


FINDLAY = _find_findlay()

# BRCA1 gene span on GRCh37, 1-based inclusive. Only used if Findlay is missing.
DEFAULT_REGION = ("17", 41196312, 41277500)
PAD = 1000  # bases added on each side of the Findlay span

UA = {"User-Agent": "GENIUS-hackathon-cache/0.1 (ScienceClaw 2026)"}


def load_findlay() -> pd.DataFrame:
    """Return Findlay SNVs with columns chrom, pos, ref, alt, score, func_class.

    The sheet has title rows above the header, so find the header row by
    looking for the 'position (hg19)' cell instead of hardcoding it.
    """
    if not FINDLAY.exists():
        sys.exit(f"No .xlsx found in {RAW}. Put the Findlay 2018 supplementary file there first.")
    raw = pd.read_excel(FINDLAY, header=None)
    hdr = next(
        (i for i, row in raw.iterrows() if row.astype(str).str.strip().eq("position (hg19)").any()),
        None,
    )
    if hdr is None:
        sys.exit("Could not find a 'position (hg19)' header in the Findlay sheet. Check the file.")
    df = pd.read_excel(FINDLAY, header=hdr)
    df.columns = [str(c).strip() for c in df.columns]
    out = pd.DataFrame(
        {
            "chrom": df["chromosome"].astype(str).str.replace("chr", "", regex=False),
            "pos": df["position (hg19)"].astype(int),
            "ref": df["reference"].astype(str).str.upper(),
            "alt": df["alt"].astype(str).str.upper(),
            "score": df.get("function.score.mean"),
            "func_class": df.get("func.class"),
        }
    )
    return out.drop_duplicates(["chrom", "pos", "ref", "alt"]).reset_index(drop=True)


def region() -> tuple[str, int, int, str]:
    """(chrom, start, end, how) covering every Findlay variant plus padding."""
    if FINDLAY.exists():
        df = load_findlay()
        chroms = df["chrom"].unique()
        if len(chroms) != 1:
            sys.exit(f"Findlay variants span several chromosomes: {chroms}")
        return chroms[0], int(df["pos"].min()) - PAD, int(df["pos"].max()) + PAD, "findlay span + padding"
    print("WARNING: Findlay file not found, using the default BRCA1 gene span.", file=sys.stderr)
    return (*DEFAULT_REGION, "default BRCA1 span")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_cache(df: pd.DataFrame, name: str, meta: dict) -> Path:
    """Write data/cache/<name>.csv plus <name>.meta.json with provenance."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.csv"
    df.to_csv(path, index=False)
    meta = {
        **meta,
        "genome_build": "GRCh37",
        "coordinates": "1-based, chrom without chr prefix",
        "rows": int(len(df)),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "sha256": sha256(path),
    }
    (CACHE / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {path} ({len(df)} rows)")
    return path
