"""Reference points that let an agent read its own number.

A conservation score of 6.457 means nothing on its own. This script works
out, from the CALIBRATION SPLIT ONLY, how phyloP and gnomAD frequency line
up with the Findlay labels, so an agent can be told:

    "6.457 is the 99th percentile here, and among calibration variants that
     conserved, 78% were loss of function."

    python scripts/calibration_refs.py

Writes data/cache/calibration_refs.json.

The test split is never read. Every number here comes from the 779
calibration variants, so nothing fitted on labels touches the variants the
panel is scored on. The script fails rather than guessing if the split file
is missing.
"""
from __future__ import annotations

import datetime as dt
import json
import sys

import pandas as pd

from cache_common import CACHE, REPO, load_findlay, write_cache  # noqa: F401  (write_cache unused, kept for parity)

SPLITS = REPO / "data" / "splits"
# Findlay's three classes. The Evo 2 BRCA1 notebook treats LOF as the
# positive class and FUNC plus INT as negative; we follow that.
POSITIVE = ("LOF",)
BINS = 5


def _variant_ids(df: pd.DataFrame) -> pd.Series:
    return "GRCh37-" + df["chrom"] + "-" + df["pos"].astype(str) + "-" + df["ref"] + "-" + df["alt"]


def _read_split(name: str) -> set[str]:
    path = SPLITS / f"{name}.txt"
    if not path.exists():
        sys.exit(f"Missing {path}. Run: python -m vus_panel.eval.split --findlay data/raw/<file>.xlsx")
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _bin_table(frame: pd.DataFrame, value: str) -> list[dict]:
    """Quantile bins of `value`, with the share of positives in each."""
    usable = frame.dropna(subset=[value])
    if len(usable) < BINS * 5:
        return []
    try:
        labels = pd.qcut(usable[value], BINS, labels=False, duplicates="drop")
    except ValueError:
        return []
    out = []
    for b in sorted(pd.unique(labels.dropna())):
        rows = usable[labels == b]
        out.append({
            "bin": int(b) + 1,
            "range": [round(float(rows[value].min()), 6), round(float(rows[value].max()), 6)],
            "n": int(len(rows)),
            "fraction_lof": round(float(rows["is_positive"].mean()), 4),
        })
    return out


def main() -> None:
    calibration = _read_split("calibration")
    test_ids = _read_split("test")
    print(f"calibration {len(calibration)} IDs, test {len(test_ids)} IDs (test is never read below)")

    findlay = load_findlay()
    findlay["variant_id"] = _variant_ids(findlay)
    cal = findlay[findlay["variant_id"].isin(calibration)].copy()
    if cal.empty:
        sys.exit("No calibration variants matched the Findlay table. Check the split and the xlsx are the same file.")
    leaked = cal["variant_id"].isin(test_ids).sum()
    if leaked:
        sys.exit(f"{leaked} calibration IDs are also in the test split. Stop and re-run the split.")
    cal["is_positive"] = cal["func_class"].isin(POSITIVE).astype(int)
    print(f"calibration variants matched: {len(cal)}, loss of function: {int(cal['is_positive'].sum())}")

    phylop = pd.read_csv(CACHE / "phylop100way.csv", dtype={"chrom": str})
    gnomad = pd.read_csv(CACHE / "gnomad_v2_1_1.csv", dtype={"chrom": str})
    merged = (cal.merge(phylop, on=["chrom", "pos"], how="left")
                 .merge(gnomad[["chrom", "pos", "ref", "alt", "ac", "an", "af"]],
                        on=["chrom", "pos", "ref", "alt"], how="left"))

    observed = merged[merged["af"].notna() & (merged["an"].fillna(0) > 0)]
    absent = merged[merged["af"].isna()]

    refs = {
        "built_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "split": "calibration only",
        "n_calibration": int(len(merged)),
        "positive_class": "LOF (Findlay func.class); FUNC and INT are negative",
        "base_rate_lof": round(float(merged["is_positive"].mean()), 4),
        "conservation": {
            "source": "phyloP 100-way vertebrate (hg19), UCSC",
            "note": "Scored per position, so every alt at one position shares a score.",
            "n_with_score": int(merged["phylop"].notna().sum()),
            "bins_low_to_high": _bin_table(merged, "phylop"),
        },
        "population": {
            "source": "gnomAD v2.1.1 (GRCh37), exomes and genomes",
            "n_observed": int(len(observed)),
            "n_not_observed": int(len(absent)),
            "fraction_lof_when_observed": round(float(observed["is_positive"].mean()), 4) if len(observed) else None,
            "fraction_lof_when_not_observed": round(float(absent["is_positive"].mean()), 4) if len(absent) else None,
            "note": (
                "Most calibration variants are absent from gnomAD, so absence barely shifts the base "
                "rate and is weak evidence on its own. Being observed at an appreciable frequency is "
                "the informative case."
            ),
            "bins_low_to_high_af_when_observed": _bin_table(observed, "af"),
        },
    }

    out = CACHE / "calibration_refs.json"
    out.write_text(json.dumps(refs, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(f"  base rate LOF: {refs['base_rate_lof']:.1%}")
    for b in refs["conservation"]["bins_low_to_high"]:
        print(f"  phyloP {b['range'][0]:>7.3f} to {b['range'][1]:>6.3f}  n={b['n']:<4} LOF {b['fraction_lof']:.1%}")
    p = refs["population"]
    print(f"  in gnomAD: n={p['n_observed']} LOF {p['fraction_lof_when_observed']}")
    print(f"  absent:    n={p['n_not_observed']} LOF {p['fraction_lof_when_not_observed']}")


if __name__ == "__main__":
    main()
