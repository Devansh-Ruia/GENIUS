"""Cache phyloP 100-way (hg19) for every base in the BRCA1 region.

    python scripts/fetch_phylop.py            # bigWig if pyBigWig works, else UCSC API
    python scripts/fetch_phylop.py --both     # fetch both ways and compare (verification)

Output: data/cache/phylop100way.csv with chrom, pos, phylop
"""
from __future__ import annotations

import argparse
import math
import sys

import pandas as pd
import requests

from cache_common import UA, region, write_cache

BW_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/phyloP100way/hg19.100way.phyloP100way.bw"
API_URL = "https://api.genome.ucsc.edu/getData/track"
API_TRACK = "phyloP100wayAll"  # hg19 track name for phyloP 100-way


def via_bigwig(chrom: str, start: int, end: int) -> pd.DataFrame:
    import pyBigWig  # often fails to install on Windows; the API path covers that

    bw = pyBigWig.open(BW_URL)
    vals = bw.values(f"chr{chrom}", start - 1, end)  # 0-based half-open
    bw.close()
    rows = [(chrom, start + i, None if (v is None or math.isnan(v)) else float(v)) for i, v in enumerate(vals)]
    return pd.DataFrame(rows, columns=["chrom", "pos", "phylop"])


def _find_intervals(obj):
    """The API nests the interval list under the chrom or track key; find it."""
    if isinstance(obj, list) and obj and isinstance(obj[0], dict) and {"start", "end", "value"} <= obj[0].keys():
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            found = _find_intervals(v)
            if found is not None:
                return found
    return None


def via_api(chrom: str, start: int, end: int, chunk: int = 20_000) -> pd.DataFrame:
    vals: dict[int, float] = {}
    for s in range(start - 1, end, chunk):  # 0-based
        e = min(s + chunk, end)
        params = {"genome": "hg19", "track": API_TRACK, "chrom": f"chr{chrom}", "start": s, "end": e}
        r = requests.get(API_URL, params=params, headers=UA, timeout=60)
        r.raise_for_status()
        intervals = _find_intervals(r.json())
        if intervals is None:
            sys.exit(f"Unexpected UCSC API response for {params}: {str(r.json())[:300]}")
        for iv in intervals:
            for p0 in range(max(iv["start"], s), min(iv["end"], e)):  # intervals may span >1 base
                vals[p0 + 1] = float(iv["value"])
    rows = [(chrom, p, vals.get(p)) for p in range(start, end + 1)]
    return pd.DataFrame(rows, columns=["chrom", "pos", "phylop"])


def summarize(df: pd.DataFrame, label: str) -> None:
    v = df["phylop"].dropna()
    print(f"[{label}] {len(df)} positions, {df['phylop'].isna().sum()} missing, "
          f"range {v.min():.2f} to {v.max():.2f}, median {v.median():.2f}")
    if len(v) and (v.min() < -25 or v.max() > 15):
        print(f"[{label}] WARNING: values outside the usual phyloP range, check the source.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--both", action="store_true", help="fetch via bigWig and API and compare")
    args = ap.parse_args()

    chrom, start, end, how = region()
    print(f"region chr{chrom}:{start}-{end} ({how}, {end - start + 1} bp)")

    bw_df = api_df = None
    try:
        bw_df = via_bigwig(chrom, start, end)
        summarize(bw_df, "bigWig")
    except Exception as exc:  # noqa: BLE001
        print(f"bigWig route failed ({exc}); using the UCSC API.")
    if bw_df is None or args.both:
        api_df = via_api(chrom, start, end)
        summarize(api_df, "API")

    if bw_df is not None and api_df is not None:
        m = bw_df.merge(api_df, on=["chrom", "pos"], suffixes=("_bw", "_api")).dropna()
        diff = (m["phylop_bw"] - m["phylop_api"]).abs()
        print(f"bigWig vs API: {len(m)} shared positions, max abs diff {diff.max():.4f}, "
              f"{(diff > 1e-3).sum()} positions differ by >0.001")

    df = bw_df if bw_df is not None else api_df
    source = BW_URL if bw_df is not None else f"{API_URL}?genome=hg19;track={API_TRACK}"
    write_cache(df, "phylop100way", {"source": source, "track": "phyloP 100-way vertebrate",
                                     "region": f"chr{chrom}:{start}-{end}", "region_basis": how})


if __name__ == "__main__":
    main()
