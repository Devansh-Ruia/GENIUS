"""Write the cache files the tool modules look for.

fetch_phylop.py and fetch_gnomad.py write working files named after their
sources. tools/conservation.py and tools/population.py look for build-tagged
TSVs instead. This converts one into the other, so the fetch scripts stay
about the source and the tools stay about the build.

    python scripts/build_tool_caches.py

Writes, for GRCh37 only (there is no liftover step, so GRCh38 stays absent):
    data/cache/conservation_brca1_GRCh37.tsv   chrom, pos, score
    data/cache/gnomad_brca1_GRCh37.tsv         chrom, pos, ref, alt, ac, an, af
"""
from __future__ import annotations

import sys

import pandas as pd

from cache_common import CACHE, region

BUILD = "GRCh37"


def main() -> None:
    chrom, start, end, how = region()
    print(f"region {chrom}:{start}-{end} ({how})")

    phylop_in = CACHE / "phylop100way.csv"
    gnomad_in = CACHE / "gnomad_v2_1_1.csv"
    for path in (phylop_in, gnomad_in):
        if not path.exists():
            sys.exit(f"Missing {path}. Run fetch_phylop.py and fetch_gnomad.py first.")

    ph = pd.read_csv(phylop_in, dtype={"chrom": str})
    ph = ph.rename(columns={"phylop": "score"})[["chrom", "pos", "score"]].dropna(subset=["score"])
    out = CACHE / f"conservation_brca1_{BUILD}.tsv"
    ph.to_csv(out, sep="\t", index=False)
    print(f"wrote {out} ({len(ph)} positions)")

    gn = pd.read_csv(gnomad_in, dtype={"chrom": str})
    cols = ["chrom", "pos", "ref", "alt", "ac", "an", "af"]
    missing = [c for c in cols if c not in gn.columns]
    if missing:
        sys.exit(f"{gnomad_in} is missing columns {missing}. Re-run fetch_gnomad.py.")
    gn = gn[cols].copy()
    # A site with an=0 was not callable there; that is "can't say", not "absent".
    gn.loc[gn["an"] <= 0, ["ac", "af"]] = pd.NA
    out = CACHE / f"gnomad_brca1_{BUILD}.tsv"
    gn.to_csv(out, sep="\t", index=False)
    print(f"wrote {out} ({len(gn)} variants, {int((gn['ac'].fillna(0) > 0).sum())} with AC > 0)")

    cov = CACHE / "gnomad_v2_1_1_coverage.csv"
    if not cov.exists():
        print("note: no coverage cache. A variant absent from the file means not observed, "
              "which is weak evidence on its own. Say so in the population agent's rationale.")


if __name__ == "__main__":
    main()
