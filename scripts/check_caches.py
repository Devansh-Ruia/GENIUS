"""Check the caches before anyone builds on them.

    python scripts/check_caches.py

1. Reference bases: Findlay ref alleles vs the hg19 sequence from UCSC.
   Catches off-by-one and strand mistakes, the most common silent bug.
2. Coverage: how many Findlay variants have a phyloP value / a gnomAD record.
3. Cross-source: gnomAD API vs BigQuery, if both caches exist.
4. Spot-check sheet: data/cache/verification.md with ~12 variants, our cached
   values, and direct UCSC and gnomAD browser links to compare by hand or with Claude.
"""
from __future__ import annotations

import sys

import pandas as pd
import requests

from cache_common import CACHE, UA, load_findlay

COMP = str.maketrans("ACGT", "TGCA")


def ucsc_sequence(chrom: str, start: int, end: int) -> str:
    """hg19 sequence for 1-based inclusive [start, end]."""
    r = requests.get("https://api.genome.ucsc.edu/getData/sequence",
                     params={"genome": "hg19", "chrom": f"chr{chrom}", "start": start - 1, "end": end},
                     headers=UA, timeout=60)
    r.raise_for_status()
    return r.json()["dna"].upper()


def check_ref(fd: pd.DataFrame) -> bool:
    chrom = fd["chrom"].iloc[0]
    lo, hi = int(fd["pos"].min()) - 2, int(fd["pos"].max()) + 2
    seq = ucsc_sequence(chrom, lo, hi)
    base = lambda p: seq[p - lo] if lo <= p <= hi else "N"  # noqa: E731
    tests = {
        "as given": fd.apply(lambda r: base(r.pos) == r.ref, axis=1).mean(),
        "shifted -1": fd.apply(lambda r: base(r.pos - 1) == r.ref, axis=1).mean(),
        "shifted +1": fd.apply(lambda r: base(r.pos + 1) == r.ref, axis=1).mean(),
        "complemented": fd.apply(lambda r: base(r.pos) == r.ref.translate(COMP), axis=1).mean(),
    }
    print("1. Findlay ref allele vs hg19 (fraction matching):")
    for k, v in tests.items():
        print(f"   {k:13s} {v:6.1%}")
    ok = tests["as given"] > 0.99
    print("   PASS: positions and strand line up with hg19" if ok else
          "   FAIL: fix coordinates before going on. The best-matching row above tells you how.")
    return ok


def main() -> None:
    fd = load_findlay()
    print(f"Findlay: {len(fd)} SNVs, chr{fd['chrom'].iloc[0]}:{fd['pos'].min()}-{fd['pos'].max()}\n")
    ref_ok = check_ref(fd)

    print("\n2. Cache coverage of Findlay variants:")
    ph_path, gn_path = CACHE / "phylop100way.csv", CACHE / "gnomad_v2_1_1.csv"
    if not ph_path.exists() or not gn_path.exists():
        sys.exit("   Run fetch_phylop.py and fetch_gnomad.py first.")
    ph = pd.read_csv(ph_path, dtype={"chrom": str})
    gn = pd.read_csv(gn_path, dtype={"chrom": str})
    m = (fd.merge(ph, on=["chrom", "pos"], how="left")
           .merge(gn[["chrom", "pos", "ref", "alt", "ac", "an", "af"]], on=["chrom", "pos", "ref", "alt"], how="left"))
    print(f"   phyloP present:   {m['phylop'].notna().mean():.1%} of variants (expect ~100%)")
    print(f"   in gnomAD:        {m['ac'].notna().sum()} variants (expect a small minority; most are absent)")
    if (CACHE / "gnomad_v2_1_1_coverage.csv").exists():
        cov = pd.read_csv(CACHE / "gnomad_v2_1_1_coverage.csv", dtype={"chrom": str})
        mc = fd.merge(cov, on=["chrom", "pos"], how="left")
        col = "exome_over_20" if "exome_over_20" in mc else mc.columns[-1]
        print(f"   exome coverage:   {mc[col].notna().mean():.1%} of positions have a coverage value")
    else:
        print("   coverage cache missing: the population agent must treat 'absent' as weak evidence")

    print("\n3. gnomAD API vs BigQuery:")
    bq_path = CACHE / "gnomad_v2_1_1_bq.csv"
    if bq_path.exists():
        bq = pd.read_csv(bq_path)
        j = gn.merge(bq, on="variant_id", how="outer", suffixes=("", "_bq"), indicator=True)
        print(f"   only in API: {(j['_merge'] == 'left_only').sum()}, only in BigQuery: "
              f"{(j['_merge'] == 'right_only').sum()}, in both: {(j['_merge'] == 'both').sum()}")
        both = j[j["_merge"] == "both"]
        for kind in ("exome", "genome"):
            a, b = both.get(f"{kind}_ac"), both.get(f"{kind}_ac_bq")
            if a is not None and b is not None:
                print(f"   {kind} AC mismatches: {((a.fillna(-1) != b.fillna(-1))).sum()}")
        print("   Small differences can come from filtered variants; large ones mean a coordinate bug.")
    else:
        print("   skipped (run fetch_gnomad.py --bigquery PROJECT to enable)")

    # 4. spot-check sheet
    seen = m[m["ac"].notna()].sort_values("af", ascending=False)
    unseen = m[m["ac"].isna()]
    pick = pd.concat([
        seen.head(3), seen.tail(1),
        unseen.sample(min(4, len(unseen)), random_state=0),
        m.nlargest(2, "phylop"), m.nsmallest(2, "phylop"),
    ]).drop_duplicates(["pos", "ref", "alt"])
    lines = [
        "# Cache spot-check", "",
        "For each row, open both links. UCSC: confirm the browser says **hg19** at the top, "
        "then hover the phyloP 100-way track at the highlighted base. gnomAD: confirm the dataset is "
        "**gnomAD v2.1.1**, then compare allele count, number and frequency. A 'variant not found' page "
        "should match an empty AC in our cache.", "",
        "| variant | our phyloP | our AC / AN / AF | UCSC | gnomAD | browser matches? |",
        "|---|---|---|---|---|---|",
    ]
    for r in pick.itertuples():
        vid = f"{r.chrom}-{r.pos}-{r.ref}-{r.alt}"
        ucsc = (f"https://genome.ucsc.edu/cgi-bin/hgTracks?db=hg19&position=chr{r.chrom}:{r.pos - 10}-{r.pos + 10}"
                f"&highlight=hg19.chr{r.chrom}:{r.pos}-{r.pos}&phyloP100wayAll=full")
        gnomad = f"https://gnomad.broadinstitute.org/variant/{vid}?dataset=gnomad_r2_1"
        afs = "absent" if pd.isna(r.ac) else f"{int(r.ac)} / {int(r.an)} / {r.af:.3g}"
        ph_v = "missing" if pd.isna(r.phylop) else f"{r.phylop:.3f}"
        lines.append(f"| {vid} | {ph_v} | {afs} | [UCSC]({ucsc}) | [gnomAD]({gnomad}) | |")
    out = CACHE / "verification.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"\n4. Spot-check sheet written to {out} ({len(pick)} variants)")
    if not ref_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
