"""Population tool: gnomAD allele frequency for the variant.

Owner: evidence-data. Reads a local cache file per genome build under
data/cache/. There is no liftover step, so a variant is only ever looked up
in the cache for its own build.

The cache is gnomAD v2.1.1 exomes plus genomes over the BRCA1 region,
written by scripts/build_tool_caches.py. Three outcomes are distinct and
must stay distinct:

    observed      the variant is in gnomAD, with a frequency
    not_observed  the site is in the cached region, the variant is not there
    no_call       gnomAD reports the site but could not call it (AN is 0)

"not_observed" is weak evidence on its own: almost every possible variant
is absent, because most have never been sequenced in anyone.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

from vus_panel import config
from vus_panel.schemas import GenomeBuild, ToolResult, Variant, reject_answer_key

TOOL = "population"
OWNER = "evidence-data"
DATASET = "gnomAD v2.1.1 (GRCh37), exomes and genomes"
CACHE_FILES: dict[str, str] = {
    "GRCh37": "gnomad_brca1_GRCh37.tsv",
    "GRCh38": "gnomad_brca1_GRCh38.tsv",
}


def cache_path(build: GenomeBuild) -> Path:
    return config.cache_dir() / CACHE_FILES[build]


def _number(raw: str | None) -> float | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


@lru_cache(maxsize=4)
def _load(path_str: str, mtime: float) -> tuple[dict[tuple[str, int, str, str], dict], set[int], int]:
    """Return {(chrom, pos, ref, alt): row}, the set of cached positions, and the row count."""
    rows: dict[tuple[str, int, str, str], dict] = {}
    positions: set[int] = set()
    with open(path_str, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            chrom = (row["chrom"] or "").strip().removeprefix("chr")
            pos = int(row["pos"])
            key = (chrom, pos, (row["ref"] or "").strip().upper(), (row["alt"] or "").strip().upper())
            rows[key] = {"ac": _number(row.get("ac")), "an": _number(row.get("an")), "af": _number(row.get("af"))}
            positions.add(pos)
    return rows, positions, len(rows)


def allele_frequency(variant: Variant) -> ToolResult:
    reject_answer_key(variant, f"tools.{TOOL}.allele_frequency")
    path = cache_path(variant.genome_build)
    if path.is_file():
        rows, positions, total = _load(str(path), path.stat().st_mtime)
        if not rows:
            return ToolResult(
                tool=TOOL,
                status="error",
                message=f"gnomAD cache at {config.display_path(path)} has no usable rows.",
            )
        if not positions or not (min(positions) <= variant.pos <= max(positions)):
            return ToolResult(
                tool=TOOL,
                status="unavailable",
                message=(
                    f"{variant.chrom}:{variant.pos} is outside the cached gnomAD region, so absence here "
                    "says nothing about how common the variant is."
                ),
                sources=[DATASET],
            )

        row = rows.get((variant.chrom, variant.pos, variant.ref, variant.alt))
        if row is None:
            return ToolResult(
                tool=TOOL,
                status="ok",
                values={"observation": "not_observed", "ac": 0, "an": None, "af": None},
                message=(
                    "This variant is not in gnomAD v2.1.1. Most possible variants are absent, so on its "
                    "own absence is weak evidence of rarity. No per-site coverage is cached, so a site "
                    "that was poorly covered cannot be told apart from one where the variant is truly rare."
                ),
                sources=[DATASET],
            )

        ac, an, af = row["ac"], row["an"], row["af"]
        if an is None or an <= 0 or ac is None or af is None:
            return ToolResult(
                tool=TOOL,
                status="ok",
                values={"observation": "no_call", "ac": None, "an": 0, "af": None},
                message=(
                    f"gnomAD lists {variant.chrom}:{variant.pos} {variant.ref}>{variant.alt} but reports no "
                    "called alleles there, so frequency is unknown rather than low."
                ),
                sources=[DATASET],
            )
        return ToolResult(
            tool=TOOL,
            status="ok",
            values={
                "observation": "observed",
                "ac": int(ac),
                "an": int(an),
                "af": af,
                "cached_variants_in_region": total,
            },
            message=(
                f"Seen {int(ac)} times in {int(an)} alleles in gnomAD v2.1.1, a frequency of {af:.3g}. "
                "A variant common in people not selected for disease is unlikely to be strongly damaging; "
                "a very rare one is only weak evidence either way."
            ),
            sources=[DATASET],
        )

    other_builds = [b for b in CACHE_FILES if b != variant.genome_build and cache_path(b).is_file()]
    if other_builds:
        return ToolResult(
            tool=TOOL,
            status="error",
            message=(
                f"gnomAD cache exists only for {', '.join(other_builds)}, but the variant is "
                f"{variant.genome_build}. There is no liftover step, so positions are not compared "
                "across builds."
            ),
        )
    return ToolResult(
        tool=TOOL,
        status="unavailable",
        message=(
            f"gnomAD cache not found at {config.display_path(path)}. "
            f"The {OWNER} subagent owns the fetch script that writes it."
        ),
    )
