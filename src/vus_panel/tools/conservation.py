"""Conservation tool: per-base conservation score at the variant position.

Owner: evidence-data. Reads a local cache file per genome build under
data/cache/. There is no liftover step, so a variant is only ever looked up
in the cache for its own build.

The cache is phyloP 100-way vertebrate from UCSC, one row per base of the
BRCA1 region, written by scripts/build_tool_caches.py. The score is per
position, not per substitution, so every alt at one position scores alike.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

from vus_panel import config
from vus_panel.schemas import GenomeBuild, ToolResult, Variant, reject_answer_key

TOOL = "conservation"
OWNER = "evidence-data"
TRACK = "phyloP 100-way vertebrate (hg19), UCSC"
CACHE_FILES: dict[str, str] = {
    "GRCh37": "conservation_brca1_GRCh37.tsv",
    "GRCh38": "conservation_brca1_GRCh38.tsv",
}


def cache_path(build: GenomeBuild) -> Path:
    return config.cache_dir() / CACHE_FILES[build]


@lru_cache(maxsize=4)
def _load(path_str: str, mtime: float) -> tuple[dict[tuple[str, int], float], float, float]:
    """Return {(chrom, pos): score}, and the region's min and max score.

    mtime is part of the key so an updated cache file is re-read rather than
    served stale from this process's memory.
    """
    scores: dict[tuple[str, int], float] = {}
    with open(path_str, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            raw = (row.get("score") or "").strip()
            if not raw:
                continue
            chrom = (row["chrom"] or "").strip().removeprefix("chr")
            scores[(chrom, int(row["pos"]))] = float(raw)
    if not scores:
        return {}, 0.0, 0.0
    values = scores.values()
    return scores, min(values), max(values)


def conservation_score(variant: Variant) -> ToolResult:
    reject_answer_key(variant, f"tools.{TOOL}.conservation_score")
    path = cache_path(variant.genome_build)
    if path.is_file():
        scores, low, high = _load(str(path), path.stat().st_mtime)
        if not scores:
            return ToolResult(
                tool=TOOL,
                status="error",
                message=f"Conservation cache at {config.display_path(path)} has no usable rows.",
            )
        score = scores.get((variant.chrom, variant.pos))
        if score is None:
            return ToolResult(
                tool=TOOL,
                status="unavailable",
                message=(
                    f"No conservation score cached for {variant.chrom}:{variant.pos}. The cache covers "
                    f"the BRCA1 region only, so a position outside it has no score."
                ),
                sources=[TRACK],
            )
        span = high - low
        percentile = (score - low) / span if span > 0 else 0.0
        return ToolResult(
            tool=TOOL,
            status="ok",
            values={
                "score": score,
                "position_percentile_in_region": round(percentile, 4),
                "region_min": low,
                "region_max": high,
                "scored_per": "position",
            },
            message=(
                f"phyloP 100-way score {score:.3f} at {variant.chrom}:{variant.pos}, against a cached "
                f"region range of {low:.3f} to {high:.3f}. Higher means the base has changed less across "
                f"species. The score is per position, so it does not distinguish one alt base from another."
            ),
            sources=[TRACK],
        )
    other_builds = [b for b in CACHE_FILES if b != variant.genome_build and cache_path(b).is_file()]
    if other_builds:
        return ToolResult(
            tool=TOOL,
            status="error",
            message=(
                f"Conservation cache exists only for {', '.join(other_builds)}, but the variant is "
                f"{variant.genome_build}. There is no liftover step, so positions are not compared "
                "across builds."
            ),
        )
    return ToolResult(
        tool=TOOL,
        status="unavailable",
        message=(
            f"Conservation cache not found at {config.display_path(path)}. "
            f"The {OWNER} subagent owns the fetch script that writes it."
        ),
    )
