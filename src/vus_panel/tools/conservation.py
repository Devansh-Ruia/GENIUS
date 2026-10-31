"""Conservation tool: per-base conservation score at the variant position.

Owner: evidence-data. Reads a local cache file per genome build under
data/cache/. There is no liftover step, so a variant is only ever looked up
in the cache for its own build. The lookup itself is not written yet, so
this module never returns a score.
"""

from __future__ import annotations

from pathlib import Path

from vus_panel import config
from vus_panel.schemas import GenomeBuild, ToolResult, Variant, reject_answer_key

TOOL = "conservation"
OWNER = "evidence-data"
CACHE_FILES: dict[str, str] = {
    "GRCh37": "conservation_brca1_GRCh37.tsv",
    "GRCh38": "conservation_brca1_GRCh38.tsv",
}


def cache_path(build: GenomeBuild) -> Path:
    return config.cache_dir() / CACHE_FILES[build]


def conservation_score(variant: Variant) -> ToolResult:
    reject_answer_key(variant, f"tools.{TOOL}.conservation_score")
    path = cache_path(variant.genome_build)
    if path.is_file():
        return ToolResult(
            tool=TOOL,
            status="unavailable",
            message=f"Conservation cache found, but lookup not implemented (owner: {OWNER}).",
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
