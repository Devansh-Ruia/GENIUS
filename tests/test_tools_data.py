"""conservation and population tools. Owner: evidence-data."""

from __future__ import annotations

import pytest

from _factories import placeholder_variant
from vus_panel.tools import conservation, population

CACHE_TOOLS = [(conservation, conservation.conservation_score), (population, population.allele_frequency)]


def _write_cache(module, build):
    path = module.cache_path(build)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("placeholder\n", encoding="utf-8")


@pytest.mark.parametrize("module, tool", CACHE_TOOLS)
def test_cache_tool_without_file_names_path_and_owner(module, tool):
    result = tool(placeholder_variant())
    assert result.status == "unavailable"
    assert module.CACHE_FILES["GRCh37"] in result.message
    assert "evidence-data" in result.message
    assert result.values == {}


@pytest.mark.parametrize("module, tool", CACHE_TOOLS)
def test_cache_tool_refuses_cross_build_lookup(module, tool):
    _write_cache(module, "GRCh38")
    result = tool(placeholder_variant(genome_build="GRCh37"))
    assert result.status == "error"
    assert "liftover" in result.message
    assert result.values == {}


@pytest.mark.parametrize("module, tool", CACHE_TOOLS)
def test_cache_tool_with_file_present_still_returns_no_values(module, tool):
    _write_cache(module, "GRCh37")
    result = tool(placeholder_variant())
    assert result.status == "unavailable"
    assert "not implemented" in result.message
    assert result.values == {}


@pytest.mark.parametrize("module", [conservation, population])
def test_cache_path_follows_data_dir_override(module, tmp_path):
    assert module.cache_path("GRCh37").parent == tmp_path / "data" / "cache"
