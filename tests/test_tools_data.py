"""conservation and population tools. Owner: evidence-data."""

from __future__ import annotations

import pytest

from _factories import placeholder_variant
from vus_panel.tools import conservation, population

CACHE_TOOLS = [(conservation, conservation.conservation_score), (population, population.allele_frequency)]

CONSERVATION_ROWS = "chrom\tpos\tscore\n17\t100\t6.400\n17\t101\t-3.200\n17\t102\t0.150\n"
POPULATION_ROWS = (
    "chrom\tpos\tref\talt\tac\tan\taf\n"
    "17\t100\tA\tG\t12\t1000\t0.012\n"
    "17\t101\tA\tG\t\t0\t\n"
    "17\t102\tC\tT\t5133\t282332\t0.01818\n"
)


def _write_cache(module, build):
    """A file that exists but holds nothing a lookup can use."""
    path = module.cache_path(build)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("placeholder\n", encoding="utf-8")


def _write_rows(module, rows, build="GRCh37"):
    path = module.cache_path(build)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rows, encoding="utf-8")
    return path


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
def test_cache_tool_with_unreadable_file_errors(module, tool):
    _write_cache(module, "GRCh37")
    result = tool(placeholder_variant())
    assert result.status == "error"
    assert result.values == {}


@pytest.mark.parametrize("module", [conservation, population])
def test_cache_path_follows_data_dir_override(module, tmp_path):
    assert module.cache_path("GRCh37").parent == tmp_path / "data" / "cache"


def test_conservation_returns_the_score_at_the_position():
    _write_rows(conservation, CONSERVATION_ROWS)
    result = conservation.conservation_score(placeholder_variant(pos=100))
    assert result.status == "ok"
    assert result.values["score"] == pytest.approx(6.4)
    assert result.values["scored_per"] == "position"
    assert result.sources


def test_conservation_scores_a_position_not_a_substitution():
    _write_rows(conservation, CONSERVATION_ROWS)
    to_g = conservation.conservation_score(placeholder_variant(pos=100, ref="A", alt="G"))
    to_t = conservation.conservation_score(placeholder_variant(pos=100, ref="A", alt="T"))
    assert to_g.values["score"] == to_t.values["score"]


def test_conservation_percentile_spans_the_cached_region():
    _write_rows(conservation, CONSERVATION_ROWS)
    highest = conservation.conservation_score(placeholder_variant(pos=100))
    lowest = conservation.conservation_score(placeholder_variant(pos=101))
    assert highest.values["position_percentile_in_region"] == pytest.approx(1.0)
    assert lowest.values["position_percentile_in_region"] == pytest.approx(0.0)


def test_conservation_position_outside_the_cache_is_unavailable():
    _write_rows(conservation, CONSERVATION_ROWS)
    result = conservation.conservation_score(placeholder_variant(pos=999))
    assert result.status == "unavailable"
    assert result.values == {}


def test_population_reports_an_observed_variant():
    _write_rows(population, POPULATION_ROWS)
    result = population.allele_frequency(placeholder_variant(pos=100, ref="A", alt="G"))
    assert result.status == "ok"
    assert result.values["observation"] == "observed"
    assert result.values["ac"] == 12
    assert result.values["an"] == 1000
    assert result.values["af"] == pytest.approx(0.012)
    assert result.sources


def test_population_separates_absent_from_no_call():
    """A site gnomAD could not call is not the same as a variant it never saw."""
    _write_rows(population, POPULATION_ROWS)
    no_call = population.allele_frequency(placeholder_variant(pos=101, ref="A", alt="G"))
    assert no_call.status == "ok"
    assert no_call.values["observation"] == "no_call"
    assert no_call.values["af"] is None

    absent = population.allele_frequency(placeholder_variant(pos=100, ref="A", alt="T"))
    assert absent.status == "ok"
    assert absent.values["observation"] == "not_observed"
    assert absent.values["af"] is None


def test_population_absence_message_warns_it_is_weak_evidence():
    _write_rows(population, POPULATION_ROWS)
    result = population.allele_frequency(placeholder_variant(pos=100, ref="A", alt="T"))
    assert "weak evidence" in result.message


def test_population_outside_the_cached_region_is_unavailable():
    _write_rows(population, POPULATION_ROWS)
    result = population.allele_frequency(placeholder_variant(pos=999, ref="A", alt="G"))
    assert result.status == "unavailable"
    assert result.values == {}