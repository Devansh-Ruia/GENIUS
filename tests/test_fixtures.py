from __future__ import annotations

import json

import pytest

from vus_panel import config
from vus_panel.api import FixtureError, load_fixtures
from vus_panel.schemas import parse_variant_id


@pytest.fixture(scope="module")
def fixtures():
    return load_fixtures()


def test_at_most_five_fixtures_all_mock(fixtures):
    assert 1 <= len(fixtures) <= 5
    assert all(r.mock is True for r in fixtures)


def test_summaries_are_marked_mock(fixtures):
    for r in fixtures:
        assert r.counselor_summary.startswith("[MOCK]")
        assert r.patient_summary.startswith("[MOCK]")


def test_fixture_ids_are_grch37_chr17(fixtures):
    for r in fixtures:
        variant = parse_variant_id(r.variant_id)
        assert variant.genome_build == "GRCh37" and variant.chrom == "17"


def test_fixtures_cover_each_display_case(fixtures):
    resolved = [r for r in fixtures if r.call != "unresolved" and r.confidence is not None]
    assert any(r.confidence >= 0.9 for r in resolved)
    assert any(0.7 <= r.confidence < 0.9 for r in resolved)
    assert any(r.confidence < 0.7 for r in resolved)
    assert any(r.call == "unresolved" for r in fixtures)
    assert any(not r.calibrated for r in fixtures)

    def revised(r):
        before = {f.agent: f for f in r.findings_initial}
        return any(
            f.status == "ok" and before[f.agent].status == "ok" and f.verdict != before[f.agent].verdict
            for f in r.findings_after_rebuttal
        )

    assert any(revised(r) for r in fixtures)


def test_loader_rejects_non_mock_and_too_many(tmp_path):
    reports = json.loads(config.FIXTURES_PATH.read_text(encoding="utf-8"))
    not_mock = [dict(reports[0], mock=False)]
    path = tmp_path / "f.json"
    path.write_text(json.dumps(not_mock), encoding="utf-8")
    with pytest.raises(FixtureError, match="mock"):
        load_fixtures(path)
    path.write_text(json.dumps(reports + [reports[0]]), encoding="utf-8")
    with pytest.raises(FixtureError, match="limit"):
        load_fixtures(path)


# Keys each real tool returns (tools/*.py), and the one the page displays on the agent card.
REAL_KEYS = {
    "evo2": {"delta", "delta_mean", "ref_total", "alt_total", "next_token_terms", "context_length",
             "score_source", "calibration_percentile", "n_calibration_reference"},
    "conservation": {"score", "position_percentile_in_region", "region_min", "region_max", "scored_per"},
    "population": {"observation", "ac", "an", "af", "cached_variants_in_region"},
    "sae": {"feature_shift"},  # planned key, documented in tools/sae.py
}
DISPLAYED = {"evo2": "delta", "conservation": "score", "population": "observation", "sae": "feature_shift"}


def test_sample_findings_carry_mock_values_shaped_like_real_ones(fixtures):
    """Every agent in a sample report looks alike on screen: a mock value and a [MOCK] explanation."""
    for r in fixtures:
        for f in [*r.findings_initial, *r.findings_after_rebuttal]:
            where = (r.variant_id, f.agent)
            assert f.rationale.startswith("[MOCK]"), where
            assert f.revision_reason is None or f.revision_reason.startswith("[MOCK]"), where
            if f.status != "ok":
                assert not f.raw_values, where
                continue
            assert f.sources == ["mock"], where
            assert set(f.raw_values) <= REAL_KEYS[f.agent], where
            assert DISPLAYED[f.agent] in f.raw_values, where


def test_sample_ids_are_not_real_split_variants(fixtures):
    """Mock numbers must never sit on a real Findlay variant that the panel is scored on."""
    splits = config.REPO_ROOT / "data" / "splits"
    real = set((splits / "calibration.txt").read_text(encoding="ascii").split())
    real |= set((splits / "test.txt").read_text(encoding="ascii").split())
    real |= set(json.loads((splits / "pilot20.json").read_text(encoding="utf-8"))["variant_ids"])
    assert not {r.variant_id for r in fixtures} & real


def test_every_sample_report_shows_sae_evidence(fixtures):
    """The sample set exists to show all four agent cards filled in; real reports keep SAE unavailable."""
    for r in fixtures:
        for f in [*r.findings_initial, *r.findings_after_rebuttal]:
            if f.agent == "sae":
                assert f.status == "ok" and "feature_shift" in f.raw_values, r.variant_id
