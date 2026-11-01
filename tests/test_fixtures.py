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


def test_sample_findings_carry_mock_values_shaped_like_real_ones(fixtures):
    """Every agent in a sample report looks alike on screen: a mock value and a [MOCK] explanation."""
    for r in fixtures:
        for f in [*r.findings_initial, *r.findings_after_rebuttal]:
            assert f.rationale.startswith("[MOCK]"), (r.variant_id, f.agent)
            if f.status != "ok":
                assert not f.raw_values
                continue
            assert f.raw_values and f.sources == ["mock"], (r.variant_id, f.agent)
            if f.agent == "sae":
                assert set(f.raw_values) == {"feature_shift"}
                assert isinstance(f.raw_values["feature_shift"], float)
