from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from vus_panel.api import app
from vus_panel.schemas import EVIDENCE_AGENTS

client = TestClient(app)
PLACEHOLDER = {"chrom": "17", "pos": 100, "ref": "A", "alt": "G", "genome_build": "GRCh37"}


def test_index_serves_the_page():
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert 'id="mock-banner"' in response.text


def test_page_never_writes_html_from_data():
    page = client.get("/").text
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write"):
        assert sink not in page


def test_review_returns_unresolved_report_with_four_unavailable_findings():
    response = client.post("/review", json=PLACEHOLDER)
    assert response.status_code == 200
    body = response.json()
    assert body["variant_id"] == "GRCh37-17-100-A-G"
    assert body["call"] == "unresolved"
    assert body["confidence"] is None
    assert body["mock"] is False
    for key in ("findings_initial", "findings_after_rebuttal"):
        assert [f["agent"] for f in body[key]] == list(EVIDENCE_AGENTS)
        assert all(f["status"] == "unavailable" for f in body[key])


def test_review_normalizes_chr_prefix():
    response = client.post("/review", json={**PLACEHOLDER, "chrom": "chr17"})
    assert response.json()["variant_id"] == "GRCh37-17-100-A-G"


@pytest.mark.parametrize(
    "payload",
    [
        {**PLACEHOLDER, "alt": "A"},
        {**PLACEHOLDER, "ref": "a"},
        {k: v for k, v in PLACEHOLDER.items() if k != "genome_build"},
        {**PLACEHOLDER, "genome_build": "hg19"},
        {**PLACEHOLDER, "extra": "field"},
        {**PLACEHOLDER, "pos": 0},
    ],
)
def test_review_rejects_invalid_variants(payload):
    assert client.post("/review", json=payload).status_code == 422


def test_fixtures_endpoint_returns_mock_reports():
    response = client.get("/fixtures")
    assert response.status_code == 200
    reports = response.json()
    assert 1 <= len(reports) <= 5
    assert all(r["mock"] is True for r in reports)
