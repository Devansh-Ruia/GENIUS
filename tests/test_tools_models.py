"""evo2 and sae tools. Owner: evidence-models."""

from __future__ import annotations

import pytest

from _factories import placeholder_variant
from vus_panel.tools import evo2, sae

BACKEND_TOOLS = [(evo2, evo2.log_likelihood_delta), (sae, sae.feature_activations)]


@pytest.mark.parametrize("module, tool", BACKEND_TOOLS)
def test_backend_tool_without_url_names_the_gpu_requirement(module, tool):
    result = tool(placeholder_variant())
    assert result.tool == module.TOOL
    assert result.status == "unavailable"
    assert "EVO2_BACKEND_URL" in result.message
    assert "FP8" in result.message
    assert result.values == {}


def test_sae_with_url_sends_nothing_and_stays_unavailable(monkeypatch):
    monkeypatch.setenv("EVO2_BACKEND_URL", "http://placeholder.invalid")
    result = sae.feature_activations(placeholder_variant())
    assert result.status == "unavailable"
    assert "not implemented" in result.message
    assert "evidence-models" in result.message
    assert result.values == {}


def test_evo2_uses_typed_client_and_returns_only_successful_values(monkeypatch):
    from test_scoring_service import service

    instance = service()
    monkeypatch.setenv("EVO2_BACKEND_URL", "http://placeholder.invalid")
    monkeypatch.setattr(evo2.ScoringClient, "score", lambda self, variant: instance.score(variant))
    variant = placeholder_variant(pos=4097, alt="C")
    result = evo2.log_likelihood_delta(variant)
    assert result.status == "ok" and result.values["delta"] == -2
    instance.backend.fail_at = len(instance.backend.calls) + 2
    result = evo2.log_likelihood_delta(variant)
    assert result.status == "error" and result.values == {}


def test_evo2_backend_failure_does_not_invent_values(monkeypatch):
    from vus_panel.scoring.client import ScoringClientError

    def fail(self, variant):
        raise ScoringClientError("offline")

    monkeypatch.setenv("EVO2_BACKEND_URL", "http://placeholder.invalid")
    monkeypatch.setattr(evo2.ScoringClient, "score", fail)
    result = evo2.log_likelihood_delta(placeholder_variant())
    assert result.status == "error" and result.values == {}
