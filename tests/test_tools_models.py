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


@pytest.mark.parametrize("module, tool", BACKEND_TOOLS)
def test_backend_tool_with_url_sends_nothing_and_stays_unavailable(module, tool, monkeypatch):
    monkeypatch.setenv("EVO2_BACKEND_URL", "http://placeholder.invalid")
    result = tool(placeholder_variant())  # the autouse socket guard fails the test on any request
    assert result.status == "unavailable"
    assert "not implemented" in result.message
    assert "evidence-models" in result.message
    assert result.values == {}
