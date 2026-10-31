"""Evo 2 tool: log-likelihood delta (alt minus ref) from Evo 2 7B.

Owner: evidence-models. The model runs on a separate GPU backend reached
through EVO2_BACKEND_URL. This module never fakes a score: until the client
exists it always reports the tool as unavailable, and it sends no request.
"""

from __future__ import annotations

from vus_panel import config
from vus_panel.schemas import ToolResult, Variant, reject_answer_key

TOOL = "evo2"
OWNER = "evidence-models"


def log_likelihood_delta(variant: Variant) -> ToolResult:
    reject_answer_key(variant, f"tools.{TOOL}.log_likelihood_delta")
    if config.evo2_backend_url() is None:
        return ToolResult(
            tool=TOOL,
            status="unavailable",
            message=(
                "EVO2_BACKEND_URL is not set. Evo 2 7B needs an FP8-capable GPU backend, "
                "and none is configured."
            ),
        )
    return ToolResult(
        tool=TOOL,
        status="unavailable",
        message=f"Evo 2 backend client not implemented (owner: {OWNER}). No request was sent.",
    )
