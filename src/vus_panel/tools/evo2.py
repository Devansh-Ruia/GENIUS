"""Evo 2 tool: log-likelihood delta (alt minus ref) from Evo 2 7B.

Owner: evidence-models. The model runs on a separate GPU backend reached
through EVO2_BACKEND_URL. Partial results remain in the evaluation client;
the panel's ToolResult only carries values when both scores succeed.
"""

from __future__ import annotations

from vus_panel import config
from vus_panel.schemas import ToolResult, Variant, reject_answer_key
from vus_panel.scoring.client import ScoringClient, ScoringClientError
from vus_panel.scoring.contracts import ScoreRequest

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
    try:
        request = ScoreRequest.from_variant(variant)
        client = ScoringClient(config.evo2_backend_url())
        response = client.score(request)
    except (ValueError, ScoringClientError) as exc:
        return ToolResult(tool=TOOL, status="error", message=f"Evo 2 scoring failed: {exc}")
    if response.status != "ok":
        return ToolResult(tool=TOOL, status="error", message=f"Evo 2 scoring failed: {response.error}")
    return ToolResult(tool=TOOL, status="ok", values=response.model_dump(mode="json"),
                      message="Evo 2 genomic-forward summed log-likelihood, ALT minus REF.",
                      sources=[f"Evo 2 service session {response.session_id}; metadata SHA256 {response.metadata_sha256}"])
