"""SAE tool: Goodfire sparse autoencoder feature activations at the variant position on Evo 2 7B.

Owner: evidence-models. Shares the Evo 2 GPU backend (EVO2_BACKEND_URL).
This module never fakes activations: until the client exists it always
reports the tool as unavailable, and it sends no request.

Planned value key: when this tool returns ok, the page shows
values["feature_shift"] on the SAE card (web/index.html AGENT_INFO). Until
then only the mock sample reports in data/fixtures/ carry that key.
"""

from __future__ import annotations

from vus_panel import config
from vus_panel.schemas import ToolResult, Variant, reject_answer_key

TOOL = "sae"
OWNER = "evidence-models"


def feature_activations(variant: Variant) -> ToolResult:
    reject_answer_key(variant, f"tools.{TOOL}.feature_activations")
    if config.evo2_backend_url() is None:
        return ToolResult(
            tool=TOOL,
            status="unavailable",
            message=(
                "EVO2_BACKEND_URL is not set. SAE features come from Evo 2 7B, which needs an "
                "FP8-capable GPU backend, and none is configured."
            ),
        )
    return ToolResult(
        tool=TOOL,
        status="unavailable",
        message=f"SAE backend client not implemented (owner: {OWNER}). No request was sent.",
    )
