"""HTTP API. Owner: frontend. The panel behind POST /review is orchestrator.py, owned by orchestration-eval.

GET /          serves web/index.html
POST /review   takes a Variant without the Findlay fields and returns a PanelReport (422 if either is present)
GET /fixtures  returns the mock PanelReports in data/fixtures/ for UI work
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import TypeAdapter, model_validator

from vus_panel import config, orchestrator
from vus_panel.schemas import ANSWER_KEY_FIELDS, PanelReport, Variant

MAX_FIXTURES = 5
_REPORTS = TypeAdapter(list[PanelReport])


class FixtureError(ValueError):
    """The fixture file breaks the fixture rules."""


def load_fixtures(path: Path = config.FIXTURES_PATH) -> list[PanelReport]:
    """Load and validate fixtures. Every one must be mock, and there are at most five."""
    reports = _REPORTS.validate_json(path.read_bytes())
    if len(reports) > MAX_FIXTURES:
        raise FixtureError(f"{len(reports)} fixtures found; the limit is {MAX_FIXTURES}")
    not_mock = [r.variant_id for r in reports if not r.mock]
    if not_mock:
        raise FixtureError(f"fixtures must all be mock: true: {not_mock}")
    return reports


class ReviewRequest(Variant):
    """The POST /review body. A Findlay key is refused even when its value is null."""

    @model_validator(mode="before")
    @classmethod
    def _reject_answer_key_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            present = [name for name in ANSWER_KEY_FIELDS if name in data]
            if present:
                raise ValueError(
                    f"request must not include {', '.join(present)}: the Findlay answer key never goes to the panel"
                )
        return data


app = FastAPI(title="VUS panel", version="0.1.0")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(config.WEB_DIR / "index.html", media_type="text/html")


@app.post("/review", response_model=PanelReport)
def review(variant: ReviewRequest) -> PanelReport:
    return orchestrator.review(variant)


@app.get("/fixtures", response_model=list[PanelReport])
def fixtures() -> list[PanelReport]:
    return load_fixtures()
