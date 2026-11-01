"""HTTP API. Owner: frontend. The panel behind POST /review is orchestrator.py, owned by orchestration-eval.

GET /          serves web/index.html
POST /review   takes a Variant without the Findlay fields and returns a PanelReport (422 if either is present)
GET /fixtures  returns the mock PanelReports in data/fixtures/ for UI work
POST /explain  answers a plain-English question about the data, the method or an open report
GET /demo      returns the saved real PanelReports in data/demo/ that the page opens with
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, TypeAdapter, model_validator

from vus_panel import config, explain, orchestrator
from vus_panel.schemas import ANSWER_KEY_FIELDS, PanelReport, Variant

MAX_FIXTURES = 5
DEMO_PATH = config.REPO_ROOT / "data" / "demo" / "panel_reports.json"
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


def load_demo(path: Path | None = None) -> list[PanelReport]:
    """Saved real panel reviews the page opens with. Every one must be mock: false; no file means none."""
    path = path if path is not None else DEMO_PATH
    if not path.is_file():
        return []
    reports = _REPORTS.validate_json(path.read_bytes())
    mock = [r.variant_id for r in reports if r.mock]
    if mock:
        raise FixtureError(f"demo reports must all be mock: false: {mock}")
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


class ExplainRequest(BaseModel):
    """A question, plus the report the reader is looking at if there is one."""

    model_config = {"extra": "forbid"}

    question: str = Field(min_length=1, max_length=explain.MAX_QUESTION_CHARS)
    report: PanelReport | None = None


class ExplainResponse(BaseModel):
    answer: str


@app.get("/fixtures", response_model=list[PanelReport])
def fixtures() -> list[PanelReport]:
    return load_fixtures()


@app.post("/explain", response_model=ExplainResponse)
def explain_route(request: ExplainRequest) -> ExplainResponse:
    """Explain the panel. This is a reading aid and never reviews a variant."""
    try:
        return ExplainResponse(answer=explain.answer(request.question, request.report))
    except explain.ExplainError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
@app.get("/demo", response_model=list[PanelReport])
def demo() -> list[PanelReport]:
    return load_demo()
