# Repository Guidelines

## Project structure and module organization

`src/vus_panel/` contains the Python package: `api.py` serves FastAPI routes, `schemas.py` defines shared contracts, `agents/` interprets evidence, `tools/` supplies evidence, and `eval/` handles splits and scoring. `orchestrator.py` coordinates the panel. Tests live in `tests/`; `web/index.html` contains the frontend with no build step. `data/fixtures/` holds mock reports; `data/reference/` holds licensed reference files and their manifest.

## Build, test, and development commands

Run from the repository root with Python 3.11 or newer:

```sh
uv venv --python 3.11                         # Create the environment
uv pip install -e ".[dev]"                    # Install package and test dependencies
.venv/Scripts/python -m pytest -q             # Run the test suite
.venv/Scripts/python -m uvicorn vus_panel.api:app --reload
```

The last command serves the API and page at `http://127.0.0.1:8000`. On macOS/Linux, replace `.venv/Scripts/python` with `.venv/bin/python`. Run focused tests by appending a path, such as `tests/test_api.py`.

## Coding style and naming conventions

Use four-space indentation, type annotations, `snake_case` functions/modules, `PascalCase` classes, and `UPPER_SNAKE_CASE` constants. Follow adjacent code and keep changes focused. Shared Pydantic contracts are frozen and reject extra fields. No formatter or linter is configured in `pyproject.toml`.

## Testing guidelines

Use pytest with `tests/test_*.py` files and `test_*` functions. Add regression tests for changed behavior. Fixtures in `tests/conftest.py` block network access, clear API environment variables, and isolate data/output directories. Preserve those guards; use mocks and temporary files. Run the full suite before submitting. No numeric coverage threshold is configured.

## Commit and pull request guidelines

Use short imperative commit subjects, matching history: "Add reference data management" or "Harden the checker guard". Open PRs against `main`; explain the problem, changes, and validation, link relevant issues, and include screenshots for UI changes. Follow ownership assignments in `README.md` and `.claude/agents/`. Obtain affected owners' reviews for shared-file changes and run the checker subagent before merging.

## Security and architecture constraints

Keep credentials, raw datasets, weights, and generated caches untracked. Export runtime settings such as `ANTHROPIC_API_KEY` and `EVO2_BACKEND_URL`; the app does not load `.env` automatically. Register public reference data in `data/reference/MANIFEST.json`. Each evidence agent may access only its own tool; the chair receives findings only. Keep Findlay answer-key fields out of agent inputs and preserve mock-data labels.
