# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A hackathon repo for a multi-agent panel that reviews BRCA1 single-nucleotide variants of uncertain significance. Four evidence agents (evo2, sae, conservation, population) each see one evidence type, revise once in a rebuttal round, and a chair returns a `PanelReport`. The answer key is the Findlay et al. 2018 saturation genome editing table. Most agents are still stubs that report `"unavailable"` and never a made-up value, so `POST /review` returns `call: "unresolved"`. README.md holds the contracts, eval plan and branch ownership.

## Commands

Python 3.11 venv managed by uv. Windows uses `.venv/Scripts/python`; macOS and Linux use `.venv/bin/python`.

```bash
uv venv --python 3.11 && uv pip install -e ".[dev]"
.venv/Scripts/python -m pytest -q                                   # full suite
.venv/Scripts/python -m pytest -q tests/test_isolation.py           # one file
.venv/Scripts/python -m pytest -q "tests/test_isolation.py::test_agents_never_receive_findlay_fields"
.venv/Scripts/python -m pytest -q -k answer_key                     # by name
.venv/Scripts/python -m uvicorn vus_panel.api:app --reload          # API and page on :8000
.venv/Scripts/python -m vus_panel.eval.split [--force]              # reads data/raw/41586_2018_461_MOESM3_ESM.xlsx
.venv/Scripts/python -m vus_panel.eval.run_eval --reports R.json --system panel [--allow-partial]
.venv/Scripts/python -m vus_panel.eval.pilot prepare [--download]   # --download fetches the workbook and NC_000017.10
.venv/Scripts/python -m vus_panel.eval.pilot run --selection data/splits/pilot20.json
```

No linter, formatter or type checker is configured. `python -m vus_panel.scoring.service materialize|serve` downloads a 13.7 GB checkpoint and needs a CUDA Evo 2 environment. It runs only on the GPU instance (docs/findlay-evo2-pilot.md), never locally or in tests.

## Architecture

- **Flow**: `api.py` `POST /review` -> `orchestrator.review` -> each evidence agent's `run(variant)` -> each agent's `rebut(own, others)` -> `agents/chair.decide` -> `PanelReport`. `agents/single_agent.review` is the baseline: all four tools, no rebuttal.
- **Contracts** are in `schemas.py` (`Variant`, `AgentFinding`, `PanelReport`, `RunRecord`, `ToolResult`). Every model is frozen with `extra="forbid"`, and the validators carry real rules: an ok finding keeps its `raw_values` and `sources` through rebuttal, a changed or withdrawn verdict needs `revision_reason`, a resolved call needs a confidence. Read them before building reports by hand.
- **Isolation**: each tool in `tools/` takes a `Variant` and returns a `ToolResult`. Each evidence agent imports only its own tool. `rebut` and the chair receive findings only, never tools or the variant.
- **Answer key**: `Variant.without_answer_key()` strips the Findlay fields in the orchestrator and single_agent. On top of that, every agent `run()`, every tool entry point and `ScoreRequest.from_variant` call `reject_answer_key`, which raises `AnswerKeyError`. That error must propagate; the orchestrator and single_agent re-raise it ahead of their broad `except`. `POST /review` returns 422 for a body containing either Findlay field.
- **Orchestrator errors**: other agent and tool exceptions, and findings that break the rebuttal rules, become status `"error"` findings. The orchestrator raises if the chair returns findings that differ from the ones it was given.
- **`scoring/`**: `contracts.py` is the frozen wire protocol. `client.py` is CPU-only and is what `tools/evo2.py` uses through `EVO2_BACKEND_URL`. `backend.py` and `service.py` are the GPU side and are imported only by the service; a test checks the client's import closure never reaches them.
- **`eval/`**:
  - `split.py`: the Findlay loader and the seed-0 split, stratified by class, ordered by sha256, 0.2 calibration.
  - `metrics.py`: AUROC always reported with coverage, LOF positive, ddof=1.
  - `run_eval.py`: refuses mock records, ClinVar mentions in test-split findings, and partial test sets unless `--allow-partial` is given. Writes `runs/<id>/`.
  - `pilot*.py` and `findlay_validation.py`: the frozen Evo 2 timing pilot (docs/findlay-evo2-pilot.md).
- **`web/index.html`**: one static page, vanilla JS and inline SVG, served by `GET /`. Report text enters the DOM only through `textContent`, `setAttribute` or `createElementNS`; `tests/test_api.py` greps the page for HTML sinks.

## Rules the tests enforce

- **Test environment** (`tests/conftest.py`, autouse): every socket connect and DNS lookup is blocked, the `ANTHROPIC_*`, `EVO2_BACKEND_URL` and `VUS_PANEL_*` variables are cleared, and `config.data_dir()` and `runs_dir()` point at tmp. A test that needs real repo files has to use `config.REPO_ROOT`.
- **Isolation** (`tests/test_isolation.py`):
  - follows each agent's transitive `vus_panel` imports against an allowlist and spies on tools at runtime
  - requires an owner for every module under `tools/` (`PRIVATE_TOOL_OWNERS`)
  - bans dynamic access anywhere in an agent's import closure: `getattr`/`vars` on imported modules, any `.modules` attribute, `exec`, `eval`, `compile`, `__builtins__`, `__import__`, and the `importlib`, `runpy`, `pkgutil` and `builtins` modules

  That closure includes `schemas.py`, `config.py` and `scoring/client.py`, so keep these constructs out of shared modules too.
- **Model strings** (`tests/test_llm.py`): `src/` must not match `claude-|opus|sonnet|haiku|fable`, case-insensitive. Get the model with `vus_panel.llm.get_model()`, which reads `ANTHROPIC_MODEL`.
- **Reference data** (`tests/test_reference_data.py`): every file in `data/reference/` has a `MANIFEST.json` entry whose sha256 and size match, is under 50 MiB, and has a non-empty source and license.
- **Fixtures**: `data/fixtures/` holds at most five reports, all `mock: true`, for UI work only. The page shows a MOCK DATA banner for them, and `run_eval` refuses them.

## Data and git

- **Public repo**: never commit `data/raw/`, `data/cache/`, `runs/`, model or SAE weights, genome-wide files, or anything whose license doesn't allow public redistribution. That's why the Findlay workbook stays in gitignored `data/raw/`.
- **Frozen bytes**: `.gitattributes` marks `data/reference/**` and `data/splits/**` as `-text` because their bytes are checksummed, and this checkout uses core.autocrlf. Don't hand-edit the frozen split artifacts (`calibration.txt`, `test.txt`, `split_meta.json`, `pilot20.json`). `split` needs `--force` to overwrite them, and the pilot refuses a freeze that doesn't match.
- **Genome builds**: Findlay positions are hg19/GRCh37. Never compare positions across builds without an explicit liftover step.
- **ClinVar**: never an input for a test-split variant.

## Branches and subagents

- **Owners**: branches d, a, k and m map to owners in README "Branch ownership".
- **Role files**: each file in `.claude/agents/` lists the files that subagent owns. Everything else, shared files included, is read-only for it.
- **Merge rule**: open a PR into main and run the `checker` subagent on the diff first. The checker is read-only, and `.claude/hooks/checker_bash_guard.py` limits its Bash to pytest and read-only git.
- **Codex**: configuration for the same role names lives in `.codex/agents/` and `AGENTS.md`.
