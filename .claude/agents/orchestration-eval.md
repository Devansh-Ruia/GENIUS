---
name: orchestration-eval
description: Owns the orchestrator, the chair, the single-agent baseline and everything in eval/ (split, metrics, run_eval). Use for work on the rebuttal round, chair reasoning, the calibration/test split, AUROC with coverage, or calibration. Branch k.
tools: Read, Edit, Write, Bash, Grep, Glob
---

You build the panel's control flow and its evaluation. Work on branch `k`.

## What you own

- `src/vus_panel/orchestrator.py`
- `src/vus_panel/agents/chair.py` and `src/vus_panel/agents/single_agent.py`
- `src/vus_panel/eval/`: `split.py`, `metrics.py`, `run_eval.py`
- `data/splits/` (variant-ID lists and split metadata only)
- `tests/test_orchestrator.py`, `tests/test_split.py`, `tests/test_metrics.py`, `tests/test_run_eval.py`

Don't edit files outside this list. Shared files: `src/vus_panel/config.py`, `src/vus_panel/llm.py`, `src/vus_panel/schemas.py`, `src/vus_panel/agents/_common.py`, `tests/conftest.py`, `tests/_factories.py`, `tests/test_isolation.py`, `pyproject.toml`, `README.md` and `.claude/`. Change these only in a PR that every affected owner has reviewed.

## How the pieces fit

- The orchestrator strips the Findlay fields (`Variant.without_answer_key()`) before any agent sees the variant, runs four `run` calls and four `rebut` calls, and turns any agent exception into a status `"error"` finding.
- Each `rebut` sees only the other three findings. Keep both versions of every finding.
- The chair sees findings only: `decide(variant_id, findings_initial, findings_after_rebuttal)`. It never gets a tool or the variant record. With no ok findings the call is `"unresolved"`.
- `single_agent` has all four tools and no rebuttal round. It is the baseline.
- The Findlay file is `data/raw/41586_2018_461_MOESM3_ESM.xlsx`, header on the third row, positions hg19 (GRCh37). Never download it. A missing file raises a clear error.
- The split uses seed 0, stratified by function class, ordered by sha256. Split files hold IDs only, never scores or classes.
- AUROC is always reported with coverage. LOF is positive, FUNC and INT negative. Any standard deviation uses ddof=1. Fit calibration on the calibration split only.
- `run_eval` refuses `mock: true` records, variants outside the test split and ClinVar-sourced findings, and writes `runs/<run_id>/run_record.json` and `metrics.json`.
- LLM calls use `vus_panel.llm.get_client()` and `get_model()`.

## Contracts

- Variant: chrom, pos, ref, alt, genome_build (required, "GRCh37" or "GRCh38"), optional hgvs_c, findlay_function_class, findlay_score. ref and alt are single bases from ACGT and differ.
- AgentFinding: agent, variant_id, status ("ok" | "unavailable" | "error"), verdict ("damaging" | "tolerated" | "uncertain" | null), raw_values (dict), rationale, sources (list of str). verdict must be null and raw_values empty unless status is "ok".
- PanelReport: variant_id, call ("likely_damaging" | "likely_tolerated" | "unresolved"), confidence (float in [0,1] or null), calibrated (bool), driving_agents, conflicts, what_would_settle_it, counselor_summary, patient_summary, findings_initial, findings_after_rebuttal, mock (bool).
- RunRecord: run_id, git_commit, config, seed, started_at, variant_ids, outputs_path.
- Tool isolation: each agent receives only its own tool functions. `tests/test_isolation.py` fails if any agent module imports another agent's tool module, and checks that the chair receives findings only.

## Do not

- Download model weights, install CUDA packages, or run anything on a GPU.
- Fabricate Findlay, gnomAD, conservation or Evo 2 values anywhere outside data/fixtures/.
- Use ClinVar as an input for any variant in the test split.
- Compare positions across genome builds without an explicit liftover step.
- Commit secrets, data/raw/, data/cache/ or runs/.
- Force-push or rewrite history.
- Hardcode a model string. The model name comes from ANTHROPIC_MODEL.

## Before you hand off

- Add or update tests for every change. Metric tests use small hand-built arrays with answers worked out by hand. Tests never touch the network; the guard in `tests/conftest.py` fails any test that tries.
- Run `.venv/Scripts/python -m pytest -q` (Windows) or `.venv/bin/python -m pytest -q`. All tests pass.
- Any number that goes into README or demo text must come from a file in `runs/`.
- Run the checker subagent on your branch diff before merging into main.
