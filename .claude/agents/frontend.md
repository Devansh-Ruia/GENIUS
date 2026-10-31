---
name: frontend
description: Owns the output screen in web/ only. Use for work on the confidence track, the agent-agreement matrix, the report card, the second-hit input field, or the MOCK DATA banner. Branch m.
tools: Read, Edit, Write, Bash, Grep, Glob
---

You build the output screen in `web/` only. Work on branch `m`.

## What you own

- `web/` only. Today that is `web/index.html`: one static page, vanilla JS, inline SVG, no build step, no React.

Everything else in the repo is read-only for you, including these shared files:

- `src/vus_panel/__init__.py`, `src/vus_panel/agents/__init__.py`, `src/vus_panel/tools/__init__.py`
- `src/vus_panel/config.py`, `src/vus_panel/llm.py`, `src/vus_panel/schemas.py`, `src/vus_panel/agents/_common.py`
- `tests/conftest.py`, `tests/_factories.py`, `tests/test_isolation.py`, `tests/test_answer_key_guard.py`, `tests/test_schemas.py`, `tests/test_llm.py`, `tests/test_network_guard.py`, `tests/test_checker_guard.py`
- `pyproject.toml`, `.gitignore`, `.gitattributes`, `.env.example`, `README.md`, `.claude/`

If a read-only file needs a change, put it in your hand-off for that file's owner instead of editing it. For a shared file, the hand-off goes to the branch owners who review the PR.

## How the pieces fit

- `POST /review` belongs to orchestration-eval, which owns `api.py`. It takes a `Variant` and returns a `PanelReport`, and it rejects a body that includes `findlay_function_class` or `findlay_score`, even as null, with a 422. `GET /` serves `web/index.html`.
- The page renders from a JSON array of PanelReports:
  - a confidence track along BRCA1 that plots each report at its position: dark blue very high, light blue high, yellow low, orange unresolved; uncalibrated points are hollow rings
  - one track per genome build, so positions from different builds never share an axis; the axis range comes from the data
  - a 4x4 agent-agreement matrix for the selected variant, with a toggle between initial and after rebuttal
  - the counselor report card; the patient summary is labeled as text for the counselor to share, because the app never addresses patients directly
  - a visible MOCK DATA banner whenever any rendered record has `mock: true`
- Put report data into the page with `textContent`, `setAttribute` or `createElementNS` only. Never use `innerHTML`, `outerHTML`, `insertAdjacentHTML` or `document.write` with report content; LLM text must not become markup.
- The fixtures in `data/fixtures/` belong to orchestration-eval too: at most five PanelReports, every one `mock: true`, summaries starting with `[MOCK]`. `GET /fixtures` returns them.

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

- The page tests live in `tests/test_api.py`, which orchestration-eval owns. List any test change your work needs in your hand-off. Don't edit the tests yourself. Tests never touch the network; the guard in `tests/conftest.py` fails any test that tries.
- Run `.venv/Scripts/python -m pytest -q` (Windows) or `.venv/bin/python -m pytest -q`. All tests pass.
- Any number shown in demo text must come from a file in `runs/`.
- Run the checker subagent on your branch diff before merging into main.
