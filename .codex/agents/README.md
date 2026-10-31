# Codex subagents

These project-local roles mirror the ownership defined in `README.md` and
`.claude/agents/`. They inherit the parent session's model and reasoning settings.

| Agent | Scope |
| --- | --- |
| `maker` | Focused implementation within explicitly assigned files |
| `checker` | Review without editing; contracts, isolation, security, tests |
| `frontend` | `web/` only; logical branch `m` |
| `evidence-data` | Conservation/population tools and agents; logical branch `a` |
| `evidence-models` | Evo 2/SAE tools, agents, and service client; logical branch `d` |
| `orchestration-eval` | Orchestrator, API, chair, baselines, evaluation, fixtures; logical branch `d` |

## Usage

Ask Codex to use a role by name, for example:

> Use the frontend subagent to fix the confidence-track layout. Have checker review the resulting diff.

> Have evidence-data inspect population-cache handling and evidence-models inspect scoring-client errors in parallel. Return findings without edits.

Assign non-overlapping files when agents edit concurrently. Branch names identify
ownership; agents must not switch a checkout shared with other agents. The parent
coordinates any separate worktrees and handles commits and integration. Review
changes to shared files with the affected owners before merging.

## Configuration and permissions

Current Codex discovers standalone `.toml` role files in `.codex/agents/`. Each
defines `name`, `description`, and `developer_instructions`; no additional model
or permission overrides are set here. See the
[official subagent documentation](https://learn.chatgpt.com/docs/agent-configuration/subagents).

Agents inherit the parent session's permissions. The checker's no-edit policy is
an instruction, not an independently enforced read-only sandbox. Claude's
frontmatter tool lists and Bash hook do not apply to these TOML roles. For a
review that needs filesystem enforcement, use a separate parent session with
read-only permissions. Do not combine that session with implementation work.

The existing Claude definitions and root `AGENTS.md` remain the repository's
reference material. Keep ownership descriptions synchronized when roles change.
