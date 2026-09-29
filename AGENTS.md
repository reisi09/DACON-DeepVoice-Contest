# Agent guide: DACON DeepVoice (236749)

Read by omp (Oh My Pi) and Codex directly, and by Claude Code through `CLAUDE.md`. Details live in the linked docs.

## Project state
- The competition ended 2026-09-29 10:00 KST. Final submission W_new: total 0.86652 (ADS 0.85195, CPS 0.99767), 29th, outside the top-15 second round.
- This repo is the record: code, notebooks, configs, results. Audio, features, weights and submission zips are not in git, and the working data (AI-Hub included) was deleted after the competition, so nothing here can be retrained or re-scored as-is.
- `repository/` keeps the original project's relative paths. `repository/AGENTS.md` holds the old VESSL workspace rules.

## Where things are
- Overview and final result: `README.md`. Code per experiment: `EXPERIMENT_INDEX.md`, `docs/CODE_MAP.md`.
- Final model and validation: `repository/docs/collab/README.md`, `repository/docs/collab/SUBMISSION_PLAN_20260929.md`.
- Agent harness (omp, Jev, plugins): `harness/README.md`.

## Working rules
- A result claim needs its source (log, CSV, LB record). Keep leaderboard scores and local metrics apart and say which one a number is.
- Write a test only when it protects a stated requirement, failure mode, invariant, numerical property, data integrity, integration between real components, reproducibility, packaging, or measured output quality. No tests for counts or coverage; no shape-only or file-exists checks when values matter. Report which failure modes were checked, not how many tests passed.
- Commit locally whenever files change. Never commit audio, weights, feature caches, zips or credentials.

## Harness
- Primary agent: omp. Interactive `omp`, headless `omp -p "…"`. Project settings: `.omp/config.yml`.
- Skills shared by omp and Claude Code go in `.claude/skills/` (omp reads it: `skills.enableClaudeProject`).
- Jev: omp judges text-only stops with TypeSafe Jev (`features.unexpectedStopDetection: smart`) when `TYPESAFE_API_KEY` is set.
- Third-party plugins are listed in `harness/plugins.json` and vendored only after the owner approves, with `python3 harness/sync_plugins.py`.
