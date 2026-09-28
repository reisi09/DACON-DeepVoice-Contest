# mert12-a100-full-submission - Work Plan

## TL;DR (For humans)
<!-- Fill this LAST, after the detailed plan below is written, so it summarizes the REAL plan. -->
<!-- Plain English for a non-engineer: NO file paths, NO todo numbers, NO wave/agent/tool names. -->

**What you'll get:** Full-data MERT layer-12 music heads trained on the existing A100, an offline submission ZIP, and an attempted official submission after artifact verification.

**Why this approach:** Refit the selected frozen-feature candidate and preserve baseline voice/presence and file-max fusion. Reuse verified features and assets to minimize GPU time.

**What it will NOT do:** Train on official hidden test data, restart the old H100 run, or claim independent validation from the merged training population.

**Effort:** Medium
**Risk:** Medium - source rights and packaged inference parity require verification.
**Decisions to sanity-check:** All eligible external music originals; three seeds; exact E6 architecture and preprocessing; actual sampling coverage.

Execution is already authorized by the user; no additional plan approval is required.

---

> TL;DR (machine): full refit, existing A100, offline ZIP, verified shutdown, official submission attempt.

## Scope
### Must have
Source and license audit; every eligible original represented in optimization; reproducible notebook; hash-bound offline package; real GPU inference parity and independent-file checks; verified result download and compute stop.
### Must NOT have (guardrails, anti-slop, scope boundaries)
No new workspace, no screen operations for VESSL, no hidden-test training or cross-file score normalization, no unsupported official-score/L4 claims, no modifications to old deadlines or unrelated dirty files.

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: tests-after with existing pytest or direct numerical checks. Cover preprocessing/serialization parity, sampler coverage, invalid inputs and archive integrity, not file existence alone.
- Evidence: <attemptDir>/task-<N>-mert12-a100-full-submission.<ext> (attemptDir = currentAttemptDir from 'omo ulw-loop status --json', .omo/evidence/ulw/<session>/<goalId>/a<attempt>; outside ulw-loop use .omo/evidence/)

## Execution strategy
### Parallel execution waves
Data audit, runtime implementation, and storage preparation run independently; paid launch waits for all three.

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| 1 | none | 4 | 2,3 |
| 2 | none | 4 | 1,3 |
| 3 | none | 4 | 1,2 |
| 4 | 1,2,3 | 5 | none |
| 5 | 4 | final | none |

## Todos
> Implementation + Test = ONE todo. Never separate.
<!-- APPEND TASK BATCHES BELOW THIS LINE WITH edit/apply_patch - never rewrite the headers above. -->
- [x] 1. Audit and lock all-data population and rights
  Inspect _workspace/eight_h100_20260917/dataset/manifest.csv and sources.csv, prior full-data manifests and license receipts. Count unique originals and cross-reference prior cohorts; exclude unknown target labels and user-excluded NSynth. Validate input hashes. Record missing rights explicitly and resolve before submission. Evidence: results/mert12_a100_20260918/data_audit.json. Commit scoped receipts.
- [x] 2. Implement refit, inference, notebook and packaging
  Reuse eight_heads.py, eight_fixed_io.py, c08_submission.py and package_redesign_submission.py. E6 layer12 64600/32300 windows, stem features, per-file mean, fixed training normalizer, seeds41/42/43 and500updates unless coverage requires a documented sampler correction. Verify real gradient/update, every eligible training row exposure, save/reload parity and per-file inference independence. Evidence: results/mert12_a100_20260918/local_verification.json. Commit only new task modules/notebook/checks.
- [x] 3. Prepare existing A100 storage and bounded watcher
  Target wsp-5i3knd3gxeox exclusively. Verify uploads/readbacks before start; stage correct bootstrap and new configuration. New run deadline no more than two hours after actual release; existing deadlines unchanged. Watch every300seconds, verify downloads then pause, preserve partial results and pause on error. Evidence: results/mert12_a100_20260918/preflight.json. No launch until tasks1 and2 pass.
- [x] 4. Train and verify actual GPU package, download and stop
  Execute prepared notebook on A100. Run offline script on representative actual audio plus silence/60s stereo; compare direct/cached MERT12 and reloaded model predictions, reordered/subset inference. Record timing and memory without claiming L4 compliance. Verify complete ZIP/member hashes and safe paths, preserve executed notebook and weights, confirm stopped/standby and watcher teardown. Evidence: training_receipt.json, submission_smoke.json, download_verification.json, resource_stop.json under the new results directory.
- [x] 5. Submit verified ZIP and report actual outcome
  Use authorized official DACON submission surface only if account/browser access exists; inspect remaining daily allowance and resulting status. If access is unavailable, deliver verified ZIP and state the exact access limitation without claiming submission. Preserve full data/training source evidence for second-round verification. Evidence: submission receipt or documented access blocker, final report and scoped local commit.

## Final verification wave
Run independent review of implementation, numerical/package evidence, scope and data claims. User has authorized execution and submission; no repeat approval gate.
- [x] F1. Plan compliance audit
- [x] F2. Code quality review
- [x] F3. Real manual QA
- [x] F4. Scope fidelity

## Commit strategy
Commit task files only; preserve .omo/evidence/v4-training/source-sha256.txt and other unrelated edits.

## Success criteria
All training/artifact claims supported by actual receipts; compute stopped after verified downloads; a ZIP usable for official submission delivered; submission result or concrete access blocker stated accurately.

## Completion record

Implementation reviewed at `e31f1f6679b7fb23d1f490b4fceb10813c22a80b`; all five lanes PASS with exact-SHA records in the ledger. Evidence-only commits do not change implementation. Final corrected package GPU QA, hashes, and stopped/standby verified. Official submission92294 accepted at2026-09-18 10:30:25 KST and grading; score follow-up remains in heartbeat h100-15 without GPU restart. No official score or L4 success claimed.
