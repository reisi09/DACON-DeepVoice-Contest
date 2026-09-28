# V4 current context and requirements gate review

## Recommendation

**APPROVE**

`EXACT_SOURCE_SHA_VERDICT: PASS`

- Required source SHA: `d2397ad8cc96667e5bc2e4132d86405afe093f4e`
- Reproduced detached HEAD: `d2397ad8cc96667e5bc2e4132d86405afe093f4e`
- Detached checkout: `/tmp/dacon-v4-review-d2397ad`
- Tree: `d02b30ae16617544d05527c8580b1fe77a97c503`
- Git state: detached and clean; no remotes configured.

## Blockers

None.

## Original intent

The user wanted actual notebook experiments on Colab A100, using pretrained full fine-tuning and fair epoch/hyperparameter comparisons against the retained baseline. The execution had to use rule-compliant data, save measured results to Drive, remain within the original two-hour allocation, avoid sealed leakage or automatic submission, and report a no-improvement outcome honestly if no trained candidate passed the fixed gates.

## Desired outcome

The expected outcome was a reproducible experimental record, not a guaranteed score gain: all attempted rows must have terminal receipts; comparable arms must share the relevant data, initialization, schedule and update budget; artifacts must be retrievable; the GPU must be released before the deadline; and failure to qualify must retain G0 without sealed refit or submission.

## User outcome review

The artifacts satisfy that outcome.

1. The original matrix completed. `.omo/evidence/v4-retry-pipeline_status.json` records G0 and P1-P4 as `completed`. `.omo/evidence/v4-retry-finished.json` records measured-in-run GPU metrics, `final: G0`, `sealed_opened: false`, and all P1-P4 gates as failed. This supports a completed no-improvement experiment rather than a fabricated improvement claim.
2. The legacy P1/P3 causal result is explicitly excluded. `.omo/evidence/v4-matched-comparison.json` records `legacy_gpu_P3_excluded: true` and `matched_complete: true`.
3. The replacement P1/P3 comparison is matched and terminal. Both arms used 1,116 eligible training rows and 210 development rows under the same 30-epoch scheduler horizon, encoder/head learning rates, seed, initial-weight SHA (`0784cf50ef834751c93f813f981ca249799c0ebac12b93950ec1c4e2a1b1be10`), seed-RNG SHA (`373d3bd3579f4e77cc224197612499148e34968d0f3a69ded66a682559b1beda`), 10 completed epochs, 350 optimizer updates and 11,160 window exposures. P1 and P3 each report `status: completed`; the differing trial-contract hashes are expected because augmentation is the treatment.
4. The runtime source closure is exact. I independently recomputed SHA-256 for all 14 entries in `.omo/evidence/v4-matched-source_hashes.json` against the detached `d2397ad` checkout; all 14 passed, including `recheck_v4_matched.py` at `130e078b393f627b871e549e695e9fe8f4061cfe10a1a16494b60d528aab825c`.
5. Drive persistence is evidenced. `.omo/evidence/v4-matched-drive_receipt.json` binds `matched_comparison.json` to SHA-256 `3821e5ba6fdc7dffebcf22c61c1ec17d590c42314a7e104bb0eaa82454813e92` and the experiment log to `9393a07989429f0ad34b27e813d64386da38f7d7c8c045dbbf5b8cd249bfec00`.
6. The resource ceiling passed. The original allocation began at Unix `1789283305`; the safe deadline was `1789290385`. The matched release request is Unix `1789290093.9671278`, which is 291.0328722 seconds before the safe deadline and 6,788.9671278 seconds after allocation origin, within 7,200 seconds. The current release UI receipts record zero sessions.
7. No disallowed promotion occurred. The original terminal record keeps G0 and leaves sealed unopened; `.omo/evidence/v4-results-summary.json` records `refit_performed: false`, `sealed_accessed: false`, and selected `G0`. No evidence claims a submission was made.

The matched results do not establish an accepted trained replacement: P1 and P3 retain the same negative OOD rank score (`-0.045`) at their selected checkpoints. That is the measured result and is consistent with the stated success criterion that a completed failed improvement attempt is valid.

## Requirements trace

| Criterion | Verdict | Evidence |
| --- | --- | --- |
| CTX-SHA: review exact requested source | PASS | `git rev-parse HEAD` in detached checkout; 14/14 runtime source hashes independently recomputed |
| CTX-REAL: real notebook/GPU experiment receipts | PASS | `v4-retry-*terminal.json`, `v4-matched-P1_terminal.json`, `v4-matched-P3_terminal.json`, A100 environment receipt |
| CTX-FAIR: fair full-FT comparison | PASS | identical initial-weight/RNG receipts, train/dev counts, epochs, updates and exposures; 17,172,493 trainable parameters per matched arm |
| CTX-BUDGET: original two-hour ceiling and release | PASS | `v4-matched-budget.json`, `v4-matched-release_requested.json`, zero-session UI receipt |
| CTX-HONEST: no fabricated improvement | PASS | failed original gates, G0 retained, matched negative OOD result, legacy P3 excluded |
| CTX-SAFETY: no sealed/refit/submission after rejection | PASS | `sealed_opened: false`, `refit_performed: false`, selected G0; no submission artifact or claim |
| CTX-DRIVE: results saved/retrievable | PASS | matched Drive receipt and per-arm Drive checkpoint/receipt paths with hashes |

## Direct programming and remove-ai-slops pass

I directly reviewed the production delta and tests under the `omo:programming`, Python, `omo:remove-ai-slops`, active Ponytail, and repository test-value criteria. The source change fixes shared execution seams for seed reset, manifest/source binding, cleanup and budget enforcement. The two added tests protect concrete failure modes: altered-manifest refusal before execution and stochastic checkpoint resume equivalence. I found no deletion-only test, requested-removal assertion, tautology, prompt/prose pin, output-derived expected value, implementation-mirroring test, needless production extraction, speculative normalization, pass-through wrapper, or new dependency that violates a stated criterion. Generated accessibility transcripts contain whitespace noise, but this does not violate an execution success criterion.

The current code-review report independently records the same skill perspectives and explicitly covers deletion-only, tautological, prompt/implementation-text, needless parsing/normalization/abstraction, and concrete failure-mode test value. Its coverage agrees with this direct pass and was not used as a substitute for it.

## Checked artifacts

- `/tmp/dacon-v4-review-d2397ad` Git HEAD, status, log, diff, and all 14 source files named by the runtime source receipt
- `.omo/plans/trainable-recovery-v4.md`
- `.omo/evidence/v4-final-current-context.md`
- `.omo/evidence/v4-current-code-review.md`
- `.omo/evidence/v4-current-review-ledger.json`
- `.omo/evidence/v4-retry-finished.json`
- `.omo/evidence/v4-retry-selection.json`
- `.omo/evidence/v4-retry-pipeline_status.json`
- `.omo/evidence/v4-retry-release_requested.json`
- `.omo/evidence/v4-matched-comparison.json`
- `.omo/evidence/v4-matched-P1_terminal.json`
- `.omo/evidence/v4-matched-P3_terminal.json`
- `.omo/evidence/v4-matched-P1_trial_receipt.json`
- `.omo/evidence/v4-matched-P3_trial_receipt.json`
- `.omo/evidence/v4-matched-budget.json`
- `.omo/evidence/v4-matched-release_requested.json`
- `.omo/evidence/v4-matched-drive_receipt.json`
- `.omo/evidence/v4-matched-environment.json`
- `.omo/evidence/v4-matched-source_hashes.json`
- `.omo/evidence/v4-results-summary.json`
- `.omo/evidence/v4-results-first-comparison.md`
- `.omo/evidence/v4-final-qa-review.md` (historical original-run review only)

## Exact evidence gaps and notes

- `v4-final-current-context.md` was captured while P3 was still running. It is stale for terminal status; the later terminal, comparison, Drive and release receipts close that gap.
- `v4-final-qa-review.md` reviews the earlier `a01a50e` run and therefore cannot verdict the matched run. It is historical evidence only.
- `v4-results-first-comparison.md` still says the matched recheck is pending. The machine-readable `v4-results-summary.json` and matched receipts contain the completed result. This is a stale prose note, not a failed execution criterion.
- The saved GPU training notebook capture is intentionally marked potentially incomplete. Terminal receipts establish training completion, and the separately executed CPU analysis notebook is the complete results notebook. No artifact claims the GPU capture itself is a fully completed notebook.
- There is no sealed/refit/submission receipt because the fixed gates selected G0. Their absence is the required branch outcome, supported by explicit `sealed_opened/accessed: false` and `refit_performed: false` fields.
- No Git remote exists, so no push provenance can be checked or required. Slack/Notion account state is outside the requested scope.

## Final decision

`recommendation: APPROVE`

`blockers: []`
