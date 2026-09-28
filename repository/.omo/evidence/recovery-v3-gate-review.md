# Recovery V3 final gate review

## recommendation

APPROVE

## blockers

None.

## originalIntent

Analyze the recorded DACON failures, explain why the shorter AST inference did not establish correctness, implement a materially different and rules-compliant recovery experiment, execute it in Colab within the limited A100 budget, save the evidence to Drive, and promote only a candidate that passes predeclared quality gates.

## desiredOutcome

An evidence-backed failure analysis plus a reproducible five-candidate comparison. The selected artifact must preserve the working baseline when every proposed change regresses, must not describe a partial diagnostic as an official score, and must leave complete predictions, timing, coverage, notebook, Drive, and GPU-release evidence.

## userOutcomeReview

The shipped outcome satisfies the request even though it did not discover a stronger model. The prior AST failure is tied to a confirmed selection error and weak generalization evidence, while its short runtime is explained by replacing PANNs, Demucs, and two DF-Arena paths with one shared AST backbone. Recovery V3 restores G0 and tests four bounded SONICS music-path variants. All four fail the predeclared development gates, so the implementation correctly retains G0 and makes no new improvement claim. The report identifies submission 88418, score 0.6909110688, as the best verified submission.

The executed notebook processes all 396 old and 88 fresh files. The five old prediction CSVs each contain 396 unique IDs with bounded probabilities; the fresh G0 CSV contains 88 unique IDs. Independent recomputation reproduces stored metrics within 1e-12, checks protected heads, verifies gap-free SONICS coverage for all 484 inputs, matches embedded notebook sources to local sources, and confirms the candidate lock preceded the fresh-data receipt. The 60-second trace covers 960,000 samples, file-order delta is 0, and two untouched-baseline parity probes are at most 2.7533936020063265e-08 apart.

Colab execution counts are sequential 1 through 12 with a final `status: finished` result. The comparison run takes 570.4202938079834 seconds; release is requested after 842.4277806282043 seconds, below the 7,200-second allocation limit. `released.png` shows 53.34 remaining units and zero active sessions after the run. The complete run is preserved under the linked Drive result folder in the final report. No candidate qualified, so the plan explicitly makes new packaging and server replay unnecessary; `finished.json` truthfully records `submission_status: not_submitted`.

## checkedArtifacts

- `.omo/plans/recovery-v3.md`
- `_workspace/recovery_v3/notepad.md`
- `.omo/evidence/recovery-preflight-code-review.md`
- `.omo/evidence/recovery_sonics_runtime/evidence.md`
- `.omo/evidence/recovery_sonics_runtime/score_audit_report.md`
- `.omo/evidence/recovery_experiment/final-validation.txt`
- `results/recovery_v3/failure_analysis.md`
- `results/recovery_v3/new_candidate_diagnostics.json`
- `results/recovery_v3/colab_20260913_044621/executed.ipynb`
- `results/recovery_v3/colab_20260913_044621/selection.json`
- `results/recovery_v3/colab_20260913_044621/candidate_lock.json`
- `results/recovery_v3/colab_20260913_044621/candidate_lock_written.json`
- `results/recovery_v3/colab_20260913_044621/fresh_opened.json`
- `results/recovery_v3/colab_20260913_044621/comparison.json`
- `results/recovery_v3/colab_20260913_044621/coverage_timing.json`
- `results/recovery_v3/colab_20260913_044621/finished.json`
- `results/recovery_v3/colab_20260913_044621/release_requested.json`
- `results/recovery_v3/colab_20260913_044621/released.png`
- all six `*predictions.csv` files in the run directory
- `verify_recovery_results.py`
- `run_recovery.py`, `recovery_selection.py`, `recovery_sonics.py`, `recovery_fusion.py`, `build_recovery_notebook.py`, `prepare_recovery_data.py`
- `test_recovery_sonics.py`, `test_recovery_fusion.py`, `test_recovery_selection.py`, `test_prepare_recovery_data.py`

## directProgrammingAndSlopPass

The current meaningful suite passes 16 tests in 4.97 seconds. Tests cover concrete risks: late-tail coverage, non-finite input rejection, checkpoint tampering, portable offline load parity, protected-head ownership, file-local fusion, immutable selection lock, holdout non-promotion, incomplete source/provider data, and the prior failed-gate promotion error. The previously noted CLI-help-only test was removed; the preflight report's historical count of 17 describes the earlier preflight state and must not be reused as the current test count.

No deletion-only test, prompt-prose pin, tautological score test, output-derived expected metric, useless existence-only test, speculative factory, single-use interface, unnecessary parsing layer, or production extraction added solely to satisfy tests was found. The two file-existence assertions in the prune test accompany the actual stale-audio removal invariant and therefore have data-integrity value. All reviewed production modules remain below 250 pure lines. No success criterion is violated by the remaining script-style typing or the unavailable local basedpyright check; actual Colab execution and independent numerical replay cover the release decision.

The preflight code review explicitly records its own `omo:programming` and `omo:remove-ai-slops` check, including tautological tests, deletion-only tests, needless abstraction, parsing/normalization, and untyped escape hatches. Direct review agrees with its blocking conclusion.

## exactEvidenceGaps

- No DACON server-side per-file trace exists for historical submissions; the report states this limitation.
- The A100 environment is Python 3.13.15 and PyTorch 2.11.0 rather than the competition server environment. Since every new candidate was rejected and no package is promoted, server replay is correctly omitted under the written plan.
- Fresh data are disjoint from the project's prior corpus but are not guaranteed disjoint from upstream pretrained-model corpora; the report states this limitation.
- `voice_presence_auc` is uncomputable on the fresh set because its 40 known labels contain only one class; the report leaves it invalid and does not present the partial diagnostic as official.
- The current local meaningful suite is 16 tests. The preflight report retains its accurate-at-the-time count of 17 from before the low-value CLI-help test was removed.

None of these gaps violates a stated success criterion.
