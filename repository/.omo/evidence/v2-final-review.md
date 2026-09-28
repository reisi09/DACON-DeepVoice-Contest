# DACON 236749 V2 final acceptance review

## recommendation

**APPROVE_WITH_NOTES**

The authoritative experiment remains the reproducible three-candidate comparison, whose immutable development-only lock selects `G0` before the fresh test is opened. The fourth column-routed hybrid is now explicitly classified as a separate exploratory analysis, not a replacement primary selection and not evidence satisfying the original pre-lock protocol. This resolves the prior acceptance blockers without altering locks or reselecting from fresh results.

## originalIntent

Diagnose the G1/G3/G4 regressions, construct source-connected and provenance-preserving development/fresh splits with unknown-label masking, compare the fixed G0 baseline against a frozen AST head and a true last-block AST fine-tune on Colab A100 within 7,200 GPU seconds, lock selection before opening the fresh slices, and deliver a self-contained offline package without submitting a new DACON entry.

## desiredOutcome

A defensible comparison of exactly `G0`, `AST_FROZEN`, and `AST_LAST_BLOCK`; an immutable development-only candidate lock preceding one fresh evaluation; actual gradient/update evidence for both AST modes; honest fresh-slice reporting; and an offline ZIP with `<=1e-5` replay parity, bounded resources, Drive persistence, and released GPU runtime.

## userOutcomeReview

The shipped artifacts provide the requested forensic diagnosis, 1,029-row provenance-connected data bank, true frozen/last-block training evidence, executed A100 notebooks, Drive receipts, honest non-universal fresh results, exact offline replay after the TF32 fix, and runtime release. The original three-candidate selector correctly chose `G0` because both AST candidates failed non-regression gates.

The final report and machine-readable decision now keep `G0` as the authoritative primary selection. `G0_AST_MUSIC`, which routes two probability columns from G0 and three from `AST_LAST_BLOCK`, is explicitly additional exploratory analysis. Its lock was written after `test_opened.json`, and the report now states that it does not satisfy the original pre-lock protocol. The historical hybrid artifacts remain intact and are scoped rather than rewritten.

No new submission was made. The final report correctly states that the hybrid regressed on `test_generator` (`0.776938 -> 0.756339`) and does not claim universal gain.

## blockers

None.

### resolved findings

1. **C3-EXACT-THREE / NO-ENSEMBLING: RESOLVED.** The primary experiment and final selection remain exactly `G0`, `AST_FROZEN`, and `AST_LAST_BLOCK`, with `G0` selected. `RESULTS_V2.md:33-37`, `README.md:83-89`, `PIPELINE_GRAPH_V2.md:70-74`, and `results/v2_evidence/final_decision.json` consistently classify `G0_AST_MUSIC` as optional exploratory analysis rather than a fourth primary candidate.
2. **C3-LOCK-BEFORE-FRESH-EVALUATION: RESOLVED FOR THE CLAIMED PRIMARY OUTCOME.** The original `candidate_lock.json` predates `test_opened.json` and selects G0. The hybrid lock still follows test opening, but the artifact no longer claims that exploratory result satisfies the original untouched-test protocol. `RESULTS_V2.md:35-41` and the correction at `.omo/evidence/v2-replay-review.md:45-46` state the chronology exactly.

## criterion review

- **C1 exact old ROC forensics: PASS.** `results/v2_forensics.json`, `results/v2_evidence/forensic_interpretation.md`, and `RESULTS_V2.md:15-17` distinguish discrete EER crossings, integer error counts, AUC, pair changes, and dev/old-holdout limits.
- **C2 provenance/source-connected splits/unknown labels: PASS.** `v2_validation/split_audit.json`, both manifests, `pcm_sha256.csv`, source inventory evidence, and the independent data review support 1,029 unique files, role counts `633/122/176/98`, component counts `197/40/58/30`, no cross-role atomic lineage, and 32 ambiguous mixture voice-authenticity labels masked rather than coerced.
- **C3 frozen head and last-block fine-tuning: PASS.** `training_report.json` records 3,845 versus 7,093,253 trainable parameters and nonzero intended gradient/update evidence; checkpoint hashes are locked. Original and final primary selection is `G0` because both AST candidates fail the declared non-regression gates. The hybrid result is separately labeled exploratory and does not alter this decision.
- **C4 offline ZIP parity/resources: PASS with limitation.** `replay_verified/packaged_verification.json` records fresh-process, socket-blocked execution for four files including 60-second audio, exact max delta `0.0`, load-inclusive runtimes `13.453s` and `30.353s`, and ZIP/expanded sizes below 10/32 GB. `precision_diagnostic.json` reproduces the original `2.2172927856445312e-5` failure and discriminates the TF32 fix without relaxing `1e-5`. L4 1,200-file runtime remains explicitly unverified; the plan requires honest reporting rather than an unsupported L4 claim.
- **C5 notebook/Drive/GPU/review cleanup: PASS.** The primary notebook has 18 executed code cells and one preserved expected package-parity error; the replay notebook has 4/4 executed code cells and zero errors. Runtime receipts total about 23.60 minutes of executed GPU work, Drive links/receipts are recorded, the final UI evidence shows zero active sessions, and no new submission occurred.

## direct remove-ai-slops and programming pass

I directly inspected the production diff and all added tests. The tests target concrete defects: tied-score ROC behavior, test-role rejection, CRC/range corruption, duplicate PCM and lineage leakage, unknown-label masking, gradient connectivity and parameter updates, best-checkpoint aliasing, sample-rate parity, checkpoint/manifest lock integrity, socket-blocked package execution, file independence, and `<=1e-5` numerical replay. I found no deletion-only test, requested-removal assertion, prose pin, tautological expected value, implementation-mirroring test, or test added merely for coverage.

Production code does contain three maintainability notes that do not violate a stated success criterion: `train_v2.py` (392 added lines), `prepare_v2_sources.py` (265), `prepare_v2_validation.py` (289), and `v2_scoring.py` (287) exceed the programming skill's 250-LOC guideline; the validation PCM CSV has CRLF lines that `git diff --check` reports as trailing whitespace; and the hybrid package duplicates the full G0 and AST assets, producing a 5.35 GB ZIP. These are NOTES, not blockers. No speculative framework, public API break, unsafe normalization, or untyped escape hatch was found that independently defeats the requested outcome.

The code/data/orchestration reports explicitly include the required programming and remove-ai-slops perspectives. The replay report also includes those perspectives. Its original provenance ordering sentence is factually wrong, but the appended chronology correction explicitly retracts that sentence and matches the timestamped artifacts.

## checked artifact paths

- `.omo/plans/dacon-v2-redesign.md`
- `/var/folders/1t/xzrjp0k52f76sd1rsd1_fbrw0000gn/T/ulw-20260913-014029.XXXXXX.md.DQVsPcTgPE`
- full `git diff 70abd7a` and current uncommitted diff/status
- `.omo/evidence/v2-code-review.md`
- `.omo/evidence/v2-data-review.md`
- `.omo/evidence/v2-orchestration-review.md`
- `.omo/evidence/v2-replay-review.md`
- `.omo/evidence/v2/final_regression_suite.log`
- `results/v2_forensics.json`
- `v2_validation/development/manifest.csv`
- `v2_validation/locked_test/manifest.csv`
- `v2_validation/split_audit.json`
- `v2_validation/pcm_sha256.csv`
- `results/v2_colab_20260912_174330/training_report.json`
- `results/v2_colab_20260912_174330/comparison.json`
- `results/v2_colab_20260912_174330/selection.json`
- `results/v2_colab_20260912_174330/candidate_lock.json`
- `results/v2_colab_20260912_174330/test_opened.json`
- `results/v2_colab_20260912_174330/hybrid_selection.json`
- `results/v2_colab_20260912_174330/hybrid_candidate_lock.json`
- `results/v2_colab_20260912_174330/hybrid_comparison.json`
- both executed notebooks under `results/v2_colab_20260912_174330/`
- all JSON receipts under `results/v2_colab_20260912_174330/replay_verified/`
- `results/v2_evidence/hybrid_pretest_protocol.json`
- `results/v2_evidence/final_decision.json`
- `results/v2_evidence/final_runtime_released.jpeg`
- `results/v2_evidence/final_runtime_released.txt`
- `RESULTS_V2.md`, `PIPELINE_GRAPH_V2.md`, and `DATA_SOURCES_V2.md`

## exact evidence gaps

- The hybrid candidate was not locked before its component predictions were made. It is correctly scoped as exploratory and is not part of the accepted primary result.
- L4 throughput for 1,200 evaluator files remains unmeasured; this is disclosed and is not a blocker under the stated criteria.
- The 5.35 GB exploratory hybrid ZIP is evidenced by the Colab replay receipt and Drive link, not retained as a local ZIP; its hash and size were not independently recomputed from local bytes in this gate.

## terminal re-review

The machine-readable `results/v2_evidence/final_decision.json` passed direct semantic checks: primary selected `G0`, primary selection unchanged, hybrid explicitly exploratory, hybrid not locked before primary test opening, and no new submission. The unchanged local official baseline ZIP hashes to `cc86f143915c95fd958a0066cbf14b0db9f4c41fd213310c3650b9fe870d0c70`; `results/pilot_20260912_160943/packaged_verification.json` records a real fresh offline G0 execution with maximum probability delta `0.0`, and the official submission receipt records score `0.6909110688` with 21m59s evaluator time. No fresh reselection, lock rewrite, or new submission was used to obtain approval.
