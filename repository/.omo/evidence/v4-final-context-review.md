# V4 final context and plan-evidence consistency review

Date: 2026-09-13

## Recommendation

`REJECT` **as a final execution or model-quality claim at the present checkpoint.** The implementation and live run are still in progress. This recommendation does not say the experiment failed scientifically; it says the plan's execution-completion artifacts do not yet exist.

## Exact source binding

- Reviewed immutable checkout: `/tmp/dacon-v4-review-a01a50e`
- Reviewed commit: `a01a50e0c774f181c20238109127c5b81cc95dc1`
- Comparison base: `f69f4bf`
- Plan: `.omo/plans/trainable-recovery-v4.md`
- `omo ulw-loop status --json`: `ULW_LOOP_PLAN_MISSING`; the required fallback report path is therefore this file.

## Original intent

After the low leaderboard result and unexpectedly short prior inference, the user wanted a real, bounded A100 experiment: genuine full fine-tuning of pretrained music encoders, explicit epoch/hyperparameter trials, roughly five pipeline comparisons, actual execution in Aside Colab, Drive persistence, safe release within the original two-hour allocation, and no automatic DACON submission.

## Desired outcome

The user should receive an honest completed experiment record. Every attempted trial must have a reproducible terminal outcome or an explicit censored reason; P1-P4 and the fixed probes must feed a predeclared development decision; a qualifying candidate alone may undergo complete 1,286-ID refit, weight lock, sealed evaluation and package replay. If no candidate qualifies, the scientific result is “no improvement found,” G0 is retained, artifacts are archived, and the A100 is released. Either scientific outcome can satisfy execution completion.

## User outcome review

The current artifacts prove meaningful partial execution only. G0 covers all 210 development rows. Six probes each completed 60 updates. P1 completed five epochs, 175 updates and 5,580 window exposures in 67.7886 seconds. The first probe and P1 fail prespecified OOD gates; no probe passes all gates. These are valid negative development observations, not a final comparison, final quality result, or official score.

P2 is running and P3/P4, final development selection, conditional refit, weight lock, sealed decision, package replay, Drive retrieval verification, and release receipt remain absent at this checkpoint. Therefore neither “improved model” nor “completed no-improvement experiment” is supportable yet.

## Blockers

1. `violatedCriterion: EXEC-TRIAL-MATRIX` — The execution criterion requires every attempted trial to have a reproducible outcome and the plan requires five primary pipeline rows with measured or explicit censored results. Current evidence contains P1 only; P2 is running and P3/P4 remain pending. `evidencePointer: .omo/evidence/v4-retry-initial-metrics.json#limits; .omo/evidence/v4-retry-P1-terminal.json`
2. `violatedCriterion: EXEC-FINAL-DECISION` — The plan requires a development selection followed, when eligible, by complete refit/lock/sealed handling, or a recorded rejection that retains G0. No final `selection.json`, recipe/refit/weight lock, sealed decision, or terminal rejection exists. `evidencePointer: .omo/plans/trainable-recovery-v4.md tasks 5 and Success criteria; absent from .omo/evidence/v4-retry-*`
3. `violatedCriterion: EXEC-ARCHIVE-RELEASE` — Execution completion requires retrievable artifacts and GPU release. The live receipt still says `setup_running`; no final Drive readback receipt or release evidence is present for the retry. `evidencePointer: .omo/evidence/v4-retry-live-run.json`

## Constraints and claim boundaries

- The original allocation remains pinned to `1789290385` as the safe deadline; any later receipt must prove it did not reset the two-hour clock.
- No sealed data may be opened before recipe/weight lock, and no rejected or incomplete candidate may be submitted. The notebook contract expressly forbids automatic submission.
- `absolute_score` is null because 110 relevant voice-fake labels and 90 voice-presence labels are unknown; `voice_presence_auc` also lacks both known classes. Any five-head or official-score claim is unsupported.
- The 1,286-ID refit is conditional on a qualifying development winner and sufficient reserved time. It is not current evidence of completed full-data refit.
- The 67.8-second P1 run and short update profile measure the V4 development workload. They do not establish official L4 package inference time or full-package runtime.
- “Full fine-tuning” here applies to the selected music detector, not every model in the multi-model submission. Current terminal artifacts report trainable parameter counts, but the final run receipt must preserve gradient/update and weight-hash evidence for the exact selected model.
- No leaderboard improvement, submission, sealed acceptance, or final model-quality claim is currently supported.

## Regulatory, data-license and source-disjoint review

The existing record already captures the applicable official constraints: 4-60 second inputs, no private-test training or cross-file normalization, offline L4 limits, archive size limits, and no automatic submission. The plan requires per-source licensing rather than inference from papers. The audited corpus reports 1,936 rows, no missing license rows, held-out providers excluded from training, derivative-parent closure, and speaker/provider/artist separation. No new missing regulatory, licensing, or source-disjoint condition was found in the recorded scope. Final approval still requires the terminal run to preserve these manifests and chronology; this review did not browse because the existing official-source-linked plan and audit artifacts directly cover the requested conditions.

## Direct programming and overfit/slop pass

I directly inspected the changed production/test scope and the referenced reviews. The large test files mostly guard real failure modes: full-source tail retention, lineage/provider leakage, optimizer ownership and actual layer updates, deterministic resume, deadline-safe partial state, P4 treatment/control separation, incomplete-refit refusal, immutable recipe/weight bindings, and sealed-prediction substitution. I found no deletion-only test, requested-removal assertion, prompt/prose pin, tautological expected value, or production extraction/normalization added only to satisfy a test.

Maintenance notes that do not violate an execution success criterion: `prepare_v4_data.py` and several orchestration modules exceed the programming perspective's 250-LOC preference; evaluation payloads use broad nested dictionaries; two `ModelKind` branches are not exhaustively matched. These are NOTES, not blockers. The earlier P4-control blocker in `.omo/evidence/v4-training-final-review.md` is superseded by `.omo/evidence/v4-control-delta-review.md`, which verifies the distinct mean/no-aux control behavior. The code-review reports explicitly cover the same programming and remove-ai-slops categories, including deletion-only, tautological, implementation-mirroring, parsing/normalization, and needless-abstraction checks.

## Checked artifacts

- `.omo/plans/trainable-recovery-v4.md`
- `.omo/evidence/v4-review-context.txt`
- `.omo/evidence/v4-retry-live-run.json`
- `.omo/evidence/v4-retry-initial-metrics.json`
- `.omo/evidence/v4-retry-G0_predictions.csv`
- `.omo/evidence/v4-retry-first-probe-terminal.json`
- `.omo/evidence/v4-retry-probe_sonics_3e-05-terminal.json`
- `.omo/evidence/v4-retry-probe_sonics_1e-04-terminal.json`
- `.omo/evidence/v4-retry-probe_ast_1e-06-terminal.json`
- `.omo/evidence/v4-retry-probe_ast_3e-06-terminal.json`
- `.omo/evidence/v4-retry-probe_ast_1e-05-terminal.json`
- `.omo/evidence/v4-retry-P1-terminal.json`
- `.omo/evidence/v4-retry-stage_budget_plan.json`
- `.omo/evidence/v4-retry-throughput_profile.json`
- `.omo/evidence/v4-data-final-review.md`
- `.omo/evidence/v4-model-review.md`
- `.omo/evidence/v4-training-final-review.md`
- `.omo/evidence/v4-control-delta-review.md`
- `.omo/evidence/v4-eval-review.md`
- `.omo/evidence/v4-live-budget-review.md`
- `v4_validation/data_protocol_audit.json`
- `v4_validation/dataset_lineage_audit.json` via its recorded review
- `v4_validation/licenses/*`
- commit diff `f69f4bf..a01a50e0c774f181c20238109127c5b81cc95dc1`

## Exact evidence gaps

- P2 terminal result or censor reason.
- P3 and P4/control terminal results or explicit not-run/censored reasons.
- Final development `selection.json` recomputed from saved predictions.
- Conditional recipe lock, complete 1,286-ID refit ledger, final weight audit/lock, and sealed-open chronology, or a terminal G0-retention record showing why these were correctly skipped.
- Final executed notebook receipt and exact Drive retrieval/hash verification.
- Package replay/timing evidence only if a candidate is accepted; otherwise an explicit plan-consistent N/A reason.
- Retry A100 release timestamp and Aside zero-active-runtime evidence.
- Final budget receipt proving completion/release before UNIX `1789290385`.

