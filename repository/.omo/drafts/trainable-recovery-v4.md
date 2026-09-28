---
slug: trainable-recovery-v4
status: design-complete
intent: clear
review_required: false
pending-action: write .omo/plans/trainable-recovery-v4.md
approach: Full fine-tuning of a music authenticity encoder, provider-diverse full-source audio, controlled augmentation and temporal MIL comparisons, under a measured 7200-second A100 budget.
---

# Draft: trainable-recovery-v4

## Components (topology ledger)
data | original audio, provenance, source groups and sealed evaluation | active | prepare_v2_sources.py; recovery_validation/provenance.json
training | all encoder layers receive optimization, bounded LR/epoch search | active | ast_multitask.py; train_v2.py
inference | within-file window integration and unchanged reference components | active | recovery_sonics.py; baseline_original/script.py
selection | no promotion after failed quality gates | active | results/recovery_v3/colab_20260913_044621/selection.json
execution | Colab, Drive, checkpoint replay and GPU release | planned | build_recovery_notebook.py
<!-- Lock the SHAPE before depth. One row per top-level component that can succeed or fail independently. -->
<!-- id | outcome (one line) | status: active|deferred | evidence path -->

## Open assumptions (announced defaults)
Current deliverable is a concrete pipeline design. Existing 2-hour A100 ceiling persists for its subsequent execution. No GPU is allocated during design.
Full fine-tuning means all learned parameters of the selected music encoder and its classifier; it does not mean retraining PANNs, Demucs and the 1B speech model.
Read-only research delegation and Metis gap review follow the planning skill. Intent is clear; the user requested careful reasoning, not the skill's separate dual high-accuracy CLI review workflow. review_required remains false; do not claim that review was performed.
<!-- Record any default you adopt instead of asking, so the user can veto it at the gate. -->
<!-- assumption | adopted default | rationale | reversible? -->

## Findings (cited - path:lines)
ast_multitask.py:85: old AST mode unfreezes only the last block, final norm and five heads.
v2_data.py:66: source preparation retains central 10 seconds. All 740 original source files still exist locally (read-only inventory checked this turn).
_workspace/v2_discovery/archive_probes/echoes_manifest.csv: 4468 rows, 12 providers; echoes_fma_unique.json: 220 matched tracks, 106 unique artists, 72 previously used and 34 candidate new artists (independently counted after Metis correction).
results/recovery_v3/new_candidate_diagnostics.json: frozen SONICS misses Echoes fake examples while reducing real false positives. A temporal root cause is not established.
https://arxiv.org/html/2603.23667v2: current Echoes provider diversity, transfer and held-out-provider evidence. v1 is obsolete for counts/license; v2 agrees with local manifest.

## Decisions (with rationale)
Preserve known-working speech/presence paths while directly replacing and training the music encoder; compare SONICS full FT versus AST full FT, augmentation, and bag-level learning separately.
Old 1029 and recovery 88 are all previously inspected. New group-disjoint dev and sealed panels must be acquired before execution. Unknown labels remain unknown.
Use matched controls and save resource-censored trials as inconclusive. No speculative throughput guarantee or guaranteed score gain.

## Scope IN
One executable research/implementation plan with sources, pipeline graph, fixed experiment matrix, loss definitions, hyperparameters, budget, acceptance and export contracts.

## Scope OUT (Must NOT have)
No new training, submission, source-code mutation or GPU spending in this design turn. No private evaluation data learning or cross-file inference statistics.

## Open questions
None blocking design. Discoverable implementation details have explicit verification tasks and fail-closed outcomes.

## Approval gate
status: authorized-by-current-request
Authorization: user explicitly instructs pipeline design now; higher-priority developer requires completing authorized reversible work without a redundant approval pause. Plan creation is authorized. Execution status is separately reported, not implied by the plan.
<!-- When exploration is exhausted and unknowns are answered, set status: awaiting-approval. -->
<!-- That durable record is the loop guard: on a later turn read it and resume at the gate instead of re-running exploration. -->

## Metis gap review integrated
Reviewer /root/v4_plan_gaps found a critical track-versus-artist count error. Direct set counting confirmed 220 tracks != 220 artists: 106 unique, 72 previously used, 34 available before acquisition checks. Revised panels to 12+12 new artists and train target70/min60; fixed provider minima and acquisition failure before GPU. Other repairs: immutable original initialization, explicit pooling/label formulas, disable vendor MixUp/masking, count full parameter/epoch exposure, final refit before sealed opening, no partial-refit promotion. This is a Metis gap review, not the separate dual high-accuracy workflow.

Second Metis review: TTA/provider intersection leaves only21 eligible artist identities (Stable19/Mubert15/intersection13); root independently reproduced these counts. Final panels are10+10,50music/210total files each, with exclusive-first feasible assignment. P1/P3 independent stopping conflict repaired by matched P1 update counts; exact tail grids/short bags fixed; complete refit forecast moved into pre-ranking eligibility. No dual-review approval is claimed.

Final design verification: six implementation tasks and four final-verifier tasks; all template headers present in order; budget sums to120minutes; final role count210; source/model/manifest references checked. Documentation only: no V4 training or GPU allocation has occurred.
Plan SHA256: 06208ce573350d777f3884912a48bb98f998fb183434af7fed97fdbee2b78bc2
