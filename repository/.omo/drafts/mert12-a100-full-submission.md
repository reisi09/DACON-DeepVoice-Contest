---
slug: mert12-a100-full-submission
status: executing
intent: clear
review_required: false
pending-action: finish prelaunch checks, execute authorized A100 refit and submission
approach: reuse selected frozen MERT12 features and baseline inference; full-data three-seed head refit
---

# Draft: mert12-a100-full-submission

## Components (topology ledger)
<!-- Lock the SHAPE before depth. One row per top-level component that can succeed or fail independently. -->
<!-- id | outcome (one line) | status: active|deferred | evidence path -->

## Open assumptions (announced defaults)
<!-- Record any default you adopt instead of asking, so the user can veto it at the gate. -->
<!-- assumption | adopted default | rationale | reversible? -->

## Findings (cited - path:lines)
Canonical music population is 1322 core/base known-label rows: 218 REAL and1104 FAKE. Parent/source-hash join covers1116 previous eligible rows,128 freshFMA and78 freshSONICS; no missing source URL or license field. All1322 decoded audio files exist relative to the canonical dataset directory. Training features exist in results/eight_h100_20260917/downloaded/unpacked/output/features.
Existing A100 wsp-5i3knd3gxeox is stopped, has empty initScript, persistent /root cluster volume and shared object volume at /root-1. Existing H100 stays stopped.
Live official rules permit public external resources subject to terms and prohibit hidden-test training/cross-file inference statistics. Official account submission page is accessible in Codex browser. Highest visible recorded score0.7396465503, submission90735; latest row91610 score0.6837822646.

## Decisions (with rationale)
Keep MERT12 encoder frozen and baseline voice/presence/file-max inference. Original balanced schedule misses109/1322 rows; use a deterministic within-class coverage prefix while preserving500updates,batch32 and optimizer. Bind exact ordered eligible ID list and manifest hash to an audit receipt. Benchmark32/128/256 with equal sample exposure on A100 only for throughput; do not infer quality or silently alter final recipe.

## Scope IN
Reproducible refit notebook, verified offline ZIP, GPU parity and single-file independence, bounded watcher, official submission.

## Scope OUT (Must NOT have)
No newworkspace; no extension of old or new recorded deadlines; no hidden-test learning; no unverified generalization/L4 claims; no unrelated edits.

## Open questions

## Approval gate
status: execution_already_authorized
User explicitly requested A100 full-data training and submission before planning began. Higher-priority autonomy/user authorization overrides the skill's additional approval workflow. No repeat approval needed for preparation, training or authorized ZIP submission.
<!-- When exploration is exhausted and unknowns are answered, set status: awaiting-approval. -->
<!-- That durable record is the loop guard: on a later turn read it and resume at the gate instead of re-running exploration. -->
