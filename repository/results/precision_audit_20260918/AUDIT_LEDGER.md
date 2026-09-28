# DACON 236749 precision audit — 2026-09-18

## Scope

User request: inspect the complete pipeline, from small details to major modeling choices, explain stagnation using actual DACON submissions, and consider substantially different methods including ViT.

Current production baseline: submission 92294, MERT12_full1322_submit.zip. Live authenticated DACON submission table observed in this session: score 0.7479394074, ADS 0.7211111111, CPS 0.9893940741, runtime 19m12s; submitted 2026-09-18 10:30:25. Package SHA-256 in submission memo: e7bebbded6fef06517db0d42421a575a0440849248de99856796ffc3e5eb5ee5.

The existing MERT12 official_submission.json and REPORT.md still describe their earlier grading state. Those files are historical records; this audit will retain its own live receipt.

## Starting state and boundary

- Git branch main, initial HEAD a0e25e5. Pre-existing tracked change: .omo/evidence/v4-training/source-sha256.txt. Numerous unrelated untracked experiment artifacts exist and must not be staged.
- Inspection uses local .venv/bin/python, current source, small ZIP members, existing checkpoints/predictions, and primary external sources.
- No paid compute, new workspace, resubmission, production code modification, model retraining, or large dataset download is part of this audit.
- Existing .debug-journal.md belongs to an earlier remote recovery task; do not overwrite or clean its artifacts.
- All new durable audit artifacts live under results/precision_audit_20260918/. They are requested evidence, not temporary instrumentation.

## Independent hypotheses

| ID | Hypothesis | Distinguishing evidence | Owner |
|---|---|---|---|
| H1 | Data lineage, label gaps or source-class confounding make local selection unreliable | Raw manifests, connected groups, source/label tables, holdout histories | data_lineage |
| H2 | Actual inference differs from training/cache or component semantics | Submitted ZIP code and real parity/supplemental predictions | inference |
| H3 | Optimization/sampler/feature choices prevent intended learning | Actual exposure by source and parent, losses, gradient/checkpoint receipts | training |
| H4 | Metrics, grouping, model selection or uncertainty calculations mis-rank candidates | Independent replay of saved predictions and selection logic | evaluation |
| H5 | Packaging/runtime differences or weak provenance misidentify the executed model | Exact ZIP member bindings, live submission memo, receipts and output schema | packaging |
| H6 | Existing representations/objectives miss useful forensic or temporal evidence | Prior controlled trials plus primary papers and official implementations | new_methods |

## Evidence labels

- Confirmed behavior: directly measured in this session or specific saved real execution, with scope stated.
- Confirmed code defect: executable reproducer or exact code path violates a concrete contract; dataset occurrence reported separately.
- Hypothesis: plausible mechanism requiring a distinguishing experiment.
- Already fixed: evidence identifies an old problem and verifies the current path changed.
- Not established: evidence insufficient, missing ground truth, unavailable hidden component results or target execution.

## Completion gates

1. Map current end-to-end behavior and identify which historical paths were inspected.
2. Every material finding has precise source, affected scope, severity, evidence level and practical next action.
3. Distinguish live official totals, local diagnostic metrics, numerical parity and model quality.
4. Include verified non-issues, unresolved questions, and useful limits of this audit.
5. Independently review the synthesized report, polish Korean without changing facts, commit only audit artifacts.

## Final audit disposition

- H1: source-label association, missing verified voice-absence class and narrow independent populations confirmed; recorded cross-split lineage/hash checks held.
- H2: current duration/aggregation limitations and edge cases confirmed; latest decoder discrepancy already repaired, no current parity or polarity defect established.
- H3: source-balanced sampler omission confirmed against saved schedule hashes; latest coverage prefix leaves 109 REAL rows seen exactly once at the beginning. Actual saved-head fit disparity measured. Corrected-sampler quality remains untested.
- H4: metric arithmetic replay held; E5 denominator metadata and generic bootstrap admission defects confirmed, selection/update-budget limits documented.
- H5: fresh full ZIP SHA and runtime/head bindings held; official submission 92294 completed. 91610 model identity remains unbound.
- H6: reviewed prior AST/EAT fullFT and temporal/partial-edit history; proposed controlled patch/localization, temporal aggregation and CNN comparisons without claiming measured gains.
- REPORT.md and eight detailed sections contain scope, findings, passed checks, unresolved questions and follow-up decisions. REVIEW.md records the independent skeptical synthesis review and CPU sampler/head replay.
- No production changes, retraining, cloud starts, uploads or new submissions were performed. Only this audit directory is eligible for the requested local commit.
- Raw DOM submission text and the extracted DF-Arena backbone retain their original trailing whitespace. The generic Git whitespace check flags those verbatim evidence files; authored report/replay files are checked separately without rewriting the source evidence.
