# Terminal code-quality review: eight H100 experiments

**Reviewed HEAD:** `2a0697edef92903f604cdb50f77ef807a940c5d0`
**Scope:** `eight_*.py`, `build_eight_h100_notebook.py`, runtime sources, plan, and downloaded output.
**Verdict:** **FAIL — REQUEST_CHANGES**
**codeQualityStatus:** **BLOCK**

## Basis checked

- The downloaded output contains E1–E8 reports, E7 checkpoints/predictions, selection locks, and raw prediction files. `output/progress.json` ends at `experiment_5`, but that progress marker is stale rather than evidence that E6–E8 failed: their reports and durable artifacts are present.
- The plan’s fixed budgets are `head_steps=500`, `lora_steps=200`, three seeds. The executed `output/config.json` matches those values.
- Source flow was traced for provenance splitting, cache/audio identity, train-only feature normalization, class-balanced schedules, selection-before-test ordering, E2/E5 labels, E8 REAL-only fitting, and E7 reset/checkpoint paths.
- Skill-perspective check ran: `omo:programming` (Python reference) and `omo:remove-ai-slops` were explicitly loaded. No deletion-only/removal-only tests, tautological tests, prompt-text tests, implementation-constant tests, untyped escape hatch, or needless production parsing/normalization was found in this review scope. The production diff does not violate either perspective in a release-blocking way.

## Findings

### CRITICAL

None found that invalidates the raw branch predictions or the numerical training paths.

### HIGH

1. **Published aggregation metadata is false for multiple experiments and must not be approved as the final report.**
   - [build_eight_h100_notebook.py](/Users/choiwuseck/Documents/ChatGPT/데이콘/build_eight_h100_notebook.py:210) writes the global manifest `component_count` and every-label denominator into every E1–E8 aggregation entry instead of the population actually consumed by that experiment.
   - [build_eight_h100_notebook.py](/Users/choiwuseck/Documents/ChatGPT/데이콘/build_eight_h100_notebook.py:212) substitutes `config["head_steps"]` whenever a report has no `fixed_step_counts`; this turns E8’s REAL-only, zero-update baseline into `fixed_updates: 500` in `final_aggregation.json`.
   - The downloaded `final_aggregation.json` consequently reports `component_count: 178` for every experiment, while `experiment_1.json`, `experiment_3.json`, `experiment_4.json`, `experiment_6.json`, and `experiment_8.json` report `256`; it also reports E8 as 500 updates although its report has `fixed_updates: null` and `class_exposure: REAL-only fit`.
   - This does not alter the stored raw predictions, but it makes the durable study-level interpretation internally contradictory. Regenerate the final aggregation from each experiment’s exact report fields and represent E8 as no optimization updates.

### MEDIUM

None.

### LOW

None.

## Verified non-findings

- `eight_heads.balanced_draws` balances both classes per batch; E1/E3 paired arms use the same seed schedule, and E3 rejects unequal draw hashes.
- E1/E3/E4/E6 train normalization is fit on train rows only. E8 fits both standardization and diagonal Gaussian on train REAL rows only, then fixes high-distance-is-fake threshold on calibration REAL rows.
- E5 selects pools on dev after calibration thresholds and before sealed-test scoring. E7 creates/restores pristine adapter-plus-classifier state around frozen evaluation and validates checkpoint reload logits.
- E2 retains raw detector scores and marks unknown voice labels separately; it does not synthesize labels for unverified backing vocals.

## Required blocker resolution

Regenerate and re-verify `output/final_aggregation.json` so that each experiment carries its own population, label denominators, component count, and fixed-update semantics. Preserve the raw experiment reports and predictions; this is a reporting correction, not grounds to discard their numerical evidence.

## Decision

```json
{
  "codeQualityStatus": "BLOCK",
  "recommendation": "REQUEST_CHANGES",
  "reportPath": ".omo/evidence/eight_h100_20260917/review_code.md",
  "blockers": [
    "final_aggregation.json falsely reports global denominators/component counts for every experiment and reports E8 as 500 updates; regenerate it from experiment-specific report fields."
  ]
}
```
