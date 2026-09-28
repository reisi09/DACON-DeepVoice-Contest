# V4 live-budget delta review

**Scope.** Read-only review of the live-budget retry delta only: `train_v4.py`,
`v4_training_refit.py`, `build_v4_notebook.py`, and the added budget regression.
No model, evaluator, optimizer, or GPU execution was reviewed or run.

**Verdict.** `codeQualityStatus: WATCH`  
**recommendation: APPROVE**  
**blockers: none**

## Evidence inspected

| Artifact | SHA-256 |
| --- | --- |
| `train_v4.py` | `497dadf37ce6ddcd4d081906bd41a110d0dd05d6bdfe0d4b5b7920427cca351a` |
| `v4_training_refit.py` | `6ccd87831ac23bca015960bbac80bfa5921c9a8a4539b773a7a3885bd9381b72` |
| `build_v4_notebook.py` | `25a3be622d9115f9a9d7318f5ee3cc5f3b02a596fb7ba8c6ce5d62958a572c2d` |
| `test_v4_training.py` | `d0a61437d785d3d3c2f021a9e30f6b12c2ee5701acd792137d6ae3203c56a764` |
| `.omo/evidence/v4-attempt1-events.txt` | `dd1fa686c978ccf10b7e5adaa72cc7a7408e305c00eb32424261baf65a8c9826` |
| `.omo/evidence/v4-attempt1-log.txt` | `8c2df70796da5bc437e660c1b15029b1fd0c2d249e28a6ae51ba0069a986afac` |
| `.omo/evidence/v4-live-run.json` | `20a454ae432b4d8d09fc78e914dbe15c2a9be7e3cb947b322cdb8b74669f694a` |
| `.omo/evidence/v4-training/live-budget-schedule.json` | `990bb3f94ac8e052571ef515e91aef4f736e0552bcbbd485d63cbc892b003619` |

The recorded first attempt establishes an actual allocation start of
`1789283305`, external safe deadline `1789290385`, and an old profile limit
that had already expired before the first batch. The failure is consistent with
that fixed profile cutoff.

The delta pins a new-run notebook to `ALLOCATION_START + 7080`
([`build_v4_notebook.py:70-76`](../../build_v4_notebook.py#L70-L76)), passes
that exact deadline to the runner ([`build_v4_notebook.py:251`](../../build_v4_notebook.py#L251)),
and keeps a watchdog termination margin ([`build_v4_notebook.py:257-261`](../../build_v4_notebook.py#L257-L261)).
For the new-run path, `budget.json` is written with the pinned origin
([`build_v4_notebook.py:112-115`](../../build_v4_notebook.py#L112-L115)).
The runner then clamps any supplied deadline to the declared deadline and the
allocation-origin cap ([`train_v4.py:53-56`](../../train_v4.py#L53-L56)).
Thus a retry cannot extend the original safe deadline.

The schedule scales the remaining time from real runner entry and preserves
the ordered profile, G0, search, refit, sealed, export, and shutdown boundaries
([`train_v4.py:43-50`](../../train_v4.py#L43-L50),
[`train_v4.py:85-95`](../../train_v4.py#L85-L95)). Refitting and sealed work
consume the schedule values rather than the former fixed absolute offsets
([`v4_training_refit.py:32-145`](../../v4_training_refit.py#L32-L145)).

Independent CPU-only checks run for this review:

* `uv run --no-project python -m pytest -q test_v4_training.py::test_delayed_setup_rebudgets_stages_without_resetting_allocation` — passed (`1 passed in 3.18s`).
* `uv run --no-project python -m py_compile train_v4.py v4_training_refit.py build_v4_notebook.py` and `git diff --check` — passed.
* A temporary generated notebook pinned to `1789283305` passed structural validation: it embeds the current `train_v4.py` and `v4_training_refit.py` byte-for-byte, has 28 cells with no pre-existing outputs, uses `SAFE_DEADLINE = ALLOCATION_START + 7080`, and produced `/tmp/v4_live_budget_review.ipynb` (SHA-256 `f967730f33076912e6dc9ad285a3b7611ee567b9e8cc9cc4de7fc3a768609c9c`).
* A direct schedule calculation at `runner_start = allocation_start + 636.5252` yielded hard/shutdown `1789290385.0`, with profile `1789284378.370949`, refit `1789289129.0684712`, sealed `1789289675.1256576`, and export `1789289948.1542509`; the boundaries are strictly ordered.

## Findings

### CRITICAL

None.

### HIGH

None.

### MEDIUM

1. **Attempt-timing evidence has a 45-second origin mismatch.**
   `test_delayed_setup_rebudgets_stages_without_resetting_allocation`
   ([`test_v4_training.py:157-166`](../../test_v4_training.py#L157-L166)) treats
   `636.5252` seconds as elapsed from allocation start, placing runner entry at
   `1789283941.5252`. The attempt log instead records first-cell start
   `1789283350.5691943` and outer runner launch `1789283972.492938` (621.9237442
   seconds after first cell); the runner's own event is later still, at
   `1789283987.0944014`. The 636.5252-second value is compatible with a
   first-cell-relative measurement, not the declared allocation origin
   `1789283305`. The live implementation reads `time.time()` at runner entry,
   so this does not permit a deadline extension and is not an approval blocker.
   Correct the test/evidence wording and use a single declared origin before
   presenting its exact stage timestamps as observed timing.

### LOW

1. **Unused parameter:** `_budget_schedule` accepts `allocation_start` but does
   not use it ([`train_v4.py:43`](../../train_v4.py#L43)). It does not affect the
   current deadline calculation; remove it in later cleanup if the call sites
   are touched again.

## Skill-perspective check

The required `remove-ai-slops` and `programming` reviews were loaded and
applied. The added regression protects a real observed failure mode (setup
consuming a fixed early profile window), rather than a deletion-only,
tautological, prose, or implementation-constant assertion. It does not use
an untyped escape hatch, brittle prompt assertion, or unnecessary production
parsing/normalization. The small duplicate `min(args.deadline, stage)` checks
remain useful defense at each downstream deadline consumer. The diff has no
blocking slop violation; the timestamp-origin discrepancy above is the only
material evidence-quality concern.
