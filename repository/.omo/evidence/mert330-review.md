# MERT-330 input-dimension experiment code review

## Scope and evidence inspected

Read-only review of `music_temporal.py`, `temporal_resolution.py`, `temporal_experiment.py`, `temporal_backbone.py`, and `test_temporal_input_dim.py`.

Independently ran:

```text
.venv/bin/python -m pytest -q test_temporal_input_dim.py test_music_temporal.py test_temporal_resolution.py
```

Result: `13 passed, 2 subtests passed`. `git diff --check` also passed.

## Findings

### CRITICAL

None.

### HIGH

None. The prior external-control blocker is resolved by the reviewed execution contract. `h100_bootstrap.sh:18-64` verifies the input ZIP, then SHA-verifies the prior 1,446-member paired-cache archive before extracting it to `reference95_cache`; `temporal_backbone.py:19-30` verifies every sealed input member before GPU extraction. The M95 heads, training receipts, tune predictions, linear control, and M95 features manifest are present in the sealed input ZIP. Each extracted M95 NPZ is checked again against its per-record receipt before use (`temporal_backbone.py:46-54`). The new corruption/missing-control test exercises the early validation failure (`test_temporal_input_dim.py:16-37`).

### MEDIUM

1. **No L4 end-to-end timing evidence exists yet.** The bootstrap validates an H100 and uses a two-hour experiment deadline (`h100_bootstrap.sh:79-87`). This is appropriate for the authorized H100 comparison but does not establish the contest L4 submission-runtime limit. Keep any later submission claim scoped to the H100 experiment until an L4 end-to-end run is performed.

### LOW

None.

## Confirmed properties

- `TemporalRecipe`, `TemporalMusic`, batching, train/validation prediction, and strict checkpoint restore now carry `input_dim`. The `d1024` architecture tag prevents treating a 1,024-input checkpoint as the legacy C08 format (`music_temporal.py:29-308`).
- The 1,024 mode is deliberately constrained to compact 64×3, GELU, one-bin, sigmoid operation, preserving existing branches and preventing accidental combination with the unrelated softmax/eight-bin study (`music_temporal.py:75-106`).
- Extraction derives the dimension from the sealed MERT config, records it in the v2 cache identity, and validates both coarse/fine array shapes plus their weighted identity (`temporal_resolution.py:39-136`).
- Historical evaluation routes a reloaded 1,024 checkpoint to `wide_records`, retaining coarse M95 records for legacy heads (`temporal_experiment.py:155-183`).

## Skill-perspective check

Ran: yes. I applied the already-loaded `omo:programming` Python perspective, `omo:remove-ai-slops` review pass, and Ponytail review perspective.

- `remove-ai-slops`: the added corruption/missing-control test and the input-dimension gradient/reload test protect real reproducibility and connectivity failures. They are not deletion-only or tautological dimension assertions.
- `programming`: no prompt tests, untyped escape hatches, or needless abstraction appeared in the scoped diff. The manifest validation puts integrity checks at the execution boundary.
- Ponytail: a single input-dimension parameter, one new architecture tag, and reuse of the sealed paired cache are proportionate to the requested comparison.

## Recommendation

- `codeQualityStatus`: **WATCH**
- `recommendation`: **APPROVE**
- `blockers`: none.
