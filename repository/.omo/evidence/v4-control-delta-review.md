# V4 P4 control delta review

**Scope:** Read-only delta review of `train_v4.py`, `v4_training_trials.py`,
and `test_v4_training.py` against
`.omo/evidence/v4-training-final-review.md`. The earlier review's only HIGH
finding was the P4 matched-control loss path. No other module or previously
approved criterion was re-audited.

## Result

```json
{
  "codeQualityStatus": "CLEAR",
  "recommendation": "APPROVE",
  "blockers": []
}
```

## Frozen source binding

The live hashes match the refreshed receipt
`.omo/evidence/v4-training/source-sha256.txt` exactly:

| File | SHA-256 |
|---|---|
| `train_v4.py` | `4597acb1516de3d707f3ffd6b81968c228d4d986e0e79547d6f360c79adbdc5a` |
| `v4_training_trials.py` | `feb2bd54f6c7ca238500c8ec2e164b6230be4bb4d96179d838603bf7d0a6fb23` |
| `test_v4_training.py` | `eab6e81fe984e244bc32dfe1785943b2053144b0116119c6a635ce47a3e29bd2` |

## Delta verification

- The continued control is still constructed with `p4=True` and
  `p4_control=True`, preserving the P4 window/microbatch/accumulation schedule
  (`train_v4.py:196-202`; `v4_training_trials.py:243-249,296-311`).
- The actual loss dispatcher now takes the control branch before P4 pooling or
  auxiliary-label handling. It computes BCE per native window, averages it
  inside each parent bag, applies that bag's weight, and returns
  (`v4_training_trials.py:101-107`). The P4 log-sum-exp pool is reachable only
  after that return (`v4_training_trials.py:109-111`), and known-window
  auxiliary BCE is below it (`v4_training_trials.py:112-124`). Therefore the
  control has no auxiliary-loss gradient.
- Full-file prediction receives `p4=False` for the control and `p4=True` for
  treatment at both fixed-update and epoch evaluation sites
  (`v4_training_trials.py:277-280,345-346`). The model's non-P4 prediction
  path averages native-window probabilities, while P4 uses normalized LSE
  logits (`v4_model.py:252-270`).
- The new regression runs `train_trial` twice over the same multi-window bag,
  preserves one update and four window exposures for each arm, verifies the
  prediction modes `[True, False]`, and requires different resulting parameter
  values (`test_v4_training.py:227-279`). This is a behavioral dispatch test;
  it would fail if control again followed treatment's objective/prediction
  configuration.

## Validation

Ran:

```text
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. \
uv run --with pytest pytest -q test_v4_training.py
```

Result: **12 passed in 5.17s**. This includes the real `train_trial` P4
treatment/control regression and the actual P4/control window-exposure test.

## Findings

### CRITICAL

None.

### HIGH

None. The prior P4-control objective blocker is resolved.

### MEDIUM

None.

### LOW

None.

## Skill-perspective check

Consulted `omo:programming` (including its Python guidance) and
`omo:remove-ai-slops`. The check ran. This delta introduces no untyped escape
hatch, brittle prompt assertion, implementation-constant test, needless
abstraction, or unnecessary boundary parsing/normalization. The new test has
practical decision value: it executes the real dispatcher and update path and
distinguishes the treatment/control configurations. No deletion-only,
tautological, or implementation-mirroring test was found. Neither skill
perspective is violated by this delta.

## Limit

No Colab/A100 execution was performed or inferred; that pending execution is
outside this source-delta approval and is not a code blocker.
