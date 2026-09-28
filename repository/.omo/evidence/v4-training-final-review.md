# V4 training final implementation-readiness review

**Scope.** Read-only review of `train_v4.py`, `v4_training.py`,
`v4_training_trials.py`, `v4_training_data.py`, `v4_training_io.py`,
`v4_training_refit.py`, `v4_batches.py`, `test_v4_training.py`, their called
model/evaluation/selection APIs, and
`.omo/plans/trainable-recovery-v4.md`. Production files were not edited.

**Result:** `BLOCK` / `REQUEST_CHANGES`.

```json
{
  "codeQualityStatus": "BLOCK",
  "recommendation": "REQUEST_CHANGES",
  "blockers": [
    "P4 matched control uses the P4 bag-plus-auxiliary loss instead of the required mean parent loss without auxiliary supervision."
  ]
}
```

## Frozen source binding

The final before/after source check matches
`.omo/evidence/v4-training/source-sha256.txt` exactly. The earlier supplied
intermediate hashes (`train_v4.py` `345848...`; `v4_training_trials.py`
`862da4...`) did not match the live source during review; the refreshed final
receipt does.

| File | SHA-256 |
|---|---|
| `train_v4.py` | `5dd66879f99748c3030f1deadafd1a3a45d1d62faca359271874d102f5ade922` |
| `v4_training.py` | `63ce9946c13883d79b2039390458ba5f2890435f992b91153ed22e7b1099fae8` |
| `v4_training_trials.py` | `253b781d78d18d05880aae9f48c798b06d40ff2531dd4f952bb1396228b6e6f6` |
| `v4_training_data.py` | `fcc6280b1eb619403e05564ce1c40736ae36cffb8139627646681aadd7f63edd` |
| `v4_training_io.py` | `04fe392dd350c9d2468ac558d1d87342e18ad9754bd229dac1fbfe9eaddad2c2` |
| `v4_training_refit.py` | `3d8fbed4efa2f046bbf7af672ace7347e65fe9dcf77d898934195c486143d243` |
| `v4_batches.py` | `52155fbeeaebb8e7072d3ec3e31e1021069e5317464e8e80624b37c241c536b1` |
| `test_v4_training.py` | `e412a03e643b518e25d42c5b3319abec5b58e93065ae26c2166234849b84e0c3` |
| `v4_model.py` | `510a36e121e49707c5cd8d42408c9ba4394eda9c547fb94f25f7608d64051a7b` |
| `v4_model_components.py` | `2991a04d2a81faf1b9c209041b32045a53b2a23dc7d360c0a18ed3ef2b422aae` |
| `v4_data.py` | `ff3745f6caf79a1ee2726360f7e38cd51dc296283a8aaba958c8d9ea7a698ccb` |
| `v4_evaluation.py` | `a3a8e307d1024f089748a1777265c1830ae22609c9df81013679499500172144` |
| `v4_selection.py` | `602f46332acdab20453e4f3e9272a21c15f6edcb5487465e39137a879d64e72c` |
| `graph_features.py` | `ee21d931a316164f1cd67bef9055afdd104f7e01141fcc890f8fe28a9ab9b137` |
| `graph_fusion.py` | `342602dc27d3311a86bff00ca6e075fa4f74d5e207859734f8b071a9757c3b9a` |
| `graph_scoring.py` | `7993699ceb926c06c000a7ed5313bfa8fd83b558f3c1e1c1e38f2fcb0f52ba96` |
| `v2_scoring.py` | `1a31cf80dd360cd50bdf7697f491f5a17c40d213ee6f7af16bd31ba871a4afdd` |
| `recovery_sonics.py` | `4882c036f504695df46162b81ca794a897acde7a671c564231eea2066b82e0ba` |
| `ast_multitask.py` | `2204c7a406f2865a9fc84ea9efb3a7ea3bfeaeb1e894eb15ef80671944c69a6b` |
| `v2_data.py` | `f030da8bd3f91bb04581a3ebcee6d76ce15f33862979adcc554802e0eeeaea0d` |

## Findings

### CRITICAL

None.

### HIGH

1. **P4's advertised matched control is trained with the P4 treatment loss,
   including its auxiliary window-label loss.**
   - `train_v4.py:196-202` constructs the continuation with both
     `p4=True` and `p4_control=True`.
   - `v4_training_trials.py:291-305` forwards `p4_control` only to
     `EpochBatches`, then always calls `_weighted_music_loss(...,
     p4=spec.p4, ...)`. It never branches on `spec.p4_control`.
   - `_weighted_music_loss` at `v4_training_trials.py:102-120` uses
     normalized log-sum-exp bag BCE and adds the `0.5 * mean-known-window`
     auxiliary term whenever `p4=True`. `v4_training_data.py:134-142` gives
     the control real/real splice window labels, so that auxiliary term is
     active rather than absent.

   The plan requires a continuation that keeps the parent objective and mean
   pooling, with no P4 auxiliary loss; the supplied implementation claim was
   also “mean loss/no aux.” The present control changes donor content but not
   the loss/pooling treatment. Any P4-versus-continuation difference therefore
   cannot support the intended causal comparison. Correct the loss path and
   add one behavioral regression test that distinguishes treatment from
   control: same prepared window inputs and updates, but control uses mean
   parent loss and contributes no window auxiliary gradient.

### MEDIUM

None.

### LOW

None.

## Verified behavior and limits

- The six previously reported defects are addressed in the final source:
  zero patience disables early stop; terminal trial results are contract and
  artifact hash bound; the G0 receipt includes audio/dependency hashes; the
  source/weight-lock closure includes the inference dependencies; refit uses
  selected best-stage epochs; and the final eligible union is 1,286 IDs
  (1,116 train + 170 dev).
- `OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=.
  uv run --with pytest pytest -q test_v4_training.py` passed **11** tests in
  5.74 seconds. It exercises real SONICS updates, checkpoint resume,
  terminal replay, refit-patience behavior, and actual P4 window exposure.
- The same CPU-thread setting with `test_v4_selection.py` passed **15** tests
  in 1.38 seconds. It exercises recipe immutability, incomplete-refit refusal,
  hash-bound weights, and sealed receipt checks. The existing combined
  evidence reports 33 passing tests, but pass count was not treated as model
  quality evidence.
- The test suite has practical decision value overall. No deletion-only,
  removal-only, tautological, prompt-text, or implementation-constant test
  was found. The P4 exposure test is meaningful, but it omits the distinct
  control-objective requirement above.
- No A100/paid run was performed or inferred. A100 throughput, VRAM, and
  end-to-end runtime remain pending the parent-controlled run; this is not a
  code blocker by itself.

## Skill-perspective check

This review consulted `omo:programming` (including its Python rules) and
`omo:remove-ai-slops`, plus Ponytail. The reviewed final diff does not show a
programming-skill issue that independently blocks this run: no newly exposed
untyped escape hatch, brittle prompt test, or needless validation/parsing was
identified in the scoped change. The slop pass found no useless test or
unnecessary production extraction/normalization. The missing P4 control-loss
branch is a correctness flaw, not an overengineering/style finding.

## Required blockers before approval

1. Implement and verify the P4 control's mean parent objective with no P4
   auxiliary term, while retaining its matched update/window schedule.
2. Reissue the source SHA-256 receipt after that change and rerun the focused
   behavioral tests, including the new P4 treatment-versus-control loss test.
