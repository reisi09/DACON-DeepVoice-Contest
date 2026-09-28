# Temporal-resolution and softmax code review

## Scope and evidence inspected

Read-only final review of `music_temporal.py`, `temporal_augmentation.py`, `temporal_experiment.py`, `temporal_resolution.py`, `test_temporal_resolution.py`, and `results/temporal_h100_resolution_20260915/h100_bootstrap.sh`.

I independently ran:

```text
.venv/bin/python -m pytest -q test_music_temporal.py test_temporal_augmentation.py test_temporal_tuning.py test_temporal_resolution.py
```

Result: `16 passed, 2 subtests passed`. `python -m py_compile` for the changed Python files and `git diff --check` also passed.

## Findings

### CRITICAL

None.

### HIGH

None.

### MEDIUM

None.

### LOW

1. The bootstrap depends on `/root/resolution_existing_audio_map.json`, outside the signed input ZIP (`h100_bootstrap.sh:25`). This is not an audio-integrity or selection blocker: it must map exactly the locked IDs and every extracted member is SHA-256 checked against `locked_split.json` before use (`h100_bootstrap.sh:26-36`). Persisting the map's digest in the final result receipt would make later provenance inspection simpler.

## Verified design and execution properties

- `pool_frames` uses disjoint, nonempty frame groups and records receptive-field-centre times. `load_pair` rejects stale IDs, malformed dimensions, nonfinite values, bad bin ordering, and a failed weighted fine-to-coarse identity (`temporal_resolution.py:21-117`).
- The high-resolution batch path repeats the audible mask, then stable-sorts flattened bins by persisted `bin_times`; it rejects a one-bin record for an eight-bin recipe (`music_temporal.py:133-161`). Crop applies the identical window selection to `bin_times`, and feature masking indexes the final 768-feature axis (`temporal_augmentation.py:38-72`). The tests exercise overlapping-tail ordering, crop consistency, immutability, and masking.
- Softmax training uses unreduced cross entropy multiplied by the pre-existing per-file weights, records the optimization loss, and maps fake probability to column 1. The two output rows receive gradient and strict reload is exercised for the softmax paths (`music_temporal.py:168-274`, `test_temporal_resolution.py:72-102`).
- The legacy `mert-residual-64x3-max-v1` tag for `R1_sigmoid` is an intentional loader-compatible representation of the mathematically identical prior C08 architecture. Run-level identity remains in the recipe/arm name and paired-cache receipts; the binned and/or softmax variants have distinct tags. This is acceptable for the stated compatibility requirement and is not a checkpoint correctness blocker.
- A single paired NPZ is also acceptable here: `music_1` and `music_8` originate from the same frozen extraction and retain separate arrays, schema, and representation hashes. Physical file separation was not required for correctness.
- The bootstrap verifies the signed input ZIP, rejects traversal, validates embedded manifest hashes, verifies all 1,446 raw audio members against locked hashes, requires Python/Torch/H100 versions, records the deadline/runtime, and persists result plus feature archives with hashes (`h100_bootstrap.sh:16-37`, `h100_bootstrap.sh:51-74`). It correctly uses the two existing object archives through the locked member mapping rather than relying on an unverified aggregate audio ZIP.

## Interpretation boundary

The four arms vary both temporal resolution and output/loss parameterization. Results should be described as a factorial `resolution × output-loss` comparison unless a marginal analysis is also reported. Probability-based tune and historical BCE should stay distinct from the optimization loss.

## Skill-perspective check

Ran: yes. I applied the already-loaded `omo:programming` Python perspective and `omo:remove-ai-slops` review pass.

- `remove-ai-slops`: the new tests cover concrete ordering, cache-integrity, crop/masking, weighted-loss, gradient, and strict-reload failure modes. They are not deletion-only, tautological, or mere constant assertions.
- `programming`: no prompt tests, untyped escape hatches, needless abstraction, or unnecessary production parsing/normalization was introduced. The paired cache preserves the required shared-extraction boundary.

The diff does not violate either skill perspective.

## Recommendation

- `codeQualityStatus`: **CLEAR**
- `recommendation`: **APPROVE**
- `blockers`: none.
