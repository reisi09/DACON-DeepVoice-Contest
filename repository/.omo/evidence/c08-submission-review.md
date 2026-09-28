# C08 submission package code review

## Scope and evidence inspected

Read-only review of `c08_submission.py`, `package_c08_submission.py`, and `test_c08_submission.py`, plus their archive/runtime imports and the newly built `submissions/C08/C08_submit.zip`.

Independently verified:

```text
.venv/bin/python -m pytest -q test_c08_submission.py
```

Result: `1 passed`. Python compilation and `git diff --check` for the three reviewed files also passed. The built archive's CRC check and receipt identify C08 seed 41, selected epoch 12, 88,257 parameters, and checkpoint SHA-256 `a32d8b51337e35a3395e83bdcc0d367d28429e7159f015a307db4369559c64b3`.

## Findings

### CRITICAL

None.

### HIGH

None. The earlier dependency-closure concern is resolved for this target environment: the official evaluator documents `scikit-learn`, `transformers`, `librosa`, `soundfile`, `demucs`, PANNs, NumPy, and SciPy as preset packages and explicitly recommends omitting pinned preset libraries from submission requirements. The three ABI-coupled Torch packages are pinned to the documented CUDA-12.8 versions. The builder now also pins selected epoch 12 and the exact C08 checkpoint SHA-256 (`package_c08_submission.py:40-45`) and verifies all six frozen MERT asset hashes against the sealed training manifest (`package_c08_submission.py:46-55`).

### MEDIUM

1. **Keep the cached and end-to-end parity claims separate.** `test_c08_submission.py:23-34` correctly establishes all-five-probability parity within `1e-6` for the 190 H100 cached records through the deployed C08 head and file fusion. The freshly unpacked CPU replay independently exercised extraction and output writing, but its historical-audio receipt reports a `VOICE_FAKE_PROB` difference of `4.61936e-05` from H100. This is an expected cross-device numerical observation, not evidence that the cached C08 path is wrong; it does mean that “all five predictions reproduce H100 within `1e-6`” must remain scoped to cached features, not claimed for fresh CPU extraction. No L4 end-to-end result is available.

### LOW

None.

## Confirmed properties

- Packaging checks the H100 checkpoint digest, C08 selection, seed 41, architecture, parameter count, and records selected epoch 12 in the receipt (`package_c08_submission.py:33-52`).
- The entrypoint validates every packaged member hash before inference and refuses a non-C08 compact-sigmoid checkpoint (`c08_submission.py:39-59`).
- The music path preserves the original 64,600-sample windows, `soxr_hq` resampling, MERT mean pooling, audible mask, scalar sigmoid C08 head, baseline voice/presence, and file-level maximum fusion (`c08_submission.py:48-81`).
- The archive is stored rather than recompressed, has a manifest, and passed the package's CRC/size checks (`package_c08_submission.py:60-96`). A freshly unpacked offline CPU run wrote a valid CSV for organizer dummy audio in 22.46 seconds, and a separate real historical-audio run completed in 17.49 seconds.

## Skill-perspective check

Ran: yes. I applied `omo:programming` with its Python reference, `omo:remove-ai-slops`, and the required Ponytail review perspective.

- `remove-ai-slops`: the cached-190-file test protects a real probability/fusion regression; it is neither a deletion-only test nor a constant pin. No needless production extraction, parsing, or normalization was added for the stated inference goal.
- `programming`: no prompt tests, untyped escape hatch, or unnecessary abstraction was introduced in the reviewed files. The target evaluator's documented preset-library contract satisfies the imports deliberately omitted from requirements.
- Ponytail: copying the existing official baseline assets and using one C08 head is appropriately minimal. Relying on the documented preset packages avoids unnecessary reinstalls.

## Recommendation

- `codeQualityStatus`: **WATCH**
- `recommendation`: **APPROVE**
- `blockers`: none.
