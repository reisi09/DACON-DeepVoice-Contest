# V5 full-data final-refit validation

Source SHA-256:

- `v5_final_refit.py`: `d0065fc01f7efa6d5fd41f2ff727635fbfbbf7de35590c36ff8a0c3ea4437453`
- `test_v5_final_refit.py`: `8d6a8b9885ebd190963f68417e5b173d79ebbec8af7d5712d065272482e5f9c6`

## Focused scenarios

Invocation:

```text
.venv/bin/pytest -q test_v5_final_refit.py test_v5_training.py test_v5_fusion.py
```

Captured result:

```text
................                                                         [100%]
16 passed in 1.48s
```

Binary observables protected by `test_v5_final_refit.py`:

1. Expert full-pool refit: every eligible ID has exposure count equal to the fixed epoch count; crop-unknown sealed/splice rows are excluded with `unknown_crop_target`; encoder and head gradients and parameter deltas are positive; `final.pt` SHA-256 matches metadata; integer state buffers retain their dtype; every completed epoch emits a machine-readable progress event.
2. Expert deadline and resume: an interrupted job has no `final.pt`, retains `checkpoint.pt`, and its resumed model tensors are bit-identical to uninterrupted training after optimizer and scheduler restoration.
3. Temporal architecture and coverage: B3 uses `TemporalFusion(input_dim=4, joint=False)`, B4 uses `TemporalFusion(input_dim=4, joint=True)`, both expose every cached-feature ID once per completed epoch, and neither records development selection.
4. Temporal deadline and resume: B3 interruption exports no final model; the resumed final state is bit-identical to uninterrupted training.
5. Existing expert/fusion regressions: the original meaningful V5 expert training and temporal loss/architecture checks remain green.

## Static validation

Invocation:

```text
uvx ruff check v5_final_refit.py test_v5_final_refit.py
```

Captured result:

```text
All checks passed!
```

Invocation `git diff --check -- v5_final_refit.py test_v5_final_refit.py` completed with exit code 0.

The production epoch constants are B1 voice/music/presence `4/5/5`, B2 `4/1/1`, and B3/B4 `20`; expert cosine scheduling keeps a 60-epoch horizon. The helper API has no epoch override.
