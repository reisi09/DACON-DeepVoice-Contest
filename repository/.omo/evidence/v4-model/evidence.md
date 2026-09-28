# V4 model adapter evidence

All local real-weight checks used CPU only with `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, PyTorch 2.7.1, and Python 3.11.15. No GPU was allocated.

## Pinned real-weight batch independence and feature contract

- Scenario: load the pinned SONICS and AST originals; run deterministic quiet/loud waveforms together, serially, and in reverse order.
- Invocation: `PYTHONHASHSEED=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest -q test_v4_model.py -k 'real_weight_batch_matches_serial'`
- Binary observable: both models remain within `1e-5`; deliberately restoring SONICS's legacy 3-D batch-wide `top_db` path must differ by more than `1e-5`.
- Captures: `parity-check.txt`, `feature_parity.json`. Measured max batch/serial deltas were `2.384185791015625e-07` (SONICS) and `5.960464477539063e-08` (AST); the injected legacy SONICS path differed by `0.029443860054016113`.

## Full-fine-tuning connectivity and updates

- Scenario: one real FP32 CPU forward/backward/SGD update on pinned weights and a deterministic quiet/loud labeled batch; inspect optimizer identity, every trainable gradient, and representative early/middle/late/head tensors.
- Invocation: `PYTHONHASHSEED=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest -q test_v4_model.py -k 'real_weight_step_updates'`
- Binary observable: exact optimizer/model parameter identity, all parameters trainable, finite nonzero early/middle/late/head gradients, and byte-changing representative tensors after the step. A detached encoder fails the explicit `gradient is not None` assertions.
- Captures: `gradient-check.txt`, `gradient_audit.json`. SONICS exposes all `17,172,493` parameters; AST exposes its `86,187,264`-parameter backbone plus `769`-parameter head. The exhaustive audit found no missing-gradient or zero-gradient trainable tensor in either model.
- The same scenario passes a tensor through SONICS's training-time augmentation seam and verifies unchanged spectrogram and targets, protecting the P1 control from upstream MixUp and SpecAugment.

## P4 temporal objective and native grids

- Scenario: normalized temperature-2 log-sum-exp pooling under complete bag duplication; bag BCE plus `0.5` known-window BCE with unknown-window masking; four explicit distinct stratified starts including a late full window.
- Invocation: `PYTHONHASHSEED=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest -q test_v4_model.py -k 'temporal_pooling or explicit_p4_starts'`
- Binary observable: pooled logits are equal within `1e-6` after whole-bag duplication; unknown segments have exactly the bag-only gradient; known segments receive auxiliary gradient; explicit windows begin and end at the requested original sample positions.
- Capture: `temporal-check.txt`.

## Paired late-fake full-file measurement

- Scenario: compare a 30-second FMA-real file with an otherwise identical file whose final five seconds come from labeled Echoes/udio fake music, using each model's full native window grid and fixed P4 pooling.
- Invocation: the numeric audit executes `predict_files(loaded, [all_real, late_fake], p4=True)` under the same two-thread environment used by the real-weight pytest scenarios.
- Binary observable: the paired prediction changes for both actual model graphs; this is a connectivity measurement, not an AST quality claim because its new head is initialized with seed `236749`.
- Capture: `late_fake_measurement.json`. Absolute probability deltas were `0.09804543852806091` (SONICS) and `0.012966811656951904` (AST).

## Regression and static checks

- Scenario: run V4 plus the adjacent existing SONICS and AST suites, preserving old V2/V3 behavior.
- Invocation: `PYTHONHASHSEED=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest -q test_v4_model.py test_recovery_sonics.py test_ast_multitask.py`
- Binary observable: `16 passed`.
- Capture: `regression-suite.txt`.
- Invocation: cached Ruff check/format, the programming skill's no-excuse checker, and `.venv/bin/python -m py_compile v4_model.py v4_model_components.py test_v4_model.py`.
- Binary observable: all checks pass and all three files compile.
- Capture: `static-checks.txt`.

CPU timing is recorded in `cpu_profile.json`; the A100 throughput and memory profile remains reserved for the parent-controlled run.

## AST sub-Kaldi tail review fix

- Scenario: run full-file inference with the real pinned AST weights on `160,001` samples, producing the native starts `(0, 160000)` and a one-sample final tail below Kaldi's 400-sample frame minimum.
- Red invocation: `PYTHONHASHSEED=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest -q test_v4_model.py -k sub_kaldi_window_tail`
- Red observable: `AssertionError: choose a window size 400 that is [2, 1]` from `torchaudio.compliance.kaldi.fbank`.
- Green invocation: the same focused command after zero-padding only the feature-extractor copy to 400 samples.
- Green observable: `1 passed`; `FilePredictions.starts == ((0, 160000),)` and the real-weight probability is finite.
- Captures: `ast-short-tail-red.txt`, `ast-short-tail-green.txt`.
- Full model suite: the same bounded environment ran `.venv/bin/python -m pytest -q test_v4_model.py` with `7 passed`; capture `pytest-real-weights-after-tail-fix.txt`.
- Adjacent regression: `.venv/bin/python -m pytest -q test_v4_model.py test_recovery_sonics.py test_ast_multitask.py` with `17 passed`; capture `regression-after-tail-fix.txt`.
- Static checks and compile remain green; capture `static-after-tail-fix.txt`.
