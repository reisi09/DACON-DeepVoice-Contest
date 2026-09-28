# V4 training integration review

Scope was read-only: `train_v4.py`, `v4_training*.py`, `v4_batches.py`, their focused tests, and the V4 plan. No owned training/data source was edited.

## Existing checks

- Invocation: `PYTHONHASHSEED=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest -q test_v4_training.py test_v4_selection.py`
- Observable: `22 passed in 5.03s`.
- Limit: these tests do not exercise terminal trial restart, patience-disabled refit beyond five epochs, P4/continuation exposure equality on the real manifest size, or stale G0 cache rejection.

## Concrete blockers

1. `v4_training_trials.py:247`: `patience=0` still stops after epoch five. After a just-improved fifth epoch, `state.epoch - best_epoch` is `1`, so `1 > 0` is true. P3 and every refit caller use zero to mean disabled. Gate the condition with `spec.patience > 0`.

2. `train_v4.py:151-162`: the current 1,116-row eligible training manifest gives P4 `1,395` fixed updates. Conventional continuation has 35 updates per epoch and its 30-epoch cap permits only `1,050`; the equality guard therefore invalidates P4. Measurement invocation: load `v4_validation/train.csv`, apply `music_supervision_rows`, and evaluate the exact formulas at lines 151 and `EpochBatches` length. Observable: `{'eligible_train': 1116, 'p4_steps': 1395, 'continuation_steps_per_epoch': 35, 'continuation_30_epoch_max': 1050, 'epochs_needed': 40}`. Set continuation epochs to the ceiling needed for `p4_steps`; retain the fixed update cap.

3. `v4_training_refit.py:48-51`: the locked refit uses `epochs_completed`, but `v4_training_trials.py:227-253` evaluates and exposes `best.safetensors`/`best_epoch`. The final refit therefore does not replay the selected recipe whenever the best precedes early stop. Use `best_epoch` for each selected stage, including both parent and P4 stages.

4. `v4_training_trials.py:155-159,191-193,249-255` and `train_v4.py:92-112`: terminal trial results exist only in memory/JSONL. Restarting a checkpoint already at `max_updates` breaks before recomputing the fixed-update evaluation and returns prior complete-epoch state instead; restarting a patience-completed trial can add another epoch. Atomically persist a contract-bound terminal result per trial and return it before the training loop.

5. `v4_training_io.py:48-52`: G0 cache reuse checks only ordered IDs and the presence of `g0_parity.json`. Changed PCM, baseline weights/code, or preprocessing can silently reuse stale predictions. Persist and verify a receipt over ordered file/PCM and baseline/preprocessing/code hashes; recompute on mismatch.

6. Checkpoint contracts at `train_v4.py:95-98` and `v4_training_refit.py:73-76` omit `v4_training.py`, `v4_training_data.py`, and `v4_batches.py`. The sealed weight lock assembled at `v4_training_refit.py:54-55,123` omits inference dependencies including `v4_data.py`, `recovery_sonics.py`, and `ast_multitask.py`. Hash the complete behavior dependency set into resume contracts and the complete inference dependency set into the weight lock.

Model API compatibility, CPU batch lifetime, device transfer, full-parameter optimizer grouping, native starts, P4 labels, and strict trained-state overlays were otherwise consistent.
