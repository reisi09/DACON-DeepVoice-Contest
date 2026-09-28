# C08 cosine classifier with label smoothing and plateau scheduling

Status: the A100 run completed on 2026-09-16 KST. See
results/cosine_a100_20260916/README.md for verified results and limits. The new
MERT330 combination scored worse than the previous MLP on both tune music EER
and historical music/file EER. No new submission artifact has been produced.

## Model

Frozen HTDemucs backing audio -> frozen MERT-330M -> ordered 1024-dimensional
window features -> LayerNorm -> Conv1d(1024, 64, 1) -> GELU -> one residual block
with three LayerNorm / Conv1d(64, 64, 3) / GELU / Dropout(0.1) stages -> masked
temporal max pooling -> L2 normalization -> cosine similarity to two learned
64-dimensional class representatives -> fixed scale 10 -> two logits.

The previous 64 -> 32 -> 2 MLP is removed. The convolutional dropout remains.
Inference uses softmax, with fake at class index 1. All-invalid sequences keep
the existing near-zero fake probability behavior.

## Training

- Recipe: `temporal_backbone.COSINE_C08`; use `replace(COSINE_C08, input_dim=1024)`
  for MERT-330M. The existing comparison runner also supports MERT-95M at 768.
- AdamW: initial learning rate 1e-4, weight decay 0.01, batch size 64.
- Training: source/class-balanced cross entropy with label smoothing 0.05.
  Binary targets are [0.975, 0.025] or [0.025, 0.975].
- Validation: original hard labels, class-balanced cross entropy, no smoothing,
  no augmentation, dropout disabled.
- ReduceLROnPlateau: mode=min, factor=0.5, threshold=1e-4, threshold_mode=abs,
  minimum learning rate 1e-6, no warmup. PyTorch `patience=4` deliberately means
  reduction at the end of the fifth consecutive non-improving epoch; the next
  epoch uses the reduced rate. This counter resets on a qualifying improvement
  or a reduction.
- Early stopping remains patience 20 with min_delta 1e-4, maximum 300 epochs.
  The checkpoint with the lowest unsmoothed validation loss is restored.
- Existing crop 0.7 and feature-mask probability 0.3 remain unchanged.
- Frozen MERT/Demucs receive no optimization. Reuse verified feature caches.

The recipe, actual per-epoch learning rates, scheduler settings, losses and
checkpoint hash are saved in `training.json`. `progress.json` distinguishes the
current epoch's learning rate from the next epoch's rate. Checkpoints store a
distinct cosine architecture tag and the scale buffer; older checkpoints retain
their loading behavior. These are inference checkpoints, not resumable optimizer
snapshots.

## Verification

```sh
OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q test_temporal_cosine.py test_temporal_input_dim.py test_music_temporal.py test_temporal_resolution.py test_temporal_augmentation.py test_temporal_experiment.py test_c08_submission.py
```

The new checks verify cosine values against independent calculations, feature
scale invariance, padding isolation, gradients through both representatives and
all temporal convolutions, training-only smoothing, 1024-dimensional save/load
parity, actual scheduler reductions/floor and their effect on optimization.
Existing checks include original C08 inference parity on real cached audio.
Synthetic training checks establish integration and numerical behavior; they do
not establish a new competition score or GPU performance.
