# MERT330 MLP submission: 1,206 approved music files

- ZIP: `M330_MLP_full1206_submit.zip` (6,267,568,297 bytes).
- SHA-256: `1f6cfd6c9feacead01d4ebcfd217d10c1796969b1f863ba4b0283d46e7432c7e`.
- Frozen MERT-v1-330M, revision `5240c2708a5acaee1007f43fb9735c7dcd0b78c9`.
- Original compact residual 64-channel/3-convolution trunk, max pooling, MLP 64→32→2, GELU and softmax.
- Dropout0.1, AdamW lr1e-4, weight decay0.01. No label smoothing or LR scheduler in this submission.
- Five source-group-disjoint folds. Each file trains four models and validates one. All 1,206 approved music files, including 90 NSynth files, receive positive loss weight when in training. NSynth mass0.05.
- Early stopping: unsmoothed balanced validation cross entropy, patience20, min_delta1e-4, max300 safety limit. Restore minimum validation loss. Selected epochs36/43/40/42/28.
- Inference averages five music probabilities, then applies baseline file fusion. Frozen audio models run once per file; only the small music heads are ensembled.

## Verification boundaries

Training verification checks source-group separation, positive-weight coverage, early stopping decisions and checkpoint reload parity. `verify_submission.py` reopens the actual ZIP and runs offline CPU inference on two files, including a real four-second music sample. Independent A100 execution uses ZIP runtime/head files and manifest-verified local pretrained assets.

A100 packaged music probability is0.5231001973; cached-feature reference is0.5231000066 (difference1.91e-7). The pooled MERT feature maximum absolute difference is4.77e-7. This confirms the packaged music path against the training environment for this sample.

CPU music probability is0.5300406575, a0.00694 difference from the historical A100 reference. It repeats exactly on CPU and is reproduced by applying the five heads to the captured CPU feature. Device and torch/librosa/numpy versions differ; the exact source of cross-environment drift has not been isolated. No claim of device-independent parity is made.

This is a bounded inference check, not a full official L4 execution or an official score. Historical validation labels do not cover the complete official voice metric. The all-data ensemble has no independent final holdout score. No competition upload has been performed.

Subsequent classifier comparisons retain their original locked train/tune/historical-test split and NSynth mass0 for matched comparisons. They do not change this ZIP.
