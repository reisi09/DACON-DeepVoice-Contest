# C08 MERT-330M comparison

This run changes the frozen pretrained music encoder from `m-a-p/MERT-v1-95M` (768 features) to `m-a-p/MERT-v1-330M` (1024 features). The classifier retains the C08 convolutional trunk: LayerNorm, projection to 64 channels, three Conv1D/GELU layers with residual connection, max pooling, followed by Linear(64,32), GELU, Dropout(0.1), Linear(32,2), and two-class softmax at inference. Training uses class-balanced cross-entropy on raw logits; fake is class 1. One mean feature vector per original audio window is used; the eight-bin extraction cache is retained only for reuse.

## Fixed protocol

- Same 1,446 source-audio hashes; 1,008 train, 248 tune, 190 historical test files.
- Frozen Demucs and MERT. Only the classifier is trained.
- Classifier dropout is now 0.1 after each of the three residual GELUs and after the MLP GELU; AdamW weight decay is 0.01 on classifier parameters. The final scalar linear head is replaced by the MLP; other training settings remain unchanged. Both backbones use seeds 41, 42, 43 (six new head fits).
- MERT95 reuses the prior verified feature cache but trains new matched regularized heads. Selected groups are `M95_mlp` and `M330_mlp`. The copied unregularized M95 checkpoints are historical inputs only; they are neither overwritten nor used as matched controls.
- Tune EER mean, seed standard deviation, then balanced BCE determine the comparison winner, rounded to 12 decimals before ordering. Historical test is reported only after selection is frozen.
- Voice and presence predictions stay fixed to the baseline. Local labels do not permit an official total-score claim.
- Two-hour compute ceiling. Download and verify results before pausing H100; retain feature caches on object storage.

## Reproduction

From the repository root, `results/mert330_h100_20260916/build_input.py` stages the source code, sealed prior controls, split, metadata and notebook. The bootstrap verifies the upload, source audio, prior paired-cache archive, Demucs assets and pinned MERT330 weights before running the notebook on H100. The runner repeats input manifest and per-cache checks before extraction.

Input notebook: `C08_MERT330_H100.ipynb`.
Launcher: `h100_bootstrap.sh`.
The original `submissions/C08/C08_submit.zip` is preserved (SHA256 `0c3c328947c2f6d83a07763ac482a945004f2bad0c3df6d6b16bb160b18f9ead`).

## Validation before dispatch

The wider-input test learns a label carried only by the extra 256 channels, verifies their gradients and prediction influence, and checks checkpoint reload parity. Missing or changed fixed controls are rejected before extraction. Existing C08 cached historical predictions remain within 1e-6 after the dimension extension. New input-dimension, experiment, resolution and C08 packaging checks passed together.

## Execution status

The input ZIP and bootstrap were uploaded successfully. Jupyter and SSH connections repeatedly timed out. One workspace restart did not restore access. No training command was dispatched; there are no new MERT330 checkpoints, extracted features, performance scores, or H100 inference measurements.

The workspace was paused to stop billing, and the recurring monitor is PAUSED. The final observed resource state, credits and original-submission hash are recorded in `connection_failure.json`. Resume from the stored input and bootstrap after server access is restored; do not interpret this prepared notebook as an executed experiment.

## Current MLP revision

The updated input is `C08_MERT330_MLP_Input.zip`, staged separately in `_workspace/mert330_mlp_input` and uploaded under `temporal/mert330-mlp-20260916/`. The updated bootstrap targets this revision. The earlier connection-failure receipt describes the unregularized attempt; no new GPU start is part of this configuration change. Local integration checks verify that weight decay changes actual training updates and dropout varies training predictions while reloaded evaluation remains deterministic.

The MLP check trains on class information present only in the added 256 feature channels, verifies gradients through the hidden layer and both class outputs, checks fake-class probability and silent-input handling, and reloads the checkpoint with matching predictions. Original sigmoid and direct-softmax checkpoints retain their architecture tags and behavior.
