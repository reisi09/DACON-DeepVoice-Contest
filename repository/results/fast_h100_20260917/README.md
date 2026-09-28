# H100 frozen-MERT330 structural ablation

Authorized 2026-09-17: one H100 80GB, at most eight wall-clock hours from workspace creation. Workspace and deadline are recorded in `workspace_run.json`. No DACON submission is part of this run.

The 1,686 original recordings are split using ten source-group-aware stratified folds, seed 41: eight folds train, one validation, one test. Existing synthetic mixtures/splices are excluded before splitting. Source groups, PCM hashes and IDs are checked across partitions. Counts are 1,327 / 187 / 172; the ratio of files is approximate because groups remain intact. Add 320 / 40 / 40 mixtures using parents only from the corresponding partition. Unknown voice authenticity stays unknown when a real voice is mixed with music whose vocal authenticity is unknown.

Six new compact M330 heads compare native-HTDemucs backing against raw audio, using seeds 41, 42, 43. Encoders remain frozen. The validation set selects the raw/stem averaging weight from 0, .25, .5, .75, 1. Three direct File MLPs use pooled raw/stem embeddings and fixed DF-Arena/PANNs evidence. Their checkpoints are selected by validation cross-entropy. Five predeclared outputs separate music-branch and File-combination changes. No test score selects a checkpoint, seed or blend weight.

The best historical submitted M330 checkpoints were trained on this source pool and are not used as fresh train-only models. This is a fresh-head comparison on historically reused data, not a new untouched external test. Test contains eight natural real FMA recordings and eleven NSynth recordings, so natural-real uncertainty remains substantial. Synthetic-mixture and natural-file metrics are reported separately. Missing voice labels prevent an official total; retain `score: null` and report eligible component metrics rather than inventing a leaderboard score.

The S2 preprocessing diagnostic holds weights and recordings fixed and compares DF-Arena predictions after the two separator input paths on 24 training music files. Input sample rates/channels are saved; results from 16 kHz mono recordings do not establish the effect on original stereo/high-rate inputs.

Practical checks: group/PCM leakage rejection, source audio and packaged-file hashes, archive CRC, finite encoder outputs, checkpoint reload prediction parity, actual File-head gradients and measured raw-feature ablation. Shared new features are cached once for all heads. Previously trained heads and submitted ZIPs remain preserved.

Monitoring: app heartbeat `h100-15` every 15 minutes plus `_workspace/fast_h100_watch.py`. On completion, download `results.zip`, verify its SHA-256 and every internal artifact, then pause the workspace. The eight-hour deadline pauses compute even if the run is incomplete. Check `download_verification.json` and `resource_stop.json` before claiming completion or stopped billing.
