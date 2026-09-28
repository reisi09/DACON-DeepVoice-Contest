# Controlled localization experiment — 2026-09-18

Status: local sampler comparison complete; additional licensed public data and GPU input bundle in preparation. GPU has not started.

## Authorization and resource limits

- User authorized local stages 1–2, followed by stages 3–4 on one existing A100; maximum compute wall time four hours.
- Reuse `wsp-5i3knd3gxeox`, A100 SXM 80GB ×1. Keep B workspace `wsp-9683432r72m7` stopped.
- Upload and read back all inputs before starting compute. The run file fixes the absolute deadline at launch; recovery cannot reset it.
- Reserve the last 600 seconds for export and verified download. Monitor every 300 seconds; stop after verification or at the original deadline.

## Local result and hypotheses

The sampler fix changes coverage from 789/831 to 831/831 without changing the class/source/generator exposure budget. Three-seed probability ensemble dev EER changes from 0.1591366906 to 0.2007194245; historical test EER remains 0.09155518395. This repairs a real defect but does not demonstrate generalization improvement. The old test has been inspected repeatedly and is diagnostic, not independent confirmation.

The next comparison tests whether localized supervision and preserving temporal evidence help compared with averaging MERT features. AST is the audio spectrogram form of ViT. Earlier AST/EAT/temporal experiments already existed: this is a controlled comparison with repaired sampling, fresh confirmation parents, known intervals, and matched exposure, not the first use of those concepts.

## Data and split freeze

- Preserve original music base split: train831/dev164/calibration200/historical test127.
- Add 128 train-only matched partial/sham examples from original train parents. Fake events and real-real editing controls use the same 10ms crossfade, gain and splice operations.
- Freeze new confirmation music parents: 32 previously unused FMA artists plus 32 unused SONICS IDs (16 Suno/16 Udio). Evaluate natural 4/8/16/30/60-second views and 16 paired partial/sham controls. These are new parents, not unseen generators.
- Fresh GuitarSet isolated guitar and LibriSpeech clean speech provide documented presence negatives. Unknown voice labels in music remain unknown.
- User additionally requested training data from Hugging Face, GitHub and Kaggle where rules permit. Final selected sources, counts, source revisions, licenses, hashes and source-disjointness audit must be recorded before input upload. No dataset is admitted merely because its repository code is open source.
- Additional selection, before model scores: JamendoMaxCaps 320 licensed tracks/135 artists; reserve all66 tracks from27 deterministically selected artists for `confirmation_external`, leaving254 real training tracks. HAIM128 ACE-Step +128 MusicGen generated tracks stay in training because some original-reference bindings are unresolved. All64 Kaggle SunoCaps prompt groups are held out in `confirmation_external`. Expected addition:510 train and130 external confirmation rows; actual acquisition/audit receipts must confirm this before launch.
- ACE-Step, MusicGen and Suno families already appear in the old Echoes data. These additions expand datasets, tracks, prompts and documented versions, not necessarily generator families. Real/fake source confounding remains and unresolved cross-catalogue aliases/reference bindings limit independence claims.
- Official rules permit publicly accessible resources with at least noncommercial use allowed, require source attribution and forbid updating models from private evaluation data. No external-data publication-date cutoff appears on the inspected rules page.
- Split by source parents/artists/speakers; transformed versions inherit all parent groups. The 64 confirmation parents are excluded from additional training.

## Matched model recipes

1. Mean: raw MERT layer12 window feature mean, train-only fitted mean/std, existing 1024→128→1 GELU MLP; file BCE.
2. MIL: same frozen MERT features and scaler, per-window MLP, temperature-1 log-mean-exp pooling; file BCE plus 0.25 known-window localization BCE.
3. AST: original AudioSet checkpoint, frozen prefix, trainable last encoder block/final norm/temporal head; frequency-pooled temporal patches, log-mean-exp file aggregation, file BCE plus 0.25 known-patch localization BCE.

The localized arms change both aggregation/architecture and objective; gains cannot be attributed solely to backbone choice.

- Common seeds 41/42/43 and corrected source→generator→parent sampler.
- Add an original831-only mean control using the same effective batch128×125 and original-only scaler. This separates the effect of expanded data from the three model recipes. Its three checkpoints are also locked before any held-out prediction, giving12 total fitted candidates.
- Effective file batch128 ×125 optimizer updates =16,000 draws per seed, matching the earlier batch32×500 sample exposure. Optimizer schedules are new; this is not numerical equivalence to 500 smaller updates.
- Benchmark actual batches32/64/128. Prefer the largest within 90% of best measured throughput and ≤80% VRAM. Use accumulation when needed; record optimizer updates and sample exposure separately.
- MERT and AST use identical separated backing audio, 64,600-sample windows at16kHz, 32,300-sample hop and deduplicated tail. Decoder fixed to torchaudio+Demucs conversion.
- AST processor stays at1024 frames and the native1214 tokens. Prefix uses bfloat16 autocast and float16 disk caching; this is lossy. Padding masks affect pooling/loss, not native AST attention.
- 25ms fbank frames receive fractional interval occupancy. Unknown truth is excluded. Labels are source/constructed-interval evidence, not sample-exact annotations of separated stems.

## Selection and reporting

- Select each seed checkpoint using dev Music EER, breaking ties with dev BCE, every25 updates. Fit feature scaling only on training rows.
- Lock all nine selected checkpoints and the dev-selected architecture before scoring historical test and fresh confirmation.
- Report every arm, three-seed ensemble, source/generator/duration/partial/sham slices and known-label coverage; do not tune on confirmation results.
- File predictions retain the fixed voice/presence branch and max fusion. Official total stays unavailable when labels are incomplete; local diagnostics are not DACON submission scores.
- Preserve exact manifests, source/model hashes, batch benchmarks, exposure schedules, predictions, selected checkpoints, runtime receipts and GPU telemetry. Export archive member hashes must match downloaded content.

Sources: [DACON rules](https://dacon.io/competitions/official/236749/overview/rules), [AST model](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593), [local sampler evidence](../local_sampler_20260918/ANALYSIS.md).
