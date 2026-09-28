# Recovery V3 execution plan

User outcome: explain AST degradation from recorded experiments, implement a targeted improvement, measure it in Colab, and preserve results in Drive. No new submission in this comparison run. Maximum A100 allocation 7,200 seconds; notebook watchdog at 7,080 seconds. Tier HEAVY.

## Evidence and hypotheses

- Official G0 score 0.6909110688 versus AST633 0.6453405503 and AST1029 0.6174819841. Latest ZIP identity matches full1029 refit.
- AST failed prior EER and companion-AUC gates. Promoting it despite those gates was a confirmed selection error.
- All 1,029 prepared files are 4–10 seconds due to upstream cropping. AST inference covers full input duration; long-file mean aggregation and short-tail padding are unvalidated design risks, not confirmed hidden-test causes.
- More epochs improved development much more than reused evaluation; source/generator slices disagree. Further identical refits have weak evidence.
- SONICS provides a music-authenticity-specific frozen representation. Its addition is an experiment, not a promised improvement.

## Locked candidate contract

Pinned SONICS source: 9156ffad151f797c71556923c4a02fa01fa8fc91.
Pinned SONICS HF model: a349d427b681fa15e3d0e22d1708108322f967b1.
Weight SHA-256: 11c384cea651c56028d6f02e04aa467c2099cf1102687353fe4dd9504c4db868. MIT. Target 1 is fake.

Five candidates in fixed tie-break order:
1. G0: unchanged official baseline; **maximum** DF-Arena music-window probability.
2. SONICS_RAW: mean SONICS raw-audio window probability replaces music authenticity.
3. SONICS_STEM: mean SONICS Demucs music-stem window probability replaces music authenticity.
4. BLEND_RAW: 0.5 G0 music + 0.5 SONICS_RAW music.
5. BLEND_STEM: 0.5 G0 music + 0.5 SONICS_STEM music.

Keep VOICE_FAKE, VOICE_PRESENT and MUSIC_PRESENT exactly G0 for every candidate. For changed candidates FILE_FAKE=max(voice_presence*voice_fake, music_presence*new_music_fake). G0 remains exactly unchanged. No ranking or statistics across private input files. No training, model updates or calibration in inference.

SONICS uses deterministic 80,000-sample windows at 16 kHz, anchored at zero and with a final full window ending at the last sample when input exceeds five seconds. Only short files are zero-padded. All retained samples are covered. Serial windows are acceptable and avoid batch-dependent decibel normalization. No pooling/candidate sweeps beyond these five.

## Selection and data

Old dev_select122 ranks candidates by explicitly partial known-subset weighted diagnostic, gain>=0.005. Candidate EER must not worsen by>0.02 and companion AUC by>0.01. Connected-group bootstrap: 2,000 samples, win_rate>=0.70 and p10>=-0.005. Old generator176/source98 provide safety vetoes under the same per-head regression tolerances; they are reused diagnostics, not fresh validation.

Select on dev first; old274 may veto the selected candidate but cannot promote a runner-up. Lock choice, formula, gates, code/model hashes and all manifest hashes before fresh labels or predictions are examined. Fresh challenge accepts/rejects only the frozen candidate against G0. On rejection retain G0; never use fresh results to choose another candidate. Preserve unknown labels and report metric coverage. No partial diagnostic is an official score.

Fresh input worker targets20 new MLAAD real/fake pairs and12–24 new FMA/Echoes artists, each real+3 fake providers. Exclude all prior atomic groups, content identities and decoded PCM overlaps. Preserve up to60seconds, record original and retained durations and licenses. These files are disjoint from our prior corpus, not guaranteed disjoint from upstream model pretraining.

## Ownership and sequence

- Runtime worker: recovery_sonics.py, recovery_fusion.py, corresponding tests. Verify pinned offline load, serial reference parity, tail sample coverage, invalid inputs and exact protected-head ownership.
- Data worker: prepare_recovery_data.py, its tests, recovery_validation/ metadata and ignored audio; acquire sources, verify lineage/PCM and provenance, create archive. No predictions.
- Experiment worker: run_recovery.py, recovery_selection.py and selection tests. Reuse original baseline operations, one separation per file, per-stage timing, atomic per-file cache, complete old396 coverage, immutable lock and fresh accept/reject.
- Parent: build_recovery_notebook.py, actual Aside Colab execution, Drive transfer, independent review, results report and local commits. Bundle a candidate for offline replay only if it passes quality gates; otherwise preserve the existing baseline archive and record failure evidence.

## Practical verification and stop condition

1. Run `.venv/bin/python -m pytest -q test_recovery_sonics.py test_recovery_fusion.py test_recovery_selection.py test_prepare_recovery_data.py`: protect late-window evidence, local model hashes, per-file independence, source separation, and refusal to promote a gate-failed candidate. Capture failing-first proof where implementation changes behavior.
2. Use actual pinned SONICS on real WAV; compare with upstream forward in FP32, maxdelta<=1e-5. On A100 compare G0 extraction with untouched official functions on two real files and log full-duration60sec coverage plus order invariance.
3. Build and upload DACON_236749_recovery_v3.ipynb; in Aside open the uploaded Colab, choose A100 and run all. Verify complete predictions, metrics, lock-before-fresh, stage timings, saved results and released GPU. Inputs/outputs under MyDrive/DACON_236749/runs_recovery and existing project root.
4. If candidate passes, fresh offline package inference must reproduce reference probabilities within1e-5, exact IDs and protected heads. Server L4 throughput remains unverified unless actually measured; A100 projections are estimates only.
5. Independent code/evidence reviewer checks real acceptance criteria; fix concrete blockers. Report accepted improvement or measured rejection honestly, preserve all evidence, commit locally and finish. No extra user approval is needed for already-authorized implementation, experiment and Drive saving.
