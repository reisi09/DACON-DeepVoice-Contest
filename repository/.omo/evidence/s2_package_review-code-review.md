# S2 EAT package code review

## Scope and evidence inspected

- New package sources: `package_s2_eat.py`, `s2_eat_submission.py`.
- Sealed training source: `results/fullft8_h100x1_20260916/downloaded/batch_worker.py` (the model/preprocessing section only; excluded the unrelated large embedded block from line 679 onward).
- Sealed trial records: `07_S2_EAT_fullft/{identity,selection,evaluation,execution_config,complete}.json` and `best.pt` hash.
- Baseline behavior and packaged files: `baseline_original/script.py`, `submissions/S2_EAT/submit.zip`, and `submissions/S2_EAT/package_verification.json`.

The `omo ulw-loop status --json` lookup returned `ULW_LOOP_PLAN_MISSING`; therefore this report uses the required fallback evidence path.

## Findings

### CRITICAL

None.

### HIGH

None.

### MEDIUM

None.

### LOW

None.

## Verified correctness points

- Checkpoint identity is correctly guarded. `package_s2_eat.py:20-22,49-54,62-64` pins the actual `best.pt` SHA-256 `81bf…c6b6`, worker SHA-256 `9c07…d727`, trial identity `0b27…7356`, and stored epoch `3` (zero-based, thus selected epoch four). These agree with the sealed `complete.json` and `best.pt` hash.
- The package uses the trial's original frozen HTDemucs path. The runtime's mono 16 kHz to native stereo HTDemucs normalization/separation/resample sequence at `s2_eat_submission.py:42-56` matches worker `batch_worker.py:237-248`; it does not load the experimental adapted separator.
- Evaluation-window selection at `s2_eat_submission.py:27-39` matches `batch_worker.py:249-268`: 80,000 samples, fixed stride candidates plus final candidate, evenly selected eight windows, and repeat padding. The training-only activity gate is intentionally absent from evaluation.
- EAT feature extraction and classifier architecture are preserved: the runtime source is extracted from the sealed worker, and its EAT fbank parameters, 512-frame pad/crop, normalization, CLS exclusion, 8-window bag forward, and `CompactHead` match worker `batch_worker.py:308-410`. The archive runtime file byte-matches the new `s2_eat_submission.py`; the head is loaded `strict=True` at `s2_eat_submission.py:91-93`.
- The five submission probabilities follow the baseline column order and formula at `s2_eat_submission.py:84-102`, with only `MUSIC_FAKE_PROB` replaced by the S2 EAT model. This matches `baseline_original/script.py:361-394`.
- Offline HF loading is explicit at `s2_eat_submission.py:15-16`, the entry script, and `fullft_model.py:57-61`. Required EAT remote-code modules and `model.safetensors` are present in the ZIP.
- The initial dependency concern does not stand as a defect: although `requirements.txt` only pins the torch stack and timm, `COMPETITION_CONTRACT.md:54` records the evaluator base image's compatible `transformers`, `demucs`, `librosa`, and `panns-inference` versions. The supplied requirements do include the non-base EAT dependency `timm==1.0.15`.
- Archive integrity evidence is concrete: `submissions/S2_EAT/package_verification.json` identifies `submissions/S2_EAT/submit.zip`, SHA-256 `19c7a66a4a82d8cb117c7a7ae6c71ed370016c1528fa2e96801105c5a5b0897c`, 6,239,402,263 bytes, and a successful ZIP CRC check. The rebuild changed ZIP metadata only; it retains the recorded sealed-artifact choice and source checks.
- Bounded offline CPU execution evidence is available in `submissions/S2_EAT/runtime_verification.json` and `offline_cpu.log`: two real inputs (one WAV and one FLAC) completed, preserved sample order, and produced all five valid output probabilities with network connections blocked and an empty HF cache. The verifier also records 314 exactly exported tensors and crop-parity lengths of 64,000, 80,000, 176,003, and 960,000 samples.

## Skill-perspective check

Ran: `omo:programming` (including its Python reference) and `omo:remove-ai-slops`, plus the required Ponytail review perspective.

- `remove-ai-slops`: no deletion-only, tautological, implementation-mirroring, or prose/prompt tests were added. No unnecessary production parsing/normalization was introduced beyond the trained EAT and baseline inference contracts.
- `programming`: no brittle prompt tests, untyped escape hatches, needless abstractions, or extra validation layers were introduced by the package diff. The dynamic-module import is necessary for the local `trust_remote_code` EAT snapshot.
- Ponytail: the package reuses the sealed worker head/extraction code and original baseline assets instead of creating an alternate inference stack. No avoidable abstraction or dependency addition was found.

## Recommendation

`codeQualityStatus: CLEAR`  
`recommendation: APPROVE`  
`blockers: []`

The package remains limited by the independently recorded absence of GPU end-to-end verification and L4 1,200-file time-limit verification; this is an evidence limit, not a code-review defect or a claim of GPU/L4 runtime parity.
