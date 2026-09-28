# Final code review — MERT12 submission build

## Verdict

- Reviewed commit: `e31f1f6679b7fb23d1f490b4fceb10813c22a80b`
- Base: `83190ff31b89c096f1dfa13742ca546c09b3c5a9`
- Static code verdict: **PASS**
- `codeQualityStatus`: **CLEAR**
- `recommendation`: **APPROVE**
- `blockers`: **none**

This is a static/package review. It does not assert that the separately-running final A100 QA completed or that the package is ready for external submission; those claims require the final QA receipt and exported-artifact hashes.

## Scope and evidence inspected

I inspected the commit diff from the stated base through the reviewed SHA, the executable callers, package manifest/receipt, and the following evidence artifacts:

- `.omo/evidence/mert12_submission_build/decoder_fix_pytest.xml` — six passing focused tests, including the hash-bound real-WAV decoder regression.
- `.omo/evidence/mert12_submission_build/parity_diagnosis/diagnosis.json` and `parity_diagnosis/fallback_cpu/all_four_receipt.json` — the pinned `torchaudio.load + demucs.audio.convert_audio` path reproduces four cached music probabilities with maximum absolute error `1.346343196928501e-07`, below the fixed `1e-5` gate.
- `submissions/MERT12_full1322_submit.json` — package SHA `e7bebbded6fef06517db0d42421a575a0440849248de99856796ffc3e5eb5ee5`, size `6268222843`, member SHA list, and CRC result.
- `results/mert12_a100_20260918/local_verification.json` and the A100 preparation receipts — only as historical/package-preparation evidence, not as replacement for the final QA.

## Correctness review

- The decoder change is applied before `separate_voice_and_music` in `mert12_submission.run` (`mert12_submission.py:149-187`). The patched symbol is the baseline module's `load_track`, which is the path used by baseline separation; this corrects the shared decoder seam rather than adding a per-format or per-caller workaround.
- `load_track_torchaudio` uses the diagnosed training-feature path (`torchaudio.load`, then `demucs.audio.convert_audio`) at `mert12_submission.py:109-120`. The package manifest binds this decoder declaration and `verify_package` rejects a package whose declaration differs (`mert12_submission.py:123-134`).
- The frozen MERT feature contract remains consistent: exact 16 kHz windows, SOXR-HQ resampling to 24 kHz, hidden state 12, per-file valid-window mean, then mean of the three serialized head probabilities (`mert12_submission.py:137-146`, `87-99`). File fusion remains `max` (`102-106`) and package validation checks that choice.
- Full-data refit protects the required numerical/data contracts: 1322 eligible original core-music files, manifest/audit binding, fixed seed/step/batch recipe, TF32 disabled, finite features, per-class coverage, nonzero gradient/update evidence, and checkpoint hashes (`mert12_full_refit.py:41-56`, `138-163`, `205-248`; `package_mert12_submission.py:41-80`).
- Packaging checks source/member hashes, duplicate/missing names, CRC, archive size, and expanded size before promotion (`package_mert12_submission.py:83-139`). The final-QA bootstrap checks the exact package, receipt, runtime source, and three checkpoint hashes before executing raw-audio smoke scenarios (`_workspace/mert12_a100_20260918/final_qa_bootstrap.sh:79-145`; `DACON_MERT12_Final_QA.ipynb`, code cell 1).
- The final-QA notebook deliberately reuses heads and does not retrain or repack. It runs original/permuted/subset raw-audio inference, enforces order/subset `<=1e-7` and cache/raw music `<=1e-5`, then runs codec/channel/duration supplemental QA. This matches the unresolved risk after the decoder diagnosis.

## Findings

### CRITICAL

None.

### HIGH

None.

### MEDIUM

None.

### LOW

None.

## Skill-perspective check

This check ran after loading `omo:programming`, `omo:remove-ai-slops`, and `ponytail:ponytail`.

- `remove-ai-slops`: no deletion-only test, prompt-text test, tautological test, implementation-constant-only test, or unneeded production parsing/normalization was found in the reviewed change. The decoder regression uses a real hash-bound WAV and disables `PATH`, which tests the actual failure mode. Coverage validates the prior unseen-row defect. Serialized-head and smoke comparison tests protect numerical/file-independence contracts.
- `programming`: no new untyped escape hatch that affects the package contract, no brittle prose-prompt assertion, and no validation added away from the relevant file/package boundary was found. The changed scripts are operationally broad but their checks bind concrete workspace, archive, hash, deadline, and device contracts; no extra abstraction was introduced for the decoder fix.
- `ponytail`: the selected repair is the smallest shared-path change: one pinned baseline loader, one manifest field, and one real decoder regression. It does not duplicate models or add a format-specific fallback layer.

## Test and verification assessment

The focused pytest receipt contains six passes. Those tests are decision-relevant: decoder-path parity, coverage/balance, exact tail handling, serialized inference parity, CSV order/subset/cache comparison, and notebook-to-parser CLI compatibility. The CPU four-file parity receipt independently supports the decoder choice. I did not rerun GPU or VESSL work, create tests, edit source, create a package, or submit anything.

The package receipt still correctly reports `gpu_end_to_end_verified: false` because it predates the final QA. Before any readiness/submission claim, review the final A100 QA's `final_qa_receipt.json`, `final_submission_smoke.json`, supplemental receipt, package/evidence ZIP hash receipts, and stopped/standby operational receipt.
