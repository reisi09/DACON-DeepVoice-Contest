# MERT12 final context and scope review

- Exact reviewed SHA: `e31f1f6679b7fb23d1f490b4fceb10813c22a80b`
- Base SHA: `83190ff31b89c096f1dfa13742ca546c09b3c5a9`
- Recommendation: **PASS / APPROVE**
- Blockers: none

## Original intent

Use the already selected MERT layer-12 follow-up on the full eligible external music population, refit the small frozen-encoder head on the existing authorized A100, build an offline submission ZIP, verify it, and submit it to DACON. Preserve evidence boundaries: no claim that the full 7,101 derived inputs were trained, no hidden-test learning, no cross-file score correction, no inferred L4 compliance, no official score or generalization claim before official evaluation, and no reuse of the abandoned noisy-OR fusion.

## Desired outcome

A traceable E6 MERT-layer-12, three-seed full-data head refit using every eligible original at least once per seed; a package locked to File-max and the proven preprocessing path; an honest report that separates completed training and diagnosis from still-pending final GPU ZIP QA and official submission.

## User outcome review

The current human report satisfies the requested context and scope boundaries. It accurately reports completed training and CPU decoder/cache parity while explicitly leaving corrected-ZIP GPU QA and official submission pending. This pending authorized execution is not treated as a failure of this context review.

### Criterion C1: selected recipe is E6 MERT layer 12

**PASS.** `results/eight_h100_20260917/FINAL_REPORT.metrics.json` identifies `E6_MERT_l12` as the source-disjoint follow-up priority and expressly limits the claim to `priority_for_source_disjoint_followup_not_adopted_as_general_robustness`. `results/mert12_a100_20260918/downloaded/unpacked/output/mert12_full/full_refit_receipt.json` records `candidate: E6_MERT_layer12`, and `submissions/MERT12_full1322_submit.json` locks `candidate: E6_MERT_layer12_full_data`, seeds 41/42/43, three checkpoint paths, File-max, and the fixed decoder. The report describes MERT-330 layer 12 with a frozen encoder and an MLP head; it does not promote E6 to a general robustness claim.

### Criterion C2: full eligible population and rights mapping

**PASS.** `_workspace/mert12_a100_20260918/upload/joined_data_audit.json` records 1,322 ordered and unique eligible IDs, REAL 218 / FAKE 1,104, with 1,116 prior rich-parent matches and 206 fresh matches. It also records `all_audio_sha256_verified: true`, `all_source_sha256_verified: true`, and `all_license_and_urls_present: true`, with source counts covering all 1,322 rows. The human report correctly calls these 1,322 original music files and explicitly says they are not all 7,101 derived inputs. The training receipt records 1,322 eligible rows; each of seeds 41/42/43 changes `before_unique: 1213` to `after_unique: 1322` and reduces 109 zero-exposure IDs to zero.

### Criterion C3: batch claim boundaries

**PASS.** `results/mert12_a100_20260918/live_metrics/downloaded/batch_benchmark.json` binds the measurements to 8,192 equal draws, batches 32/128/256, five warm-ups, and scope `throwaway frozen-head throughput only; not MERT extraction or quality selection`; it preserves final recipe batch 32. The report reproduces the measured times, throughput, and allocated-memory values, explains that batch 256 changes optimizer-update count for equal samples, and explicitly denies convergence or score-improvement evidence. It does not present allocated PyTorch memory as total process memory.

### Criterion C4: root cause versus abandoned hypotheses

**PASS.** `.omo/evidence/mert12_submission_build/parity_diagnosis/diagnosis.json` records the initial maximum cache/raw probability error `0.00019127128483675815`, same-stem batch-versus-single probability error at most approximately `1.58e-9`, bitwise-equal CPU features for Transformers 4.51.3 versus 4.57.6, and both TF32 flags false after model initialization. It identifies the environment-dependent FFmpeg versus torchaudio/Julius Demucs input path as the isolated cause. `.omo/evidence/mert12_submission_build/parity_diagnosis/fallback_cpu/all_four_receipt.json` reproduces all four hash-bound files and gives maximum fallback/cache probability error `1.346343196928501e-7`, within the unchanged `1e-5` gate. The report presents batch geometry and Transformers-version differences as unsupported explanations, does not revive the incorrect cuDNN-TF32 explanation, and correctly notes that the shared decoder change prevents claiming voice predictions stayed identical to the failed FFmpeg candidate.

### Criterion C5: fusion and unsupported claims

**PASS.** `submissions/MERT12_full1322_submit.json` locks `file_fusion: max`; `mert12_submission.py` implements `max(voice_present * voice, music_present * music)` and does not implement noisy-OR. The report names File-max, keeps voice/presence weights unchanged, and contains no claim of independent validation, generalization improvement, L4 compliance, or a new official score. It explicitly states that no independent validation remains, the observed checks do not establish an official score or generalization, and the A100 smoke run is not an official L4 full-runtime validation.

### Criterion C6: honest current operational state

**PASS.** `submissions/MERT12_full1322_submit.json` records `gpu_end_to_end_verified: false`; the current report separately states that the corrected ZIP's final A100 verification and official submission remain pending. It preserves the corrected ZIP identity as 6,268,222,843 bytes and SHA-256 `e7bebbded6fef06517db0d42421a575a0440849248de99856796ffc3e5eb5ee5`, consistent with the diagnosis resolution and final review context. It does not misrepresent the earlier training notebook or failed recovery QA as final success.

## Direct programming and AI-slop review

The production diff was inspected directly under the `programming` and `remove-ai-slops` criteria. The new validation and parsing are at real trust boundaries: manifests, model checkpoints, ZIP members, audio inputs, CSV identity/order, deadlines, and remote export receipts. No speculative interface, factory, dependency, duplicate normalization layer, or deletion-only production change was found. The tests protect concrete failure modes: all-row exposure under class balance, decoder-path stability on real hash-bound audio, serialized-head parity, file order/subset independence, ZIP/runtime CLI compatibility, and bounded supplemental-QA provenance. They are not tests of a requested deletion, tautological constant checks, or coverage-only assertions. The supplemental-QA orchestration test uses a small fake runtime, but its purpose is the real contract for source provenance, long/stereo fixture construction, receipt production, and duration bounds; it does not claim model quality. No slop finding violates a success criterion.

The existing code review at `.omo/evidence/mert12-a100-full-submission-code-review.md` explicitly records both skill perspectives and the same overfit/slop classes: it states that tests are neither deletion-only, tautological, nor implementation-mirroring, and that production parsing/validation is required at asset/archive boundaries. Its original scope was prelaunch, so it does not replace this direct final pass.

## Checked artifacts

- `/tmp/mert12-final-review-e31f1f6` at exact SHA `e31f1f6679b7fb23d1f490b4fceb10813c22a80b`
- `.omo/evidence/mert12_submission_build/final_review_context.md`
- `results/mert12_a100_20260918/REPORT.md`
- `.omo/plans/mert12-a100-full-submission.md`
- `.omo/drafts/mert12-a100-full-submission.md`
- `results/eight_h100_20260917/FINAL_REPORT.metrics.json`
- `_workspace/mert12_a100_20260918/upload/joined_data_audit.json`
- `results/mert12_a100_20260918/downloaded/unpacked/output/mert12_full/full_refit_receipt.json`
- `results/mert12_a100_20260918/live_metrics/downloaded/batch_benchmark.json`
- `.omo/evidence/mert12_submission_build/parity_diagnosis/diagnosis.json`
- `.omo/evidence/mert12_submission_build/parity_diagnosis/fallback_cpu/all_four_receipt.json`
- `.omo/evidence/mert12_submission_build/parity_diagnosis/transformers_cpu/comparison.json`
- `submissions/MERT12_full1322_submit.json`
- `mert12_full_refit.py`, `mert12_submission.py`, `mert12_batch_parity_probe.py`, `package_mert12_submission.py`
- `test_mert12_submission.py`, `test_mert12_supplemental_qa.py`
- `.omo/evidence/mert12-a100-full-submission-code-review.md`
- Commit diff and `git diff --check` for base-to-reviewed SHA

## Evidence gaps and notes

- Final corrected-ZIP GPU end-to-end QA is pending. This is explicitly disclosed and belongs to the continuing authorized execution, so it is not a blocker for the requested context/scope review.
- Official DACON submission status and score are pending. No contrary success claim appears, so this is not a blocker.
- The H100 cache did not preserve a literal process-level FFmpeg/ffprobe PATH receipt. Numerical reproduction isolates the effective decoder behavior across four hash-bound files; the report states the actual evidentiary boundary and does not overclaim process-level provenance.
- The earlier code-review report is prelaunch rather than a post-diagnosis final review. This direct review covers the later decoder diagnosis, corrected package metadata, tests, and current report, so there is no missing-coverage blocker.

## Final recommendation

**PASS / APPROVE.** No reviewed claim violates the requested success criteria or exceeds its supporting evidence. Pending final GPU QA and official submission must remain described as pending until their own receipts exist.
