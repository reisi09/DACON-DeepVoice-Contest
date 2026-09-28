# MERT12 final goal gate review

## recommendation

**APPROVE / PASS**

- Reviewed commit: `e31f1f6679b7fb23d1f490b4fceb10813c22a80b`
- Base commit: `83190ff31b89c096f1dfa13742ca546c09b3c5a9`
- Review checkout: `/tmp/mert12-final-review-e31f1f6`
- Final package: `submissions/MERT12_full1322_submit.zip`
- Package bytes: `6268222843`
- Package SHA-256: `e7bebbded6fef06517db0d42421a575a0440849248de99856796ffc3e5eb5ee5`
- Official submission: DACON submission `92294`, submitted `2026-09-18 10:30:25 KST`, visible status `채점 중`

## blockers

None. Every stated success criterion has artifact-backed completion evidence.

## originalIntent

Use the already selected E6 MERT layer-12 candidate, refit its frozen-encoder music heads on every eligible external-data original on the existing authorized A100, investigate the user's request to use the 80 GB GPU with larger batches without turning throughput measurements into a quality claim, build and verify an offline submission ZIP, stop compute after verified downloads within the fixed deadline, and make the authorized official DACON submission. Preserve the evidence boundaries: no hidden-test training, no cross-file correction, no independent-validation claim after merging all local splits, and no official score or L4-runtime claim before official evaluation.

## desiredOutcome

A reproducible three-seed full-data refit in which all 1,322 eligible originals participate in optimization for every seed; a hash-bound ZIP using File-max and the diagnosed training-compatible decoder; real packaged CUDA inference proving cached/raw parity and file-order/subset independence; supplemental silence, long-stereo, and MP3 execution; verified downloads followed by stopped/standby A100 state; and an actual DACON submission receipt.

## userOutcomeReview

The shipped outcome satisfies the requested goal.

- Training coverage is complete: the A100 receipt records seeds `41`, `42`, and `43`, each with 500 updates at batch 32, `after_unique: 1322`, no zero-exposure IDs, nonzero gradients for every trainable tensor, and nonzero parameter updates. The final package binds the three resulting checkpoint hashes.
- The larger-batch request was evaluated on the 80 GB A100 with equal 8,192-draw benchmarks at batches 32, 128, and 256. Batch 256 increased measured frozen-head throughput, but it would reduce optimizer updates for the same draw count and supplied no convergence evidence; retaining batch 32 for the selected recipe is accurately disclosed and does not violate the request.
- Decoder drift was corrected at the shared Demucs input seam. The corrected package pins `torchaudio.load + demucs.audio.convert_audio`. Four-file CPU parity had maximum error `1.346343196928501e-7`; the final packaged CUDA QA improved the cache/raw maximum error to `3.971516315814938e-8`, below the unchanged `1e-5` gate.
- Original four-file, permuted four-file, and strict two-file subset runs produced `max_probability_error: 0.0`, establishing the named file-independence criterion. Supplemental CUDA QA passed for silence, a 60-second stereo FLAC, and MP3, with finite probabilities and explicit GPU/RSS receipts. A genuine executed QA notebook is preserved.
- The downloaded submission ZIP was independently verified as the exact expected `6268222843`-byte artifact with SHA-256 `e7bebbded6fef06517db0d42421a575a0440849248de99856796ffc3e5eb5ee5`; 45 submission members and 23 evidence members were verified. The workspace was then observed as `standby/stopped` with reason `archive_verified_exported`.
- The original absolute deadline `1789695277.734545` was preserved. Attempt 6 reconstructed the exact ZIP from already uploaded material, reran QA without retraining, completed QA at `1789694685.452305`, verified downloads at `1789694878.1788192`, and verified stop at `1789694931.738518`, all before that deadline.
- DACON accepted the exact package as submission `92294`. The receipt binds the package SHA, one confirmation click, completion dialog, timestamp, and current `채점 중` status. The official score, ADS/CPS values, and official L4 result remain pending; no success claim is made for them.

## successCriteria

| Criterion | Result | Evidence |
|---|---|---|
| C1 — authorized A100 full eligible-data refit | PASS | `gpu_training_receipt.json`; `joined_data_audit.json`; three checkpoint hashes |
| C2 — every eligible original represented in optimization | PASS | 1,322 unique rows per seed; zero missing exposure IDs |
| C3 — larger-batch/80 GB request evaluated honestly | PASS | `batch_benchmark.json`; batches 32/128/256 on `NVIDIA A100-SXM4-80GB`; final recipe 32 with stated boundary |
| C4 — reproducible offline package and fixed preprocessing | PASS | package/member manifest, fixed decoder, three head hashes, local and downloaded exact ZIP SHA |
| C5 — real packaged CUDA parity and adversarial QA | PASS | `final_qa_receipt.json`, `final_submission_smoke.json`, `final_supplemental_qa/`, executed notebook |
| C6 — original deadline preserved | PASS | `workspace_run.json`, attempt-6 dispatch, QA/download/stop timestamps all before `1789695277.734545` |
| C7 — verified downloads before stopped/standby | PASS | `download_verification.json` followed by `resource_stop.json` (`standby/stopped`) |
| C8 — actual official submission | PASS | `official_submission.json`, submission `92294`, exact package SHA, `채점 중` |
| C9 — claim boundaries preserved | PASS | `REPORT.md` and receipts disclose no independent validation, score, generalization, or L4-runtime conclusion |

## direct programming and remove-ai-slops review

I directly reviewed the base-to-head diff, production scripts, focused tests, and final receipts under the `programming`, `remove-ai-slops`, and Ponytail criteria. The tests protect concrete defects and decisions: missing-row sampler exposure, exact tail handling, serialized-head parity, the real hash-bound decoder path, file-order/subset independence, notebook/parser compatibility, and bounded supplemental-QA provenance. I found no deletion-only test, requested-removal test, prompt/prose pin, tautological expected value, shape-only confidence test, implementation-mirroring test, or unnecessary production extraction/parsing/normalization that violates a success criterion. The operational and packaging validation remains at real archive, model, audio, CSV, deadline, and remote-receipt trust boundaries.

The largest changed production files are in the programming warning band (`mert12_full_refit.py` 243 pure LOC; `mert12_submission.py` 239; `mert12_a100_launch.py` 227) but remain under the 250-pure-LOC defect threshold. This is a maintenance note, not a stated-criterion blocker. The final code review explicitly records the same skill-perspective and overfit/slop classes in `.omo/evidence/mert12_submission_build/review_final_code.md`; the earlier prelaunch review also covers them in `.omo/evidence/mert12-a100-full-submission-code-review.md`.

## checkedArtifactPaths

- `/tmp/mert12-final-review-e31f1f6` at exact commit `e31f1f6679b7fb23d1f490b4fceb10813c22a80b`
- `.omo/evidence/mert12_submission_build/final_review_context.md`
- `.omo/plans/mert12-a100-full-submission.md`
- `results/mert12_a100_20260918/REPORT.md`
- `.omo/evidence/mert12_submission_build/review_final_code.md`
- `.omo/evidence/mert12_submission_build/review_final_context.md`
- `.omo/evidence/mert12_submission_build/review_final_manual.md`
- `.omo/evidence/mert12_submission_build/review_final_security.md`
- `.omo/evidence/mert12-a100-full-submission-code-review.md`
- `_workspace/mert12_a100_20260918/upload/joined_data_audit.json`
- `results/mert12_a100_20260918/live_metrics/downloaded/gpu_training_receipt.json`
- `results/mert12_a100_20260918/live_metrics/downloaded/batch_benchmark.json`
- `.omo/evidence/mert12_submission_build/parity_diagnosis/diagnosis.json`
- `.omo/evidence/mert12_submission_build/parity_diagnosis/fallback_cpu/all_four_receipt.json`
- `submissions/MERT12_full1322_submit.json`
- `submissions/MERT12_full1322_submit.zip`
- `results/mert12_a100_20260918/downloaded/evidence/output/final_qa_receipt.json`
- `results/mert12_a100_20260918/downloaded/evidence/output/final_submission_smoke.json`
- `results/mert12_a100_20260918/downloaded/evidence/output/final_supplemental_qa/supplemental_qa.json`
- `results/mert12_a100_20260918/downloaded/evidence/output/DACON_MERT12_Final_QA_Executed.ipynb`
- `results/mert12_a100_20260918/download_verification.json`
- `results/mert12_a100_20260918/resource_stop.json`
- `results/mert12_a100_20260918/official_submission.json`
- Base-to-head diff, `git diff --check`, Python compilation, and focused test receipts; direct rerun in the complete main checkout: `7 passed`

## exactEvidenceGaps

- Official evaluation is still running. Submission `92294` has no score, ADS, or CPS yet, so the artifact does not establish an official improvement over submission `90735` (`0.7396465503`). This is not a blocker because the success criterion required actual submission or an access blocker, not a completed official score.
- The A100 QA is not an official L4 full-private-test runtime measurement. No L4 compliance claim is approved.
- All local splits were merged for the authorized refit, so there is no remaining independent validation population. Package integrity and parity do not establish generalization quality.

## finalDecision

**PASS / APPROVE.** No blocker can be tied to a failed stated success criterion. The requested full-data refit, batch investigation, offline package verification, fixed-deadline operation, post-download shutdown, and actual official submission are all supported by exact artifacts.
