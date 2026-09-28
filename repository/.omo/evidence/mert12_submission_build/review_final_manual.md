# MERT12 final manual QA

Verdict: **PASS**

Reviewed commit: `e31f1f6679b7fb23d1f490b4fceb10813c22a80b`  
Review base: `83190ff31b89c096f1dfa13742ca546c09b3c5a9`  
Final GPU QA: attempt 6, package-only inference, no training rerun.

The final attempt-6 export is hash-bound: `export_receipt.json` reports exit code 0, package SHA-256 `e7bebbded6fef06517db0d42421a575a0440849248de99856796ffc3e5eb5ee5`, 6,268,222,843 bytes, and evidence ZIP SHA-256 `c9cc1bfd5abe460b8011af7745dde4db64f4f43c25cd00e266020ecb0f2d9166`. I independently verified every exported evidence-member hash and ZIP CRC.

## manualQa

### surfaceEvidence

| scenario id | criterion reference | surface | exact invocation | verdict | artifactRefs |
|---|---|---|---|---|---|
| MERT12-TRAIN-01 | training1322 coverage, seeds, gradients, updates, checkpoint bindings | JSON receipts, serialized checkpoints, CPU readback | `./.venv/bin/python - <<'PY'` reading `full_refit_receipt.json`, `gpu_training_receipt.json`, `training_manifest.json`, `joined_data_audit.json`; SHA-256 of each `head_seed*.pt`; `torch.load(..., weights_only=True)` parameter count | PASS | A1, A2, A3, A4 |
| MERT12-PKG-01 | final ZIP hash, member manifest, decoder binding | ZIP artifact and manifest | `shasum -a 256 submissions/MERT12_full1322_submit.zip`; `./.venv/bin/python - <<'PY'` with `zipfile.ZipFile`, manifest-member SHA-256 checks, and `testzip()` | PASS | A5, A6, A7 |
| MERT12-QA-01 | cached/raw parity and independent-file behavior | CUDA packaged smoke inference plus CSV comparison | `/root/V6_L40S/.venv/bin/python /root/mert12_a100_20260918/final_package/script.py ... --device cuda` for original, permuted, and subset inputs; then `mert12_submission.py compare ...` | PASS | A8, A9, A10 |
| MERT12-QA-02 | supplemental silence, duration/channel, and codec behavior | CUDA packaged supplemental inference | `/root/V6_L40S/.venv/bin/python /root/mert12_a100_20260918/source/mert12_supplemental_qa.py --package-root /root/mert12_a100_20260918/final_package --source-audio /root/mert12_a100_20260918/input/supplemental_real_source.wav --source-sha256 f11541207692ae85606b95019ab828f2c554bdcd334759f7964c81793b63960e --output /root-1/mert12-a100-20260918/output/final_supplemental_qa --deadline-epoch 1789695277.734545 --runtime-python /root/V6_L40S/.venv/bin/python` | PASS | A8, A11, A12 |
| MERT12-QA-03 | genuine executed final QA notebook | Jupyter execution artifact and log | `/opt/conda/bin/jupyter-nbconvert --to notebook --execute /root/mert12_a100_20260918/source/DACON_MERT12_Final_QA.ipynb --ExecutePreprocessor.kernel_name=mert12-final-qa --ExecutePreprocessor.timeout=7200 --output /root-1/mert12-a100-20260918/output/DACON_MERT12_Final_QA_Executed.ipynb` | PASS | A8, A13 |

Training verification found 1,322 unique eligible IDs, class counts REAL 218 / FAKE 1,104, and corrected coverage of all 1,322 IDs for seeds 41, 42, and 43. Each seed used 500 steps at batch size 32 on CUDA; the recorded prior sampler had 1,213 unique IDs and 109 zero-exposure IDs, while the coverage prefix produced zero after-training zero-exposure IDs. All recorded first gradients are positive and all parameter-update L2 values are positive. Each serialized head has 131,329 parameters and its SHA-256 matches the training and package receipts.

The package manifest binds candidate `E6_MERT_layer12_full_data`, seeds `[41, 42, 43]`, 1,322 training rows, File-max fusion, and decoder `torchaudio.load+demucs.audio.convert_audio`. The ZIP contains 45 members; all 44 manifest member hashes match the bytes read from the ZIP and `testzip()` returned no CRC error.

Attempt-6 smoke comparison passed with 4 original files, a reversed 4-file permutation, and a strict 2-file subset. Independent CSV readback reproduced maximum order/subset prediction error `0.0`; the cached/raw music probability error was `3.971516315814938e-08`, below the fixed `1e-5` tolerance. All prediction values were finite and within `[0, 1]`.

Supplemental QA passed 3 cases: `qa_silence_4s` (4 s mono WAV), `qa_real_stereo_60s` (first 60 s of the pinned real source as 16 kHz stereo FLAC), and `qa_codec_mp3_6s` (MP3, tested). CUDA inference completed with finite probabilities in `[0.0, 0.9999964237]`, positive GPU allocation/reservation and RSS receipts, and no quality claim.

The executed notebook is 7,460 bytes, has one code cell with execution count 1 and two outputs, and the log contains `FINAL_QA_COMPLETE` plus the supplemental `status: passed` result. The final evidence archive contains 23 hash-bound files and passed independent manifest/hash verification.

### adversarialCases

| scenario id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
|---|---|---|---|---|---|
| MERT12-ADV-01 | training1322 coverage | unseen eligible IDs / sampler gap | Every eligible ID appears at least once for every seed; no zero-exposure IDs remain | PASS | A1, A2 |
| MERT12-ADV-02 | checkpoint and package binding | stale or mismatched checkpoint/manifest | Checkpoint bytes, receipt hashes, package hashes, seed metadata, and decoder choice agree | PASS | A2, A4, A6, A7 |
| MERT12-ADV-03 | independent-file parity | file-order and strict-subset dependence | Predictions remain identical by ID under permutation and subset execution | PASS | A8, A9 |
| MERT12-ADV-04 | cached/raw parity | decoder-path drift | Corrected packaged raw-audio probability remains within `1e-5` of cached reference | PASS | A8, A10, A14 |
| MERT12-ADV-05 | supplemental runtime QA | silence, long stereo input, and MP3 codec | Runtime completes on CUDA and emits finite probabilities in `[0,1]` with memory receipts | PASS | A8, A11, A12 |
| MERT12-ADV-06 | genuine execution | unexecuted or simulated notebook result | Executed notebook has execution count/output and completion log tied to the attempt-6 export | PASS | A8, A13 |

### artifactRefs

| id | kind | description | path |
|---|---|---|---|
| A1 | receipt | Full-data refit contract, 1,322 IDs, seeds, 500 steps, batch 32 | `/Users/choiwuseck/Documents/ChatGPT/데이콘/results/mert12_a100_20260918/downloaded/unpacked/output/mert12_full/full_refit_receipt.json` |
| A2 | receipt | Per-seed CUDA coverage, first gradients, parameter updates, checkpoint hashes | `/Users/choiwuseck/Documents/ChatGPT/데이콘/results/mert12_a100_20260918/downloaded/unpacked/output/mert12_full/gpu_training_receipt.json` |
| A3 | data-audit | Eligible ID list and REAL/FAKE/source audit | `/Users/choiwuseck/Documents/ChatGPT/데이콘/results/mert12_a100_20260918/downloaded/unpacked/output/mert12_full/joined_data_audit.json` |
| A4 | checkpoint-set | Serialized seed41/42/43 heads, each 131,329 parameters | `/Users/choiwuseck/Documents/ChatGPT/데이콘/results/mert12_a100_20260918/downloaded/unpacked/output/mert12_full/head_seed41.pt` (and `head_seed42.pt`, `head_seed43.pt`) |
| A5 | package | Final submission ZIP, 6,268,222,843 bytes | `/Users/choiwuseck/Documents/ChatGPT/데이콘/submissions/MERT12_full1322_submit.zip` |
| A6 | manifest | ZIP package receipt and local CRC/hash receipt | `/Users/choiwuseck/Documents/ChatGPT/데이콘/submissions/MERT12_full1322_submit.json` |
| A7 | manifest | Decoder-fixed manifest used by the final package | `/Users/choiwuseck/Documents/ChatGPT/데이콘/_workspace/mert12_a100_20260918/MERT12_fixed_manifest.json` |
| A8 | export | Attempt-6 evidence ZIP; 24 ZIP members including 23 hash-bound evidence files and `EXPORT_SHA256.json` | `/Users/choiwuseck/Documents/ChatGPT/데이콘/results/mert12_a100_20260918/downloaded/evidence.zip` |
| A9 | comparison | Final original/permuted/subset CSV comparison and cached/raw error | `/Users/choiwuseck/Documents/ChatGPT/데이콘/results/mert12_a100_20260918/downloaded/evidence/output/final_submission_smoke.json` |
| A10 | runtime-receipts | Final CUDA smoke receipts and CSVs for original/permuted/subset | `/Users/choiwuseck/Documents/ChatGPT/데이콘/results/mert12_a100_20260918/downloaded/evidence/output/final_smoke_original/` (also `final_smoke_permuted/`, `final_smoke_subset/`) |
| A11 | supplemental-report | Silence, 60 s stereo FLAC, and MP3 supplemental QA result | `/Users/choiwuseck/Documents/ChatGPT/데이콘/results/mert12_a100_20260918/downloaded/evidence/output/final_supplemental_qa/supplemental_qa.json` |
| A12 | supplemental-runtime | Supplemental CUDA, memory, decoder, and submission hash receipt | `/Users/choiwuseck/Documents/ChatGPT/데이콘/results/mert12_a100_20260918/downloaded/evidence/output/final_supplemental_qa/runtime_receipt.json` |
| A13 | notebook | Genuinely executed final QA notebook and completion log | `/Users/choiwuseck/Documents/ChatGPT/데이콘/results/mert12_a100_20260918/downloaded/evidence/output/DACON_MERT12_Final_QA_Executed.ipynb` and `/Users/choiwuseck/Documents/ChatGPT/데이콘/results/mert12_a100_20260918/downloaded/evidence/final_qa.log` |
| A14 | diagnosis | Earlier mismatch diagnosis and decoder reproduction evidence; final QA supersedes the earlier failed FFmpeg candidate | `/Users/choiwuseck/Documents/ChatGPT/데이콘/.omo/evidence/mert12_submission_build/parity_diagnosis/diagnosis.json` |
| A15 | export-verification | Independent readback of all 45 package members and 23 evidence files | `/Users/choiwuseck/Documents/ChatGPT/데이콘/results/mert12_a100_20260918/download_verification.json` |
| A16 | resource-state | Attempt-6 workspace stopped after verified export; observed `standby/stopped` | `/Users/choiwuseck/Documents/ChatGPT/데이콘/results/mert12_a100_20260918/resource_stop.json` |

The receipts explicitly retain `independent_validation: false` and `official_l4_validated: false`; this audit therefore passes artifact integrity and requested runtime behavior without claiming independent validation, an official DACON score, or L4 compliance.
