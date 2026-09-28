# PANNs R3 submission error: NumPy 1.26 compatibility

The submitted archive failed with `AttributeError: module 'numpy._core' has no attribute 'multiarray'`. The user-provided DACON screenshot identifies the submission at 2026-09-26 20:37:55. The same exception was reproduced from the original packaged `Engine` at `panns_submission.py:101` with real NumPy 1.26.4 and PyTorch 2.7.1 on CPU.

## Fix

Use `np.core.multiarray.scalar` in the restricted checkpoint-loader allowlist. Retain `weights_only=True`, the explicit pickle alias, all model weights, decision readouts, and selection metadata. No retraining or inference algorithm change.

Official environment: https://dacon.io/competitions/official/236749/overview/evaluation lists NumPy 1.26.4, PyTorch 2.7.1+cu128, Python 3.11.15 and L4. The previous A100 QA instead used NumPy 2.2.6 and PyTorch 2.9.1+cu128, so its success did not cover this compatibility failure.

## Observed verification

- Original archive SHA-256: `5717f10be02e18ee5ef1fffcc4435883940c93b28bd9df512ff125ab39a9103a`.
- NumPy 1.26.4 wheel came from official PyPI, verified against published SHA-256 `edd8b5fe47dab091176d21bb6de568acdd906d1887a4584a15a9a96a1dca06ef`. It was extracted into a task-local directory; the existing environment was not changed.
- Real execution used the existing macOS network-denying sandbox, offline Hugging Face settings, four CPU threads and a 600-second process limit. No GPU or new paid compute.
- Before: original packaged Engine raised the screenshot's exact AttributeError.
- After: all real encoders and trained heads initialized with NumPy 1.26.4 / PyTorch 2.7.1.
- Actual `script.py` then processed three existing hash-verified audio samples, using default input/output paths and only `--device cpu`, and wrote the required five probability columns in 46.939812541 seconds.
- Compare with the original runtime on the same CPU under NumPy 2.4.6: maximum absolute probability difference `1.1102230246251565e-16` (preselected tolerance `1e-5`). All probabilities finite and within [0, 1]. See `verification.json`.
- New ZIP: `PANNs_R3_submit_20260926_fix1.zip`, 6,577,761,151 bytes, SHA-256 `6efcb0186b36a08f4e47653097f8972bc310da08761fa245f4c6edfcc123468b`.
- All 58 ZIP members passed SHA-256 and CRC reads, manifest coverage and size bounds. Only `model/runtime/panns_submission.py` and `model/manifest.json` changed; the other 56 members match the original byte-for-byte. Every new ZIP member also matches the extracted package that was executed.
- The existing training packager correctly rejected the local reference lock because its hash differs from the cloud training reference. No lock or selected-weight identity was rewritten. The repair streamed the verified original archive and replaced exactly the two members above.

## Limits and completion

The confirmed submission crash is fixed. This is macOS CPU validation of the relevant NumPy/PyTorch API versions, not an exact Linux CUDA environment reproduction. The A100-to-CPU outputs differed by up to 0.001394, so CPU parity is reported separately from GPU results. L4 peak memory, 1,200-file runtime, long-input coverage and competition acceptance remain unverified. The package was not resubmitted. LSP was unavailable; the user had previously declined installing it. Python source parsing, real execution and `git diff --check` were used.

Plan complete: identify/reproduce the reported failure; correct its runtime producer; execute the corrected entrypoint; preserve weights and verify the replacement archive. Submission ZIPs and raw audio are excluded from Git.
