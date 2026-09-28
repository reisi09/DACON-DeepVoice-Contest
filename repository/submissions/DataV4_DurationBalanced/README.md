# DataV4 duration-balanced submission

Submit **submit.zip** to DACON 236749.

- Archive: 6,350,720,613 bytes (expanded 6,350,711,037); SHA-256 `05ae786864dc282de6ae750a451bc297a870be34f2d1e971b14429eca0e47419`.
- Roots: `model/`, `script.py`, `requirements.txt`; entrypoint `python script.py` (`data/test/` -> `output/submission.csv`).
- Base: Existing9504_MIL archive (SHA-256 `b19b88de…95aa`); frozen assets, FFmpeg, script.py, requirements.txt and unchanged runtime modules copied byte-for-byte.
- Replaced: six heads (data v4 train split, `--duration-balance`, seeds 41/42/43, validation-only selection), their lock/identity/metrics, `presence_head.json`, `file_ensemble.json`, `full_bank_submission.py` (decision layer), new `decision_heads.py`.
- Held-out test (6,450 rows, unseen generators/domains; known-label subset): score 0.877 vs 0.818 for Existing9504. This is not a leaderboard result.

## Verification

`verify_datav4_submission.py` ran the actual extracted `script.py` on the 10 real parity originals under a network audit guard with an empty HF cache (macOS CPU): runtime manifest hash gate passed, ID order and ranges valid, reversed 2-file subset identical (max diff 0.0), decision layer active (presence moved up to 0.918 from legacy PANNs max; FILE not the plain max rule on 10/10). 66.7 s for 10 files on CPU.

FILE_FAKE_PROB is the val-fitted CDF mean of three per-file fusion rules, so it is a rank score, not a calibrated probability; EER is rank-based.

Not validated: L4/CUDA throughput, Linux FFmpeg. No DACON submission was made.

## Reproduce

```sh
.venv/bin/python package_datav4_submission.py
.venv/bin/python verify_datav4_submission.py
```
