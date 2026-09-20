# Existing9504 MIL submission

Submit **submit.zip** to DACON 236749. The earlier Handoff ZIP is not the submission archive.

- Archive: 6,347,673,463 bytes; expanded 6,347,664,181 bytes.
- SHA-256: `b19b88de1b23ccb8e2863d30d9a755e991419ff36b39ac629d73ff26f65895aa`.
- Entrypoint: `python script.py`; archive roots are `model/`, `script.py`, `requirements.txt`.
- Exact selected H100 checkpoints: music and voice seeds 41, 42, 43; hashes match the validation-only candidate lock.
- Dataset: 9,504 rows; train 6,652, validation 1,426, test 1,426.
- Frozen HTDemucs base, MERT330M layer12 WindowMIL, DF-Arena fc5, PANNs; file score remains max of presence-weighted component scores.
- Runtime modules come from the training handoff. Only the inference candidate name changed; no learned parameters, fusion weights, normalization or inference batch logic were tuned during packaging.
- Linux x86-64 FFmpeg is bundled from the training handoff; requirements add torchvision 0.22.1+cu128 to the competition's preinstalled environment.

## Verification

Archive member SHA-256 and CRC verified. Actual `script.py` from the extracted ZIP ran on 7 original audio files under a Python socket/network audit guard and empty Hugging Face cache, compared with the locked H100 predictions. Maximum absolute difference: 0.0007349252700805664; predefined tolerance 0.002, relative tolerance 0.

A reversed 2-file subset reproduced the same predictions exactly (maximum difference 0.0). Three historical gain-preserving samples were checked separately through the packaged Engine's `preserve_gain=True` path: maximum difference 0.0003068248430887488. The public CLI always applies the standard peak cap for new inputs; it does not look up sample IDs or use evaluation data.

QA ran locally on macOS CPU, torch/torchaudio 2.7.1. This is not Linux/CUDA/L4 or full evaluation-set throughput verification. Linux FFmpeg execution was not validated locally. The installed local numpy/soundfile versions differ from the competition defaults; see submit.json. No GPU was restarted for this packaging work.

Local validation file EER is 7.9944%; test file EER 7.4334%. The test has 1,001 historically exposed rows and is not independent; these are not official leaderboard results. No DACON submission was made.

## Reproduce

```sh
.venv/bin/python package_existing9504_submission.py --handoff results/existing9504_h100_20260920/completed/DACON_Existing9504_H100_Handoff.zip --output submissions/Existing9504_MIL/submit.zip
.venv/bin/python verify_existing9504_submission.py
```

The verifier executes the actual ZIP, checks ID order, finite probability ranges and raw-audio parity, and removes its unpacked 6 GB runtime copy after success. Existing submission archives are unchanged.
