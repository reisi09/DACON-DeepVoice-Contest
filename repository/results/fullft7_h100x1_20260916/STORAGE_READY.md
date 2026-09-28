# H100 × 1 / 7 experiments — staged inputs only

Object volume: dacon-v6-training (objvol-a5gxdg7er70i)
Mount this volume at /shared. Asset prefix: fullft7/h100x1-20260916/.
Copy MERT_EAT_FullFT_H100x1_7Experiments.ipynb to a writable persistent working directory (for example /root in a persistent workspace) and open it there.
Use Python 3.11 and one H100 80GB. Keep outputs/checkpoints on persistent storage with sufficient free space.

The notebook defaults to NO installation, extraction, checks, preprocessing, or training.
When you choose to run: install the pinned dependencies if needed and restart the kernel; set RUN_STAGE_INPUTS=True to extract original audio; then RUN_PREPARE=True and RUN_TRAIN=True for sequential trials 01–07.

Existing source archives are reused without uploading duplicate audio:
- v6/V6_H100_RunBundle.zip
- c3/full-final-20260915/C3_Final_Input.zip

existing_audio_map.json maps the 1,356 required IDs to exact archive members. The notebook validates extracted SHA-256 hashes against its embedded locked split.
model_manifest.json records the pinned MERT/EAT revisions and hashes of all local model weights and code. Model downloads are disabled at runtime; dependency installation still needs network access.

H100 × 1 planning rate: 2.98 credits/hour. Session limits: 48 credits and 16 hours. These are not measured runtime guarantees or account balances. New HTDemucs preprocessing takes additional time. Stop/Pause the VESSL allocation manually when done; stopping Python alone does not stop billing.

This single-GPU revision has NOT been executed, tested, or trained, per user instruction. No GPU was started while staging these files.
