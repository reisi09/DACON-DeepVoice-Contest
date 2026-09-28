# Round 14 — extra validation (AT-ADD Track 1?) and an error audit before the last 3 submissions (2026-09-28 15:15 KST)

Files: code_r14/ (current code of every step; train_backbone_heads.newrows.diff = today's trainer change), new_data_summary.csv,
new_eval_newtest.json (E1/E2 results of round 13), e2_subsets.txt, LICENSES.md. No AI-Hub data here.

## What was built today (all measured)
1. Rows: atadd_rows.py (AT-ADD T2 subset, 34,000 rows; train pool = T2 train+dev, newtest = T2 eval pool), open_rows.py (HAIM B3/B4/B6,
   MUSDB18-HQ 3 crops/song, AI-OpenBMAT 60 s, GTSinger-ko, SD-RR, D1 remixes), assign_splits.py (lineage splits), openbmat_extra.py (+400).
2. Features: dump_features.py runs the packaged S7 Engine up to the heads. Parity (parity.py) vs the training caches on 24 bank rows (H100) and
   10 parity files (Mac MPS): cos >= 0.99999 for every key, voice_silent 24/24; presence vs runtime predictions max |d| 3e-8 (H100), 1.8e-7 (MPS).
   Incidents: 3 workers per H100 -> CUDA OOM -> rows claimed by lock files failed silently (1,772 + 2,590 stale locks, cleared and redone);
   now 2 workers/GPU, OOM releases the lock and retries; 8 rows of 40,222 failed (GTSinger, CUDA/NCCL error) and are dropped by new_rows().
   Presence for AT-ADD newtest rows dumped before the presence key existed was back-filled (10,000/10,000).
3. Trainer: per-row npz (npz_path), add_new_rows (NEW_TRAIN / NEW_VAL), P1 final() adds every new row as training (newval -> val with
   NEW_VAL=1) and puts new npz into the EAT/SSLAM/WavLM indexes; weighting.py SPEECH gained atadd_speech and gtsinger_ko_speech.
4. DB protocol results: round 13 table (s7_new E1 +0.0064, E2 +0.0285; W_new E1 +0.0033, E2 +0.0451; W_ref worse everywhere).
5. P1 finals trained on x1 H100 (fp32, TF32 off): s7_new (val official FILE 0.02736 / VOICE 0.0219 / MUSIC 0.05131), W_new.
6. Packages (Mac): package_datav4_submission.py; verify_datav4_submission.py (offline CPU); w_runtime_parity.py (training path vs runtime
   on 10 parity files; s7_new worst V 0.00018 / M 0.00089 PASS); VESSL CUDA check pending (verify_s8_vessl.py). 19 members differ from S7
   (model/trained/*, manifest.json, runtime/full_bank_submission.py = the S8-era runtime that keeps S7 behaviour without S8 files).

## Questions (Korean, short, VERIFIED / INFERRED / UNKNOWN, with file:line for code issues)
(a) AT-ADD Track 1 (HF xieyuankun/AT-ADD-Track1, gated auto, speech: train/dev/progress/eval + labels/meta) or any other held-out set:
    can it give a MORE PRECISE proxy of the DACON test than E1/E2 tonight? Which subset, how many rows, what would it decide?
    Constraints: needs the user's HF token again; extraction ~18 audio-s/s per H100; decision deadline ~23:00; slots at 00:00-10:00.
(b) Audit for errors that could make a package wrong or an evaluation misleading. Look especially at:
    - labels in atadd_rows.row / open_rows.mk (VOICE/MUSIC/FILE and masks; HAIM B6 VOICE_FAKE=1; MUSDB vocal presence; remix labels),
    - leakage between train and newtest (AT-ADD speakers/singers shared across its own splits; openbmat groups use only the first music
      source track; remix donor filter), and whether newval rows can reach training,
    - trainer integration (npz_path for OLD / cache lookups, window-count checks, every_row val rows, label completion ordering, weighting of
      the new sources), P1 final using new rows but val unchanged,
    - evaluation (new_eval: tables from old val only, FILE fixed 0/9/1, presence from npz for newtest, bootstrap on groups),
    - packaging (pkg_chain.sh: heads fetched from the VESSL volume, receipt/sha flow, runtime parity reference, what is NOT verified).
    Give the concrete fix and whether it blocks submitting s7_new / W_new tomorrow.
