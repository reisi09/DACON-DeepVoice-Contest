# Skeptic critique of the backbones study (2026-09-27, written about 15:50 KST)

Verdict: **DOWNGRADE.** Starting the MuQ (+EAT) extraction still stands. It is the only way to get evidence, and the code
is sound. The ship rule (decision step 4) is not: the check below shows it admits a candidate that adds no new encoder.
The extraction manifest is also too small for the final package.

## What I checked and could not refute
- **Held-out selection.** None found. `choice_lock.json` (15:21:34 KST) predates every new-encoder feature. The only
  held-out numbers computed so far are reproductions of R2's already-reported 0.88643. `ads.py` writes its val lock
  before phase 2.
- **Plumbing.** The diffs of `run.py`/`ads.py` against R2 contain only the new groups, row pinning and the new-cache row
  filter. The reproduction (max |diff| 0.0, same best epochs) is genuine.
- **Runtime preprocessing parity.** The extractor uses the same load, peak cap 0.6, HTDemucs, `window_layout`/`windows`
  and per-window `librosa.resample(..., 'soxr_hq')` as `full_bank_submission.Engine.predict` (lines 89-106). So reusing
  MERT's 24 kHz windows for MuQ at runtime gives identical input. MuQ gets the raw resampled audio; MERT additionally
  goes through its processor's normalisation. That difference is correct for each model.
- **EAT variable-length input.** Fixed position encodings are sliced to 200 tokens (25 time x 8 freq, time-major). This
  matches the model's own `extract_features` (rel 2e-4; vendored vs original timm code 0.0).
- **Cost comparison.** The CPU timing of MERT runs all 24 layers (`output_hidden_states=True`), and so does the runtime.
  The 2.17x ratio is therefore like for like.
- **Leakage and "seen" data.** I found none that affects labels. MSD (MuQ pretraining) is not a bank source. Real music
  in the bank comes from OpenMIC/FMA, Jamendo, MUSAN and constructed mixes. FakeMusicCaps rows are fakes only; their
  MusicCaps/AudioSet real counterparts, which EAT saw, are not bank rows.
- **Baseline.** S4-like (R2 all-chosen) is the correct reference.

## F1 (changes decision step 4): the 0.002 val gate admits a no-new-encoder "null"
Question: does the study's unchanged `ads.py` grid (replace / fuse w 0.5, 1, 2 / voice add w .25, .5, 1) with
GATE = 0.002 accept candidates that bring no new encoder? My candidates were existing R2 heads on the same 31,270 pinned
rows, 3 seeds, DB protocol: MERT-only music heads L3_6_9_12, L3_6_9, concat8, sls8, and extra voice heads
voice_df resmlp_long and voice_ad mlp_long.

Setup: the lock was written before the run (`critique_check/choice_lock.json`, 15:45:35 KST). I ran a copy of the
study's `ads.py` in `critique_check/`; only HERE differs, and NEWCACHE is `_workspace/backbone_cache`, so the row filter
is a no-op. It took 33 s CPU and 0.53 GB RSS. Output is in `critique_check/null.out` and `ads_report__*.json`.

Official ADS, DB protocol, rows val 3,680 / test 4,822:

| config | val ADS | vs baseline |
|---|---|---|
| baseline (S4-like) | 0.93845 | - |
| music fuse MERT L3_6_9 w1 (and w2) | 0.94112 | **+0.00267 (passes the gate)** |
| music fuse MERT concat8 w1 | 0.94102 | +0.00257 (passes) |
| music fuse MERT L3_6_9_12 w0.5 / concat8 w0.5 | 0.94029 | +0.00184 |
| music replace (any MERT variant) | 0.92872-0.93786 | negative |
| voice add voice_ad mlp_long w1 (best voice null) | 0.93970 | +0.00125 (fails) |

The study's own rule therefore picks "music fuse music__L3_6_9 w1". Its held-out result, reported once as a transfer
diagnostic, was: **ADS 0.88663 vs 0.88643, delta +0.0002, 95% group CI [-0.0031, +0.0040], p(delta <= 0) = 0.45.**
Music EER improved (.1847 to .1805), but FILE EER got worse (.0834 to .0855).

Interpretation: a second, correlated MERT head fused through the val CDF gains about +0.0027 val ADS and nothing on
held-out. Part of this is FILE-ensemble refit on val and part is the max over 12 grid cells. Any "MuQ fuse" win of
0.002-0.003 val ADS would be indistinguishable from this null. The rule as written would ship it: the package cost is
0.62 GB and a few L4 minutes, and the evidence would amount to zero.

Required correction, pre-registered before the MuQ features exist:
1. Add the MERT-only null candidates (`music__L3_6_9`, `music__concat8`, `music__L3_6_9_12` from R2) to the same
   `ads.py` call.
2. The MuQ music choice must beat max(baseline, best null config) by at least 0.002, which today means val ADS
   >= 0.94312. MuQ-voice must beat max(baseline, best voice null 0.93970) + 0.002, i.e. >= 0.94170.
3. Ship only if the single held-out pass gives a group-bootstrap p(delta <= 0) <= 0.05 against the baseline. Also report
   the delta against the val-best null. This is a go/no-go on one pre-registered choice, not a selection.

## F2 (changes decision step 1): the manifest covers 31,270 of the 42,039 rows needed to ship
`_workspace/backbone_cache` now has all 42,039 rows. The orchestrator's 16:00 KST note in `research-state.md` says S4
was trained on 100% of rows under the P1 protocol. `manifest_backbone_rows.csv` equals `rows_r2.txt` (31,270 IDs;
verified) and misses 10,769 rows: 1,251 val, 1,628 test and 7,890 train.

A shipped MuQ head would then be trained on 74% of rows while the rest of S4 uses 100%. Its val-CDF fusion table would
also lack a quarter of the val rows. Closing that gap later would need a second H100 pass of about 50 min.

Fix: append the 10,769 rows after the current 31,270, so the study still completes first. Put the new val/test rows
first, then music-labelled train rows, then the rest. Launch with NGPU=2, or on one H100 set MINUTES to about 210. At
S3's 3.47 files/s, 42,039 rows take about 200 H100-min.

## F3: runtime risk is overstated; the fallback item is stale
- **L4 runtime.** S3's L4 runtime was measured at 24 min (user, `research-state.md` 16:00 update); it is not
  "estimated 22-35". Calibration: S3 minus P1 (17.5 min) = 6.5 min for AntiDeepfake-1B, which is 2.04 MERT units by
  FLOPs, so one MERT unit is about 3.2 L4-min. MuQ on both stems plus EAT is 2.76 units, about +9 min, so S4+new
  encoders comes to about 31-33 min. The noisy CPU-QA ratio (+26% for AD) gives about 31 min too. Even a 2x error
  (about 42 min) fits 60 min. This is an estimate; the GPU/CPU cost ratio is unverified.
- **Fallback.** S4 removed the time fallback. Decision step 6 says to "extend the 48-minute fallback", which should be
  dropped. A time-based skip makes a file's output depend on its queue position (claim L1-F15, per-file independence).
- **What to log instead.** Have the extraction log per-stage H100 seconds (it already does) and use them to check the
  MuQ/MERT ratio on GPU.

## F4: operational blockers the decision does not mention
- **VESSL credit.** The 16:00 note gives $5.57, about 1.87 H100-hours at $2.98/h. The planned run is 2.5 H100-h
  (31,270 rows), or about 3.3 H100-h with F2. Money is no constraint per the user, but the credit must be topped up
  before launch, or the workspace stops mid-run. The manifest order limits the damage; it does not remove it.
- **Download size.** New features take 40,960 B per window (MuQ 2 x 7 x 1024 + EAT 6 x 1024, fp16). The existing cache
  takes 18,944 B per window and is 3.3 GB, so expect about 5.3 GB (31,270 rows) or about 7.1 GB (42,039 rows) to pull.
- **EAT pass.** EAT adds about 0.6 MERT units, roughly 15-20% of wall time. With a single GPU, EAT=0 on the first pass
  reaches the music-complete point sooner; with NGPU >= 2, keep EAT=1.

## Minor
- **Lock scope.** `ads.py` keys its lock by the argument string. A second call with a different run list writes a new
  lock and takes a second held-out look. Record the exact A2-winner invocation in `choice_lock.json` before phase 2, or
  add a "held-out already computed" guard.
- **MuQ token-accuracy check.** 0.0 on 8 codebooks is also what chance predicts: about 0.24 expected hits per codebook
  over about 250 tokens, and P(all eight are 0) is about 0.14. The check is uninformative, as the study says; the mel
  statistics are the real input check.
- **Source archiving.** Sofia and AT-ADD were "fetched" but not archived in `sources/`, so the key ablation numbers
  cannot be re-verified offline. The "without MuQ 93.9 vs without MERT 94.6" gap is 0.7 F1 on one test set, which is
  weak for ranking MuQ over MERT. The useful reading is complementarity: MuQ adds 3.3 F1 on top of MERT. Medium
  confidence is fair.

## Corrected decision
1. Top up VESSL credit. Then run `run_vessl.sh` with a manifest of the 31,270 pinned rows (current order) followed by
   the 10,769 new rows, NGPU=2 (or MINUTES of about 210 on one H100), EAT=1 only with 2 or more GPUs.
2. Screen and combine as in rule A. In rule B, include the MERT-only null fusions. MuQ must clear max(baseline, best
   null) + 0.002 val ADS; today that is 0.94312 for music and 0.94170 for voice. Ship only if the single held-out pass
   gives p(delta <= 0) <= 0.05.
3. Package without a time fallback. The L4 estimate is about 31-33 min for S4 + MuQ x2 + EAT (calibrated on S3's
   measured 24 min).
