> SUPERSEDED IN PART (2026-09-27 14:50 KST): written before S3 scored LB ADS 0.7853015873 and before the time-only GPU
> budget directive. C.4, C.5, C.7, E and F are stale (credit-based rejections, S3-not-yet-scored, gap arithmetic from P1).
> Current verdicts were returned to the orchestrator; new evidence here: ad2b_check.py/.out.txt, rank_check.py/.out.txt.

# Review v3: feasibility, compliance, deadline (2026-09-27 14:35 KST)

Read-only review. Nothing outside this folder was modified. `pkg/` holds small runtime/table files extracted from
`submissions/DataV4_S3_Backbones/submit.zip` (unzip -p, read-only). One CPU check (`sat_check.py`, numpy, <1 s).
One small official page fetched: https://huggingface.co/datasets/mueller91/MLAAD/resolve/main/README.md (2026-09-27).

## A. S3 package (DataV4_S3_Backbones, sha 062fc236)
- Size: zip 8,286,541,720 B, expanded 8,286,530,714 B (submit.json). Limits 10 GB / 32 GB (official_text.txt:42,177);
  packager enforces 10,000,000,000 B (package_datav4_submission.py:128). Headroom about 1.71 GB.
- Output range: packaged `component_fusion.json` voice `ours` table max 0.9999953916, ad max 0.99954508, file tables
  max <= 0.99999929. fuse() <= ours[-1], so VOICE can never exceed 1; `sat_check.out.txt` shows fused(1,1) = 0.99999539.
  The R2 saturation crash therefore does NOT affect S3.
- New code already ran on CUDA: `data_v4/extract_backbones.py:48,70` builds the same `AntiDeepfake` and calls the same
  `windows()` used at runtime (full_bank_submission.py:134), on H100 for 31,270 files, including multi-batch (>34 s) files.
  What is new on L4 is co-residency with DF-Arena and wall time.
- Runtime bound: AD encoder config equals the DF-Arena SSL config (48 layers, 1280, 5120, conv 7x512; AD
  antideepfake_config.json vs df_arena_1b/facebook/wav2vec2-xls-r-1b/config.json). AD uses 50% overlap windows
  (1.0x at 4 s, 1.33x at 10 s, 1.75x at 30 s, 1.93x at 60 s of DF-Arena segment count, script.py:157-165 vs
  antideepfake_runtime.py:97-99), batched 16 vs DF-Arena batch 1. So AD time <= ~1.93 x DF-Arena time.
  With the only recorded L4 runtime (DurBal 17m32s, memory file; P1 has no recorded runtime but the same code path),
  the pessimistic bound (all 17.5 min = DF-Arena) is AD <= 33.8 min, S3 <= ~51 min: fallback would skip AD for about
  the last 6% of files, and no timeout (remaining files take P1 time only, ~66 s). Realistic: 23-35 min (lane4).
  Timeout needs AD > ~6.8 s/file on average, i.e. >8x the whole P1 per-file cost: implausible.
- QA coverage gap: the 10 QA files are 3.1-32.8 s, all mono, one 32 kHz mp3 (qa/originals); no 60 s file, no stereo.
  P1 already covered stereo/format on L4, and the AD path only sees the mono voice stem.
- CPU QA timing is noisy: P1 67.3 s vs S2a 82.4 s for identical runtime code, S3 84.9 s. Not an L4 estimate.

## B. Independence of the 48-min fallback
- Rule 2-4 (official_text.txt:101-104) bans using other files' info/predictions/statistics. 417885-2 adds "각 평가 파일을
  개별적으로 추론하더라도 동일한 결과가 산출되어야". The fallback depends on elapsed time, not other files' content, and
  only fires past 48 min. If the DACON-displayed runtime of S3 is < 48 min, it provably never fired for that run.
- Action: note the displayed runtime; in the next package either drop the guard (if S3 runtime < ~35 min) or keep it
  and disclose. Not a blocker.

## C. Next candidates
1. S4 = R2 MLP voice heads (DF-Arena MLP + AD ResMLP), P1 refit.
   - Mandatory code: MLP/ResMLP construction in load_heads (full_bank_submission.py:34) and AD heads (:71-72), and a
     clip of fused VOICE (:136) or inside component_fusion.fuse. sat_check: with 128/2,828 val scores == 1.0 the table
     top is 1+1e-9 and 4.3% of saturated files exceed 1 even with random AD, so over 1,200 files a crash is near-certain
     without the clip. R2's ADS already uses min(1, fuse) (r2/ads.py:64-65), so the estimate includes the resulting ties.
   - Missing evidence: no `fullvoice_p1__mlp_long` run (r2/runs listing), and the official-ADS figure (+0.0138) used
     the backbone-row DF MLP (voice_df), not the all-row DF MLP that would ship. Run both before packaging (CPU, minutes).
   - Fits size (+~1 MB) and runtime (heads negligible). Build time: S3 heads 07:40 -> zip 07:42 -> QA 07:44, so
     packaging is minutes; total 2-4 h including code edits. Depends on S3's L4 success (same AD path).
   - Strongest failure: LB transfer. Head-only changes moved LB ADS by <= +-0.002 and val->LB paired ratios were
     0.32/0.39; the 0.63 ratio is one data-change pair. Expect +0.002 to +0.009 LB ADS, not +0.09.
2. R1 pooling (voice LME + music mean): tested only on Linear heads; mean pooling did not help S3's MERT-6+12 music on
   held-out; AD heads and MLP heads untested. Needs 3 runtime edits (full_bank_submission.py:114,120,135) + retrain +
   refit. Do not bundle into S4 without a DB-protocol held-out check of the exact combined config.
3. R3 song-aware VOICE: per-file OK, size/runtime OK. Strongest failure: fake-voice + real-music miss 0.084 -> 0.238
   (n 214) and real-voice + fake-music FPR 0.105 -> 0.337 (r3/cross_component_check.out.txt:17-18). AI covers (cloned
   vocals over real backing) are a natural case for a deepvoice-crime test set. Deciding it by one LB A/B is a probe
   of test composition (417702-4). Reject for a slot.
4. R3 E2 (EAT-large or MuQ into MUSIC): needs VESSL GPU (audio only there; paid; needs user authorization), weight
   download, possibly extra pip deps (install errors do not count, runtime import errors do). 1.71 GB headroom fits
   one ~0.3B encoder, not two, and not fp32 AD. No evidence on our data; credit $9.85 allows about one attempt.
5. SSL fine-tuning: ~3.3 H100-h of credit in total, no iteration budget, spend authorization needed. Reject for this
   deadline.
6. Data: Music Arena (1,000 local files, 1.9 GB, CC BY 4.0, rights_decisions.json training_allowed true) is the only
   clean pack. The existing S2-B bank is 289 AI-Hub 71470 + 851 Jamendo + 773 Music Arena train rows
   (_workspace/s2_extra/bank_B_train.csv); Jamendo is training_allowed false (TDM unresolved) and AI-Hub is unresolved
   for round 2, so S2-B as built must not ship. Music Arena has 273 VOICE_FAKE=1 rows, which would need AD embeddings
   for the S3/S4 voice heads. AI-Hub D2-D4: AI-Hub policy (usagepolicy.txt:75) bans provision to non-approved parties
   and needs agreement for export abroad; 417280-1 bans data whose license restricts third-party provision. Reject for
   any candidate that could become the best (= final) submission.
7. Backbone completion on VESSL ($3-3.6): small gain, spends the credit that round-2 data export needs.

## D. Cross-cutting compliance and round 2
- MLAAD: data v4 uses mueller91/MLAAD (data_v4/sources.py:57,60), 8,775 of 42,039 bank rows. The card says "published
  strictly for non-commercial academic research use, under the CC-BY-NC 4.0 license" (fetched 2026-09-27). The
  project's own rights review marks MLAAD `training_allowed: false, research_scope_unresolved`
  (_workspace/lb_admission_20260924/rights_decisions.json). 417857 allows NC models but asks to comply with other stated
  conditions. This applies to P1, S3 and S4 equally; document it or get a ruling before round 2.
- Round-2 data files: materialized audio lives on the VESSL /root cluster volume (DESIGN.md 8.1-8.2, output ~30 GB).
  Exporting it needs a workspace session, so keep a credit reserve; do not spend all $9.85 on experiments.
- Final = best-ever score. Whatever wins must be reproducible from training code + data; P1-protocol heads trained on
  CPU reproduce exactly (R1, R2), which helps.

## E. Deadline and slots
- Now 2026-09-27 14:35 KST; deadline 2026-09-29 09:59:59 (~43.4 h). 3/day; reset time UNVERIFIED (FAQ 93 body
  inaccessible). P1 and S2a used 2 of 09-27's slots. Whether S3 was already submitted (and scored below P1, still
  scoring, or failed) is UNVERIFIED; LB at 13:33 shows only P1. Check the submission tab first.
- P1: submitted ~01:07, score known by ~02:10 (memory file). With 8.3 GB uploads and queue congestion, a runtime-changing
  package should not be first submitted on 09-29; use 09-29 only for code already proven on L4.

## F. Gap arithmetic
Top-15 needs +0.0899 LB ADS. Best-case sum of measured held-out levers (S3 +0.0083, S4 +0.0138 over S3, R1 ~+0.005,
R3 +0.0046; not proven additive) is about +0.03 held-out ADS, i.e. about +0.01-0.02 LB ADS at ratios 0.32-0.63:
rank ~100-90 (r4 rank map: +0.010 -> 100, +0.030 -> 76). Top-15 is not reachable with evidence-backed levers.
