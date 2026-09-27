# R2: head optimization and multi-layer features (2026-09-27 13:52-14:10 KST, CPU only, Apple M5 Pro, 4 torch threads)

Everything is in `research/top15/work/r2/`. No project file, cache or submission was modified.

## Method
- `run.py` is a copy of `data_v4/train_backbone_heads.py::train` with extra knobs: max_epochs, patience, lr, files/update,
  architecture and MERT layer set. It records the best epoch (the original returns the last one), update counts,
  wall time and per-epoch val history. Objective, loss_weights, duration_balance, AdamW wd .01, clip 1 and the
  (val EER, val BCE) lexicographic selection are unchanged. Groups:
  - `voice_df`: DF-Arena fc5 inputs from the old cache (`items_for(..., dfarena=True)`), backbone rows.
  - `voice_ad`: AntiDeepfake-1B embeddings, backbone rows.
  - `music`: MERT layers, backbone rows.
- `full_voice.py`: the shipped voice head (DF-Arena, all rows of `_workspace/cache_v4`). Protocol `db` is train, val
  early stop, held-out. Protocol `p1` adds held-out to train, so it gives val only.
- Protocol: `PROTOCOL.md` fixed the val-only choice rule before any run and logs the amendments with timestamps.
  `choice_lock.json` was written at 14:04:45 KST, before `report.py test` computed any held-out EER.
- Official ADS: `ads.py` uses the exact `ads_compare.py` pipeline (shared presence, FILE ensemble refit on val, voice
  CDF fusion ours 1 : ad 0.5), then `boot.py` runs the group bootstrap (2,215 groups, 1,000 reps) and the per-source
  breakdown.

## Reproduction checks (the copy matches the originals)
- R2 `base` seed-mean scores vs `_workspace/audit_20260927/ads_compare_scores.json`: max |diff| is 0.0 for voice_df and
  voice_ad, and 6.6e-5 for music_6_12. The S3-like baseline ADS reproduces 0.87267 exactly (`ads_report.json`).
- Per-seed values:
  - music [6,12] seed 41 val 0.11905 (music_layer_seeds.json: 0.1191).
  - voice_ad 0.04850 at 40 epochs, voice_df 0.05088 at 37 epochs (screen_report.json).
  - `fullvoice_p1 base` equals `_workspace/p1_refit_cpu/results.json`: best epochs 39/24/50, val 0.049175/0.050964/0.049175.
  - `fullvoice_db base` val equals the DurBal H100 results (0.050768/0.050964/0.050768). Seed 42's best epoch is 50
    here vs 44 on H100, because H100 ran bf16 autocast and this run is fp32 CPU.

## Results (val = choice metric; held-out = report only)
See `table.md` (generated from `choice_lock.json` + `report.json`).

The post-lock confirmation `fullvoice_db mlp_long` (the val-chosen voice_df architecture on all rows) gave:
- val ens 0.0317 vs 0.0510 base;
- held-out ens 0.0815 vs 0.1101, seeds 0.0819/0.0872/0.0858.

Official held-out ADS (backbone rows, n=4,822; `ads_report.json`, `boot.json`):

| config | val ADS | held-out ADS | held-out FILE/VOICE/MUSIC EER | delta vs S3-like, 95% group CI |
|---|---|---|---|---|
| S3-like (Linear DF + Linear AD, music 6+12) | 0.92912 | 0.87267 | .0996/.1106/.1847 | — |
| DF voice head -> MLP | 0.93849 | 0.88161 | .0882/.0945/.1847 | +0.0089 [0.0053, 0.0123] |
| AD voice head -> ResMLP | 0.92955 | 0.87438 | .0975/.1073/.1847 | +0.0017 [-0.0003, 0.0045] |
| both MLP | 0.93845 | 0.88643 | .0834/.0823/.1847 | +0.0138 [0.0092, 0.0174] |

Per-source voice-route errors, each at that model's own EER threshold (`boot.json`), DF Linear -> DF MLP:

| held-out cell | Linear | MLP |
|---|---|---|
| HAIM fakes (miss) | .409 | .281 |
| constructed_enhanced reals (FP) | .242 | .135 |
| constructed_mix reals (FP) | .207 | .103 |
| FLEURS-ko reals (FP) | .114 | .096 |
| MLAAD fakes (miss) | .044 | .024 |
| MUSAN-music reals (FP) | .326 | .277 |
| constructed_mix fakes (miss) | .105 | .126 (worse) |

Length dependence (n-weighted within-source Spearman of score vs samples, held-out reals, full rows) is unchanged:
0.241 Linear vs 0.262 MLP (`logs/length_bias.out`).

## Latent bug found (blocks deploying a saturating head)
`decision_heads.quantile_table` adds `i*1e-12`. When a head outputs exactly 1.0, the table top becomes 1+1e-9.
`component_fusion.fuse` then returns a value above 1 through `np.interp`. The runtime (`full_bank_submission.py:136`)
passes it unclipped to `validate_predictions` (line 167), which raises. That is a runtime error, and it burns a slot.

The MLP heads do saturate: voice_df mlp_long has 128 val and 58 held-out seed-mean scores equal to 1.0. The current
Linear heads peak at 0.999997, so S3 is unaffected. The runtime also builds `nn.Linear(1280, 2)` with strict loading for
both voice_* (line 34) and voice_ad_* heads, so an MLP checkpoint would raise SubmissionError until the code changes.

## Timing / batch
- Load: all npz for one group takes 20-40 s.
- Train, per seed: 5-38 s on CPU. A full 3-seed run takes 49-117 s wall.
- Peak RSS: 0.9-1.5 GB for voice, 1.3-2.1 GB for music with 2-4 layers, 3.6 GB for 8-layer fp16.
- Updates/epoch at 256 files: voice (backbone rows, 18,260 items) 72; voice (full, 24,623) 97; P1 (29,762) 117;
  music (6,981) 28.
- Batch size, AD Linear at 150/10:

  | batch | updates/epoch | val | held-out |
  |---|---|---|---|
  | 1024 | 18 | 0.0482 | 0.1432 |
  | 256 | 72 | 0.0482 | 0.1369 |
  | 64 | 286 | 0.0443 | 0.1336 |

  lr 3e-4 at batch 256 gives the same 0.1336 as batch 64. A larger batch at fixed lr and epochs means fewer updates and a
  worse result. A GPU is irrelevant for heads; only backbone feature extraction needs one.

## Limits
- All held-out numbers are DB protocol (train only). A shipped package would refit on train+held-out (P1 protocol),
  which has no clean estimate. The held-out-to-LB transfer ratio is about 0.3-0.6 (lane2), so +0.0138 held-out ADS
  suggests roughly +0.004 to +0.009 LB ADS. That is UNVERIFIED.
- The MLP hyperparameters are fixed and untuned: hidden 256, GELU, no dropout, lr 1e-4, 150/10. The SLS variant is a
  plain softmax-weighted layer sum (alpha init uniform, lr 1e-3). Per-layer gating as in the SLS paper was not tried.
- Held-out sources are few groups (lane3), so the per-source rates are indicative only.
