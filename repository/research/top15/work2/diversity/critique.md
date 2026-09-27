# Skeptic critique of the diversity study (written 2026-09-27 ~15:45 KST)

Scope: I inspected the code (`subsets.py`, `run.py`, `ads.py`, `pilot.py`, `relatives.py`, `coverage.py`, `lock.py`), the lock files, the run JSONs and the reports. I ran one check with no training: `critique_check.py` (23 s CPU, 0.5 GB RSS; output in `critique_check.json` and `logs/critique_check.log`). It recomputes official ADS **per seed** from the saved predictions under the study's own protocol (DB, R2 row universe, FILE refit on val, shared presence, voice seed k paired with music seed k).

## What holds up
- **Protocol hygiene is good.**
  - The lock (15:23:25 KST) came before `report.json` (15:25:15) and `coverage.json` (15:23:42).
  - `coverage.py` and `pilot.py test` both assert that the lock exists.
  - The R2 reproduction is exact (max |diff| 0.0 for both voice_df and music).
  - Budgets are matched: every Q/D and LOFO/CTRL pair has identical per-stratum row counts (asserted in `subsets.py`), and group counts are similar (for example Q50 has 5,734 groups and D50 has 5,565 to 6,072).
- **Two post-lock items are harmless.**
  - `runs/pilot_voice__plus_arena.json` has an mtime of 15:32:51, after the lock. Its content matches the pre-lock log (`logs/pilot_voice.log`, 15:19: val EER .02809/.03366, ensemble .03021), so no selection happened.
  - Repeated EER values across different models (for example 0.07837 for both DF-only and S4-fused +arena) are not a bug. At the crossing, fpr ≈ fnr, so voice EER is close to k/1556 (1,556 fakes, 2,285 reals on the voice route).
- **The missing-score default is shared with R2.** The DF and AD members have no score for 835 val and 981 held-out rows (music-route items), and these default to 0. This matches the R2/ads_compare pipeline, so the study inherits it and did not introduce it.
- **No leakage found in the Music Arena pilot.** The added rows are sonauto-v2-2 and v3-preview (source music_arena). The held-out misses they target are HAIM Lyria-3 Pro and fx_encoder. The generators and the source differ.

## What does not hold up

### 1. The CIs ignore seed and subset-draw noise, which is as large as the effects
Every CI in the study comes from a group bootstrap over held-out groups with the models fixed. That does not include retraining noise. The noise, measured per seed on held-out official ADS:
- Single-seed spread within a config: SD 0.0029 ADS.
  - Voice EER alone can swing 0.024 between two seeds (CTRL_mlaad: 0.1028 vs 0.0791).
- Spread between subset draws (SD of the 2-seed means of the 3 D draws): 0.0093 at 25%, 0.0071 at 50%, 0.0033 at 75%.

Every Q arm is a **single draw** (d0). If Q varies as much as D does (a conservative assumption), the draw-noise SE of Q − D-mean is about 0.0107, 0.0082 and 0.0039:
- **25%:** Q − D = +0.0097 per-seed-mean (+0.0111 as reported from the ensemble), z ≈ 0.9. The secondary finding "only the extreme 25% cut hurts" is **not supported**. It rests on one lucky Q25 draw.
- **50%:** the primary null (−0.0001) cannot exclude a diversity cost of about 0.015 ADS. "No cost" is overstated. The fair reading is "no detectable cost; the study could only have detected a cost of roughly 0.02 or more".
- **75%:** SE about 0.004. A cost above about 0.006 is unlikely. This is the only fraction where "diversity is not the limiter" is reasonably backed.

### 2. "Quantity flattens toward 100%" is contradicted by the study's own numbers
Held-out ADS gain per doubling of train rows (DB, per-seed means):

| arm | 25→50% | 50→75% | 75→100% |
|---|---|---|---|
| D-mean (3 draws) | +0.0144 | +0.0127 | +0.0118 |
| Q (1 draw) | +0.0041 | +0.0099 | +0.0170 |

The D-mean curve is nearly log-linear at about +0.012 ADS per doubling. The Q curve even gets steeper. The "flattening" is a linear-axis reading plus the high Q25 draw.

### 3. The study's quantity number points to a free lever that the decision text discourages
At 15:29 KST `_workspace/backbone_cache` held **42,039 files, i.e. every data-v4 row**. The 10,769 new rows split as train 7,890, val 1,251 and held-out 1,628, spread across the same sources.
- Train rows go from 22,768 to 30,658 (×1.35, 0.43 doublings).
- At about 0.011 to 0.012 per doubling, that prices retraining the AD and MERT heads on the complete cache at **≈ +0.005 held-out ADS** (DB-protocol scale; the LB-transfer ratio has ranged from about 0.6 to 1.9×).
- This needs no GPU (head training runs on CPU), and it is the largest data lever the study's evidence supports.

The decision says "Do not spend GPU time on generic data diversity or quantity … expected gain 0.005 or less". That wording must not be used to argue against the full-cache refit. The 0.005 is a central estimate, not an upper bound.

Caveat: results on the extended cache use a different held-out set (6,450 rows instead of 4,822). They need their own lock and a fresh baseline, and are not comparable with R2.

### 4. The Music Arena voice gain is mostly an ensemble artifact from one seed
Paired per-seed held-out official ADS, +arena minus base, seeds 41/42/43:

| config | held-out deltas | held-out mean (SD) | val mean |
|---|---|---|---|
| DF-only | +0.0032, −0.0002, 0.0000 | +0.0010 (0.0019) | −0.0005 |
| S4-like fusion (R2 AD ResMLP at weight 0.5) | +0.0014, +0.0002, −0.0003 | +0.0004 (0.0008) | −0.0010 |

- "Better in 3/3 seeds" is true for route voice EER (−0.0097, −0.0018, −0.0026). It is not true for ADS, where only seed 41 moves.
- Seed 41 is also the run that early-stopped at epoch 85; the others stopped at 23 to 32.
- The 3-seed ensemble EER drop (0.0901 → 0.0784) and ADS +0.0026 [+0.0003, +0.0039] are about 2.5× the per-seed gain, and their CI omits seed noise.
- "The gain vanishes in S4 *because* the AD head was not retrained" is an untested causal claim. Even DF-only, the per-seed gain is only about +0.001.
- "Val is structurally blind to data additions" is not shown by this pilot. There is barely any held-out gain for val to be blind to. The general point, that component-level val cannot score new-generator additions, is still plausible.

### 5. The piggyback plan's gate uses held-out for selection
Step 4 says "add only if fused voice EER improves ≥0.005 on both that val and held-out". That is selection on a held-out set the study admits "has been consulted many times". The gate should be the leave-generator-out val only, with held-out reported once afterwards.

Given point 4 (S4-fused per-seed +0.0004), the piggyback's expected value is below its own gate and is not worth the VESSL setup, extraction and audio staging.

### 6. Minor points
- Exp 3's "music is limited by representation, not coverage" is an inference from equal miss rates on seen and unseen generators (0.186 vs 0.175). "Seen" generators arrive from different sources under the source-held-out split, so this is only nominal coverage. The inference is plausible but should stay at medium-low confidence.
- The Exp 2 ADS LOFO−CTRL CIs (for example MLAAD −0.0085 [−0.012, −0.005]) are too narrow for the same reason as point 1. The large family-miss effects (HAIM vocals +0.14, MLAAD +0.055) are probably real, but the 0.03 to 0.04 ones (FLEURS-ko, real music) could be seed noise.

## Verdict
**Downgrade.** "Ship nothing from the diversity pilots" survives. The following do not:
- the 25% diversity effect
- "quantity flattens"
- the size of the Music Arena gain
- "val is structurally blind"
- the held-out-gated piggyback

**Corrected decision.**
1. Ship nothing from the diversity or Music Arena pilots, drop the piggyback, and acquire no new external data.
2. Treat the now-complete backbone cache (42,039 rows, +35% train rows) as the one data lever the study supports. Retrain the S4 AD and MERT heads on it on CPU with a prior of ≈ +0.005 ± 0.005 held-out ADS. Verify under a fresh DB lock on the extended split before any slot is spent.
