# Skeptic critique of the stacking study (2026-09-27, KST)

Scope: I read `rule.json`, `choice_lock.json`, `run.py` (diffed against R2), `ads.py`, the logs, `val_report.json`,
`test_report.json`, the SA patch, and the runtime (`full_bank_submission.py`, `component_fusion.py`, `voice_heads.py`).
I ran one cheap check on the held-out predictions the study had already saved: `critique_check.py` -> `critique_check.json`.
It took 6 s on CPU with 0.54 GB peak RSS, and it trains nothing.

Unless stated otherwise, every number is DB protocol: heads trained on the train split, early-stopped on val, 3 seeds
(41/42/43) with seed-mean scores, on the frozen 31,270 backbone rows. They use the RT protocol for official ADS, with
held-out n = 4,822 rows / 2,215 groups.

## Verdict
**The main decision survives: ship S5 = S4 unchanged.** The optional SA slot is weaker than reported, and its
evidence should be downgraded.

## Attacks that failed (decision unaffected)
- **Selection on held-out.** `ads.py val` builds `Evaluator(('val',))`: it loads only val scores and val presence, and
  it builds tables and FILE fits from val only. The mtimes are in order:
  - `rule.json` 15:04:25
  - runs 15:08-15:15
  - `choice_lock.json` 15:19:30
  - `presence_test.json` 15:20
  - `test_report.json` 15:23

  One wording issue: `run.py` saves *unscored* test predictions at training time (`test_preds`, `every.test`), as R2
  did. So the lock's note "no held-out prediction ... had been computed" is literally false. No held-out *metric* was
  computed, so this is not a leak.
- **Prior held-out knowledge shaping the rule.** The gate (+0.001, P <= 0.10) was written after R1's s3music and R3's
  SA held-out results were known. It rejected both the add-on that held-out disfavoured (POOL_m) and the one it favoured
  (SA). There is no sign of a rule tuned toward an outcome.
- **Wrong baseline.** The S4 member scores equal R2's run files exactly (max |diff| 0.0 on val and test). AC reproduces
  R2: val 0.93845, held-out 0.88643.
- **Unmatched budgets.** Every variant used the same cap and patience (voice 150/10, music 50/5), and no seed hit the
  cap (logs). The pooled variants therefore got the same training budget as S4.
- **Seed noise.** POOL_m per-seed val music EER: mean .1197/.1042/.1113 vs LME .1191/.1152/.1158, so 2 of 3 seeds are
  better. The val bootstrap ignores seed variance, and adding it would only raise P. POOL_m's rejection stands.
- **Leakage via the growing cache.** It is handled by `frozen_rows.json`, and the reproduction is exact. The presence
  heads come from the 09-23 DB `train/presence_head.json`, not from P1, so the SA gate `Pm` has not seen held-out labels.
- **Runtime feasibility of S4.** `full_bank_submission.py` already builds heads through `voice_heads.build(arch)`, and
  `component_fusion.fuse` now clips to [0, 1]. That fixes R2's saturation bug (self-check included). S4 needs no code
  change.

## Attacks that partly land (they support the decision or qualify claims)
1. **POOL_m's val gain is partly an in-sample FILE artefact.** On POOL_m alone the default FILE fit adopts the grid
   (`noisy_or 10`, in-sample on val). With FILE fixed to equal weights, POOL_m's val RT ADS falls from 0.94175 to 0.94040,
   i.e. +0.0027 instead of +0.0040 vs S4 0.93773. So about a third of the headline val gain is in-sample optimism that the
   "predictions fixed" bootstrap cannot see. The gate's P = 0.127 is therefore optimistic, which reinforces the
   rejection. The held-out POOL_m x FILE interaction (-0.0024) is the same effect seen out of sample.
2. **AC is label-leaky.** Under AC, a row with no voice (or music) label gets component score 0, and FILE sees that.
   This works as oracle presence. RT removes it: S4 held-out is AC 0.88643 vs RT 0.88457, a gap of 0.0019. The study
   correctly lets RT decide. Anyone quoting R2's 0.88643 as S4's held-out ADS should know it includes about +0.002 of
   this leak.
3. **The RT "voice = 0 when silent" branch is never exercised.** Every run file has 0 silent rows in val (3,680) and in
   test (4,822). RT therefore equals "every head scores every row". The `not silent` guard in the SA patch is untested.

## Attack that lands: the SA held-out gain rests on one source (`critique_check.json`)
The question: is S4+SA's held-out RT gain (+0.00463; group bootstrap CI [0.0026, 0.0068]) broad, or one source?

- **Dropping one source at a time.** Removing HAIM turns the gain into **-0.00063**, and voice EER *worsens* by +0.0032.
  Removing any other source leaves +0.0030 to +0.0064.
- **Where the gain comes from.** Held-out HAIM has 335 fake-voice rows: Lyria-3 Pro 168 and fx_encoder_style_transfer 167.
  All of them are labelled VOICE_FAKE = MUSIC_FAKE = 1, i.e. the vocal is fake exactly when the song is. Train and val
  contain **zero** HAIM fake-voice rows (VOICE_FAKE is blank or 0). This explains why val cannot see the gain.
- **The group bootstrap overstates certainty.** It treats HAIM's 478 groups as independent evidence. The held-out split
  is source-held-out, so sources are the right resampling unit. A source-cluster bootstrap (18 sources, 1,000 reps,
  993 with defined EERs) gives dADS CI **[-0.0022, +0.0111], P(d <= 0) = 0.35**. The effect is real on HAIM but does
  not generalise across sources.
- **Consequence under P1.** P1 moves these very HAIM rows into training. Outside HAIM, SA is slightly negative, and val
  (other song sources) is negative as well (-0.0005 RT). So the expected P1/LB gain of SA is about 0 or negative, unless
  the official test holds many fully generated songs whose vocals are labelled fake **and** that the P1 voice heads still
  miss. That share is UNVERIFIED.
- **Untested fallback.** In the 48-min fallback (AD skipped) the patch would fuse ours + music only. That combination was
  never evaluated.

## Corrections to the study's text
- "Song-aware VOICE is the only held-out winner, +0.0046 [0.0026, 0.0068]": add that this is 100% one held-out source
  (HAIM Lyria-3 Pro / fx_encoder fakes). Without HAIM it is -0.0006, and the source-cluster CI is [-0.0022, +0.0111].
- "POOL_m val +0.0040": about 0.0013 of it is in-sample FILE grid adoption. With equal FILE weights it is +0.0027.
- "Exact config": `train_backbone_heads.py` also requires `--cache` and `--out`. `--final` reads whatever the live
  backbone cache holds (now more than 31,270 rows), so the shipped S4 is trained on more rows than the evaluated S4.
  This is expected under P1, but it means the packaged heads are not the evaluated ones.

## Recommendation
Ship S5 = S4 as locked. Do not spend a slot on SA on this evidence. If a slot is spare, a val-legal SA test would need val
(or P1-CV) rows containing fully generated songs with fake-labelled vocals. The current val has none.
