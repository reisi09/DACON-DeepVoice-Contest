# Skeptic critique of work2/depth (2026-09-27, written after the study's 15:33 KST lock and reports)

Verdict: the decision survives. Deeper voice and music heads should not ship. Two parts of the write-up need correcting:
(1) the "train the DF MLP on all rows" recommendation is not a change, because S4 already does it; (2) "val agrees" for B
over A is weak, because the val and held-out gains come from different components. The group bootstrap also ignores
training-seed noise, and the one check I reran measures it.

## What I checked and found sound
- **Held-out isolation.** `report.py val` reads only `val_eer` and `val_ens_eer` (lines 36-37, 46-60). The run logs print
  only val numbers. File mtimes run in order: last run 15:32:50, `choice_lock.json` 15:33:05, `report_test.out` 15:33:17,
  `ads_report.json` 15:33:38. Nothing points to a held-out number being computed before the lock.
- **Amendment (15:23:45).** It was written after DF `resln` (15:16:47) and `resln_d1` (15:20:23) had finished, and it
  settles exactly that tie: two equal-size, non-incumbent heads, broken by lower val. Only val was known at that time, and
  lower val is the natural tie-break. The amended rule gives `(n_params, not incumbent, val ens)` (report.py:59).
  Low risk, and it has no effect on a "do not ship" decision.
- **Reproduction.** I reran the ads pipeline (DB protocol, frozen 31,270 rows, 3-seed mean) in my own script. It gives
  A 0.88643, B 0.88851, C 0.88628, the same as `ads_report.json`. The copied `run.py` changes only architectures and
  data loading. `train()` is unchanged, and the R2 heads reproduce with max |diff| 0.0.
- **Budgets.** All variants use 150/10, and none hit the cap (best epochs 8-80). `mil128` at 150/10 is identical to
  `mil128_r2` at 50/5. C−B is the matched comparison (both use the all-row DF head), and the study reports it next to
  the row-mismatched C−A. One limitation: lr (1e-4 voice, 1e-3 music) and wd were not tuned per architecture. "Deeper
  does not help" is therefore shown only under R2's optimizer settings. That limits the claim but does not refute the
  decision.
- **Coverage.** On the frozen ADS rows, A's DF head (`r2 voice_df__mlp_long`) and B's DF head (`fullvoice_db__mlp256`)
  cover the same 2,845 val and 3,841 held-out voice rows. The Fv=0 defaults are therefore identical in both configs.
  A and B have the same held-out FILE EER (0.08343) even though their FILE scores differ on all 4,822 rows
  (corr 0.996), so the match is a coincidence of discrete counts, not a bug.
- **Runtime.** `component_fusion.py:24` clips to [0, 1]. The strict-loading issue R2 found is fixed:
  `full_bank_submission.py:34` builds heads through `voice_heads.build(record['arch'])`, which supports
  linear/mlp/resmlp. `resln` is not supported (it raises ValueError), so shipping C would also need a runtime change.
  That is one more reason not to ship C.
- **Resources.** Measured wall time was two lanes of about 27 min each, the longest process about 15 min, and at most
  1.62 GB per process. This is within the limits.

## Correction 1: B is S4's existing practice, not a new decision
`submissions/DataV4_S4_MLPHeads/submit.json` says: "all heads retrained P1 protocol on 100% rows (train + held-out 37,108
/ val 4,931)". S4 already trains its DF MLP on all rows, and it also trains the AD and music heads on all rows, since the
backbone cache now holds 42,039 files. The A config (DF head on the 18,260-row backbone subset) was never what S4
ships. So B−A (+0.0021 held-out ADS) does not justify a change. It corrects the DB-protocol estimate of S4's DF
component upward. Neither A nor B is an exact DB proxy for S4, because this study kept AD and music on the frozen
31,270 rows by design. An S4 proxy with AD and music on all rows has not been measured.

## Correction 2: "val agrees" for B over A is weak
- DF route val EER, 3-seed mean, on the 2,845 frozen val rows: A's subset head **0.03127** vs B's all-row head
  **0.03207**. Val slightly prefers A.
- Fused val VOICE EER: A 0.0215 vs B 0.0217, flat to slightly worse for B (ads_report.json).
- B's val ADS edge (0.93964 vs 0.93845) comes entirely from val FILE EER, 0.0474 → 0.0449. The FILE rule stayed
  "equal", so this is not grid overfitting. On held-out, though, FILE is unchanged (0.0834) and all of the gain comes
  from voice (0.0823 → 0.0719). The val and held-out gains sit in different components.

B's support therefore rests on held-out evidence plus a prior: it was a pre-registered reference comparison, not a
selection among many options, and R2's learning curve showed voice still improving with more data. That is enough to
keep all-row training, which S4 already does. It should not be described as val-corroborated.

## Rerun check: per-single-seed official ADS (DB protocol, same pipeline, ≈29 s CPU)
Question: does the ranking survive training-seed noise? The group bootstrap holds the models fixed. The AD `resln_d1`
held-out seed sd is 0.014, so that source of noise is not small. Script and output:
`/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/depth_critique/perseed.py`
(`perseed.json`). Each row pairs seed s of every component. Presence, FILE refit on val and fusion ours 1 : ad .5 are
unchanged. n = 4,822 held-out rows.

| held-out ADS | seed 41 | seed 42 | seed 43 | 3-seed mean (as reported) |
|---|---|---|---|---|
| A R2 both-MLP | 0.88607 | 0.88311 | 0.88665 | 0.88643 |
| B all-row reference | 0.88747 | 0.88654 | 0.88862 | 0.88851 |
| C depth chosen | 0.88566 | 0.88758 | 0.88165 | 0.88628 |
| B − A | +0.0014 | +0.0034 | +0.0020 | +0.0021 |
| C − B | −0.0018 | +0.0010 | −0.0070 | −0.0022 |
| B + DF resln − B | +0.0005 | −0.0004 | −0.0043 | −0.0018 |
| B + AD resln_d1 − B | −0.0033 | +0.0014 | −0.0021 | −0.0016 |

- B > A holds in 3 of 3 seeds on held-out. On val, B > A holds in only 2 of 3 seeds (seed 42: val ADS 0.93484 vs
  0.93655).
- C − B ranges from −0.0070 to +0.0010 across seeds. That spread is wider than the bootstrap half-width, so the reported
  CI [−0.0058, +0.0016] understates the total uncertainty. Still, 2 of 3 seeds are negative and none is clearly
  positive, so the check does not overturn the decision. The DF resln per-seed route EER on held-out (0.0752/0.0739/
  0.0893) shows the same seed-43 collapse.
- Val ADS favors C in every seed (C 0.9377/0.9389/0.9395 vs B 0.9370/0.9348/0.9347). This repeats the study's main
  finding: val (random-group split) rewards head capacity that does not carry over to the source-held-out split.

## Residual risks (not refutations)
- Only one optimizer setting was tried for the deep heads, and the fusion weight (ad 0.5) and FILE recipe were not
  re-tuned for heads with a different score scale. The DF resln route gain (0.0828 → 0.0771 on the frozen held-out
  rows) is lost in fusion and FILE. The cause is not diagnosed; score-scale interaction in max/noisy-or is plausible.
  Re-tuning would be a new val-only study, and the deadline argues against it.
- Held-out has few groups per source (lane3). The per-source shifts in boot.json are indicative only.
- The val-to-held-out mismatch for deep heads (val ADS +0.0028 for C over B) is a warning for any other lane that selects
  capacity on val.
