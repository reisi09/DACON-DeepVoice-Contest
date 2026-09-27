# Evaluation-code audit (2026-09-27 19:50 KST)

Scope: every evaluation path used for a ship/reject decision today: `graph_scoring.official_metrics`,
`work2/backbones/ads.py` (S5 gate), `work3/integration/fit_ft_fusion.py` (ftC1 gate), `work4/eval.py` (voice-data L),
`data_v4/train_backbone_heads.py final()` (package fusion tables / FILE ensemble), `full_bank_evaluate.score_rows`.

| # | Check | Result | Impact |
|---|---|---|---|
| 1 | `official_metrics` reads label values, not `*_mask` | bank: every mask=0 cell has an empty value (0 exceptions over 42,039 rows x 5 labels) -> unknown never scored | none |
| 2 | EER / AUC implementation | nearest-point EER over the full tied ROC; `ads.py`/`Boot` sklearn EER equals it (fit_ft_fusion asserts bootstrap ADS == official ADS) | none |
| 3 | row/score alignment (`zip` without strict) | `score_rows` returns exactly one dict per row in order (missing cache raises); `validate_predictions` and `Boot.ads` assert ID order | none |
| 4 | **AC semantics: missing member score -> 0** (`ads.py`, `eval.py`) | val: 835 rows without a voice score, 2,394 without a music score; FILE EER under AC is ~0.004 optimistic vs runtime (RT) | S5 re-scored under RT (`rt_check.py`): +0.0122 [+0.0048, +0.0197], P(>0)=1.0 (AC +0.0111) -> **S5 decision stands** |
| 5 | reproduction of every reported number | rt_check: S4 RT 0.93773/0.88457, S4 AC 0.93845/0.88643, S5 AC 0.95475/0.89755 reproduced exactly; retrained S5 music head preds identical (max diff 0.0); work4 base == R2 runs and == stacking every-row preds (0.0) | none |
| 6 | **`work4/eval.py` val_completed metric** | rule-L-completed val rows had no prediction and were scored 0 (item lists hold only originally labelled rows) -> metric measured nothing about them | **defect**, fixed in `eval2.py` (own predictions `valc_*`, RT every-row); L passes its locked AC gate after the fix |
| 7 | ftC1 gate | RT and AC both negative (-0.0040 / -0.0036) | rejection robust |
| 8 | S5 package = S4 + music only | voice fusion tables byte-identical to S4; FILE ensemble refit on val (grid rule, also applied inside the DB estimate) | none |
| 9 | label-completion rule L | FakeMusicCaps caption IDs match bank `orig_id`; 1,028 train rows completed (logged in the run files) | none |
| 10 | held-out reuse | the data v4 test split has been consulted by many studies; each gate here is one pre-registered comparison, but best-of-many optimism remains | caveat |

Voice-data variant L (label completion of AI-generated vocals, no new data):
- AC (locked protocol): held-out +0.0024 [-0.0002, +0.0057], P(>0) 0.967; val_completed voice EER 0.0343 -> 0.0333 -> gate passes.
- RT: held-out +0.0020 [-0.0007, +0.0058], P(>0) 0.922, P(<-0.003) 0.001; val_completed 0.0335 -> 0.0346.
- Direct effect on the 162 completed val rows: AUC vs real songs 0.905 -> 0.919 (DF), 0.903 -> 0.935 (AD); real-song
  mean score 0.06 -> 0.14. Small, uncertain gain; bundle with the next package rather than spend a slot on it alone.
