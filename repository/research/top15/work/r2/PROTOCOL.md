# R2 pre-declared protocol (written 2026-09-27 before any R2 training run)

Data/protocol: DB protocol on backbone-cache rows (`_workspace/backbone_cache`, 31,270 rows): heads trained on the data v4
train split only, early stopping on val (selection key = (val route EER, val BCE) lexicographic, as in
`data_v4/train_backbone_heads.py::train`), held-out test (source-held-out split) used for reporting only.
Seeds 41/42/43 for every variant. Objective, loss weights, duration balance, AdamW wd .01, clip 1 unchanged.

Choice rule (fixed before running):
- Per group (voice_df, voice_ad, music) the metric is the val route EER of the 3-seed-mean probability ("val ens").
- A variant is declared better than its group baseline only if val ens EER improves by >= 0.002 AND the mean of the
  per-seed best val EERs also improves. Otherwise the baseline is kept (ties go to the simpler/baseline variant).
- Among variants passing the gate, the lowest val ens EER wins.
- The choice is written to `choice_lock.json` (timestamp) BEFORE held-out EERs are computed. Held-out route EERs are
  then reported for every variant, but are not used to revise the choice.
- Official held-out ADS is computed only for the val-chosen winners vs the ads_compare baseline.

Amendment (2026-09-27 13:59:02 KST, before any held-out EER was computed): added group fullvoice_db (shipped
DF-Arena voice head, all rows of the old cache_v4, DB protocol) under the same rule, and fullvoice_p1 (P1 protocol, val only,
no held-out estimate possible because the held-out rows are training rows).

Amendment (2026-09-27 14:02:09 KST, after val results of the first runs, before any held-out EER was computed):
added voice_df resmlp_long and mlp_long (the same MLP heads on DF-Arena fc5 inputs) because the AD MLP heads moved val a lot.

Post-lock confirmation (2026-09-27 14:07:58 KST, AFTER held-out EERs were seen): fullvoice db mlp_long = the
val-chosen voice_df architecture (mlp_long) re-run unchanged on all rows of the old cache. Single pre-specified config,
reported whatever it shows; not a selection step.
