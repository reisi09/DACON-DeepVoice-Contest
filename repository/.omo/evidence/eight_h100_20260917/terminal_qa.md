# Terminal QA attempt6

## Verdict

The unpacked attempt6 artifacts pass independent numerical, export, notebook, checkpoint, and selection-lock checks. Two report-level aggregation fields block a clean final-aggregation claim and are captured in the JSON sidecar; the archive itself was not changed.

## Evidence

- `experiment_1.json`, `experiment_3.json`, `experiment_4.json`, `experiment_6.json`, `experiment_8.json`, and `e7/report.json` all report `status=complete`. Learned head experiments use seeds `41,42,43` and `fixed_updates=500`; E7 records `200` optimizer updates per learned arm per seed. E2 and E5 are deterministic diagnostics with `fixed_updates=0`; E8 is a deterministic REAL-only density fit with `fixed_updates=null`.
- Every selection lock records `test_used=false`; E7 records `locked_before_test=true`. E7 reports unchanged frozen probes, changed LoRA adapters where enabled, nonzero adapter gradients, and exact reloaded development logits. Independent checkpoint loading found nine finite checkpoints; each has 96 LoRA-B tensors and a `(2,1280)` classifier. Head-only checkpoints retain zero LoRA-B values; LoRA CE/KL checkpoints have 96 nonzero LoRA-B tensors.
- The executed notebook has 19 cells, 9 code cells, execution counts 1 through 9, and zero error outputs. `run.log` retains three pre-recovery operational failures (missing `imageio_ffmpeg`, missing ffmpeg executable, and a missing old `asset_hashes.json` path); later recovery evidence exists, and these errors do not occur in the executed notebook outputs.
- `EXPORT_SHA256.json` lists 7,314 members and all 7,314 independently recomputed hashes match. There are 7,101 feature NPZs. Every feature identity audio hash matches manifest SHA `fc21a54a7d9aeec69575ad1b6a9c0d0c35910f13ba317fde434587686309b537`; required arrays match identity declarations, and all feature arrays are finite.
- 77 saved-prediction primary EERs were recomputed with `roc_curve(drop_intermediate=False)`, `argmin(abs(FPR-FNR))`, and `(FPR+FNR)/2`. E1/E3/E4/E5/E6/E8 and all E7 prediction files match the recorded EERs exactly; maximum absolute delta is `0`. E3 music primary comparisons use the recorded `opus_telephone` slice (127 test rows, 104 fake, 23 real).

## Report-level findings

1. `final_aggregation.json` reports `dataset.component_count=178` and repeats 178 for each experiment. Running the exact notebook DSU code against the sealed 7,101-row manifest produces 178 because singleton `group_id` atoms are never initialized: the loop only calls `union(parts[0], part)` for `parts[1:]`. A corrected DSU that initializes every group atom produces **256** connected components. This is a metadata aggregation defect; no predictions or fitted checkpoints were changed.
2. `final_aggregation.json` reports E8 `fixed_updates=500`, while raw `experiment_8.json` reports `fixed_updates=null` and has `fit_rows=139`, `calibration_real_rows=31`, and a deterministic REAL-only density procedure. The final field is a generic fallback and should be corrected to `null` (or an explicit deterministic-fit semantic).

Top-level generic label denominators are known mixed FILE/VOICE metadata and were not used to judge branch metrics. The stale `progress.json` stage is `experiment_5`; completion is established from all eight complete reports, executed notebook outputs, and export hashes.

## Exact invocations

The full command transcript and outputs are in [terminal_qa.log](terminal_qa.log). The machine-readable matrix, exact EER deltas, corrected sidecar values, and `manualQa` matrix are in [terminal_qa.json](terminal_qa.json).

Artifact source: [attempt6 unpacked output](../../../results/eight_h100_20260917/downloaded/unpacked/output) and [export receipt](../../../results/eight_h100_20260917/downloaded/export_receipt.json).
