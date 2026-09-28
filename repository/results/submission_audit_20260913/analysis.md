# Submission audit — 2026-09-13

Aside Computer Use read the live DACON submission table. All three submissions have scores; an early runtime failure is not supported. Exact UI values are in `submission_table.txt` and structured values in `audit.json`.

| Submission | Model | Score | ADS | CPS | Seconds |
|---|---|---:|---:|---:|---:|
| 88418 | baseline | 0.6909110688 | 0.6577539683 | 0.9893249735 | 1319 |
| 88630 | AST epoch17, 633-file training | 0.6453405503 | 0.6212619048 | 0.8620483598 | 81 |
| 88673 | AST epoch17, full1029 refit | 0.6174819841 | 0.5881111111 | 0.8818198413 | 82 |

The downloaded `DACON_Sumbit_1.zip` matches the full1029 package SHA-256 and its inner model SHA-256. A wrong local renamed ZIP is ruled out. Server-side per-file logs are unavailable.

The baseline runs PANNs presence detection, HTDemucs source separation and DF-Arena on separated voice and music (`baseline_original/script.py`). The AST replacement uses one backbone with five heads and averages predictions over windows (`ast_multitask.py`, `v2_inference.py`). This substantially reduced inference work explains the direction of the runtime difference; individual stage timing was not measured. Evaluation does not repeat training, so epoch count does not determine server inference duration.

Official scoring is 0.9 ADS + 0.1 CPS: https://dacon.io/competitions/official/236749/overview/evaluation . Relative to AST 633, full1029 loses 0.0278585662 total: ADS contributes -0.02983571433, while CPS contributes +0.00197714815. This identifies the metric responsible, not the specific causal mechanism inside the model.

The prior selection artifact (`results/v2_colab_20260912_174330/selection.json`) selected G0 because neither AST candidate passed all gates, including EER and companion AUC regression checks. Promoting AST based on its partial-label development proxy despite these failures was not justified. Full1029 includes former validation files, leaving no independent holdout; its training loss cannot establish generalization.

Potential mechanisms needing controlled validation are loss of separation and specialized fake detection, and overfitting the small partially labeled development distribution. Neither is proven by the public score alone. More epochs or a longer server run is not evidence of improved quality. Retain baseline 88418 as the best observed submitted model; restore a source/generator-disjoint validation comparison before promoting another candidate. No new submission or training run was performed during this audit.
