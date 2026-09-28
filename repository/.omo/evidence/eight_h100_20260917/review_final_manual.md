# Manual QA re-review — corrected aggregation metadata

Review SHA: `3970207e336bae7b72c2c261a4c54129f57ad79c`

Overall verdict: **PASS**.

This is a bounded corrective re-review of the aggregation metadata. The two requested corrective scenarios passed. The canonical corrected summary preserves experiment-specific populations and update semantics, while the immutable archive and executed notebook remain unchanged. No GPU, API, broad native check, or archive mutation was performed.

## Corrected canonical experiment metadata

The authoritative corrected summary is `results/eight_h100_20260917/corrected_summary/final_aggregation.corrected.json`. Every experiment is `status=complete`, and its metadata agrees with its durable native report:

| experiment | component_count | fixed_updates | seeds / update semantics | fit kind |
|---|---:|---|---|---|
| E1 | 256 | 500 | 41, 42, 43 | — |
| E2 | 4 | 0 | deterministic; no seeds | — |
| E3 | 256 | 500 | 41, 42, 43 | — |
| E4 | 256 | 500 | 41, 42, 43 | — |
| E5 | 4 | 0 | deterministic; no seeds | — |
| E6 | 256 | 500 | 41, 42, 43 | — |
| E7 | 10 | frozen 0; each learned arm 200 per seed / 600 total | 41, 42, 43 | — |
| E8 | 256 | `null` | deterministic; no seeds | `non_iterative_closed_form_real_only_gaussian` |

The corrected summary records global `component_count=256`, role-specific branch population metadata, `correction_scope=report_metadata_only_no_refit_no_rescore`, `official_score=null`, `fresh_holdout_claim=false`, and `robustness_claim=false`.

## manualQa

### surfaceEvidence

| scenario id | criterion reference | surface | exact invocation | verdict | artifactRefs |
|---|---|---|---|---|---|
| CA1 | F1; corrected aggregation regression | `test_eight_corrected_aggregation.py` against downloaded E1–E8 reports, manifest, and population sidecar | `.venv/bin/python -m pytest -q test_eight_corrected_aggregation.py` | PASS | R1, R2 |
| CA2 | F1/F3/F4; corrected summary and immutable artifacts | canonical corrected summary, correction receipt, sealed `results.zip`, and executed notebook | `.venv/bin/python .omo/evidence/eight_h100_20260917/verify_corrected_aggregation.py` | PASS | R3, R4, R5, R6, R7 |
| CA3 | F3; numerical-score continuity from prior terminal QA | receipt-bound `.omo/evidence/eight_h100_20260917/terminal_qa.json` | `.venv/bin/python -c 'load terminal_qa.json and correction_receipt.json; assert eer_recompute comparisons=77, max_abs_delta=0, mismatches=[]; assert terminal_qa SHA equals receipt binding'` | PASS | R8, R9 |

CA1 output: `1 passed in 1.06s`. Its meaningful regression checks recompute the complete-manifest connected-component count, compare corrected metadata against each native report, verify branch populations against the hashed sidecar, and enforce E8's non-iterative semantics.

CA2 output: `VALID global=256 sealed=178 E8=null archive_and_notebook_immutable=true`. Observed archive SHA-256 is `681d41de10397d31a371daaa276a4a90d19f04789b85e01fdd6f936399dd23d2` and executed-notebook SHA-256 is `bb28445e88a4d963aa7969fddba03e2c88badf4d969249e2456aca6b08acee91`; both match the correction receipt. The corrected-summary SHA-256 is `d2e306516e82e20345213483f078ca52efa8801e6b7eac43c713982e94df070a`.

CA3 output: `VALID EER comparisons=77 max_abs_delta=0 mismatches=0 terminal_qa_sha_bound=true`. This preserves the existing exact 77-EER evidence without rerunning the broad numerical audit.

### adversarialCases

| scenario id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
|---|---|---|---|---|---|
| ADV-CA1 | F1 | stale aggregation metadata | Corrected global and experiment-specific fields must agree with the complete manifest, native reports, and sidecar; E8 must remain `fixed_updates=null` with its non-iterative fit kind. | PASS | R2, R3, R4, R5 |
| ADV-CA2 | F4 | correction-induced archive/notebook mutation | A metadata-only correction must leave the downloaded archive and executed notebook byte-identical to their receipt-bound hashes. | PASS | R3, R4, R5, R6, R7 |
| ADV-CA3 | F3 | numerical regression after metadata correction | Corrective metadata work must not alter the previously verified saved-prediction EERs. | PASS | R4, R8, R9 |

### artifactRefs

| id | kind | description | path |
|---|---|---|---|
| R1 | execution log | Fresh CA1 pytest invocation and output | `.omo/evidence/eight_h100_20260917/corrected_aggregation_regression_manual.log` |
| R2 | source/test | Meaningful corrected-summary regression test | `test_eight_corrected_aggregation.py` |
| R3 | corrected summary | Canonical post-run metadata summary, authoritative for populations and update semantics | `results/eight_h100_20260917/corrected_summary/final_aggregation.corrected.json` |
| R4 | receipt | Correction receipt, source-report hashes, immutable hashes, and EER binding | `results/eight_h100_20260917/corrected_summary/correction_receipt.json` |
| R5 | execution log | Fresh CA2 verifier invocation and output | `.omo/evidence/eight_h100_20260917/corrected_aggregation_integrity_manual.log` |
| R6 | archive | Immutable downloaded result archive; SHA-256 `681d41de10397d31a371daaa276a4a90d19f04789b85e01fdd6f936399dd23d2` | `results/eight_h100_20260917/downloaded/results.zip` |
| R7 | notebook | Immutable executed notebook; SHA-256 `bb28445e88a4d963aa7969fddba03e2c88badf4d969249e2456aca6b08acee91` | `results/eight_h100_20260917/downloaded/unpacked/output/DACON_Eight_H100_Executed.ipynb` |
| R8 | terminal QA | Existing 77-EER exact recomputation record: comparisons `77`, max delta `0`, mismatches `0` | `.omo/evidence/eight_h100_20260917/terminal_qa.json` |
| R9 | execution log | Fresh receipt-bound read-only CA3 cross-check | `.omo/evidence/eight_h100_20260917/terminal_qa_eer_binding_manual.log` |
