# Manual QA review — eight H100 run

Review commit: `2a0697edef92903f604cdb50f77ef807a940c5d0`

Overall verdict: **FAIL**. The H100 execution, archive, notebook, feature, plot, and E7 evidence pass the hands-on checks below. The exported `final_aggregation.json` is stale/internally inconsistent: it reports `component_count=178` for every experiment while the corresponding experiment reports and current `connected_components()` execution report `256`; it also reports `E8.fixed_updates=500` while `experiment_8.json` records `fixed_updates=null` and `seeds=[]`. This blocks an overall PASS until the aggregation sidecar is regenerated and rechecked.

## manualQa

### surfaceEvidence

| scenario id | criterion reference | surface | exact invocation | verdict | artifactRefs |
|---|---|---|---|---|---|
| M1 | F3/F4; export integrity | `results/eight_h100_20260917/downloaded/results.zip` and `EXPORT_SHA256.json` | `(cd results/eight_h100_20260917/downloaded && sha256sum -c results.zip.sha256)`; `.venv/bin/python - <<'PY'` using `zipfile.ZipFile.testzip()`, `EXPORT_SHA256.json` coverage, and SHA-256 of every declared member `PY` | PASS | A1, A2 |
| M2 | F3; notebook execution through final output | `DACON_Eight_H100_Executed.ipynb` | `.venv/bin/python - <<'PY'` loading the notebook JSON, enumerating code cells, execution counts, error outputs, and final output types `PY` | PASS | A3 |
| M3 | F3; feature identity, schema, and counts | extracted `output/features/*.npz` plus extraction receipts | `.venv/bin/python - <<'PY'` reading the 7,101-row manifest, loading every NPZ, checking identity `ID`/`audio_sha256`, `validate_feature_arrays()`, finite values, and summing window rows `PY` | PASS | A4, A5, A6 |
| M4 | F3; prediction evidence and plot | all `predictions_*.csv`, E7 JSON predictions, `prediction_evidence_counts.png` | `.venv/bin/python - <<'PY'` checking prediction identifiers, labels, finite/range-bounded score fields and row counts `PY`; `view_image` on `results/eight_h100_20260917/downloaded/unpacked/output/prediction_evidence_counts.png` | PASS | A7, A8 |
| M5 | F3; native execution evidence | H100 preflight, telemetry, and E7 native parity | `.venv/bin/python - <<'PY'` parsing `gpu_preflight.json`, `admission_E7.json`, and the four-column telemetry CSV for finite samples and nonzero memory `PY` | PASS | A9, A10, A11 |
| M6 | F3; E7 optimization connectivity and checkpoint reload | E7 admission/report, curves, and checkpoints | `.venv/bin/python - <<'PY'` loading all nine checkpoints with `torch.load(..., weights_only=True)`, checking archive hashes, finite tensors, 200-step curves, nonzero gradients, `reloaded_dev_logits_exact`, and trainable-arm flags `PY` | PASS | A12, A13, A14, A15 |
| M7 | F1/F3; final aggregation fidelity | `final_aggregation.json` versus each `experiment_N.json` | `.venv/bin/python - <<'PY'` comparing `component_count`, `fixed_updates`, and `seeds` for E1/E3/E4/E6/E8 `PY` | FAIL | A16, A17, A18 |
| M8 | F4; stopped workspace and restored bootstrap | run/stop/operational receipts | `.venv/bin/python - <<'PY'` cross-checking archive SHA/bytes/member count, `workspace_stopped`, `phase=standby`, and restored bootstrap SHA across receipts `PY`; `ps -p 58324 -o pid=,stat=,command=` and process scan for eight runner/watch commands | PASS | A19, A20, A21 |

Observed M1 values: archive SHA-256 `681d41de10397d31a371daaa276a4a90d19f04789b85e01fdd6f936399dd23d2`, 292,178,089 bytes, 7,314 declared members, ZIP member count 7,315 including `EXPORT_SHA256.json`, CRC test `None`, coverage equal, and zero member hash mismatches.

Observed M2 values: 19 total cells, 9 code cells, execution counts `1..9`, zero unexecuted code cells, zero error outputs; the last code cell has `display_data` and `execute_result` output.

Observed M3 values: 7,101 manifest rows and 7,101 feature files; zero missing/extra files, zero identity failures, zero schema failures, zero non-finite arrays, and 29,275 validated windows. Extraction receipt independently reports `records=7101`, `requested_windows=29275`, `processed_windows=29275`, `status=complete`.

Observed M4 values: CSV row counts are E1 music 127 + voice 48, E2 15,864, E3 music 508 + voice 192, E4 127, E5 648, E6 Anti 48 + MERT 127, E7 JSON predictions 10 × 600, and E8 291. The inspected chart is titled `Durable prediction evidence by experiment`, labels E1 through E8, and displays those evidence counts; it is treated as a row-count chart, not a quality result.

Observed M5 values: `gpu_preflight.json` reports one `NVIDIA H100 80GB HBM3`; telemetry has 457 finite samples, 455 samples with nonzero reported memory, utilization range 0–100, and memory range 0–12,979 MiB. E7 native parity reports three real audio cases and max logit error `2.956390380859375e-05`.

Observed M6 values: nine checkpoints load as 194 finite tensors each and all match `EXPORT_SHA256.json`; all nine curves contain steps 1–200 with 200 nonzero gradient norms. E7 admission reports `adapter_changed=true`, `frozen_probe_unchanged=true`, `nonzero_lora_b_gradients=96`, and `restored_logits_exact=true`; the report has `reloaded_dev_logits_exact=true` for all nine arms, with adapters trainable only for the six LoRA arms.

Observed M7 failure: `final_aggregation.json` reports component count 178 for E1–E8. The corresponding E1/E3/E4/E6/E8 reports each contain component count 256. The same comparison reports `final E8 fixed_updates=500` versus `experiment_8.json fixed_updates=null`; E8 has REAL-only fit and no optimizer seed schedule. Mismatches: E1/E3/E4/E6/E8 component count, plus E8 fixed updates.

Observed M8 values: archive SHA, bytes, and member count match across operational and downloaded run receipts; `resource_stop.json` reports `phase=standby` and `pause_verified=true`; the restored bootstrap SHA is `d027558a7774b4581c6ed1485ed9e94f8bec5b0b8fc88687aa5366e520b07a94`; no matching local runner/watcher process remained.

### adversarialCases

| scenario id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
|---|---|---|---|---|---|
| ADV1 | F3/F4; archive safety | one-byte compressed-member corruption | ZIP integrity checking must reject the mutated member | PASS | A1 |
| ADV2 | F3; notebook reliability | hidden notebook execution error | any code-cell `error` output or unexecuted code cell must fail the review | PASS | A3 |
| ADV3 | F3; feature cache integrity | stale/mismatched feature identity or invalid array contract | identity/audio hash, required profile keys, alignment, masks, and finite-value checks must reject invalid records | PASS | A4, A5, A6 |
| ADV4 | F3; checkpoint portability | corrupted or incompatible checkpoint state | every exported checkpoint must load, contain finite tensors, and match the declared member digest | PASS | A1, A12, A13 |
| ADV5 | F1/F3; metadata drift | stale aggregation summary | final aggregation fields must agree with the durable per-experiment reports and the source computation | FAIL | A16, A17, A18 |
| ADV6 | F3; numerical output integrity | non-finite/out-of-domain prediction output | prediction score fields must be finite; probability fields must stay in `[0,1]`; distance fields must be nonnegative | PASS | A7, A8 |

### artifactRefs

| id | kind | description | path |
|---|---|---|---|
| A1 | archive | Downloaded result archive and sidecar checksum | `results/eight_h100_20260917/downloaded/results.zip` |
| A2 | manifest | Per-member SHA-256 export manifest | `results/eight_h100_20260917/downloaded/unpacked/EXPORT_SHA256.json` |
| A3 | notebook | Executed notebook with final output | `results/eight_h100_20260917/downloaded/unpacked/output/DACON_Eight_H100_Executed.ipynb` |
| A4 | receipt | Full feature extraction count receipt | `results/eight_h100_20260917/downloaded/unpacked/output/extraction.json` |
| A5 | receipt | Role-specific feature preflight and identity receipts | `results/eight_h100_20260917/downloaded/unpacked/output/feature_preflight.json` |
| A6 | data receipt | Sealed manifest row/audio-hash receipt | `results/eight_h100_20260917/data_receipt.json` |
| A7 | predictions | E1–E8 CSV prediction exports | `results/eight_h100_20260917/downloaded/unpacked/output/predictions_*.csv` |
| A8 | image | Durable prediction evidence count plot | `results/eight_h100_20260917/downloaded/unpacked/output/prediction_evidence_counts.png` |
| A9 | receipt | Native GPU preflight | `results/eight_h100_20260917/downloaded/unpacked/output/gpu_preflight.json` |
| A10 | telemetry | H100 utilization/memory telemetry | `results/eight_h100_20260917/downloaded/unpacked/output/gpu_telemetry.live.csv` |
| A11 | receipt | E7 H100 admission, parity, and gradient rollback | `results/eight_h100_20260917/downloaded/unpacked/output/e7/admission_E7.json` |
| A12 | receipt | E7 training report and reload flags | `results/eight_h100_20260917/downloaded/unpacked/output/e7/report.json` |
| A13 | curves | E7 200-step gradient/loss curves | `results/eight_h100_20260917/downloaded/unpacked/output/e7/curves/` |
| A14 | checkpoints | Nine E7 serialized arm checkpoints | `results/eight_h100_20260917/downloaded/unpacked/output/e7/checkpoints/` |
| A15 | lock | E7 selection lock | `results/eight_h100_20260917/downloaded/unpacked/output/e7/selection_locked.json` |
| A16 | aggregate | Exported final aggregation with stale fields | `results/eight_h100_20260917/downloaded/unpacked/output/final_aggregation.json` |
| A17 | reports | Durable E1/E3/E4/E6/E8 reports used for comparison | `results/eight_h100_20260917/downloaded/unpacked/output/experiment_*.json` |
| A18 | source | Current connected-component and fixed-update construction | `eight_fixed_io.py`, `eight_ablation.py`, `build_eight_h100_notebook.py` |
| A19 | operational receipt | Archive/download/stop/bootstrap status | `results/eight_h100_20260917/terminal_operational_receipt.json` |
| A20 | stop receipt | Verified standby state | `results/eight_h100_20260917/resource_stop.json` |
| A21 | restoration receipt | Restored original bootstrap digest | `results/eight_h100_20260917/bootstrap_restored.json` |
