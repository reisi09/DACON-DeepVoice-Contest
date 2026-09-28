# V4 current manual QA review

검토 대상 source commit은 `d2397ad8cc96667e5bc2e4132d86405afe093f4e`이며, detached checkout `/tmp/dacon-v4-review-d2397ad`의 HEAD와 일치한다. 현재 분석 산출물은 `.omo/evidence/v4-results-analysis.ipynb` 301387 bytes, SHA-256 `d050883fb71fe6572879deb9d85da5dc7f006e3f7979761b40a7e582855c0418`이다.

`QA_VERDICT: PASS` — source closure, 실제 P1/P3 terminal receipts, paired initialization, 10 epoch data, CPU analysis execution, G0 fallback, budget release, and UI zero-session evidence were independently checked. `CANDIDATE_RELEASE_VERDICT: NOT_RELEASED` — fixed gates selected `G0` with `no_candidate_passed`; this is the required no-improvement branch, not evidence of a trained candidate release.

GPU training notebook capture itself is not treated as a complete current-cell execution: `v4-matched-notebook_capture.json` records `current_cell_may_be_incomplete=true`. Training completion is established by the terminal receipts, and the separate CPU analysis notebook is fully executed with eight code cells and zero notebook errors.

## manualQa

### surfaceEvidence

| scenario id | criterion reference | surface | exact invocation | verdict | artifactRefs |
|---|---|---|---|---|---|
| V4-CUR-S01 | exact source closure | immutable detached checkout + matched runtime manifest | `python3 - <<'PY'` read `.omo/evidence/v4-matched-source_hashes.json`, SHA-256 hashed each of its 14 entries under `/tmp/dacon-v4-review-d2397ad`, asserted `len(manifest)==14` and no mismatch; `git -C /tmp/dacon-v4-review-d2397ad rev-parse HEAD` | PASS | A1, A2 |
| V4-CUR-S02 | actual training completion and epoch schema | `.omo/evidence/v4-matched-P1_terminal.json`, `.omo/evidence/v4-matched-P3_terminal.json` | `python3 - <<'PY'` loaded both `result` objects and asserted `status=completed`, `epochs_completed=10`, `completed_epoch_updates=350`, `completed_epoch_window_exposures=11160`, ten reports with epochs 1..10, per-epoch exposures 1116..11160, and optimizer steps 35..350 | PASS | A3, A4 |
| V4-CUR-S03 | fair matched initialization and equal work | P1/P3 trial receipts and matched comparison | `python3 - <<'PY'` compared P1/P3 `seed`, `seed_rng_sha256`, `initial_weights_sha256`, and `source_contract_hash`; then checked `matched_complete=true`, both arms 10 epochs/350 updates/11160 exposures | PASS | A5, A6 |
| V4-CUR-S04 | same-epoch numerical comparison | terminal epoch reports + `v4-results-summary.json` | `python3 - <<'PY'` read each terminal receipt's epoch-10 `FILE_EER`, `MUSIC_EER`, `seen_D`, `ood_D`, and `rank`, computed P3−P1, and asserted equality with `matched_recheck.epoch10_P3_minus_P1` | PASS | A3, A7 |
| V4-CUR-S05 | unknown-label and score boundary | matched terminal evaluation records | `python3 - <<'PY'` traversed all 20 matched epoch records and asserted `score is None`, `voice_eer unknown_counts=110`, and `voice_presence_auc unknown_counts=90` | PASS | A3, A7 |
| V4-CUR-S06 | final selection, G0 fallback, no refit/sealed/submission | current CPU summary and original final selection receipts | `python3 - <<'PY'` asserted `selection={reason:no_candidate_passed, selected:G0}`, `new_training_performed=false`, `refit_performed=false`, `sealed_accessed=false`, and `overall_score is None` | PASS (safe fallback) | A7, A8 |
| V4-CUR-S07 | analysis artifact execution and provenance | `.omo/evidence/v4-results-analysis.ipynb` | `python3 - <<'PY'` parsed the notebook, asserted 21 cells, execution counts 1..8, and zero `output_type=error`; `shasum -a 256 .omo/evidence/v4-results-analysis.ipynb` recorded the exact artifact hash | PASS | A9, A10 |
| V4-CUR-S08 | original allocation ceiling and release | budget receipt + Colab release UI | `python3 - <<'PY'` asserted release `1789290093.9671278 < safe_deadline 1789290385` and elapsed from allocation origin `1789283305` within 7200 seconds; `rg -n '사용 가능|사용률|활성 세션|연결되어 있지 않습니다|세션 관리' .omo/evidence/v4-colab-ui/v4-matched-recheck-final-resources-zero-session-accessibility.txt` | PASS | A11, A12, A13 |
| V4-CUR-S09 | analysis artifact Drive persistence and raw-byte integrity | Drive analysis folder readback | `python3 - <<'PY'` loaded `.omo/evidence/v4-results-drive-readback.json`, asserted 8/8 `drive_raw_readback_match=true`, and recomputed local hashes for the notebook, summary, and `V4_analysis_replay_bundle.zip` | PASS | A9, A15, A16 |

### adversarialCases

| scenario id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
|---|---|---|---|---|---|
| V4-CUR-A01 | exact source binding | stale or substituted source | A source mismatch must be detectable; all 14 runtime hashes must match the requested immutable checkout | PASS — 14/14 match at `d2397ad8cc96667e5bc2e4132d86405afe093f4e` | A1, A2, A5 |
| V4-CUR-A02 | real execution | partial, fabricated, or stale terminal claim | Each arm must have a terminal completed status, ten epoch records, 350 updates, 11160 exposures, and receipt-bound output paths | PASS — P1 and P3 satisfy all checks | A3, A4 |
| V4-CUR-A03 | fair comparison | independent initialization mismatch | A causal comparison requires common seed/RNG/initial-weight evidence and equal work | PASS — common seed/RNG/initial weights and equal updates/exposures recorded | A4, A5, A6 |
| V4-CUR-A04 | epoch comparison | comparing different selected epochs as treatment effect | The treatment delta must be computed at the same epoch, with selected-checkpoint updates kept distinct | PASS — epoch 10 P3−P1 is `FILE_EER 0.000000`, `MUSIC_EER +0.012143`, `seen_D +0.037500`, `OOD_D 0.000000`, `rank 0.000000` | A3, A7, A9 |
| V4-CUR-A05 | label integrity | unknown labels converted into a synthetic official score | Unknown counts and partial metrics must remain explicit; aggregate score must remain null | PASS — unknown voice labels remain 110/90 and score is `null` | A3, A7 |
| V4-CUR-A06 | candidate gate safety | failed candidate promoted over baseline | `no_candidate_passed` must retain G0 and avoid claiming a candidate release | PASS — final selection is `G0`; trained candidate release is not claimed | A8 |
| V4-CUR-A07 | promotion boundary | refit, sealed evaluation, or submission after rejection | Rejection must leave refit, sealed access, and submission path inactive | PASS — `refit_performed=false`, `sealed_accessed=false`, `new_training_performed=false`; no submission claim | A7, A8 |
| V4-CUR-A08 | artifact honesty | incomplete GPU notebook capture presented as fully executed | An incomplete capture must not support a complete GPU notebook claim; a separately executed analysis artifact may support analysis only | PASS — capture is explicitly marked incomplete; CPU analysis has eight executed code cells and zero errors | A9, A14 |
| V4-CUR-A09 | resource cleanup | leaked Colab session or late release | Release must be requested before the safe deadline and UI must show zero active sessions/disconnected runtime | PASS — release `1789290093.9671278`; UI shows 47.54 compute units, approximately 0/hour, 0 active sessions, disconnected | A11, A12, A13 |
| V4-CUR-A10 | analysis artifact persistence | Drive upload altered or truncated an analysis artifact | Raw-byte readback must match the local notebook, report inputs, plots, and reproducibility bundle | PASS — 8/8 Drive readback items match, including notebook and 1811815-byte ZIP | A9, A15, A16 |

## artifactRefs

| id | kind | description | path |
|---|---|---|---|
| A1 | source manifest | 14 matched runtime source SHA-256 entries | `.omo/evidence/v4-matched-source_hashes.json` |
| A2 | dispatch receipt | Matched run provenance binding the requested source commit to the run and 14-source verification | `.omo/evidence/v4-matched-recheck-dispatch.json` (checkout: `/tmp/dacon-v4-review-d2397ad`) |
| A3 | terminal receipt | P1 completed terminal result with ten epoch records, metrics, checkpoint and exposure receipts | `.omo/evidence/v4-matched-P1_terminal.json` |
| A4 | terminal receipt | P3 completed terminal result with ten epoch records, metrics, checkpoint and exposure receipts | `.omo/evidence/v4-matched-P3_terminal.json` |
| A5 | trial receipts | Shared seed, RNG state, initial weights, source contract, and arm-specific trial contracts | `.omo/evidence/v4-matched-P1_trial_receipt.json`, `.omo/evidence/v4-matched-P3_trial_receipt.json` |
| A6 | matched comparison | `matched_complete`, equal run work, selected checkpoints, and legacy P3 exclusion | `.omo/evidence/v4-matched-comparison.json` |
| A7 | analysis summary | Current numerical comparison, null overall score, G0 selection, no refit, no sealed access, and source commits | `.omo/evidence/v4-results-summary.json` |
| A8 | original selection receipt | Actual GPU run outcome: `final=G0`, `reason=no_candidate_passed`, `sealed_opened=false` | `.omo/evidence/v4-retry-finished.json`, `.omo/evidence/v4-retry-selection.json` |
| A9 | executed analysis artifact | CPU analysis replay with embedded tables/plots; 301387 bytes, SHA-256 `d050883fb71fe6572879deb9d85da5dc7f006e3f7979761b40a7e582855c0418` | `.omo/evidence/v4-results-analysis.ipynb` |
| A10 | analysis verification receipt | Builder invocation, artifact hashes, eight executed code cells, zero notebook errors, matched observables | `.omo/evidence/v4-results-verification.json` |
| A11 | runtime budget receipt | Allocation origin, safe deadline, and matched release request | `.omo/evidence/v4-matched-budget.json`, `.omo/evidence/v4-matched-release_requested.json` |
| A12 | browser accessibility receipt | Colab resources panel showing available units, usage rate, zero active sessions, and disconnected runtime | `.omo/evidence/v4-colab-ui/v4-matched-recheck-final-resources-zero-session-accessibility.txt` |
| A13 | browser screenshot | Redacted Colab resources panel; 52563 bytes, SHA-256 `f014bf1b0214150f0f94bc67f7730c640f96fec483576a8d5d090571fdc5eb95` | `.omo/evidence/v4-colab-ui/redacted/v4-matched-recheck-final-resources-zero-session.jpg` |
| A14 | capture limitation receipt | GPU notebook capture is non-empty but explicitly may be incomplete | `.omo/evidence/v4-matched-notebook_capture.json` |
| A15 | Drive readback receipt | Eight analysis files, plots and ZIP all have `drive_raw_readback_match=true` | `.omo/evidence/v4-results-drive-readback.json` |
| A16 | reproducibility bundle | CPU analysis replay bundle; 1811815 bytes, SHA-256 `70f8e598d67627b9790f5d06ce4bd4c02860a4a432271d6d9bed0e1a925b2794` | `_workspace/v4_results_report/V4_analysis_replay_bundle.zip` |

## Decision

The current V4 QA evidence is complete and internally consistent for the requested source and matched recheck. The immutable source is exactly `d2397ad8cc96667e5bc2e4132d86405afe093f4e`; the analysis artifact is exactly `d050883fb71fe6572879deb9d85da5dc7f006e3f7979761b40a7e582855c0418`. P1 and P3 both completed ten epochs with 350 updates and 11160 exposures under shared initialization receipts. At the same epoch 10, P3's OOD D and rank equal P1's, while MUSIC EER is 0.012143 higher and seen D is 0.037500 higher. Unknown labels prevent an official five-head aggregate score, so the score remains `null`.

The fixed first-run gates selected G0 with `no_candidate_passed`. No full refit, sealed evaluation, or submission was performed. The matched recheck is an analysis result and does not automatically replace G0. Release was requested before the safe deadline, and the final UI artifact shows zero active sessions and a disconnected runtime.
