# V4 final manual QA review

`sourceSHA`: `a01a50e0c774f181c20238109127c5b81cc95dc1`

`EXACT_SOURCE_SHA_VERDICT: PASS` for notebook/source byte closure. The actual bounded Drive run is terminal, with `EXPERIMENT_CANDIDATE_RELEASE_VERDICT: FAIL` because selection retained G0 (`no_candidate_passed`). The safe G0 fallback itself passed. The legacy P1/P3 causal comparison is `FAIL` as a reproducibility claim: the pinned runner declares `SEED` but does not initialize Python, NumPy, or Torch RNGs before independent trials. The later seed-fix work is a separate source/run and is excluded from this exact SHA verdict.

The audit was read/test only. No Colab UI action, GPU allocation, production mutation, or sealed child/data fetch was performed. Drive reads were limited to the provided run folder and discovered non-sealed children.

## manualQa

### surfaceEvidence

| scenario id | criterion reference | surface | exact invocation | verdict | artifactRefs |
|---|---|---|---|---|---|
| V4-S-01 | source byte closure | pinned checkout + notebook | `python3` read-only audit parsed `DACON_236749_trainable_v4.ipynb`, compiled each `%%writefile`, and byte-compared all payloads with `/tmp/dacon-v4-review-a01a50e` at `a01a50e0c774f181c20238109127c5b81cc95dc1` | PASS | A1, A2 |
| V4-S-02 | source receipt binding | allowed Drive root | `google_drive_list_folder(url=provided root, top_k=100)` then `google_drive_fetch(source_hashes.json, download_raw_file=true)` | PASS | A1, A3 |
| V4-R-01 | actual G0 coverage | Drive run `20260913_074847` | `google_drive_fetch(pipeline_status.json, download_raw_file=true)` | PASS | A4 |
| V4-R-02 | real training/update completion | Drive training outputs | read local copies of terminal receipts for P1/P2/P3/P4 and `epoch_metrics.csv`; P1=175, P2=350, P3=175, P4=1395 total updates | PASS | A5, A6 |
| V4-R-03 | exposure/coverage integrity | Drive run receipts | `google_drive_fetch(p4_comparison.json, epoch_metrics.csv, drive_receipt.json, download_raw_file=true)` | PASS | A6, A7 |
| V4-R-04 | gradient/update proof | actual profile and training path | `google_drive_fetch(throughput_profile.json, download_raw_file=true)`; profile records finite gradient norms and non-zero absolute parameter updates for real SONICS/AST updates | PASS | A8 |
| V4-R-05 | selection and release behavior | Drive run root | `google_drive_fetch(selection.json, finished.json, release_requested.json, download_raw_file=true)` | PASS (safe fallback); candidate release FAIL | A9, A10, A11 |
| V4-R-06 | executed notebook preservation | Drive run root | `google_drive_fetch(executed_notebook_capture.json, executed.ipynb, drive_receipt.json, download_raw_file=true)` | FAIL for complete-execution claim; capture is non-empty but says `current_cell_may_be_incomplete=true` | A7, A12 |
| V4-R-07 | runtime shutdown evidence | Drive run root + UI receipt | read `release_requested.json` and the saved zero-session UI accessibility/screenshot receipt | PASS | A11, A13 |
| V4-R-08 | unknown-label handling | selection/evaluation outputs | read absolute evaluation fields in `v4-retry-selection.json` and terminal receipts; voice unknown counts remain explicit and `score=null` | PASS | A9, A14 |

### adversarialCases

| scenario id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
|---|---|---|---|---|---|
| V4-A-01 | exact source binding | stale or substituted notebook/source | notebook hash and every embedded source payload must match the pinned checkout and runtime source receipt | PASS | A1, A2, A3 |
| V4-A-02 | training completion | partial or fabricated run | each claimed pipeline must have a terminal status, update count, epoch records, and saved checkpoint evidence | PASS | A4, A5, A6 |
| V4-A-03 | gradient connectivity | no-op optimizer path | real profile update must show finite gradient norm and non-zero absolute parameter update | PASS | A8 |
| V4-A-04 | matched P4 exposure | unequal treatment/control exposure | treatment and control must have identical optimizer updates and window exposures before comparing objectives | PASS | A6 |
| V4-A-05 | candidate gate safety | candidate fails OOD/resource/quality gates | retain G0 and record `no_candidate_passed`; do not claim candidate release | PASS | A9, A10 |
| V4-A-06 | unknown labels | incomplete label coverage | report partial metrics and null aggregate score; never synthesize an overall score | PASS | A9, A14 |
| V4-A-07 | causal reproducibility | independent P1/P3 initialization | common RNG initialization or an equivalent matched-seed receipt is required for a causal comparison | FAIL: pinned `train_v4.py` only declares `SEED` at lines 32 and never seeds `random`, NumPy, or Torch before `execute` at lines 188-198; checkpoint RNG restore does not make independent starts common | A15, A5 |
| V4-A-08 | executed artifact integrity | incomplete notebook capture | a completion claim requires the saved executed notebook to represent a complete current cell | FAIL: `executed_notebook_capture.json` records `current_cell_may_be_incomplete=true` | A12 |
| V4-A-09 | sealed evaluation boundary | premature sealed promotion | when no candidate passes, sealed evaluation must remain unopened and final must remain G0 | PASS for observed outcome: `finished.json` has `sealed_opened=false` and final `G0`; the root budget log does show pre-run `v4_sealed.zip` archive verification, so this is not evidence of a strict archive-staging no-touch boundary | A10, A16 |
| V4-A-10 | release cleanup | leaked Colab session | after terminal completion the release request and zero-session UI receipt must both exist | PASS | A11, A13 |

## artifactRefs

| id | kind | description | path |
|---|---|---|---|
| A1 | static receipt | Independent notebook hash, 19/19 embedded source byte matches, 19/19 compile results, and control-flow checks | `.omo/evidence/v4-final-qa-source-closure.txt` |
| A2 | source | Pinned notebook and source checkout audited at the exact commit | `/tmp/dacon-v4-review-a01a50e/DACON_236749_trainable_v4.ipynb` |
| A3 | Drive receipt | Root `source_hashes.json` cross-check; Drive file id `1A-io0C5EkV_GTAvZJWdQ81mjDNqdHy5K` | `.omo/evidence/v4-final-qa-source-closure.txt` |
| A4 | Drive receipt | `pipeline_status.json`, Drive file id `1ueh2fZFFZOb-J0n0MGPnxWW5IKQVPnNX` | `.omo/evidence/v4-retry-pipeline_status.json` |
| A5 | terminal receipts | P1/P2/P3/P4/P1_continued terminal results with checkpoint hashes and update/exposure totals | `.omo/evidence/v4-retry-P1-terminal.json`, `.omo/evidence/v4-retry-P2-terminal.json`, `.omo/evidence/v4-retry-P3-terminal.json`, `.omo/evidence/v4-retry-P4-terminal.json`, `.omo/evidence/v4-retry-P1_continued-terminal.json` |
| A6 | Drive/runtime receipt | `epoch_metrics.csv` and `p4_comparison.json` with real update and exposure equality | `.omo/evidence/v4-final-qa-runtime-receipt.txt` |
| A7 | Drive receipt | `drive_receipt.json` hashes and `epoch_metrics.csv`; Drive ids `1pUVgt6lES-1Ut94lH3jNvgtIRmOhYiUM`, `116h4mxc-tILseYr7c7rBzyJU2fF_pOCj` | `.omo/evidence/v4-final-qa-runtime-receipt.txt` |
| A8 | profile receipt | Real SONICS/AST gradient norm, absolute update, inference, and parameter-count measurements | `.omo/evidence/v4-retry-throughput_profile.json` |
| A9 | selection receipt | `selection.json` records candidate gate outcomes, unknown counts, null aggregate score, and `reason=no_candidate_passed` | `.omo/evidence/v4-retry-selection.json` |
| A10 | terminal receipt | `finished.json` records `final=G0`, `sealed_opened=false`, and measured GPU metrics | `.omo/evidence/v4-retry-finished.json` |
| A11 | shutdown receipt | `release_requested.json` with elapsed time and completion timestamp | `.omo/evidence/v4-retry-release_requested.json` |
| A12 | executed artifact | Saved executed notebook and capture metadata; Drive ids `13hl-PkyK1I4LwmLnV1d3IOLSISCOwOk5`, `1iLrG4S3mVa5Y8RjSLl_UciHGNM-6h9hR` | `.omo/evidence/v4-final-qa-runtime-receipt.txt` |
| A13 | UI receipt | Saved zero-session release screenshot/accessibility transcript from the exclusive UI lane | `.omo/evidence/v4-colab-ui/v4-retry-release-zero-sessions-accessibility.txt` |
| A14 | evaluation receipt | Initial metrics and selection evidence retain unknown labels and null overall score | `.omo/evidence/v4-retry-initial-metrics.md`, `.omo/evidence/v4-retry-selection.json` |
| A15 | source limitation | Exact line-level proof of declared-but-unused `SEED` and independent P1/P3 execution | `/tmp/dacon-v4-review-a01a50e/train_v4.py:32`, `/tmp/dacon-v4-review-a01a50e/train_v4.py:188` |
| A16 | budget receipt | Actual event chronology including `retained_g0`, `runner_finished`, and pre-run archive verification | `.omo/evidence/v4-final-qa-runtime-receipt.txt` |

## Decision

The pinned notebook and its 19 embedded source files are closed against `a01a50e0c774f181c20238109127c5b81cc95dc1`, and the original Drive run has real G0, probe, training, exposure, gate, fallback, and shutdown evidence. The run does not support a released trained candidate: all four candidates failed selection gates and G0 was retained. The P1/P3 legacy causal comparison must be treated as invalid until the separate matched seed-fix recheck is completed. The saved executed notebook is preserved but cannot support a complete current-cell execution claim because its own capture receipt marks it potentially incomplete.
