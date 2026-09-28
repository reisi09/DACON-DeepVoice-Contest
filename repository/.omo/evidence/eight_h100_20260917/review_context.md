# Context and requirements review — eight H100 run

## Verdict

**FAIL** at exact HEAD `2a0697edef92903f604cdb50f77ef807a940c5d0`.

The underlying archive contains completed measured artifacts for E1–E8, and the operational receipts support use of the existing `wsp-9683432r72m7` H100 x1 workspace, archive verification, return to `standby`, and bootstrap restoration. The user-facing notebook summary artifact is nevertheless internally false: `final_aggregation.json` reports the wrong connected-component count for every experiment and reports 500 fixed updates for E8, which is a non-iterative REAL-only fit. A later unsealed sidecar corrects population denominators but does not correct the executed notebook/archive summary. Completion therefore cannot be presented as a precise final result yet.

## Original intent and requirements checked

- Execute the precisely preregistered E1–E8 comparisons on H100 x1 in the existing workspace only.
- Do not perform or claim an L4 run or competition submission; future submission remains outside authorization.
- Preserve the original absolute deadline across retries.
- Preserve source-versus-derivative provenance and expose historical reuse, unreviewed annotations, source confounding, and limited-component uncertainty.
- Deliver an executed notebook and durable per-file evidence whose final aggregation truthfully describes arms, populations, labels, seeds, steps, limitations, and selection locks.
- Verify the downloaded archive, stop compute, restore the original bootstrap, and leave no unintended task process.

## Evidence inspected

- `.omo/plans/eight-h100-20260917.md`
- `AGENTS.md`
- Git HEAD and scoped history: `2a0697e`, `2fa9bd3`, `23c4ebb`, `bb98205`, `94fe82c`
- `.omo/evidence/eight_h100_20260917/actual_manifest_admission.md`
- `.omo/evidence/eight_h100_20260917/actual_manifest_admission_counts.json`
- `results/eight_h100_20260917/downloaded/unpacked/output/final_aggregation.json`
- `results/eight_h100_20260917/downloaded/unpacked/output/experiment_1.json` through `experiment_8.json`
- `results/eight_h100_20260917/downloaded/unpacked/output/e7/report.json`
- `results/eight_h100_20260917/postrun_e2/summary.md`
- `results/eight_h100_20260917/postrun_e2/verification_receipt.json`
- `results/eight_h100_20260917/postrun_e2/fixed_experiment_population_denominators.json`
- `results/eight_h100_20260917/download_verification.json`
- `results/eight_h100_20260917/resource_stop.json`
- `results/eight_h100_20260917/bootstrap_restored.json`
- `results/eight_h100_20260917/downloaded/run_state.json`

## Passing requirements

1. **All eight experiment IDs have measured artifacts.** `experiment_1.json`–`experiment_8.json` report `status=complete`; E7 has its separate native report, checkpoints, curves, predictions, selection lock, teacher binding, and admission receipt.
2. **The execution resource and cleanup match scope.** `resource_stop.json` identifies `wsp-9683432r72m7`, `phase=standby`, and `pause_verified=true`; `bootstrap_restored.json` reports `restored=true`; `download_verification.json` verifies 7,314 members and archive SHA-256 `681d41de10397d31a371daaa276a4a90d19f04789b85e01fdd6f936399dd23d2`.
3. **No L4/submission claim was found in the reviewed final artifacts.** `final_aggregation.json` keeps `official_score` null and limits robustness claims; the plan explicitly excludes L4 and submission.
4. **Source and derivative limits are substantially recorded.** The actual-manifest audit independently resolves 256 provenance components with zero split collisions. Experiment claim limits and the E2 post-run report explicitly retain historical/source confounding, unreviewed backing uncertainty, separator effects, constructed-diagnostic scope, and null CIs for one-component slices.
5. **E2's preregistered music/file supplement exists as a verified post-run analysis.** Its receipt binds the input manifest, frozen checkpoints, lock-before-prediction ordering, output hashes, and unchanged inputs. It also declines a causal fake-backing claim because real-backing controls are large.

## Blocking findings

### B1 — Final aggregation reports false provenance-component counts

**Violated requirement:** The shared result contract and success criterion require the sealed final aggregation to report the actual population and provenance-aware uncertainty for every experiment; completion claims must be precise and must not disguise invalid population metadata.

**Evidence:** `actual_manifest_admission.md` and `actual_manifest_admission_counts.json` independently compute **256** global connected components. The primary `experiment_1.json`, `experiment_3.json`, `experiment_4.json`, `experiment_6.json`, and `experiment_8.json` also report `component_count=256`; E2 and E5 correctly report their probe-slice counts as 4. In contrast, `final_aggregation.json` reports `component_count=178` for **E1–E8**, including E2 and E5. The post-run `fixed_experiment_population_denominators.json` acknowledges that top-level generic metadata was wrong and supplies role-specific corrections, but it is outside the verified archive and does not repair the executed notebook's final summary.

**Required correction:** Regenerate a corrected final aggregation/notebook summary from the verified primary reports and role-specific population sidecar, clearly distinguishing global manifest components from experiment/slice-specific components. Seal and hash the corrected summary as a post-run correction without altering the original archive.

### B2 — Final aggregation falsely describes E8 as a 500-update experiment

**Violated requirement:** Every final result must accurately report fixed-step counts and the actual arm; E8 is specified as a REAL-only Gaussian/distance fit, not a trained 500-step head.

**Evidence:** `experiment_8.json` has `fixed_updates=null`, while `final_aggregation.json` states `fixed_updates=500` for E8. The aggregation's own `class_exposure="REAL-only fit"` and `estimand="REAL-only distance behavior"` contradict that 500-update value.

**Required correction:** Set E8 update metadata to `null` or an explicit non-iterative/not-applicable value in the corrected sealed aggregation, and preserve the distinction from E1/E3/E4/E6 head training and E7's 200 updates.

## Concrete missed or incomplete requirements

- The plan's final verification items F1–F4 remain unchecked in the plan, and no final sealed compliance artifact reconciles the post-run correction with the executed notebook summary.
- The corrected denominator/component sidecar is not part of `download_verification.json`'s verified 7,314 archive members. That is acceptable as a non-destructive post-run correction only if the final deliverable explicitly names it as authoritative, hashes it, and stops presenting the stale aggregation as authoritative.
- The final aggregation flattens role-specific populations into one generic component count. This erases the required distinction between global provenance components, experiment populations, and one-component speech slices even though the primary reports and sidecar contain that distinction.
- E2's music/file analysis was completed after the GPU archive. The final deliverable must explicitly identify it as a verified CPU post-run supplement over frozen archived features/checkpoints, rather than imply it was inside the executed notebook archive.

## Slop and overfit review

Direct review under `remove-ai-slops`, `programming`, Ponytail, and the repository test-value rule found no new test-count or coverage claim that should substitute for artifact evidence. The reviewed checks target real failure modes: provenance leakage, selector population, cache identity, archive corruption, deadline preservation, adapter gradients/checkpoint changes, and packaged-output integrity. No blocker is based on style or architecture preference. The blocking defects are report-integrity failures tied directly to the stated final-result contract.

## Approval condition

Approval needs a sealed, hash-addressed corrected final summary/notebook addendum that reconciles the primary experiment reports and post-run E2/denominator artifacts, reports correct role-specific component counts, marks E8 updates as not applicable, and keeps the original archive immutable. No new GPU run, L4 run, submission, or archive mutation is required.
