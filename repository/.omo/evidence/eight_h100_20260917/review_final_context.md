# Final bounded context and scope gate review

## recommendation

**APPROVE** at exact commit `3970207e336bae7b72c2c261a4c54129f57ad79c`.

## blockers

None.

## originalIntent

Execute all eight preregistered comparisons on one H100 in the existing VESSL workspace `wsp-9683432r72m7`; retain the original absolute deadline; export an executed notebook and durable per-file evidence; verify the download; return the workspace to stopped/standby; restore the original bootstrap; and report the result within the known provenance, label, speaker, and runtime limits. No L4 run, competition submission, official score, or broad generalization claim was authorized.

## desiredOutcome

A truthful final report and authoritative aggregation for E1-E8 that preserve the immutable GPU archive, distinguish global and experiment-specific provenance populations, describe E8 as a non-iterative REAL-only fit, retain all evidence limits, and document verified archive download and workspace shutdown.

## userOutcomeReview

The corrected deliverable resolves both blockers from `review_context.md`. `final_aggregation.corrected.json` explicitly supersedes the immutable sealed aggregation by its SHA-256 and is authoritative for dataset component count, experiment population metadata, and fixed-update semantics. It reports the full manifest provenance graph as 256 connected components, retains E2/E5 at 4 components and E7 at 10 speaker components, and reports E8 with `fixed_updates: null` and `fit_kind: non_iterative_closed_form_real_only_gaussian`. The correction receipt binds the corrected artifact, errata, raw E1-E8 reports, population sidecar, original archive, and executed notebook by SHA-256; it states that no numerical score, raw report, archive member, notebook, prediction, checkpoint, or fitted model changed.

`FINAL_REPORT.md` names the corrected aggregation as canonical and states that the archived aggregation's 178-component and E8 500-update fields are superseded. It does not claim an official DACON result, L4 validation, fresh independent holdout performance, or population generalization. It explicitly retains historical exposure, source-label confounding, unreviewed speech labels, unverified singing validity, and the two-speaker speech-test limitation. The E2 music/file addition is identified in the corrected aggregation as a verified CPU post-run supplement outside the immutable GPU archive.

Operational scope remains satisfied: `terminal_operational_receipt.json` preserves `started_epoch=1789648960` and `immutable_deadline_epoch=1789663360`; `resource_stop.json` records `phase=standby` and `pause_verified=true` for `wsp-9683432r72m7`; `bootstrap_restored.json` records restoration; and `download_verification.json` binds the 292,178,089-byte, 7,314-member archive with SHA-256 `681d41de10397d31a371daaa276a4a90d19f04789b85e01fdd6f936399dd23d2`.

The exact commit changes correction source, one high-signal regression test, reports/results, operational evidence, and plan/ledger records. It does not change the immutable downloaded archive or experiment raw reports. The unrelated pre-existing `.omo/evidence/v4-training/source-sha256.txt` modification remains outside commit `3970207e336bae7b72c2c261a4c54129f57ad79c` and was preserved.

## criterionDisposition

- `SC-ALL8`: PASS — durable reports mark E1-E8 complete; the correction does not fabricate or rerun results.
- `SC-REPORT-INTEGRITY`: PASS — global components are 256, experiment-native populations are retained, and E8 updates are null.
- `SC-EVIDENCE-BOUNDARY`: PASS — official score is null; robustness/fresh-holdout claims are false; historical, unreviewed-label, source-confounding, and two-test-speaker limits remain explicit.
- `SC-OPERATIONS`: PASS — the existing workspace, original deadline, verified archive download, standby state, and restored bootstrap are evidenced.
- `SC-SCOPE`: PASS — the correction is metadata/report/source-only with one targeted regression check; no GPU/API operation, external communication, new workspace, L4 run, submission, or unrelated v4 inclusion occurred in the reviewed commit.

## directProgrammingAndRemoveAiSlopsPass

I directly reviewed the production diff and `test_eight_corrected_aggregation.py` under the `programming`, `remove-ai-slops`, Ponytail, and repository practical-decision-value criteria. The test protects the concrete prior regression against the actual manifest, immutable raw reports, correction sidecar, and source hashes. It verifies population semantics, supersession identity, E8 non-iterative semantics, and archive immutability. It is not a deletion-only, removal-only, prompt-text, tautological, implementation-mirroring, shape-only, or existence-only test. The production change removes a faulty local DSU implementation by reusing `eight_fixed_io.connected_components` and reads experiment metadata directly rather than adding a new abstraction. No unnecessary parser, normalizer, extraction, dependency, or speculative production layer was added.

The existing `review_code.md` explicitly records a `programming` and `remove-ai-slops` pass covering deletion-only, removal-only, tautological, prompt-text, implementation-constant, untyped-escape, and unnecessary parsing/normalization classes. Its FAIL verdict applies to the earlier commit `2a0697edef92903f604cdb50f77ef807a940c5d0`; the exact reviewed correction commit resolves its sole blocker without modifying the raw numerical evidence.

## reproducedEvidence

- `.venv/bin/python -m pytest -q test_eight_corrected_aggregation.py` — `1 passed in 0.69s`.
- `.venv/bin/python .omo/evidence/eight_h100_20260917/verify_corrected_aggregation.py` — `VALID global=256 sealed=178 E8=null archive_and_notebook_immutable=true`.
- Corrected aggregation SHA-256: `d2e306516e82e20345213483f078ca52efa8801e6b7eac43c713982e94df070a`.
- Errata SHA-256: `637357229575aa5ca1d29691200e7bb690e54f1a66c8b0249c03ed26af977993`.
- Final report SHA-256: `f5f17283493c015e3e0ce4a00b1b4adff4ad6038474eae2caee699e602c744ee`.
- Terminal QA SHA-256: `a52194a1386f54ba74bbe4a8c0b0fd00c7ae83b36b05606bccdcd8882e73a6b1`.

## checkedArtifactPaths

- `.omo/plans/eight-h100-20260917.md`
- `.omo/evidence/eight_h100_20260917/review_context.md`
- `.omo/evidence/eight_h100_20260917/review_code.md`
- `.omo/evidence/eight_h100_20260917/review_goal.md`
- `.omo/evidence/eight_h100_20260917/review_manual.md`
- `.omo/evidence/eight_h100_20260917/terminal_qa.md`
- `.omo/evidence/eight_h100_20260917/terminal_qa.json`
- `.omo/evidence/eight_h100_20260917/corrected_aggregation_integrity.log`
- `.omo/evidence/eight_h100_20260917/corrected_aggregation_regression.log`
- `.omo/evidence/eight_h100_20260917/verify_corrected_aggregation.py`
- `results/eight_h100_20260917/FINAL_REPORT.md`
- `results/eight_h100_20260917/corrected_summary/final_aggregation.corrected.json`
- `results/eight_h100_20260917/corrected_summary/correction_receipt.json`
- `results/eight_h100_20260917/corrected_summary/executed_notebook_errata.md`
- `results/eight_h100_20260917/postrun_e2/fixed_experiment_population_denominators.json`
- `results/eight_h100_20260917/terminal_operational_receipt.json`
- `results/eight_h100_20260917/download_verification.json`
- `results/eight_h100_20260917/resource_stop.json`
- `results/eight_h100_20260917/bootstrap_restored.json`
- `build_eight_h100_notebook.py`
- `test_eight_corrected_aggregation.py`
- Git delta for `3970207e336bae7b72c2c261a4c54129f57ad79c^..3970207e336bae7b72c2c261a4c54129f57ad79c`

## exactEvidenceGaps

- No live VESSL API re-query was performed in this bounded read-only review. The success criterion requires verified shutdown evidence, which is present in the saved stop and operational receipts; live re-query was excluded by the assignment.
- The immutable executed notebook still contains the original stale aggregation output. This is intentionally preserved and is explicitly superseded by the SHA-bound corrected aggregation and errata, satisfying the prior approval condition without mutating the verified archive.
- Earlier review artifacts retain FAIL verdicts for commit `2a0697edef92903f604cdb50f77ef807a940c5d0`. They are historical evidence of the defects, not current verdicts for exact commit `3970207e336bae7b72c2c261a4c54129f57ad79c`.
