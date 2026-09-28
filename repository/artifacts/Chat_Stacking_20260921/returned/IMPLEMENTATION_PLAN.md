# Stacking implementation plan

Goal: replace FILE_FAKE_PROB only, using current frozen experts; return CPU-verifiable code rather than a trained-performance claim.
Authority: input 00/02/03/04, current submission snapshot, then current training snapshot. User explicitly requests continuous execution without approval gates or Work/remote tasks.

- [x] Audit current numerical functions, checkpoints, cache identities, diagnostics, exposure registry, timebases.
- [x] Add numerical-source-preserving CPU adapter, strict feature contracts and overlap alignment; test actual six heads and ten caches.
- [x] Add connected-parent/source audit and original-first split planning; reject stale/unknown downstream exposure and cross-split relations.
- [x] Implement small-grid nested grouped meta-CV, train-only preprocessing and XGBoost inner early stopping; keep diagnostic/synthetic data quarantined.
- [x] Implement safe persistence, candidate lock and separate OOD command, file-only predictions and extended cache identity.
- [x] Patch current Engine with opt-in telemetry and a separate GPU extraction/file inference entrypoint; default four component outputs remain unchanged.
- [x] Execute meaningful CPU tests and baseline replay; document failures/fixes and NOT_RUN external work; hash and package.

Architecture: a small isolated Python package under tools/stacking; one opt-in current-runtime patch. Numerical definitions and scorer are extracted/copied verbatim from hash-verified current sources, avoiding unrelated transformer imports in CPU mode. No reference/legacy runtime is imported.

Decisions: require finite complete feature sets rather than impute unmeasured values; use 4-feature F0 and ~32-feature F1. F2 is separately enabled. LR/ExtraTrees use skops with a fixed type policy; XGBoost uses its own JSON model format. No pickle deployment. Parent-equal weighting is explicit and sensitivity to row weighting is separate. OOD selection is provenance-defined, not a random rename of current test.

Review focus: duplicate/mixed parent relations; cache IDs not recoverable by filenames; silent voice vs unavailable feature; short-repeat vs zero-padding; XGBoost stopping and model selection touching evaluation groups; masks absent/unknown; snapshot identity vs self-asserted hashes; changed input ordering; synthetic artifacts accidentally deployable.

## Completion ledger
CPU implementation and specified local checks executed. Safe LR/ET serialization code exists but execution remains NOT_RUN due to absent skops. New-data training/production OOD/GPU/L4 remain NOT_RUN. See VERIFICATION.json; no independent reviewer or remote tasks were dispatched.
