# V4 evaluation and selection evidence

## Implemented surface

- `v4_evaluation.py`: exact six-column prediction validation, 210-row role panel validation, protected G0 outputs, fixed FILE formula, exact `graph_scoring` ROC metrics, unrenormalized D, connected-group paired bootstrap, and dev/sealed reports.
- `v4_selection.py`: P1-P4 development gates/ranking, 1.25 refit forecast gate, winner-bound recipe lock, exact eligible-ID refit audit, transitive code/weight/input hashes, sealed-open and inference receipts, and single-candidate sealed accept/reject.
- `test_v4_selection.py`: decision-valued regression scenarios only.
- `optionalpackage_v4_submission.py` was not created because no accepted sealed V4 winner exists yet.

## Scenarios and observables

Invocation: `.venv/bin/pytest -vv test_v4_selection.py`

Artifact: `.omo/evidence/v4-eval/selection_scenarios.txt`

Binary observables: fifteen named scenarios pass, covering negative OOD veto despite strong seen results; valid dual-panel selection with absolute score remaining null under partial truth; pre-ranking 1.25 refit reserve rejection; exact unrenormalized D=0.288; protected-output and FILE-formula rejection; actual independently rounded `graph_predictions(...)["G0"]` compatibility; immutable/resumable recipe lock; failed/resource-ineligible recipe-lock rejection; missing prediction ID rejection; exposure-ledger and eligible-ID-union refit rejection; weight hash drift rejection; runner-up refusal; and sealed prediction provenance rejection.

Invocation: `.venv/bin/pytest -q test_graph_fusion.py test_graph_scoring.py test_v2_scoring.py test_v4_selection.py`

Artifact: `.omo/evidence/v4-eval/pytest.txt`

Binary observable: `40 passed, 8 subtests passed`, including the real graph-fusion rounding path, reused exact ROC implementation, and connected-group scoring helpers.

Invocation: `.venv/bin/python -m py_compile v4_evaluation.py v4_selection.py test_v4_selection.py`

Artifact: `.omo/evidence/v4-eval/compile.txt`

Binary observable: explicit `PASS` record.

Artifact hashes: `.omo/evidence/v4-eval/sha256.txt`.

Tool availability: `.omo/evidence/v4-eval/tooling.txt`; Ruff and basedpyright were unavailable and were not installed.

Whitespace check: `.omo/evidence/v4-eval/diff_check.txt`.
