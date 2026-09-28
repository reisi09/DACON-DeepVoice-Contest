# V4 model adapter code review

## Scope and source identity

- Goal reviewed: Task 3 of `.omo/plans/trainable-recovery-v4.md`: full-fine-tuning adapters, safe batching, P4 pooling, and full-duration native-grid inference.
- Reviewed files: `v4_model.py`, `v4_model_components.py`, and `test_v4_model.py`.
- Frozen source hashes reproduced at review time:
  - `v4_model.py`: `510a36e121e49707c5cd8d42408c9ba4394eda9c547fb94f25f7608d64051a7b`
  - `v4_model_components.py`: `2991a04d2a81faf1b9c209041b32045a53b2a23dc7d360c0a18ed3ef2b422aae`
  - `test_v4_model.py`: `444573fff75c7115454d50ba93cc067c8ef83a29e5a3d9264413679d74144697`
- Review performed read-only. No GPU or UI action was used. The A100 profile remains pending and is not treated as a failure of this CPU implementation review.

## Evidence independently checked

- `PYTHONHASHSEED=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest -q test_v4_model.py`: **7 passed in 8.54s**.
- Adjacent regression: the same environment ran `test_v4_model.py test_recovery_sonics.py test_ast_multitask.py`: **16 passed in 11.53s**.
- `py_compile` passed for all three reviewed files; `git diff --check` was clean.
- Re-inspected `.omo/evidence/v4-model/{gradient_audit.json,feature_parity.json,late_fake_measurement.json,cpu_profile.json}` and their referenced text logs. The real CPU measurements support full parameter ownership/representative updates, per-example SONICS dB parity, and P4 pooling connectivity. They do not exercise the failing AST tail below.
- `ruff` and `basedpyright` are unavailable in the active environment, so the historical `static-checks.txt` claim was inspected but not independently rerun.

## Findings

### CRITICAL

None.

### HIGH

None.

### MEDIUM

None.

### LOW

1. `v4_model.py:63` and `v4_model.py:139` discriminate `ModelKind` using `if` rather than the exhaustive `match` convention required by the loaded programming skill. Current `ModelKind` has only the two covered values, so this is not a current behavioral failure; it would become a silent default-to-AST grouping/branch if a new model kind were added.

## Skill-perspective check

- **Programming:** ran. The adapter is strongly typed at its public data containers, avoids `Any`/casts/broad catches, verifies model assets before deserializing, and has no untyped escape hatch found in scope. It violates the skill's exhaustive-literal-dispatch convention at the two LOW locations above. More materially, the AST short-tail path is an unhandled runtime contract break.
- **remove-ai-slops:** ran. The tests are not deletion-only, tautological, prompt-text, or implementation-constant-only checks: real-weight parity, all-layer gradient/update, disabled augmentation, pooling duplication invariance, and late-window content each guard a stated failure mode. The production additions are required adapter seams rather than needless data extraction/parsing/normalization. However, the suite omits the practical AST short-tail failure mode, which allowed the HIGH defect through.
- **Ponytail:** ran. No unnecessary dependency or speculative abstraction was introduced in the reviewed adapter. The model file has 242 nonblank/noncomment lines and the component file has 89, both within the applicable size ceiling.

## Positive verification

- `load_model` uses verified local SONICS assets and strict local AST hashes before loading pretrained backbones; optional trained-state overlays use strict state loading.
- The real CPU test confirms every named parameter is optimizer-owned and that early, middle, late, and classifier representatives receive finite nonzero gradients and update.
- The SONICS amplitude-to-dB wrapper is tested against serial/reversed quiet/loud batches, including a deliberate restoration of the batch-coupled behavior.
- P4 bag pooling is normalized under whole-bag duplication and masked segment labels affect only known segments.
- The prior AST tail blocker is fixed at `v4_model.py:148-151`: each AST processor input is zero-padded only to Kaldi's 400-sample floor. A real pinned-AST check on a 160,001-sample file now retains `((0, 160000),)`, produces `(2, 1024, 128)` features, and returns finite logits. `test_v4_model.py:216-228` locks that observable behavior without broadening the suite with a redundant case.

## Decision

- `codeQualityStatus`: **WATCH**
- `recommendation`: **APPROVE**
- `reportPath`: `.omo/evidence/v4-model-review.md`
- `blockers`: none.
