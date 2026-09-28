# V5 temporal fusion evidence

Invocation: `.venv/bin/pytest -vv test_v5_fusion.py`

Captured artifact: `.omo/evidence/v5-fusion-scenarios.log`

Binary observables:

- Exact 9-state analytic marginals and conditional component fake probabilities: `test_joint_probabilities_are_exact_state_marginals PASSED`.
- Unknown-label marginal likelihood, `FILE_FAKE=0` joint constraint, finite nonzero gradients: `test_partial_labels_marginalize_unknown_states_and_backpropagate PASSED`.
- Padded-window invariance in eval mode: `test_padding_cannot_change_a_file_prediction_in_eval_mode PASSED`.
- Per-file batch independence: `test_batch_members_never_fuse_across_files PASSED`.
- Non-joint legacy `max(presence * conditional_fake)` identity: `test_nonjoint_file_fake_keeps_original_max_identity PASSED`.
- Order and timestamp sensitivity through learned temporal fusion: `test_window_order_and_timestamps_can_change_learned_fusion PASSED`.
- Complete cached-feature epoch with partial labels, every-example exposure, full nonzero gradient connectivity, and returned/saved-best parity: `test_training_consumes_partial_examples_and_packages_best_state PASSED`.
- Expired budget with reserved final evaluation and explicit zero completed epochs: `test_expired_budget_returns_an_explicit_zero_epoch_result PASSED`.
- Overall result: `8 passed`.

Static invocations and artifacts:

- `uvx ruff check v5_fusion.py v5_fusion_training.py test_v5_fusion.py` -> `All checks passed!` in `.omo/evidence/v5-fusion-ruff.log`.
- `uvx basedpyright --level error v5_fusion.py v5_fusion_training.py test_v5_fusion.py` -> `0 errors` in `.omo/evidence/v5-fusion-basedpyright.log`.
- Python no-excuse checker -> `no violations in 3 file(s)` in `.omo/evidence/v5-fusion-no-excuse.log`.
- Pure LOC check -> core 170, training 228, and tests 132 in `.omo/evidence/v5-fusion-loc.log`.
