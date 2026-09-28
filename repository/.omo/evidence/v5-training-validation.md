# V5 expert training validation

Implementation SHA-256: `1ae034949aa708189459be687d762be65186d54dac2cdb4930e727e658056cef`

Test SHA-256: `e32eb4576c318a6ef170466b29b30d4fc1fdec5ab37cd27d57968028055f65ce`

## Scenarios

1. Global class balancing under microbatch 1
   - Invocation: `.venv/bin/python -m pytest -q test_v5_training.py test_v5_dataset.py test_v5_experts.py --junitxml=.omo/evidence/v5-training-pytest.xml`
   - Binary observable: the summed weight of three negative singleton batches equals the weight of one positive singleton batch for a 3:1 population.
   - Artifact: `.omo/evidence/v5-training-pytest.xml`, testcase `test_balanced_loss_gives_each_known_class_equal_mass`.

2. Full-file evaluation coverage
   - Invocation: same pytest invocation.
   - Binary observable: a 9-sample input with a 4-sample window produces starts `(0, 4, 5)`, and max pooling includes the tail-anchored score.
   - Artifact: `.omo/evidence/v5-training-pytest.xml`, testcase `test_evaluation_uses_all_windows_and_tail_anchor`.

3. Optimizer connectivity and completed-epoch accounting
   - Invocation: same pytest invocation.
   - Binary observable: a real small `nn.Module` autograd graph completes 2 epochs/4 optimizer updates at microbatch 1, freezes BatchNorm running statistics, restores training mode after the epoch-1 evaluation, reports positive first-step encoder gradient and parameter delta, emits structured audit/epoch progress events, exposes each source twice, writes reloadable best/final FP32 states, and differs from initialization. The model is explicitly a surrogate and is not evidence about production encoder quality.
   - Artifact: `.omo/evidence/v5-training-pytest.xml`, testcase `test_train_expert_updates_encoder_and_counts_only_complete_epochs`.

4. Adapter integration
   - Invocation: same pytest invocation.
   - Binary observable: V5 dataset masking/composition and voice/music/presence adapter contracts pass alongside the trainer, including real local SONICS evaluation parity and encoder gradient.
   - Artifact: `.omo/evidence/v5-training-pytest.xml`; 11 tests, 0 failures, 0 errors, 0 skipped, 6.112 seconds.

5. Static Python lint
   - Invocation: `uvx ruff check v5_training.py test_v5_training.py --output-format json --output-file .omo/evidence/v5-training-ruff.json`
   - Binary observable: exit 0 and the result is the empty diagnostic array `[]`.
   - Artifact: `.omo/evidence/v5-training-ruff.json`.

6. Final completed epoch selection under budget stop
   - RED invocation: `.venv/bin/python -m pytest -q test_v5_training.py::test_budget_stop_evaluates_and_selects_last_completed_epoch --junitxml=.omo/evidence/v5-training-budget-red.xml`
   - RED observable: 1 failure; the old policy reported 3 completed epochs but evaluated only epoch 1.
   - GREEN invocation: the same focused test after the scheduling repair, writing `.omo/evidence/v5-training-budget-green.xml`.
   - GREEN observable: 1 passed; the nine-second injected budget completes and evaluates epochs 1 and 2, stops before epoch 3, and the improving epoch-2 state is byte-equal to `best.pt` after reload.
   - Artifacts: `.omo/evidence/v5-training-budget-red.xml` and `.omo/evidence/v5-training-budget-green.xml`.

Production DF-Arena and PANNs backward/runtime audits require the actual baseline assets and accelerator run. They were not inferred from the surrogate test and remain a runtime gate for the parent execution.
