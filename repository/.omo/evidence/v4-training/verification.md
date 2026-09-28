# V4 training runner verification

- Scenario: real pinned SONICS training, best-checkpoint binding, actual 210-row development evaluator wiring, and idempotent terminal replay.
  Invocation: `OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run --no-project python -m pytest -vv test_v4_training.py::test_train_trial_real_update_best_binding_and_terminal_resume`
  Observable: `PASSED`; one full-model optimizer update is recorded, the best epoch/checkpoint SHA agrees, and a second `train_trial` call returns the bound terminal result without another update.
  Artifact: `.omo/evidence/v4-training/focused-state-transitions.log`

- Scenario: P4 treatment and matched continuation use equal actual window exposure while different objectives remain wired.
  Invocation: the focused pytest command recorded in `focused-state-transitions.log`, including `test_p4_and_control_have_exact_real_window_exposure` and `test_train_trial_wires_distinct_p4_treatment_and_parent_control`.
  Observable: both tests `PASSED`; treatment/control update and window counts match, while actual `train_trial` updates diverge because treatment uses normalized log-sum-exp plus known-window auxiliary loss and control uses per-bag mean parent loss without auxiliary supervision. Prediction modes are treatment `p4=True`, control `p4=False`.
  Artifact: `.omo/evidence/v4-training/focused-state-transitions.log`

- Scenario: real AdamW checkpoint resume and deadline interruption.
  Invocation: the same focused pytest command, including `test_real_optimizer_resume_matches_uninterrupted_next_updates` and `test_deadline_saves_partial_state_without_success_claim`.
  Observable: both tests `PASSED`; resumed parameters match uninterrupted training and a deadline saves one partial update without an epoch-success claim.
  Artifact: `.omo/evidence/v4-training/focused-state-transitions.log`

- Scenario: adjacent model, evaluation/selection, locking, and training integration regression.
  Invocation: `OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run --no-project python -m pytest -q test_v4_training.py test_v4_model.py test_v4_selection.py`
  Observable: `34 passed in 11.94s`.
  Artifact: `.omo/evidence/v4-training/pytest-v4-final.log`

- Scenario: standalone PEP 723 CLI packaging and hard deadline behavior.
  Invocations: `uv run train_v4.py --help`; then the full required CLI with `--deadline 0` and the local validation/assets paths.
  Observable: help exits successfully with all six required flags; expired execution exits `1` with `TimeoutError: V4 deadline already reached` and writes stage, budget, and failure artifacts.
  Artifacts: `.omo/evidence/v4-training/cli-help.txt`, `.omo/evidence/v4-training/expired-cli.exit`, `.omo/evidence/v4-training/expired-cli.stderr`, `.omo/evidence/v4-training/expired_cli_verified2/`.

- Scenario: supervised coverage accounting.
  Invocation: load the actual train/dev manifests through `v4_data.load_manifest` and `music_supervision_rows`.
  Observable: train `1516` manifest / `1116` music-supervised; dev `210` manifest / `170` music-supervised; exact refit union `1286`.
  Artifact: `.omo/evidence/v4-training/eligibility-counts.txt`

- Source binding: the runner, all behavior imports, and focused test hashes are recorded in `.omo/evidence/v4-training/source-sha256.txt`.
- Execution boundary: no A100 run or model-quality result is claimed by these CPU integration checks. Actual GPU timing, memory, candidate metrics, refit, and sealed acceptance remain pending the parent-controlled notebook run.

## Live allocation timing correction

- Scenario: the Python runner event occurred at `1789283987.0944014`, which is 636.5252 seconds after first-cell start and 682.0944 seconds after the A100 allocation. This is later than the old absolute `allocation + 8 minutes` profile cutoff.
  Invocation: `test_delayed_setup_rebudgets_stages_without_resetting_allocation` uses allocation `1789283305`, first-cell start `1789283350.5691943`, the observed Python runner event, and original safe deadline `1789290385`.
  Observable: the old profile cutoff is confirmed expired; the scaled schedule supplies usable ordered profile/G0/search stages, keeps distinct refit/sealed/export reserves, and ends exactly at the original safe deadline. A retry-provided later deadline is clamped and cannot reset the two-hour allocation.
  Artifacts: `.omo/evidence/v4-training/live-budget-schedule.json`, `.omo/evidence/v4-training/pytest-v4-live-budget.log`, `.omo/evidence/v4-training/live-budget-source-sha256.txt`.
