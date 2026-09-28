# Models lane evidence

Scope: CPU-only integration of the portable trained-head/OOF API. These checks do not establish model quality, H100/L4 runtime, sealed performance, DACON score, or leaderboard improvement.

## Invocation and result

- Exact command: `.venv/bin/pytest -q test_redesign_models.py --basetemp .omo/evidence/redesign-notebook-20260917/models-fixture`
- Binary observable: exit code 0; `7 passed in 1.68s`.
- Captured log: `.omo/evidence/redesign-notebook-20260917/models-pytest.txt`.
- Cross-lane command: `.venv/bin/pytest -q test_redesign_data.py test_redesign_features.py test_redesign_models.py test_redesign_evaluation.py`
- Binary observable: exit code 0; `22 passed in 2.39s`.
- Captured log: `.omo/evidence/redesign-notebook-20260917/models-redesign-suite.txt`.

## Requirement scenarios and artifacts

1. Zero-init baseline parity and raw causality after optimization
   - Scenario: a seeded unchanged `TemporalMusic` control and `MusicResidual` receive identical stem input; initial bag logits must be bit-identical. After three real optimizer updates, permuting only raw inputs must change logits.
   - Test: `test_zero_residual_preserves_control_then_raw_changes_trained_output`.
   - Observable: exact initial `torch.equal`; post-update `not torch.allclose(..., atol=1e-7)`.
   - Artifact: `models-pytest.txt`.

2. Unknown-label masking
   - Scenario: every target is unknown.
   - Test: `test_unknown_targets_have_zero_gradient`.
   - Observable: differentiated logits have exactly zero gradient.
   - Artifact: `models-pytest.txt`.

3. Matched A/B versus C/D exposure
   - Scenario: two pure/mixed pair slots and one shared unpaired row are materialized for A and D; a missing counterpart is injected.
   - Test: `test_pair_slots_replace_in_place_and_shortages_fail`.
   - Observable: pure and mixed IDs replace in place, draw counts and paired seconds match exactly, and the shortage raises.
   - Artifact: `models-pytest.txt`.

4. Actual callable music training, capacity control, OOF refits, and checkpoint reload
   - Scenario: `run_experiments(... arms=('D',), seeds=(41,), epochs=1, folds=2, device='cpu')` executes a full D fit, two outer OOF refits with separate internal group early stopping, and a real D raw=stem capacity fit.
   - Test: `test_run_experiments_refits_group_oof_and_reloads_checkpoint`.
   - Observable: every fold's atomic held-out groups are disjoint from both trained and early-stop groups; each row appears in OOF; checkpoint reload predictions agree within `1e-7`.
   - Artifacts:
     - `models-fixture/test_run_experiments_refits_gr0/run/summary.json`
     - `models-fixture/test_run_experiments_refits_gr0/run/oof/D_seed41.json`
     - `models-fixture/test_run_experiments_refits_gr0/run/oof/D_ensemble.json`
     - `models-fixture/test_run_experiments_refits_gr0/run/predictions/D_seed41.json`
     - `models-fixture/test_run_experiments_refits_gr0/run/predictions/D_raw_equals_stem_seed41.json`

5. Resume identity rejects stale truth and feature content
   - Scenario: a completed checkpoint is revisited first with one label changed under the same IDs, then with one raw feature value changed.
   - Test: `test_checkpoint_identity_rejects_relabel_and_changed_features`.
   - Observable: both attempts raise checkpoint identity mismatch instead of silently returning the old model.
   - Artifact: `models-fixture/test_checkpoint_identity_rejec0/identity/best.pt` plus `models-pytest.txt`.

6. Eligible multi-seed voice residual is used, reloadable, and exactly weight-bound
   - Scenario: known real/fake singing and telephone event labels pass the gate. `run_experiments(... seeds=(41,42))` fits two voice residuals, refits them OOF, overlays voice windows onto D, and recomputes File probability. Public sealed inference reloads both aligned music/voice checkpoint lists. The second voice checkpoint is then modified without changing seed 1.
   - Test: `test_eligible_voice_candidate_changes_voice_and_file_and_reloads`.
   - Observable: voice and File probabilities both differ from the frozen candidate; validation and OOF share the recomputable voice family identity; the reloaded ensemble reproduces predictions and the complete upstream identity exactly; ordered `seed_voice_checkpoint_sha256` contains both actual files. Changing only seed 2 leaves the family identity and seed-1 hash unchanged, changes the exact full identity and seed-2 hash, and the production `open_sealed` API rejects the old lock using a valid direct 64-hex panel identity fixture matching the data-audit contract.
   - Artifacts:
     - `models-fixture/test_eligible_voice_candidate_0/voice-run/summary.json`
     - `models-fixture/test_eligible_voice_candidate_0/voice-run/predictions/D_voice_seed41.json`
     - `models-fixture/test_eligible_voice_candidate_0/voice-run/predictions/D_voice_ensemble.json`
     - `models-fixture/test_eligible_voice_candidate_0/voice-run/oof/D_voice_seed41.json`
     - `models-fixture/test_eligible_voice_candidate_0/voice-run/voice/41/voice.pt`
     - `models-fixture/test_eligible_voice_candidate_0/voice-run/voice/42/voice.pt`
     - `models-fixture/test_eligible_voice_candidate_0/changed.json`
     - `models-fixture/test_eligible_voice_candidate_0/lock.json`
     - `models-exact-weight-receipt.json`

7. Multi-seed OOF members survive relation fitting and final application
   - Scenario: `run_experiments(... arms=('D',), seeds=(41,42), epochs=1, folds=2, device='cpu')` emits the seed ensemble, then the real `fit_relation` and `apply_relation` APIs consume it.
   - Test: `test_two_seed_oof_members_fit_and_apply_relation`.
   - Observable: two full `oof_members` retain independent fold/checkpoint proofs, the ensemble is their exact numeric mean within `1e-12`, and the fitted relation applies only to the matching validation family.
   - Artifacts:
     - `models-fixture/test_two_seed_oof_members_fit_0/two-seed/oof/D_ensemble.json`
     - `models-fixture/test_two_seed_oof_members_fit_0/two-seed/predictions/D_ensemble.json`
     - `models-fixture/test_two_seed_oof_members_fit_0/two-seed/relation.json`
     - `models-fixture/test_two_seed_oof_members_fit_0/two-seed/applied.json`

## Implementation boundaries

- The original `music_temporal.py` and `TemporalMusic` checkpoint keys were not edited. `StemMusicWrapper.encode_hidden` calls the public normalization, projection, residual blocks, and output layers.
- Global validation rows are prediction-only. Every OOF model excludes the row's full transitive parent component; internal early-stop groups are also absent from that model's trained groups and recorded separately.
- Absolute deadlines are checked at epoch and batch boundaries. Interrupted artifacts are explicitly marked restart-from-seed and `exact_resume=false`.
- Prediction artifacts separate exact checkpoint hashes from recomputable upstream family payloads. Sealed inference must lock the exact full seed checkpoints even when OOF/validation compatibility uses the family identity.
- `redesign_train` keeps the public prediction imports stable while `redesign_predictions.py` owns inference/artifact composition and `redesign_checkpoint.py` owns content-bound serialization; all production modules are below the 250 pure-line limit.
