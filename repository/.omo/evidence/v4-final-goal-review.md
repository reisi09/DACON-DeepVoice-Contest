# V4 final goal review

## recommendation

**FAIL** at the evidence cutoff for this review.

Reviewed immutable code commit: `a01a50e0c774f181c20238109127c5b81cc95dc1`  
Required base: `f69f4bf`  
Notebook SHA-256: `9e40c7c58bdc262f98699e0fef351ed15a01a8b8be7b296760bdfeead51a413c`  
Runner SHA-256: `fc399dfa456f34f12ccf25851c6a4213f4fb2beaa6a5bc1e849f01c2a488bc3a`

The fixed implementation and the completed portion of the A100 run are credible. The actual experiment is still incomplete in the available artifacts: P3, P4, final development selection, conditional refit, weight lock, sealed decision, final Drive receipt, and GPU release have no terminal evidence. This is an execution-completion failure, not a claim that the already completed trials are invalid.

## originalIntent

Run the actual V4 Colab A100 experiment, not merely generate a notebook: compare G0 plus four trainable pipelines, fully fine-tune SONICS and AST, use matched P1/P3 and parent/P4 controls, search only the 1,116 music-eligible training rows and conditionally refit on the 1,286 eligible train+dev rows, keep development and sealed roles disjoint, preserve unknown labels, avoid an invented overall score, save retrievable artifacts to Drive, stay within the original A100 allocation ending at Unix `1789290385`, buy no quota, release the GPU, and never submit automatically.

## desiredOutcome

One reproducible terminal outcome for G0, P1, P2, P3 and P4, followed by a development-only selection. If and only if a candidate passes every development and resource gate, fully refit the frozen recipe on all eligible train+dev rows, lock weights, open sealed data once, evaluate, and replay/package an accepted result. If no candidate qualifies or refit cannot finish, retain G0 without opening sealed data unnecessarily. In either case, save the executed notebook and readable/raw receipts to the named Drive run folder and prove the A100 was released within the original allocation.

## userOutcomeReview

The delivered static artifact matches the requested experiment shape. The code exposes G0/P1/P2/P3/P4, full trainable parameter groups, exact eligible-data filtering, matched exposure checks, one-way recipe/refit/weight locks, and conditional sealed access. The notebook explicitly says it never submits to DACON and its watchdog retains the original allocation deadline.

The live evidence proves an actual A100 40 GB run in Drive folder `1-8rmMbN058rMUbtlrGrdPnpWby9ljpF6`, run `20260913_074847`, bound to the reviewed commit and notebook hashes. It proves G0 inference on all 210 development rows, six completed 60-update probes, P1 completed for five epochs, 175 optimizer updates and 5,580 window exposures in 67.7885858249997 seconds, and P2 completed for ten epochs and 350 total updates in 398.31263099399985 seconds. P2's selected checkpoint is epoch 4 at 140 updates and 4,464 selected-checkpoint window exposures; its full run records 11,160 window exposures. SONICS reports 17,172,493 trainable parameters and AST reports 86,188,033; the profile records nonzero gradients and parameter updates for both.

The measured P1 result is a valid negative quality result so far: it fails the prespecified OOD gain, OOD file EER, OOD music AUC, OOD bootstrap win-rate and OOD bootstrap p10 gates. This does not itself violate execution success, because the plan expressly permits a failed improvement attempt with G0 retained. The blocker is that the experiment has not reached the terminal selection/rejection path in the supplied evidence.

No overall score is claimed: the independent metric artifact records `absolute_score: null`, explains the unknown voice labels and invalid single-class voice-presence AUC, and reports partial coverage. No sealed access is reported. This is consistent with the intended evidence boundary.

## blockers

1. **violatedCriterion: SC-EXEC-1 — every attempted trial has a reproducible terminal outcome and all five primary pipeline rows have a measured result or explicit not-run/censored reason.**
   - Observation: G0, six probes, P1, and P2 are complete; P3-P4 and final selection remain pending. No terminal artifact supplies measured or censored P3 or P4 rows.
   - evidencePointer: `.omo/evidence/v4-retry-P2-terminal.json`; `.omo/evidence/v4-retry-live-run.json`; absent terminal P3/P4 receipts and `comparison.json`/`selection.json` for Drive run `20260913_074847`.

2. **violatedCriterion: SC-EXEC-2 — artifacts are saved/retrievable and the GPU is released within the original 7,200-second allocation.**
   - Observation: the run folder and intermediate readbacks are identified, but there is no final Drive retrieval/semantic receipt for the completed run and no release artifact after this retry. The last supplied retry UI artifact is a running heartbeat, not a released runtime.
   - evidencePointer: `.omo/evidence/v4-retry-live-run.json`; `.omo/evidence/v4-colab-ui/v4-retry-run-heartbeat-17m27.png`; missing retry terminal `drive_receipt.json` and retry release screenshot/accessibility receipt before Unix `1789290385`.

3. **violatedCriterion: SC-F4 — report measured improvement, rejection, or incomplete budget honestly, and never promote or submit a failed candidate.**
   - Observation: intermediate reporting is honest, but no final outcome exists yet. There is no final selection showing G0 retention or candidate selection, no conditional refit conclusion, and no final no-submission/release receipt.
   - evidencePointer: `.omo/evidence/v4-retry-initial-metrics.json` fields `p1_initial_main.failed_gates` and `limits`; missing retry `selection.json`, `final_weight_audit.json`/explicit no-refit terminal reason, and final run receipt.

## checkedArtifactPaths

- `.omo/plans/trainable-recovery-v4.md`
- `/tmp/dacon-v4-review-a01a50e` at `a01a50e0c774f181c20238109127c5b81cc95dc1`, diff base `f69f4bf`
- `train_v4.py`, `v4_training_trials.py`, `v4_training_refit.py`, `v4_selection.py`, `v4_evaluation.py`, `v4_model.py`, `prepare_v4_data.py`, `build_v4_notebook.py`
- `test_v4_training.py`, `test_v4_selection.py`, `test_v4_model.py`, `test_v4_data.py`
- `DACON_236749_trainable_v4.ipynb`
- `.omo/evidence/v4-retry-live-run.json`
- `.omo/evidence/v4-retry-stage_budget_plan.json`
- `.omo/evidence/v4-retry-throughput_profile.json`
- `.omo/evidence/v4-retry-G0_predictions.csv`
- `.omo/evidence/v4-retry-first-probe-terminal.json`
- `.omo/evidence/v4-retry-probe_sonics_3e-05-terminal.json`
- `.omo/evidence/v4-retry-probe_sonics_1e-04-terminal.json`
- `.omo/evidence/v4-retry-probe_ast_1e-06-terminal.json`
- `.omo/evidence/v4-retry-probe_ast_3e-06-terminal.json`
- `.omo/evidence/v4-retry-probe_ast_1e-05-terminal.json`
- `.omo/evidence/v4-retry-P1-terminal.json`
- `.omo/evidence/v4-retry-P2-terminal.json` (SHA-256 `81cf8452fcafc3c538395aec716d27be3dd53ff250c3f2a75071b2b1d8aa14a8`)
- `.omo/evidence/v4-retry-initial-metrics.json`
- `.omo/evidence/v4-live-budget-review.md`
- `.omo/evidence/v4-training-final-review.md`
- `.omo/evidence/v4-control-delta-review.md`
- `.omo/evidence/v4-model-review.md`
- `.omo/evidence/v4-data-final-review.md`
- `.omo/evidence/v4-eval-review.md`
- `.omo/evidence/v4-colab-ui/v4-colab-ui-manual-qa.md`
- `.omo/evidence/v4-colab-ui/v4-retry-run-heartbeat-17m27.png`

## direct programming and remove-ai-slops pass

The direct pass found no success-criterion-blocking code slop. The added tests exercise practical decisions: real full-model gradients and updates, batch/serial parity, full-source temporal crops, data-role leakage rejection, matched P4/control exposures and distinct objectives, immutable recipe and weight locks, incomplete-refit rejection, and sealed chronology. They are not deletion-only, requested-removal, prompt-prose, tautological, or file-existence-only tests. A few broad dictionaries, modules above the skill's preferred 250 LOC ceiling, and maintenance-level typing concerns remain NOTES because no stated user criterion fails from them.

The independent code/data/model/evaluation reviews explicitly record the same `omo:programming` and `omo:remove-ai-slops` perspectives and cover deletion-only, tautological, implementation-mirroring, prompt-text, unnecessary parsing/normalization, and test decision value. Their static approvals do not replace the missing live terminal evidence.

## exactEvidenceGaps

- Terminal P3 matched-control result or explicit `not_run`/`censored` reason.
- Terminal P4 plus parent-control result or explicit `not_run`/`censored` reason.
- Final `comparison.json`/equivalent five-row result and `selection.json`.
- If a candidate is selected: recipe lock, complete 1,286-row refit exposure ledger, final weight audit, weight lock, sealed-open/inference receipts, sealed acceptance, and offline package replay if accepted.
- If no candidate is selected: explicit terminal G0-retained decision and proof sealed stayed unopened.
- Final Drive receipt retrieved and hash/semantic checked for this retry run.
- Executed notebook capture and proof of runtime release by Unix `1789290385`.

The report should be re-opened only if a final run receipt supplies these terminal artifacts. Intermediate progress after P1 cannot be inferred as completion.
