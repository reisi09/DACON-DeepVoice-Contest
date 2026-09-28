# Code quality review: fullft8

## Scope and evidence inspected

- Goal: run the original seven-trial full-fine-tuning protocol plus a train-only
  fixed-teacher HTDemucs remix adaptation and one fresh-pretrained S0 downstream
  classifier trial, within the H100 x1 credit limit.
- Baseline: `3fd30fa`.
- Reviewed changed files: `build_fullft8_notebook.py`,
  `MERT_EAT_HTDemucs_H100x1_8Experiments.ipynb`, `fullft_separator.py`,
  `fullft8_runtime.py`, `fullft8_start.sh`, `test_fullft_separator.py`, and
  `test_fullft8_integration.py`.
- Inspected the original `MERT_EAT_FullFT_H100x1_7Experiments.ipynb`, generated
  worker construction, and installed `.venv` Demucs 4.0.1 source. In particular,
  `demucs.apply.BagOfModels` aggregates member predictions source-wise, while
  the local `htdemucs.yaml` contains exactly one `955717e8` member. The custom
  differentiable forward therefore has the same aggregation semantics for this
  checkpoint and avoids `apply_model`'s deliberate `no_grad` model call.

## Findings

### CRITICAL

None.

### HIGH

None.

### MEDIUM

None.

### LOW

None.

## Correctness and risk assessment

- Worker serialization is coherent: the generated notebook installs the embedded
  separator module in `sys.modules` before worker execution, and the worker source
  includes that cell. The adapted output writes the same worker source and changes
  both separator checkpoint identity and audio recipe before preparing its cache.
- The adaptation receives only `S0.train`; connected parent components split its
  pseudo-validation subset from optimization. S0 tune/test remain outside separator
  fitting. The MERT/EAT choice uses only S0 tune EER, then balanced BCE, before
  trial 08.
- The separator checkpoint selection records step 0 but selects among trained
  candidates only. `best` begins at infinity, so the first positive validation
  step necessarily writes `best.pt`; downstream checkpoint loading cannot fail for
  the initially suspected “all later steps worse than step 0” case.
- Checkpoint/cache identity binds source hashes, model/checkpoint hashes, recipe,
  crop parameters, and embedded separator code hash. Artifact mirroring omits
  partial files and uses a temporary remote name before replacement.
- Runtime preflight uses the real local HTDemucs, then runs MERT and EAT backward
  and an optimizer update on H100 before the expensive run. The final runtime check
  expects seven standard trials plus the adapted trial and does not count separator
  completion as a classifier result.

## Validation observed

- `test_fullft_separator.py` passed: parent-disjoint remix, differentiable bag
  update/gradient gate, and held-parent selection.
- `test_fullft8_integration.py` passed: embedded worker import identity, cache
  invalidation, artifact content checks, and partial-file mirroring.
- `py_compile` passed for all reviewed Python files; diff whitespace check passed.
- No full pretrained H100 execution was claimed by these checks. The runtime
  preflight is the appropriate first live gate.

## Skill-perspective check

The `omo:programming` and `omo:remove-ai-slops` skills were loaded and applied.
No untyped escape hatch, needless abstraction, brittle prompt test,
implementation-mirroring test, deletion-only test, or needless production parsing
was found in the reviewed scope. The separator module exceeds the nominal size
threshold but explicitly documents why it is a standalone notebook-embedded
adaptation module; its responsibilities are tightly coupled and the exception is
not being used to conceal unrelated code.

## Verdict

- `codeQualityStatus`: CLEAR
- `recommendation`: APPROVE
- `blockers`: none

## Post-preflight recovery addendum

Reviewed only the recovery changes to `build_fullft8_notebook.py`,
`test_fullft8_integration.py`, `fullft8_start.sh`, and `fullft8_runtime.py`.

- The generated `MusicClassifier` now calls Transformers'
  `get_cached_module_file` for every local snapshot Python module before
  `AutoModel.from_pretrained`. This is the required narrow fix for EAT's
  `modeling_eat -> eat_model -> model_core` relative-import chain under an empty
  `HF_MODULES_CACHE`; it does not change model weights or training behavior.
- The added cold-cache integration check launches a fresh interpreter with an
  empty module cache, constructs the real local EAT model snapshot, and confirms
  encoder parameters remain trainable. It is a practical regression check for the
  observed preflight failure, rather than a prompt or implementation-text test.
- The launch script retains inputs, model assets, and the virtual environment when
  present, while moving an earlier completion marker aside and recording a fresh
  bootstrapping state. The runtime records `running` before spawning the worker,
  then mirrors it, so a failed or active attempt is distinguishable from the prior
  completion marker.
- Validation rerun: `test_fullft8_integration.py` passed, including the real
  empty-`HF_MODULES_CACHE` EAT load; Python compilation and whitespace checks
  passed for the modified Python files.

No CRITICAL, HIGH, MEDIUM, or LOW finding was introduced. The prior verdict
remains `codeQualityStatus: CLEAR` and `recommendation: APPROVE`; blockers remain
none. The previously loaded programming and remove-ai-slops perspectives still
apply, and this narrow recovery diff introduces no violation under either.

## Batching addendum

Reviewed against `9cc89eb`, limited to the batch-four / accumulate-two change,
its generated notebook, and `test_fullft8_batching.py`.

- The effective update remains eight examples. `batch_loss_weight` gives the two
  full microbatches weights `4/8` each; for a final partial update it divides
  every present example by that final group's actual size. `steps_per_epoch` is
  the corresponding ceiling of `len(train) / 8`, so scheduler updates and
  effective-example accounting remain aligned.
- The updated trial identity includes batch size and worker count. This correctly
  prevents resuming a pre-change trial with the new scheduler/update schedule;
  the planned boundary after the already-running trial is therefore necessary and
  sufficient for the new settings.
- Recreating a spawn DataLoader each epoch copies the current `AudioBags.epoch`
  into workers, and the per-row crop seed already incorporates that epoch. The
  weighted sampler remains independently seeded per epoch. `persistent_workers`
  is disabled, so no prior-epoch dataset copy survives.
- The numerical-equivalence portion of the added test directly checks the
  non-divisible final group. The worker/crop portion exercises spawned loaders;
  on this macOS review host it did not return within the 30-second command window
  and was stopped, so that portion is not treated as a completed local validation.
  This is not a launch blocker for the Linux H100 plan, whose benchmark gate will
  run the real loader path before later trials are dispatched.

No serious correctness risk was found. `codeQualityStatus` remains `CLEAR`,
`recommendation` remains `APPROVE`, and blockers remain none.
