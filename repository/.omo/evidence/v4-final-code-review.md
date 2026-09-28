# V4 final code-quality review

## Verdict

```json
{
  "reviewedCommit": "a01a50e0c774f181c20238109127c5b81cc95dc1",
  "baseCommit": "f69f4bf",
  "codeQualityStatus": "BLOCK",
  "recommendation": "REQUEST_CHANGES",
  "blockers": [
    "P1/P3 do not have a controlled common RNG seed or initialization receipt, so P3 cannot support the promised augmentation-only comparison against the already completed P1."
  ]
}
```

## Scope and binding

Reviewed the immutable checkout `/tmp/dacon-v4-review-a01a50e` at exact SHA
`a01a50e0c774f181c20238109127c5b81cc95dc1`, against `f69f4bf`. `git diff
--check` is clean. The checkout source hashes reproduce the live receipt's
runner hash: `train_v4.py` is
`fc399dfa456f34f12ccf25851c6a4213f4fb2beaa6a5bc1e849f01c2a488bc3a`.

I independently traced orchestration, full-FT optimizer ownership, checkpoint
and selected-weight handling, matched P3/P4 flows, refit gates, and live retry
artifacts. Prior reviews were treated only as leads.

The retry receipt binds the live work to this commit and reports no sealed
access. Its stage plan preserves the original safe deadline
`1789290385`. It records six completed fixed-update probes and a completed P1:
5 epochs, 175 updates, 5,580 window exposures, 17,172,493 trainable
parameters, and 67.788585825 seconds. P2--P4, selection, refit, weight lock,
and sealed evaluation are still pending. The referenced Drive weight objects
are not present locally, so their reported hashes establish a receipt, not an
independent local deserialization check.

## Findings

### CRITICAL

None.

### HIGH

1. **The P1/P3 augmentation comparison is not controlled for RNG state or
   initialization.** `train_v4.py:32` declares `SEED`, but `run()` never applies
   it to Python, NumPy, Torch, or CUDA. Each `execute()` call independently
   loads the model (`train_v4.py:146-156`); P1 and P3 are then run sequentially
   (`train_v4.py:188-203`). P3 matches P1 only on completed update/window
   counts. The experiment contract requires P1/P3 to differ only in augmentation
   while sharing initialization, data, LR, steps, seed, and pooling. Dropout or
   any other Torch/CUDA stochastic operation therefore remains a confound, and
   no seed/initial-state receipt can reconstruct or validate the completed P1.

   This invalidates a future P1-versus-P3 causal claim, rather than merely
   reducing reproducibility. Do not rank P3 as an augmentation effect against
   the completed P1. Before continuing that comparison, rerun both arms from a
   recorded common initial state with the same recorded RNG state (and preserve
   the two receipts), or explicitly remove the augmentation-control claim and
   prevent P3 from being promoted on that comparison.

### MEDIUM

1. **The cold forecast spends only the minimum P1/P3 evidence despite measured
   headroom.** `train_v4.py:180-187` converts a one-update profile into a P3
   epoch estimate with a factor of 16, then forces the cap to five epochs. The
   live P1 completed those five 35-update epochs in 67.79 seconds, whereas its
   allotted one-third main slot in the retry plan is 672.23 seconds. This is a
   large conservative overestimate, not a runtime failure: the five-epoch
   minimum is satisfied and no eligibility gate passed. It does mean the
   current prospective schedule discards useful pre-sealed search budget and
   cannot be presented as a measured capacity limit. Reforecast from the
   observed BF16 main-run timing before deciding any extension; preserve the
   fixed validation/refit reserve and do not reinterpret the five-epoch result
   as a 30-epoch comparison.

2. **Several changed Python modules exceed the loaded slop-review size
   threshold, increasing audit cost in the critical orchestration path.** Pure
   nonblank/noncomment counts are 277 for `train_v4.py`, 350 for
   `v4_training_trials.py`, 260 for `v4_selection.py`, and 344 for
   `test_v4_training.py`. This is a maintainability concern, not a separate
   demonstrated behavior failure. The highest-risk split candidate is
   `v4_training_trials.py`, which combines trial lifecycle, loss dispatch,
   prediction, artifacts, and terminal resume.

### LOW

1. `train_v4.py:43-50` rescales canonical stage proportions from the runner's
   remaining time. This correctly prevents deadline extension, but its stage
   labels no longer mean the stated absolute allocation minute boundaries. The
   retry artifact makes that distinction visible; keep reporting those values
   as rescaled ceilings, not observed original-stage timing.

## Test and evidence assessment

The previously reported 35 passing tests were not accepted as quality proof.
I attempted an independent combined run from the immutable checkout, but its
`.venv/bin/python` is absent, so no local test result is claimed. Existing
artifacts show the meaningful intended checks (real-weight update/reload,
resume trajectory, deadline stop, P4 objective dispatch, promotion locks), but
they do not exercise the missing P1/P3 common-seed invariant. No new shallow
test was created.

The P4 control implementation itself is materially sound on source review:
the control branch returns mean per-parent native-window BCE with no auxiliary
term (`v4_training_trials.py:101-108`), uses parent pooling for prediction
(`v4_training_trials.py:277-280,345-346`), and checks equal completed
update/window exposure in the runner (`train_v4.py:199-203,230-233`). Full FT
is also wired through all named parameters in `v4_training.py:48-71` and each
live P1 receipt reports the expected 17,172,493 trainable parameters. These
facts do not repair the P1/P3 seed defect.

## Skill-perspective check

This check ran after loading `omo:programming` (including the Python guidance),
`omo:remove-ai-slops`, and Ponytail. The diff violates the programming
perspective through the uncontrolled experiment state and violates the
remove-ai-slops size guidance through the oversized changed modules. I found
no deletion-only test, test that merely proves a requested removal, brittle
prompt test, tautological test, or unnecessary production parsing/normalizing
layer in the reviewed training path. The P4 dispatch regression is
decision-valued rather than an implementation-constant pin.

## Required blockers before approval

1. Restore a valid P1/P3 matched-control experiment: common recorded initial
   model state and common recorded Python/NumPy/Torch/CUDA RNG state, with a
   behavioral check that fails when the common state is not used. Because P1
   has already completed without this receipt, rerun P1 and P3 for any claimed
   augmentation comparison, or exclude P3 from promotion and describe it only
   as an uncontrolled diagnostic.
2. Reissue source and execution receipts for whichever path is chosen before
   ranking, refitting, or opening sealed data.
