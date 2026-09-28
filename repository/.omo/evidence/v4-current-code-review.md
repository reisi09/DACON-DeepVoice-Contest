# V4 current code-quality review

## Verdict

**PASS** — `CLEAR` / `APPROVE`.

- Reviewed commit: `d2397ad8cc96667e5bc2e4132d86405afe093f4e`
- Base delta: `a01a50e0c774f181c20238109127c5b81cc95dc1..d2397ad8cc96667e5bc2e4132d86405afe093f4e`
- Reviewed tree: `d02b30ae16617544d05527c8580b1fe77a97c503`
- Detached checkout was clean; `git diff --check` reported no Python-diff whitespace errors.

## Source binding

The exact runtime wrapper SHA-256 is
`130e078b393f627b871e549e695e9fe8f4061cfe10a1a16494b60d528aab825c`.
The executed notebook SHA-256 is
`3d465f9285c887dbe16db10deaaef7d70e721a301380fab98cfd201a1cc661c1`.

The 14-file full executed source closure recorded in
`.omo/evidence/v4-matched-source_hashes.json` is:

```text
130e078b393f627b871e549e695e9fe8f4061cfe10a1a16494b60d528aab825c  recheck_v4_matched.py
a5b59f5b95792e0075c2fbb2afa9949b015219b9bac4d4686f6f63214dc2bf51  v4_training_trials.py
63ce9946c13883d79b2039390458ba5f2890435f992b91153ed22e7b1099fae8  v4_training.py
fcc6280b1eb619403e05564ce1c40736ae36cffb8139627646681aadd7f63edd  v4_training_data.py
52155fbeeaebb8e7072d3ec3e31e1021069e5317464e8e80624b37c241c536b1  v4_batches.py
ff3745f6caf79a1ee2726360f7e38cd51dc296283a8aaba958c8d9ea7a698ccb  v4_data.py
510a36e121e49707c5cd8d42408c9ba4394eda9c547fb94f25f7608d64051a7b  v4_model.py
2991a04d2a81faf1b9c209041b32045a53b2a23dc7d360c0a18ed3ef2b422aae  v4_model_components.py
a3a8e307d1024f089748a1777265c1830ae22609c9df81013679499500172144  v4_evaluation.py
7993699ceb926c06c000a7ed5313bfa8fd83b558f3c1e1c1e38f2fcb0f52ba96  graph_scoring.py
1a31cf80dd360cd50bdf7697f491f5a17c40d213ee6f7af16bd31ba871a4afdd  v2_scoring.py
4882c036f504695df46162b81ca794a897acde7a671c564231eea2066b82e0ba  recovery_sonics.py
2204c7a406f2865a9fc84ea9efb3a7ea3bfeaeb1e894eb15ef80671944c69a6b  ast_multitask.py
f030da8bd3f91bb04581a3ebcee6d76ce15f33862979adcc554802e0eeeaea0d  v2_data.py
```

`v4-matched-source_hashes.json` contains the same 14 names and hashes as
`v4-training/recheck-source-sha256.txt`. The in-wrapper remedy contract lists
the 13 modules reachable by the P1/P3 runner and omits unused
`ast_multitask.py`; the notebook's runtime source receipt supplies the full
14-file embedded closure. This is a reporting asymmetry only, not a training
behavior or source-binding gap.

## Independent review

The previous P1/P3 blocker is repaired at the shared execution seam.
`recheck_v4_matched.py:85-93` resets the seed before each model construction,
and `v4_training_trials.py:273-295` resets Python/NumPy/Torch/CUDA RNG before
each trial, fingerprints its already-seeded initial model, and writes the
receipt before training. Runtime receipts show identical P1/P3 seed-RNG SHA
`373d3bd3579f4e77cc224197612499148e34968d0f3a69ded66a682559b1beda` and
identical initial-weight SHA
`0784cf50ef834751c93f813f981ca249799c0ebac12b93950ec1c4e2a1b1be10`.
The code correctly limits the claim: CUDA bitwise equivalence is explicitly
not claimed.

`recheck_v4_matched.py:56-69` rejects a changed G0 CSV, changed train/dev
manifest, wrong eligible/development counts, or misordered G0 IDs before model
load. The added adversarial test modifies the real train manifest by one byte
and proves the refusal happens before a remedy contract exists
(`test_v4_training.py:116-132` in the reviewed source). This is a meaningful
input-integrity regression, rather than a constant-mirroring test.

The added dropout resume test (`test_v4_training.py:448-490`) exercises the
real checkpoint RNG restore path and compares the resulting parameter
trajectory with uninterrupted stochastic training. It protects the stated
reproducibility failure mode; it is not deletion-only, tautological, or a
prompt/implementation-text test. Evidence records the focused three-test run
as passing in 5.68 seconds. I did not rerun broad tests because the requested
runtime experiment was active and the cited targeted evidence covers the
changed failure modes.

The notebook builder change fixes the write target/cwd agreement at
`build_v4_notebook.py:30,49,99-101` and the malformed f-string cleanup
dictionary at `:110`. Builder smoke evidence independently exercises the
embedded-write path and forced failure cleanup. The 10-epoch budget is applied
at `recheck_v4_matched.py:23` and reflected in notebook methods text. P1's
actual receipt reports `completed`, 10 epochs, 350 updates, and 11,160 window
exposures. P3 runtime completion is outside this code-lane verdict.

## Findings

### CRITICAL

None.

### HIGH

None.

### MEDIUM

None.

### LOW

None.

## Skill-perspective check

This check ran after loading `omo:programming` (including its Python guidance)
and `omo:remove-ai-slops`, and applying the active Ponytail scope-control
criteria. The diff does not violate either skill perspective: SHA/count checks
occur at the external-data boundary; no needless parsing, normalization,
abstraction, untyped escape hatch, or brittle prompt test was added. The two
new tests address concrete manifest-drift and stochastic-resume failures.

## Decision

**codeQualityStatus:** `CLEAR`  
**recommendation:** `APPROVE`  
**reportPath:** `.omo/evidence/v4-current-code-review.md`  
**blockers:** None.
