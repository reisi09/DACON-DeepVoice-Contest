# DACON 236749 V2 Diagnosis, AST Fine-Tuning, and Colab Comparison Plan

## TL;DR
> Summary:      Diagnose the G1/G3/G4 regressions at their discrete ROC operating points, build a provenance-locked source-disjoint validation bank, and compare the official G0 baseline against a frozen AST multi-task head and a real final-block AST fine-tune in one bounded Colab run. Candidate selection uses development data only; two fresh challenge slices are opened once after the selected checkpoint hash is locked, and the selected offline package plus all evidence are saved to the existing Drive folder.
> Deliverables:
> - Exact ROC-crossing and pair-ranking forensic report for G0/G1/G3/G4
> - New 400–800-row provenance-preserving voice/music/mixed validation bank with explicit unknown labels
> - Pinned AST multi-task model, frozen-head candidate, and final-block fine-tune candidate
> - Development-only bootstrap selection record and immutable candidate lock
> - New Colab notebook and executed Drive run; the original notebook and baseline remain intact
> - Self-contained offline submission ZIP and honest untouched-test/runtime report
> Effort:       Large
> Risk:         High - fresh licensed audio acquisition, discrete EER instability on small groups, and Colab/L4 runtime differences can prevent a defensible improvement claim even when implementation succeeds.

## Scope
### Must have
- Recompute and explain the observed G1 voice, G3 music/file, and G4 dev-to-old-holdout behavior using full tied-score ROC points, selected EER crossings, false-positive/false-negative counts, AUC, and changed positive/negative pair rankings; do not infer a single cause from aggregate metrics.
- Treat all 225 existing rows, including the former 65-row holdout, as development material. Construct fresh `test_generator` and `test_source` slices from atomic groups that never occur in development.
- Retain `0`, `1`, and unknown labels as distinct values. Unknown component presence/authenticity must never be coerced to `0` or used in a masked loss/metric.
- Acquire and preserve the exact files used for training/validation, their SHA-256 values, upstream revision/URL, license/attribution, and atomic source lineage for secondary evaluation.
- Compare exactly three candidates: official G0; pinned AST with a frozen encoder and newly trained five-head classifier; the same AST with only `encoder.layer[-1]`, final `layernorm`, and the heads trainable.
- Pin `MIT/ast-finetuned-audioset-10-10-0.4593` at revision `f826b80d28226b62986cc218e5cec390b1096902` and `transformers==4.57.6`; use 16 kHz, 128 mel bins, 1024 frames, AudioSet mean `-4.2677393`, std `4.5689974`, and `(x-mean)/(2*std)` preprocessing.
- Select a checkpoint from `dev_select` only, write its SHA-256 and selection inputs to an immutable lock, then evaluate `test_generator` and `test_source` exactly once without reselection or hyperparameter changes.
- Run the comparison in a new Colab notebook with a maximum 7,200-second GPU allocation, epoch/file checkpoints saved directly to `MyDrive/DACON_236749/runs_v2/<timestamp>`, a fresh offline package execution, and explicit runtime release.
- Preserve file-independent inference: every output row may depend only on that audio file and frozen model parameters.
- Report observed fresh-test results honestly. Success means a valid comparison and package, not a promised score gain.

### Must NOT have (guardrails, anti-slop, scope boundaries)
- Do not modify or overwrite `DACON_236749_five_graphs.ipynb`, `baseline_original/`, `results/pilot_20260912_160943/`, or the existing G0–G4 runtime behavior.
- Do not use private evaluation audio, private-test statistics, pseudo-labeling, cross-file calibration, batch-level normalization, or another evaluation file's predictions.
- Do not count the old 65-row holdout as fresh evidence, and do not inspect fresh-test metrics before `candidate_lock.json` exists.
- Do not use ATA Echoes rows as unambiguous generated-music labels; use TTA only. Do not infer voice absence from a music tag, title, or missing annotation.
- Do not add WavLM, full-encoder fine-tuning, custom attention pooling, ensembling, a new experiment framework, or speculative abstractions in this plan. AST is the single new backbone; G0 is the fixed reference.
- Do not duplicate original audio merely to reach a row count. Deduplicate by decoded PCM SHA-256 while retaining all provenance edges.
- Do not call a partial-label proxy an official DACON score, treat A100 timing as L4 compliance, or treat a passing test suite as evidence of model quality.
- Do not allocate a Colab GPU until the new dataset ZIP, pinned AST assets, notebook, and Drive destination have passed CPU-side preflight.

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: TDD + pytest; each failing-first test protects a concrete leakage, numerical, gradient-connectivity, checkpoint, or offline-package failure mode.
- QA policy: every task has agent-executed scenarios
- Evidence: `<attemptDir>/task-<N>-<slug>.<ext>` — under ulw-loop, `<attemptDir>` is the `currentAttemptDir` from `omo ulw-loop status --json` (`.omo/evidence/ulw/<session>/<goalId>/a<attempt>`); outside ulw-loop use `.omo/evidence/`

## Execution strategy
### Parallel execution waves
> Target 5-8 tasks per wave. <3 per wave (except final) = under-splitting.
> Extract shared dependencies as Wave-1 tasks to maximize parallelism.

Wave 1 (no dependencies):
- Task 1: implement ROC/pair forensics and development-only robust selection
- Task 2: acquire and lock the exact public source files and licenses
- Task 3: implement the pinned AST multi-task model and gradient modes

Wave 2 (after Wave 1):
- Task 4: construct the development and untouched challenge manifests
- Task 5: implement the bounded two-candidate training/checkpoint engine
- Task 6: implement file-independent AST inference and offline packaging

Wave 3 (after Wave 2):
- Task 7: generate the new resumable Colab notebook
- Task 8: upload assets and execute the real Colab comparison to Drive
- Task 9: open the locked test slices once, verify the package, and publish the evidence-limited result

Critical path: Task 2 -> Task 4 -> Task 7 -> Task 8 -> Task 9

### Dependency matrix
| Task | Depends on | Blocks | Can parallelize with |
|------|------------|--------|----------------------|
| 1    | none       | 5, 7   | 2, 3                 |
| 2    | none       | 4      | 1, 3                 |
| 3    | none       | 5, 6   | 1, 2                 |
| 4    | 2          | 7, 8   | 5, 6                 |
| 5    | 1, 3       | 7, 8   | 4, 6                 |
| 6    | 3          | 7, 8   | 4, 5                 |
| 7    | 1, 4, 5, 6 | 8      | none                  |
| 8    | 4, 5, 6, 7 | 9      | none                  |
| 9    | 8          | F1-F4  | none                  |

## Todos
> Implementation + Test = ONE task. Never separate.
> Every task MUST have: References + Acceptance Criteria + QA Scenarios + Commit.

- [ ] 1. Diagnose discrete-EER regressions and freeze the selection rule

  What to do: Add `v2_scoring.py` and `test_v2_scoring.py`. Reuse `graph_scoring.official_metrics` as the official-metric oracle; add full ROC operating-point output (`threshold`, `fpr`, `fnr`, integer `fp/fn/tp/tn`), pair-order changes, and a 2,000-resample lineage-component bootstrap with seed `20260913`. Build lineage components with union-find over every `group_id` constituent so a mixture connects all of its parents; resample connected components, not clips or individual parent IDs (the old dev has 43 atomic groups but 31 connected components, and the old holdout has 21 atomic groups but 17 components). Emit `results/v2_forensics.json` from the existing manifest/predictions, including G1 voice crossing `FP/FN 1/1 -> 2/2`, G3 music crossing `3/6 -> 4/7`, and G3/G4 file crossing `3/5 -> 4/6`. Implement one development-only selector over `G0`, `AST_FROZEN`, and `AST_LAST_BLOCK`: normalize the current available DACON weights over computable metrics; require candidate-minus-G0 proxy `>=0.005`, bootstrap win rate `>=0.70`, 10th-percentile delta `>=-0.005`, no EER worse than G0 by more than `0.02`, and no companion AUC worse by more than `0.01`. Break exact ties in the fixed order `G0`, `AST_FROZEN`, `AST_LAST_BLOCK`. Reject any selection input containing `test_generator` or `test_source`. The selector returns G0 with a machine-readable reason whenever a gate fails.
  Must NOT do: Do not interpolate EER, select from old/fresh test roles, rank candidates by AUC alone, or label G1/G3 as globally worse when their AUC is equal or better.

  Parallelization: Can parallel: YES | Wave 1 | Blocks: [5, 7] | Blocked by: []

  References (executor has NO interview context - be exhaustive):
  - Pattern:  `graph_scoring.py:91-110` - exact tied-score ROC and nearest-point EER implementation to preserve.
  - Pattern:  `graph_scoring.py:113-167` - relevant-label masking, coverage, invalid/partial handling, and official weighted score.
  - Pattern:  `run_experiment.py:110-143` - existing partial proxy and G0 fallback that the robust selector replaces for v2 only.
  - Test:     `test_graph_scoring.py:27-130` - sklearn differential, tie, unknown-label, and malformed-submission patterns.
  - Artifact: `_workspace/v2_discovery/failure_forensics/metrics.json` - full current G0/G1/G3/G4 dev/holdout AUC/EER and pair changes.
  - Artifact: `results/pilot_20260912_160943/comparison.json` - old proxy selection and incomplete official score.
  - External: `https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.roc_curve.html` - published ROC behavior used by the competition formula.

  Acceptance criteria (agent-executable only):
  - [ ] `uv run --with pytest --with numpy --with scikit-learn==1.8.0 pytest -q test_v2_scoring.py` passes after the test is first observed failing without the v2 implementation.
  - [ ] `uv run --with numpy --with scikit-learn==1.8.0 v2_scoring.py diagnose --manifest validation/manifest.csv --predictions results/pilot_20260912_160943 --output results/v2_forensics.json` records G1 holdout voice AUC `0.9842657342657343 > 0.9807692307692308` while EER changes `0.041958041958041946 -> 0.08391608391608389`, and G3 holdout music AUC remains `0.7210526315789474` while EER changes `0.3078947368421052 -> 0.3842105263157895`.
  - [ ] A fixture with `test_generator` rows passed to `select` raises `ValueError: selection accepts development roles only`; a fixture with one adverse discrete crossing but improved global AUC is reported with both facts rather than collapsed to one verdict.

  QA scenarios (MANDATORY - task incomplete without these):
  > Name the exact tool AND its exact invocation - not "verify it works". Browser use: in Codex, use `browser:control-in-app-browser` first when available and no authenticated/persistent user browser profile is required; otherwise use Chrome to drive the page, or agent-browser (https://github.com/vercel-labs/agent-browser) when Chrome is unavailable. Computer use: OS-level GUI automation for a non-browser desktop app.
  ```
  Scenario: Reproduce the non-monotonic EER/AUC regressions
    Tool:     bash
    Steps:    Run `mkdir -p "$(omo ulw-loop status --json 2>/dev/null | jq -r '.currentAttemptDir // ".omo/evidence"')"; uv run --with numpy --with scikit-learn==1.8.0 v2_scoring.py diagnose --manifest validation/manifest.csv --predictions results/pilot_20260912_160943 --output "$(omo ulw-loop status --json 2>/dev/null | jq -r '.currentAttemptDir // ".omo/evidence"')/task-1-roc-forensics.json"`.
    Expected: JSON contains the exact G1/G3 values above plus integer crossing counts and changed pair lists for every compared head.
    Evidence: <attemptDir>/task-1-roc-forensics.json   (attemptDir = currentAttemptDir from `omo ulw-loop status --json`, .omo/evidence/ulw/<session>/<goalId>/a<attempt>)

  Scenario: Reject test leakage during model selection
    Tool:     bash
    Steps:    Run `uv run --with pytest --with numpy --with scikit-learn==1.8.0 pytest -q test_v2_scoring.py -k rejects_test_roles | tee "$(omo ulw-loop status --json 2>/dev/null | jq -r '.currentAttemptDir // ".omo/evidence"')/task-1-roc-forensics-error.txt"`.
    Expected: The test passes only when selection raises the exact development-only error before reading test metrics.
    Evidence: <attemptDir>/task-1-roc-forensics-error.txt
  ```

  Commit: YES | Message: `feat(metrics): diagnose ROC crossings and lock dev selection` | Files: [v2_scoring.py, test_v2_scoring.py, results/v2_forensics.json]

- [ ] 2. Acquire and provenance-lock the bounded public source pool

  What to do: Add `prepare_v2_sources.py`, `test_prepare_v2_sources.py`, and `DATA_SOURCES_V2.md`; write downloaded originals only under `_workspace/v2_sources/`. Voice: from MLAAD-tiny revision `9143e5ea709575ebab6bec52840a1043aada7bb1`, use the fixed primary order `Cartesia.ai/Sonic-3, Higgs-Audio-V2, MeloTTS, FireRedTTS-2.0, Index-TTS-1.5, Index-TTS-2.0, Llasa-1B, Mars5, OpenVoiceV2, VoxCPM-0.5B`, with `Llasa-8B` then `RVC` as deterministic replacements; retain at most 20 decoded >=4-second matched real/fake pairs per accepted generator and require 10 accepted generators. Exclude every PCM/source group already present in `validation/manifest.csv`. Music: consume `_workspace/v2_discovery/echoes_fma_unique.json`, which contains 220 unambiguous license-eligible FMA/Echoes matches from 106 artists after old-artist exclusion; take one normalized-title/artist track per artist and select exactly 60 distinct-artist reference tracks by SHA-256 order. For the first 40 references retain the FMA original plus TTA fakes from `acestep`, `musicgen`, and `audioldm`; for the final 20 retain the FMA original plus TTA fakes from `suno`, `udio`, and `elevenlabs`, yielding 240 paired-content music files. If a specified provider/reference member is absent, advance to the next SHA-ordered unused reference; do not change providers or reduce the count. Extract Echoes members by HTTP byte range with local-header parsing, DEFLATE method 8, size, and CRC verification; parse the ranged FMA metadata member with BZIP2 method 12 and CRC verification; download actual FMA audio and preserve each track's eligible non-ND CC license. Add exactly 100 NSynth test notes selected as four SHA-ordered acoustic/non-vocal samples from each of 25 distinct instruments; preserve the 406,437,840-byte source parquet and describe these as voice-absence/instrument-note controls, not representative songs. Acquire up to 40 AIME generated rows from pinned revision `b84d4be5eda830b6eb714998569dba73530f2601` only as a development fallback if Task 4 cannot form both music classes from the paired pool; record zero AIME use when the paired pool suffices. Write `source_inventory.json` containing every retained file's URL, upstream revision/ETag, byte range if used, SHA-256, decoded PCM SHA-256, license, attribution, source ID, generator/provider, and source-group ID.
  Must NOT do: Do not download the full 8.6 GB Echoes ZIP, use ATA rows, reuse old atomic groups, accept a file on HTTP status alone, infer absent voice, or substitute URLs/scripts for the actual retained training files.

  Parallelization: Can parallel: YES | Wave 1 | Blocks: [4] | Blocked by: []

  References (executor has NO interview context - be exhaustive):
  - Pattern:  `prepare_validation.py:48-72` - retrying immutable downloads, SHA-256, decode, finite-value, duration, and crop behavior.
  - Pattern:  `prepare_validation.py:95-130` - matched MLAAD originals/fakes and retained notices/provenance.
  - Pattern:  `prepare_validation.py:134-173` - unknown-label preservation and per-file music license capture.
  - Artifact: `_workspace/v2_discovery/archive_probes/echoes_entries.json` - 4,509 ZIP64 entries with offsets, compressed sizes, compression method, and CRC.
  - Artifact: `_workspace/v2_discovery/archive_probes/echoes_manifest.csv` - CRC-verified Echoes schema and 4,468 generated rows/296 reference groups/12 providers.
  - Artifact: `_workspace/v2_discovery/echoes_fma_unique.json` - 220 unambiguous eligible FMA references from 106 artists after track-ID deduplication and old-group exclusion.
  - Artifact: `_workspace/v2_discovery/archive_probes/fma_raw_tracks.csv` - CRC-verified 122 MB FMA metadata extracted from the ranged BZIP2 ZIP member.
  - Artifact: `_workspace/v2_discovery/nsynth_test.parquet` - 4,096-row source file with 1,667 eligible acoustic non-vocal notes from 25 instruments.
  - External: `https://huggingface.co/datasets/mueller91/MLAAD-tiny/tree/9143e5ea709575ebab6bec52840a1043aada7bb1` - pinned voice source.
  - External: `https://huggingface.co/datasets/Octavian97/Echoes/tree/14b0c76c6a691c42fadfab9fb6a4eb1ee8c628a2` - pinned Echoes TTA/ATA fields, source-reference metadata, and CC BY-SA 4.0 release.
  - External: `https://huggingface.co/datasets/jg583/NSynth/tree/5ba57c8114d9d4d7cf076c4cb3c8f508f105fca8` - pinned 4-second 16 kHz note parquet used for the controls.
  - External: `https://magenta.withgoogle.com/datasets/nsynth` - official acoustic/electronic/synthetic source tags, instrument-family tags, and license.
  - External: `https://huggingface.co/datasets/disco-eth/AIME/commit/b84d4be5eda830b6eb714998569dba73530f2601` - pinned AIME release and generated-track license.
  - External: `https://github.com/mdeff/fma#acknowledgments-and-licenses` - FMA per-track license requirement.

  Acceptance criteria (agent-executable only):
  - [ ] `uv run --with pytest --with numpy --with pyarrow==21.0.0 --with scipy==1.17.1 test_prepare_v2_sources.py` passes after failing-first cases for a wrong range offset/CRC, repeated PCM, old source group, missing license, and unknown generator.
  - [ ] `uv run --with numpy --with pyarrow==21.0.0 --with scipy==1.17.1 prepare_v2_sources.py --output _workspace/v2_sources` completes with 10 voice generators, exactly 60 new distinct-artist FMA originals plus three paired TTA fakes per original, 100 NSynth controls from 25 instruments, and no decoded PCM/source-group overlap with `validation/manifest.csv`.
  - [ ] Every `source_inventory.json` record resolves to an actual retained file whose computed SHA-256 and decoded PCM SHA-256 match; generator/model/license/revision fields are nonempty; AIME records are marked `development_only`; unknown voice labels remain null.

  QA scenarios (MANDATORY - task incomplete without these):
  > Name the exact tool AND its exact invocation - not "verify it works". Browser use: in Codex, use `browser:control-in-app-browser` first when available and no authenticated/persistent user browser profile is required; otherwise use Chrome to drive the page, or agent-browser (https://github.com/vercel-labs/agent-browser) when Chrome is unavailable. Computer use: OS-level GUI automation for a non-browser desktop app.
  ```
  Scenario: Build the bounded source inventory from immutable public artifacts
    Tool:     bash
    Steps:    Run `uv run --with numpy --with pyarrow==21.0.0 --with scipy==1.17.1 prepare_v2_sources.py --output _workspace/v2_sources --report "$(omo ulw-loop status --json 2>/dev/null | jq -r '.currentAttemptDir // ".omo/evidence"')/task-2-source-lock.json"`.
    Expected: The report meets all count/provenance gates, extracts only selected Echoes members, and lists zero old-group or PCM overlaps.
    Evidence: <attemptDir>/task-2-source-lock.json   (attemptDir = currentAttemptDir from `omo ulw-loop status --json`, .omo/evidence/ulw/<session>/<goalId>/a<attempt>)

  Scenario: Detect a corrupt ranged ZIP member
    Tool:     bash
    Steps:    Run `uv run --with pytest --with numpy --with pyarrow==21.0.0 pytest -q test_prepare_v2_sources.py -k corrupt_crc | tee "$(omo ulw-loop status --json 2>/dev/null | jq -r '.currentAttemptDir // ".omo/evidence"')/task-2-source-lock-error.txt"`.
    Expected: The test passes only when one flipped compressed byte raises `ValueError: Echoes member CRC mismatch` and no final audio file is written.
    Evidence: <attemptDir>/task-2-source-lock-error.txt
  ```

  Commit: YES | Message: `feat(data): lock v2 public audio sources` | Files: [prepare_v2_sources.py, test_prepare_v2_sources.py, DATA_SOURCES_V2.md]

- [ ] 3. Implement the pinned five-head AST model and two training modes

  What to do: Add `ast_multitask.py` and `test_ast_multitask.py`. Load the processor and `ASTModel` from a supplied local checkpoint directory pinned to revision `f826b80d28226b62986cc218e5cec390b1096902`; explicitly pass 16 kHz and verify 128 mel bins/1024 frames/mean/std. Use `pooler_output`, which is final-layer-normalized `(CLS + distillation token)/2`, and five independent `Linear(768, 1)` heads in official submission-column order. For prepared 4–12 second clips use one processor input; for arbitrary inference audio use at most three deterministic 10.24-second start/center/end crops and take the per-head maximum probability within that file. Implement masked BCE: compute training-only `pos_weight = clamp(n_negative/n_positive, 0.25, 4.0)` per head, exclude unknown targets and component-authenticity targets unless the corresponding presence label is known `1`, normalize each nonempty head independently, then mean the available head losses. Define exactly `AST_FROZEN` (heads trainable, all AST parameters frozen) and `AST_LAST_BLOCK` (heads plus `encoder.layer[-1]` and final `layernorm` trainable); request SDPA when available.
  Must NOT do: Do not add WavLM, alter the AST patch embedding, use classifier logits from the AudioSet checkpoint, unfreeze more than the final block/layernorm, or allow gradients through frozen parameters.

  Parallelization: Can parallel: YES | Wave 1 | Blocks: [5, 6] | Blocked by: []

  References (executor has NO interview context - be exhaustive):
  - Pattern:  `graph_features.py:55-85` - file-local extraction and aligned evidence boundary.
  - API/Type: `graph_fusion.py:33-39` - exact five prediction names/order required downstream.
  - Test:     `test_graph_fusion.py:31-147` - high-signal numerical, order-independence, holdout-mask, and corrupt-input patterns.
  - External: `https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593/commit/f826b80d28226b62986cc218e5cec390b1096902` - pinned BSD-3-Clause safetensors checkpoint.
  - External: `https://github.com/huggingface/transformers/blob/753d61104116eefc8ffc977327b441ee0c8d599f/src/transformers/models/audio_spectrogram_transformer/feature_extraction_audio_spectrogram_transformer.py#L71-L90` - 16 kHz/128 mel/1024 frame/mean/std defaults.
  - External: `https://github.com/huggingface/transformers/blob/753d61104116eefc8ffc977327b441ee0c8d599f/src/transformers/models/audio_spectrogram_transformer/feature_extraction_audio_spectrogram_transformer.py#L144-L159` - exact normalization formula.
  - External: `https://github.com/huggingface/transformers/blob/753d61104116eefc8ffc977327b441ee0c8d599f/src/transformers/models/audio_spectrogram_transformer/modeling_audio_spectrogram_transformer.py#L369-L381` - final normalization and token pooling.
  - External: `https://github.com/huggingface/transformers/blob/753d61104116eefc8ffc977327b441ee0c8d599f/src/transformers/models/audio_spectrogram_transformer/modeling_audio_spectrogram_transformer.py#L310-L323` - SDPA support.

  Acceptance criteria (agent-executable only):
  - [ ] `uv run --with pytest --with torch==2.7.1 --with torchaudio==2.7.1 --with transformers==4.57.6 --with numpy pytest -q test_ast_multitask.py` passes after the masked-loss and gradient-connectivity tests are first observed failing.
  - [ ] On a tiny real `ASTConfig`, backward in `AST_FROZEN` yields finite nonzero head gradients and no AST gradients; backward in `AST_LAST_BLOCK` yields finite nonzero gradients only for heads, `encoder.layer[-1]`, and final `layernorm`.
  - [ ] Changing only an unknown target or a component target whose presence is `0` leaves the scalar loss bit-identical; changing a known eligible target changes the loss.
  - [ ] Reordering a two-file batch or changing the other file leaves each file's probabilities within `1e-7`; a 60-second input creates exactly three crops and no cross-file state.

  QA scenarios (MANDATORY - task incomplete without these):
  > Name the exact tool AND its exact invocation - not "verify it works". Browser use: in Codex, use `browser:control-in-app-browser` first when available and no authenticated/persistent user browser profile is required; otherwise use Chrome to drive the page, or agent-browser (https://github.com/vercel-labs/agent-browser) when Chrome is unavailable. Computer use: OS-level GUI automation for a non-browser desktop app.
  ```
  Scenario: Prove intended fine-tuning gradients reach only the final AST block
    Tool:     bash
    Steps:    Run `uv run --with pytest --with torch==2.7.1 --with torchaudio==2.7.1 --with transformers==4.57.6 --with numpy pytest -q test_ast_multitask.py -k 'gradient or masked_loss' | tee "$(omo ulw-loop status --json 2>/dev/null | jq -r '.currentAttemptDir // ".omo/evidence"')/task-3-ast-gradients.txt"`.
    Expected: Both modes satisfy the exact trainable-parameter and masked-loss assertions above.
    Evidence: <attemptDir>/task-3-ast-gradients.txt   (attemptDir = currentAttemptDir from `omo ulw-loop status --json`, .omo/evidence/ulw/<session>/<goalId>/a<attempt>)

  Scenario: Reject a processor/checkpoint contract mismatch
    Tool:     bash
    Steps:    Run `uv run --with pytest --with torch==2.7.1 --with transformers==4.57.6 --with numpy pytest -q test_ast_multitask.py -k rejects_processor_mismatch | tee "$(omo ulw-loop status --json 2>/dev/null | jq -r '.currentAttemptDir // ".omo/evidence"')/task-3-ast-gradients-error.txt"`.
    Expected: The test passes only when altered sampling rate/mel/frame metadata raises `ValueError: AST processor contract mismatch` before inference.
    Evidence: <attemptDir>/task-3-ast-gradients-error.txt
  ```

  Commit: YES | Message: `feat(model): add bounded AST multitask fine-tuning` | Files: [ast_multitask.py, test_ast_multitask.py]

- [ ] 4. Build leakage-proof development and untouched challenge manifests

  What to do: Add `prepare_v2_validation.py` and `test_prepare_v2_validation.py`; consume Task 2's locked inventory and the existing bank. Build union-find lineage components from every `group_id` constituent before any assignment, so mixtures and every transitive parent remain inseparable. Relabel all existing rows as development and repartition their connected components into `dev_train`/`dev_select` by the fixed SHA-256 rule `digest(sorted_component_ids) % 5 == 0 -> dev_select`, otherwise `dev_train`. For new voice, deduplicate decoded originals, retain at most 20 matched pairs per accepted generator, choose `VoxCPM-0.5B` and `OpenVoiceV2` as `test_generator` providers when viable (otherwise the last two viable names in Task 2's fixed ordered list), and reserve their selected lineage components from every development role. From the remaining providers, assign source-prefix/speaker components by hash: 15% to `test_source`, 15% to `dev_select`, remainder to `dev_train`. For music, assign the final 20 distinct-artist FMA references and their `suno`/`udio`/`elevenlabs` TTA counterparts to `test_generator`; assign the first 40 references and their `acestep`/`musicgen`/`audioldm` TTA counterparts by artist component hash to `dev_train`, `dev_select`, and `test_source` using the same percentages. Use AIME only if the development music class gate would otherwise fail, and never in either fresh challenge. Put NSynth instruments, not notes, in exactly one role; use them as known `FILE_FAKE=0`, `MUSIC_FAKE=0`, `VOICE_PRESENT=0`, `MUSIC_PRESENT=1` controls. Create exactly 64 new mixtures, balanced across the four real/fake voice × real/fake music combinations and 0/10 dB SNR, using constituents from the same role and recording both atomic groups. Cap the final bank at 800 rows by SHA order after preserving all challenge minima; require at least 400 rows, at least 20 lineage components in each challenge slice, both classes for all five metrics across the two challenge slices combined, and write separate `v2_validation/development/manifest.csv` and `v2_validation/locked_test/manifest.csv` plus `split_audit.json` and `_workspace/v2_validation.zip`.
  Must NOT do: Do not random-split rows, cross an atomic source/speaker/artist/instrument/mixture constituent between roles, expose locked-test paths to training APIs, duplicate PCM, fill unknown labels, or use a source group in both generator and source challenges.

  Parallelization: Can parallel: YES | Wave 2 | Blocks: [7, 8] | Blocked by: [2]

  References (executor has NO interview context - be exhaustive):
  - Pattern:  `prepare_validation.py:62-64` - deterministic source-group hashing.
  - Pattern:  `prepare_validation.py:101-125` - real/fake pairing and shared source lineage.
  - Pattern:  `prepare_validation.py:176-199` - same-split mixtures and constituent lineage.
  - Pattern:  `prepare_validation.py:202-224` - existing leakage, pairing, duration, PCM, and transform checks.
  - API/Type: `graph_fusion.py:22-30` - existing label and group fields to extend with v2 roles.
  - API/Type: `graph_scoring.py:81-88` - `0`/`1`/unknown parsing contract.
  - Data:     `DATA_SOURCES.md:3-34` - old bank counts, known confounds, unknown labels, and source-group limits.

  Acceptance criteria (agent-executable only):
  - [ ] `uv run --with pytest --with numpy --with scipy==1.17.1 pytest -q test_prepare_v2_validation.py` passes after failing-first cases for old/new group overlap, PCM duplication, mixed-role constituents, unknown coercion, and training visibility of locked paths.
  - [ ] `uv run --with numpy --with scipy==1.17.1 prepare_v2_validation.py --sources _workspace/v2_sources/source_inventory.json --old validation --output v2_validation --zip _workspace/v2_validation.zip` produces 400–800 unique-PCM rows, 64 balanced mixtures, >=20 atomic groups per challenge, and zero atomic-group overlap among all four roles.
  - [ ] Running the builder twice yields identical manifest and split-audit SHA-256 values; every mixture's role equals every constituent's role; every blank input label remains blank.

  QA scenarios (MANDATORY - task incomplete without these):
  > Name the exact tool AND its exact invocation - not "verify it works". Browser use: in Codex, use `browser:control-in-app-browser` first when available and no authenticated/persistent user browser profile is required; otherwise use Chrome to drive the page, or agent-browser (https://github.com/vercel-labs/agent-browser) when Chrome is unavailable. Computer use: OS-level GUI automation for a non-browser desktop app.
  ```
  Scenario: Construct deterministic development and locked-test slices
    Tool:     bash
    Steps:    Run `uv run --with numpy --with scipy==1.17.1 prepare_v2_validation.py --sources _workspace/v2_sources/source_inventory.json --old validation --output v2_validation --zip _workspace/v2_validation.zip --audit "$(omo ulw-loop status --json 2>/dev/null | jq -r '.currentAttemptDir // ".omo/evidence"')/task-4-split-audit.json"`.
    Expected: The audit meets every count/class/source-disjointness condition above and records hashes for both separate manifests.
    Evidence: <attemptDir>/task-4-split-audit.json   (attemptDir = currentAttemptDir from `omo ulw-loop status --json`, .omo/evidence/ulw/<session>/<goalId>/a<attempt>)

  Scenario: Reject a mixture whose constituent crosses roles
    Tool:     bash
    Steps:    Run `uv run --with pytest --with numpy --with scipy==1.17.1 pytest -q test_prepare_v2_validation.py -k cross_role_mixture | tee "$(omo ulw-loop status --json 2>/dev/null | jq -r '.currentAttemptDir // ".omo/evidence"')/task-4-split-audit-error.txt"`.
    Expected: The test passes only when the adversarial manifest raises `ValueError: atomic source group crosses roles`.
    Evidence: <attemptDir>/task-4-split-audit-error.txt
  ```

  Commit: YES | Message: `feat(data): build source-disjoint v2 challenge splits` | Files: [prepare_v2_validation.py, test_prepare_v2_validation.py]

- [ ] 5. Train frozen-head and final-block AST candidates under one bounded protocol

  What to do: Add `train_v2.py` and `test_train_v2.py`. Accept only `development/manifest.csv`; reject locked-test roles/paths. Seed Python/NumPy/Torch with `20260913`, disable TF32, use bfloat16 autocast on supported CUDA, one deterministic train crop plus gain in `[-6,+6]` dB, and deterministic center evaluation. Cache frozen AST pooler outputs by audio SHA + checkpoint SHA + processor signature. Train `AST_FROZEN` for at most 8 epochs with AdamW, head LR `3e-4`, batch 16, weight decay `0.01`; train `AST_LAST_BLOCK` for at most 6 epochs with head LR `1e-4`, final-block/layernorm LR `1e-5`, batch 8, gradient accumulation 2, weight decay `0.01`, and gradient-norm clip `1.0`. Evaluate `dev_select` after each epoch with Task 1 metrics, save epoch/optimizer/RNG state atomically to the Drive run directory, and retain the best development epoch per candidate. Enforce a shared hard deadline: do not start a candidate with <20 minutes remaining, stop only at an epoch boundary, reserve the final 20 minutes for selection/packaging, and cap total GPU time at 7,200 seconds. Run Task 1's selector, copy exactly one checkpoint to `selected/`, and write `candidate_lock.json` with candidate/checkpoint/config/development-manifest hashes before any test evaluation.
  Must NOT do: Do not early-stop or tune from fresh challenge metrics, compute class weights outside `dev_train`, overwrite a completed epoch, silently resume a different config, or claim improvement before the lock and test report exist.

  Parallelization: Can parallel: YES | Wave 2 | Blocks: [7, 8] | Blocked by: [1, 3]

  References (executor has NO interview context - be exhaustive):
  - Pattern:  `graph_features.py:17-21` - atomic JSON writes.
  - Pattern:  `graph_features.py:93-120` - content/signature cache provenance and file-boundary deadline behavior.
  - Pattern:  `run_experiment.py:51-75` - CLI inputs, deadline rejection, and environment capture.
  - Pattern:  `run_experiment.py:102-143` - fit-before-score orchestration and machine-readable selection result.
  - API/Type: `ast_multitask.py:ASTMultiTask` - five-head model and exact trainable-mode contract from Task 3.
  - API/Type: `v2_scoring.py:select_candidate` - fixed development-only gate from Task 1.

  Acceptance criteria (agent-executable only):
  - [ ] `uv run --with pytest --with torch==2.7.1 --with torchaudio==2.7.1 --with transformers==4.57.6 --with numpy --with scikit-learn==1.8.0 pytest -q test_train_v2.py` passes after failing-first tests for locked-test ingestion, cache-signature drift, wrong-config resume, and candidate-lock ordering.
  - [ ] A two-epoch tiny-AST CPU integration run writes restorable checkpoints whose resumed third epoch produces the same weights/metrics as uninterrupted training within `1e-6` and whose lock hash matches the selected checkpoint bytes.
  - [ ] The run report lists actual trainable parameter names/counts, epoch wall time, peak allocated/reserved memory, dev-only metric/coverage/ROC summaries, and a machine-readable reason when G0 remains selected.

  QA scenarios (MANDATORY - task incomplete without these):
  > Name the exact tool AND its exact invocation - not "verify it works". Browser use: in Codex, use `browser:control-in-app-browser` first when available and no authenticated/persistent user browser profile is required; otherwise use Chrome to drive the page, or agent-browser (https://github.com/vercel-labs/agent-browser) when Chrome is unavailable. Computer use: OS-level GUI automation for a non-browser desktop app.
  ```
  Scenario: Resume deterministic training without changing the selected bytes
    Tool:     bash
    Steps:    Run `uv run --with pytest --with torch==2.7.1 --with torchaudio==2.7.1 --with transformers==4.57.6 --with numpy --with scikit-learn==1.8.0 pytest -q test_train_v2.py -k resume_parity | tee "$(omo ulw-loop status --json 2>/dev/null | jq -r '.currentAttemptDir // ".omo/evidence"')/task-5-training-resume.txt"`.
    Expected: Resumed and uninterrupted tiny-AST runs match within `1e-6`, and the checkpoint SHA equals `candidate_lock.json`.
    Evidence: <attemptDir>/task-5-training-resume.txt   (attemptDir = currentAttemptDir from `omo ulw-loop status --json`, .omo/evidence/ulw/<session>/<goalId>/a<attempt>)

  Scenario: Prevent a fresh-test leak into training
    Tool:     bash
    Steps:    Run `uv run --with pytest --with torch==2.7.1 --with transformers==4.57.6 --with numpy pytest -q test_train_v2.py -k rejects_locked_test | tee "$(omo ulw-loop status --json 2>/dev/null | jq -r '.currentAttemptDir // ".omo/evidence"')/task-5-training-resume-error.txt"`.
    Expected: The test passes only when the trainer rejects the locked manifest before dataset construction with `ValueError: training accepts development manifest only`.
    Evidence: <attemptDir>/task-5-training-resume-error.txt
  ```

  Commit: YES | Message: `feat(train): compare frozen and final-block AST` | Files: [train_v2.py, test_train_v2.py]

- [ ] 6. Build file-independent AST inference and the minimal offline package

  What to do: Add `v2_inference.py`, `package_v2_submission.py`, and `test_package_v2_submission.py`. Load only the selected AST checkpoint/processor from local paths, discover `data/test` audio, follow `data/sample_submission.csv` ID order, run each file independently through Task 3's deterministic crop rule, validate exact six-column rows, and atomically write `output/submission.csv`. Package only `model/ast/`, `model/runtime/{ast_multitask.py,v2_inference.py,graph_scoring.py,choice.json}`, root `script.py`, and empty `requirements.txt`; do not copy the 5 GB PANNs/Demucs/DF baseline. Reject traversal/duplicates/missing weights, enforce ZIP <10 GB and expanded <32 GB, set `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1`, and perform fresh extraction plus reference parity at `<=1e-5`. Record A100 timing separately and leave `L4_60_minute_compliance` as `unverified` unless an actual L4 1,200-file run is available.
  Must NOT do: Do not package training data/optimizer states, access the network, reuse another file's statistics, include the old three-model assets, or infer server compliance from projected/A100 timing.

  Parallelization: Can parallel: YES | Wave 2 | Blocks: [7, 8] | Blocked by: [3]

  References (executor has NO interview context - be exhaustive):
  - Pattern:  `package_submission.py:31-78` - offline flags, file order, row validation, and atomic output script.
  - Pattern:  `package_submission.py:81-95` - duplicate/path traversal/reserved path archive validation.
  - Pattern:  `package_submission.py:98-154` - atomic ZIP build and compressed/expanded limits.
  - Pattern:  `package_submission.py:157-184` - fresh unpacked execution and exact output parsing.
  - API/Type: `graph_scoring.py:12-17` - exact six submission columns.
  - Contract: `COMPETITION_CONTRACT.md:42-58` - root layout, Python 3.11/L4 limits, offline runtime, and per-file independence.
  - Artifact: `results/pilot_20260912_160943/packaged_verification.json` - prior fresh-package parity evidence format.

  Acceptance criteria (agent-executable only):
  - [ ] `uv run --with pytest --with torch==2.7.1 --with torchaudio==2.7.1 --with transformers==4.57.6 --with numpy pytest -q test_package_v2_submission.py` passes after failing-first tests for missing local weights, path traversal, wrong ID order, network fallback, cross-file dependence, and notebook/package drift.
  - [ ] A tiny local checkpoint package has exactly required root entries, an empty requirements file, no optimizer/data, and fresh offline predictions matching direct `v2_inference.py` output within `1e-5` after reversing/changing unrelated files.
  - [ ] Deleting one required config/weight file makes `python script.py` fail with a named local-asset error before any connection attempt.

  QA scenarios (MANDATORY - task incomplete without these):
  > Name the exact tool AND its exact invocation - not "verify it works". Browser use: in Codex, use `browser:control-in-app-browser` first when available and no authenticated/persistent user browser profile is required; otherwise use Chrome to drive the page, or agent-browser (https://github.com/vercel-labs/agent-browser) when Chrome is unavailable. Computer use: OS-level GUI automation for a non-browser desktop app.
  ```
  Scenario: Run a freshly unpacked package with network endpoints disabled
    Tool:     bash
    Steps:    Run `HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --with pytest --with torch==2.7.1 --with torchaudio==2.7.1 --with transformers==4.57.6 --with numpy pytest -q test_package_v2_submission.py -k fresh_offline_parity | tee "$(omo ulw-loop status --json 2>/dev/null | jq -r '.currentAttemptDir // ".omo/evidence"')/task-6-offline-package.txt"`.
    Expected: Fresh `script.py` completes with exact IDs/header and maximum direct/package probability delta <=`1e-5` without a download.
    Evidence: <attemptDir>/task-6-offline-package.txt   (attemptDir = currentAttemptDir from `omo ulw-loop status --json`, .omo/evidence/ulw/<session>/<goalId>/a<attempt>)

  Scenario: Fail closed when a local AST asset is absent
    Tool:     bash
    Steps:    Run `uv run --with pytest --with torch==2.7.1 --with transformers==4.57.6 --with numpy pytest -q test_package_v2_submission.py -k missing_asset | tee "$(omo ulw-loop status --json 2>/dev/null | jq -r '.currentAttemptDir // ".omo/evidence"')/task-6-offline-package-error.txt"`.
    Expected: The test passes only when packaged execution raises `FileNotFoundError: required local AST asset missing` and the socket sentinel records zero connection attempts.
    Evidence: <attemptDir>/task-6-offline-package-error.txt
  ```

  Commit: YES | Message: `feat(package): ship minimal offline AST runtime` | Files: [v2_inference.py, package_v2_submission.py, test_package_v2_submission.py]

- [ ] 7. Generate a new resumable Colab notebook without touching the pilot

  What to do: Add `build_v2_notebook.py` and `test_v2_notebook.py`, and generate `DACON_236749_v2_ast.ipynb`. The notebook must mount `MyDrive/DACON_236749`, preflight `open.zip`, `v2_validation.zip`, the pinned AST asset directory/checksums, free Drive space, source hashes, and all CPU tests before CUDA/model allocation work; create a unique `runs_v2/<timestamp>` directory without overwriting; install/pin only missing experiment packages; embed the exact Task 1/3/5/6 source; run development training and selection; write `candidate_lock.json`; then run the two locked challenges once; build and freshly execute the selected package; save logs, checkpoints, predictions, metrics, environment, timing/memory, executed notebook, and final status to Drive. Use a 7,200-second watchdog, reserve 20 minutes for lock/test/package/save, resume matching checkpoints, and call `runtime.unassign()` in both success and terminal-error paths. The notebook title/file/path must be distinct from the original five-graph notebook.
  Must NOT do: Do not allocate before Drive assets exist, overwrite the old notebook/run, silently continue after failed preflight, expose test metrics before the lock cell, or leave an assigned runtime after terminal completion.

  Parallelization: Can parallel: NO | Wave 3 | Blocks: [8] | Blocked by: [1, 4, 5, 6]

  References (executor has NO interview context - be exhaustive):
  - Pattern:  `build_notebook.py:15-43` - notebook generation, Drive run directory, watchdog, and runtime release.
  - Pattern:  `build_notebook.py:45-74` - dependency/data/hash preflight and embedded source cells.
  - Pattern:  `build_notebook.py:75-97` - bounded subprocess execution and resumable logs.
  - Pattern:  `build_notebook.py:98-133` - fresh package parity, finished marker, and release.
  - Artifact: `results/pilot_20260912_160943/finished.json` - prior final status/evidence format.
  - Contract: `README.md:21-24` - current checkpoint/parity evidence and A100/L4 limitation.

  Acceptance criteria (agent-executable only):
  - [ ] `.venv/bin/python build_v2_notebook.py && .venv/bin/python -c "import nbformat; n=nbformat.read('DACON_236749_v2_ast.ipynb',4); nbformat.validate(n); assert not any(c.get('outputs') for c in n.cells if c.cell_type=='code')"` succeeds.
  - [ ] A static AST/nbformat audit proves preflight precedes training, `candidate_lock.json` write precedes either locked manifest read, package verification follows selection, every terminal path invokes `runtime.unassign()`, and no cell references `DACON_236749_five_graphs.ipynb` as an output target.
  - [ ] Running the notebook's CPU preflight against a missing `v2_validation.zip` stops before the first CUDA/model-load event with `FileNotFoundError: Upload v2_validation.zip before allocating GPU work`.

  QA scenarios (MANDATORY - task incomplete without these):
  > Name the exact tool AND its exact invocation - not "verify it works". Browser use: in Codex, use `browser:control-in-app-browser` first when available and no authenticated/persistent user browser profile is required; otherwise use Chrome to drive the page, or agent-browser (https://github.com/vercel-labs/agent-browser) when Chrome is unavailable. Computer use: OS-level GUI automation for a non-browser desktop app.
  ```
  Scenario: Build and structurally validate the new notebook
    Tool:     bash
    Steps:    Run `.venv/bin/python build_v2_notebook.py && .venv/bin/python -m pytest -q test_v2_notebook.py | tee "$(omo ulw-loop status --json 2>/dev/null | jq -r '.currentAttemptDir // ".omo/evidence"')/task-7-notebook-build.txt"`.
    Expected: Notebook schema, ordering, old-artifact preservation, lock-before-test, watchdog, and release-path assertions pass.
    Evidence: <attemptDir>/task-7-notebook-build.txt   (attemptDir = currentAttemptDir from `omo ulw-loop status --json`, .omo/evidence/ulw/<session>/<goalId>/a<attempt>)

  Scenario: Stop before GPU work when a Drive asset is missing
    Tool:     bash
    Steps:    Run `.venv/bin/python -m pytest -q test_v2_notebook.py -k missing_preflight_asset | tee "$(omo ulw-loop status --json 2>/dev/null | jq -r '.currentAttemptDir // ".omo/evidence"')/task-7-notebook-build-error.txt"`.
    Expected: The simulated cell sequence emits the exact missing-upload error and records no CUDA/model-load call.
    Evidence: <attemptDir>/task-7-notebook-build-error.txt
  ```

  Commit: YES | Message: `feat(colab): generate resumable AST comparison notebook` | Files: [build_v2_notebook.py, test_v2_notebook.py, DACON_236749_v2_ast.ipynb]

- [ ] 8. Execute the real Colab comparison and persist the complete Drive run

  What to do: Before requesting an accelerator, upload `_workspace/v2_validation.zip`, the pinned AST local asset bundle/checksum manifest, and `DACON_236749_v2_ast.ipynb` to Drive folder ID `1jKgz89YyiNRFIAsmM7canCtm0QODanNT`; preserve `open.zip`, `oldvalidation.zip`, and the original notebook. Open the new notebook in Colab, choose A100 when available, run all cells, and monitor the Drive log/checkpoints. If interrupted, reopen the same run ID and resume only from a matching checkpoint/config; do not create a second training selection. Let the notebook train both candidates within the 7,200-second ceiling, select/lock on development, evaluate fresh slices once, build/fresh-run the selected package, write `finished.json`, and release the runtime. Sync the executed notebook and small JSON/CSV/log evidence back to `results/v2_<timestamp>/`; keep large source audio/checkpoints/ZIP on Drive and record their Drive IDs and SHA-256 values.
  Must NOT do: Do not use Chrome when the authenticated Aside/local Computer Use path is available, alter hyperparameters after seeing a result, rerun fresh challenges for selection, submit to DACON, delete old Drive assets, or leave the runtime assigned.

  Parallelization: Can parallel: NO | Wave 3 | Blocks: [9] | Blocked by: [4, 5, 6, 7]

  References (executor has NO interview context - be exhaustive):
  - Pattern:  `README.md:3-4` - existing Colab/Drive destinations.
  - Pattern:  `build_notebook.py:21-43` - mounted Drive run layout and watchdog.
  - Pattern:  `build_notebook.py:75-97` - subprocess monitoring and cached resume behavior.
  - Pattern:  `build_notebook.py:126-133` - finished marker and runtime release.
  - Artifact: `results/pilot_20260912_160943/environment.json` - actual environment capture format.
  - Artifact: `results/pilot_20260912_160943/finished.json` - successful prior Drive completion format.

  Acceptance criteria (agent-executable only):
  - [ ] Computer Use confirms the three new assets in folder `1jKgz89YyiNRFIAsmM7canCtm0QODanNT` without modifying `open.zip`, `oldvalidation.zip`, or the original notebook.
  - [ ] The Drive run contains `environment.json`, `source_inventory.json`, `split_audit.json`, both candidate configs/checkpoints/logs, development predictions/metrics, `candidate_lock.json`, both fresh challenge predictions/metrics, selected package ZIP, `packaged_verification.json`, executed notebook, and `finished.json` with `status=finished`.
  - [ ] `candidate_lock.json` timestamp precedes both test result files; its checkpoint SHA matches the packaged weights; total GPU allocation is <=7,200 seconds; Colab shows zero active runtimes afterward.

  QA scenarios (MANDATORY - task incomplete without these):
  > Name the exact tool AND its exact invocation - not "verify it works". Browser use: in Codex, use `browser:control-in-app-browser` first when available and no authenticated/persistent user browser profile is required; otherwise use Chrome to drive the page, or agent-browser (https://github.com/vercel-labs/agent-browser) when Chrome is unavailable. Computer use: OS-level GUI automation for a non-browser desktop app.
  ```
  Scenario: Complete the real Colab run and verify persisted evidence
    Tool:     computer-use
    Steps:    In Aside/Computer Use, open Drive folder `https://drive.google.com/drive/folders/1jKgz89YyiNRFIAsmM7canCtm0QODanNT`; upload the three Task 7 assets; open `DACON_236749_v2_ast.ipynb` in Colab; select A100; invoke Run all; follow the displayed `runs_v2/<timestamp>` path until `FINISHED`; open that folder and verify every required artifact; open Colab resource status and disconnect/delete the runtime; capture the run folder and zero-runtime state.
    Expected: One run reaches `status=finished`, every required artifact exists under the same run ID, and no active runtime remains.
    Evidence: <attemptDir>/task-8-colab-drive.png   (attemptDir = currentAttemptDir from `omo ulw-loop status --json`, .omo/evidence/ulw/<session>/<goalId>/a<attempt>)

  Scenario: Resume an interrupted run without changing experiment identity
    Tool:     computer-use
    Steps:    If the actual run is interrupted, reopen the same notebook/run path, execute the resume cell, and inspect `resume.json`, `candidate_lock.json`, and epoch logs; if no interruption occurs, execute the notebook's CPU interruption fixture and save its log instead of forcing a live interruption.
    Expected: Completed epochs are reused only when config/data/model hashes match, no duplicate candidate lock is created, and a mismatch stops with `Checkpoint provenance mismatch`.
    Evidence: <attemptDir>/task-8-colab-drive-error.txt
  ```

  Commit: YES | Message: `exp(colab): record AST comparison run` | Files: [results/v2_<timestamp>/]

- [ ] 9. Verify the locked challenges and publish the evidence-limited V2 result

  What to do: Consume only the completed Task 8 run. Verify that the candidate lock predates test reads and matches the package. Report `test_generator` and `test_source` separately with exact EER operating points, companion AUC, coverage/unknown counts, atomic-group bootstrap intervals, source/provider slices only where both classes exist, changed pairs versus G0, time/memory, and package parity. Do not revise the selected checkpoint when a challenge regresses; record the observed regression and keep the locked candidate. Run the selected package on the official dummy inputs and a representative multi-file validation subset with network disabled; compare to notebook predictions `<=1e-5`. Write `RESULTS_V2.md`, update `README.md` with links to the new notebook/Drive run while retaining the old links, and retain `submission_candidate.zip` only when package checks pass. State `L4_60_minute_compliance: unverified` unless measured on actual L4; a DACON upload remains outside this plan.
  Must NOT do: Do not merge challenge slices into one flattering headline, omit regressions/unknown coverage, retune after test, claim causal explanation from slices, claim official score from incomplete truth, or upload the package to DACON.

  Parallelization: Can parallel: NO | Wave 3 | Blocks: [F1, F2, F3, F4] | Blocked by: [8]

  References (executor has NO interview context - be exhaustive):
  - Pattern:  `RESULTS.md:15-43` - prior result/evidence/limitation reporting style.
  - Pattern:  `README.md:35-42` - completed-run summary and explicit limits.
  - Pattern:  `graph_scoring.py:113-167` - coverage/partial/official-score boundaries.
  - Pattern:  `package_submission.py:157-184` - real fresh-package execution.
  - Contract: `COMPETITION_CONTRACT.md:42-75` - offline/L4/submission and actual training-data obligations.
  - Artifact: `results/pilot_20260912_160943/packaged_verification.json` - parity report schema.
  - Artifact: `results/v2_<timestamp>/candidate_lock.json` - immutable selection identity from Task 8.

  Acceptance criteria (agent-executable only):
  - [ ] `uv run --with numpy --with scikit-learn==1.8.0 v2_scoring.py verify-run --run results/v2_<timestamp> --output .omo/evidence/task-9-final-report.json` returns `lock_precedes_tests=true`, `checkpoint_package_sha_match=true`, both challenge reports, and no post-test selection event.
  - [ ] `HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python <fresh-unpack>/script.py` completes on the staged multi-file data and `packaged_verification.json.max_probability_delta <= 1e-5`.
  - [ ] `RESULTS_V2.md` contains separate development, generator-challenge, source-challenge, package, runtime, data-rights, and limitations tables whose values are machine-checked against JSON artifacts; improvement language appears only when the corresponding observed locked-test comparison supports it.

  QA scenarios (MANDATORY - task incomplete without these):
  > Name the exact tool AND its exact invocation - not "verify it works". Browser use: in Codex, use `browser:control-in-app-browser` first when available and no authenticated/persistent user browser profile is required; otherwise use Chrome to drive the page, or agent-browser (https://github.com/vercel-labs/agent-browser) when Chrome is unavailable. Computer use: OS-level GUI automation for a non-browser desktop app.
  ```
  Scenario: Audit the immutable selection and fresh challenge report
    Tool:     bash
    Steps:    Run `uv run --with numpy --with scikit-learn==1.8.0 v2_scoring.py verify-run --run results/v2_<timestamp> --output "$(omo ulw-loop status --json 2>/dev/null | jq -r '.currentAttemptDir // ".omo/evidence"')/task-9-final-report.json"`.
    Expected: Lock/test order and hashes pass, both challenge slices retain distinct metrics/coverage, and any regression is present in the report without reselection.
    Evidence: <attemptDir>/task-9-final-report.json   (attemptDir = currentAttemptDir from `omo ulw-loop status --json`, .omo/evidence/ulw/<session>/<goalId>/a<attempt>)

  Scenario: Reject a report generated from a post-test replacement checkpoint
    Tool:     bash
    Steps:    Copy the small run metadata to a temporary directory, change its packaged checkpoint hash, and run `uv run --with numpy --with scikit-learn==1.8.0 v2_scoring.py verify-run --run <tampered-run> --output /tmp/should-not-exist.json 2>&1 | tee "$(omo ulw-loop status --json 2>/dev/null | jq -r '.currentAttemptDir // ".omo/evidence"')/task-9-final-report-error.txt"; test ! -e /tmp/should-not-exist.json`.
    Expected: Verification exits nonzero with `Selected checkpoint changed after lock` and creates no accepted report.
    Evidence: <attemptDir>/task-9-final-report-error.txt
  ```

  Commit: YES | Message: `docs(results): publish locked AST challenge evidence` | Files: [RESULTS_V2.md, README.md, results/v2_<timestamp>/]

## Final verification wave (MANDATORY - after all implementation tasks)
> Runs in PARALLEL. ALL must APPROVE. Surface results to the caller and wait for an explicit "okay" before declaring complete.
- [ ] F1. Plan compliance audit - every task done, every acceptance criterion met
- [ ] F2. Code quality review - diagnostics clean, idioms match, no dead code
- [ ] F3. Real manual QA - every QA scenario executed with evidence captured
- [ ] F4. Scope fidelity - nothing extra shipped beyond Must-Have, nothing Must-NOT-Have introduced

## Commit strategy
- One logical change per commit. Conventional Commits (`<type>(<scope>): <subject>` body + footer).
- Atomic: every commit builds and passes tests on its own.
- No "WIP" / "fix typo squash later" commits on the final branch - clean up before merge.
- Reference the plan file path in the final commit footer: `Plan: .omo/plans/dacon-v2-redesign.md`.

## Success criteria
- All Must-Have shipped; all QA scenarios pass with captured evidence; F1-F4 approved; commit history clean.

## Execution amendments and completion map (2026-09-13)

The original three-candidate primary protocol is complete and its locked selection
remains G0. Root accepted a larger 1,029-row bank before GPU fitting because the
740 acquired new sources and 64 balanced mixtures supplied materially stronger
source-connected coverage; counts and hashes are in DATA_SOURCES_V2.md.
G2 was also included in the exact forensic replay.

Task 1 maps to v2_scoring.py and results/v2_forensics.json; tasks 2/4 to
prepare_v2_sources.py, prepare_v2_validation.py, v2_data.py and the sealed
v2_validation manifests; tasks 3/5 to ast_multitask.py/train_v2.py; tasks 6/7 to
v2_inference.py, package_v2_submission.py, run_v2.py/build_v2_notebook.py; tasks
8/9 to the actual executed Colab artifacts under results/v2_colab_20260912_174330
and RESULTS_V2.md. Scenario evidence is under .omo/evidence/v2. Literal draft
example command names were replaced by these implemented entry points.

The primary package check exposed a real TF32 mismatch. An actual second A100
allocation confirmed its cause with fresh-process toggles; the runtime now
uses the same settings as training. Offline replay receipts include unchanged
weights, exact parity, input order, a 60-second input, and explicit L4 limits.

A development-driven additional G0_AST_MUSIC route is retained as exploratory.
Its protocol preceded root fresh-result observation, but its lock did not
precede the primary program's test opening. It is not accepted as a replacement
for the original G0 selection or as an equally preregistered fourth comparison.
No fresh-result hyperparameter fitting or new DACON submission occurred.
