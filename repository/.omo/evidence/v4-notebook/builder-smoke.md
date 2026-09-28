# V4 notebook builder smoke evidence

| Success criterion | Scenario | Invocation | Binary observable | Captured artifact |
| --- | --- | --- | --- | --- |
| Source-embedded notebook is structurally valid | Build from the current final data receipt into an isolated temporary notebook | `uv run --with nbformat python -` importing `build_v4_notebook`, overriding only `INPUTS` and `NOTEBOOK`, then calling `main()` | `NOTEBOOK_BUILDER_SMOKE_OK 28 /tmp/DACON_236749_trainable_v4_builder_smoke.ipynb` | `/tmp/DACON_236749_trainable_v4_builder_smoke.ipynb` |
| Generated notebook has no fabricated result output | Inspect every code cell in that generated notebook | Same invocation; `assert all(not cell.outputs ...)` | Assertion passed | `/tmp/DACON_236749_trainable_v4_builder_smoke.ipynb` |
| Generated notebook carries the required execution contract | Search its generated cell sources for the CLI flags, `rank_score`, release call, and Colab notebook-capture request | Same invocation | All required strings were found by assertions | `/tmp/DACON_236749_trainable_v4_builder_smoke.ipynb` |
| Runtime source import closure is available in an isolated dependency environment | Import the V4 CLI without starting training | `uv run --with torch --with transformers --with safetensors --with imageio-ffmpeg python -c 'import train_v4; print("IMPORT_CLOSURE_OK")'` | `IMPORT_CLOSURE_OK` | This report and terminal capture for the current attempt |

## Final delivery build

| Success criterion | Scenario | Invocation | Binary observable | Captured artifact |
| --- | --- | --- | --- | --- |
| Final notebook embeds actual Drive receipts | Build after combining the four Drive-readback V4 receipts with the retained AST and SONICS receipts | `uv run build_v4_notebook.py` | Builder exited `0`; `INPUT_ARCHIVES` embeds six archives | `DACON_236749_trainable_v4.ipynb` |
| Notebook remains clean and contains every declared source | Parse/validate the final notebook and compare `%%writefile` targets with `build_v4_notebook.SOURCES` | `uv run --with nbformat python -` validation script | `FINAL_NOTEBOOK_VALID 28 19 192975` | `DACON_236749_trainable_v4.ipynb` |
| Drive receipt is structurally complete | Check six archive records, metadata-verified IDs for all 40 V4 parts, unique part names, and expected AST/SONICS entries | `jq -e ... v4_validation/drive_inputs.json` | Three `true` values | `v4_validation/drive_inputs.json` |
| Standalone CLI imports and exposes all required flags | Import the runner and print its parser help after the portable metadata correction | `uv run train_v4.py --help` and `uv run --with soundfile python -c 'import train_v4; print("IMPORT_CLOSURE_OK")'` | Help lists `--data --baseline --sonics --ast --output --deadline`; `IMPORT_CLOSURE_OK` | Current terminal captures |

## Source refresh

The trainer changed V4 training sources after the first build. The builder was rerun rather than treating the older embedded source cells as current. The repeated semantic check returned `FINAL_NOTEBOOK_REFRESH_VALID 28 19 195885`.

Final SHA-256: `build_v4_notebook.py` `177fce3d1445f7d84c1d874ed580902719a3ea56d736706503f1df742656787f`; `DACON_236749_trainable_v4.ipynb` `5882188ddc0f1863e8a55b55ede44e95e7cfda5035ea8ea348f03ec890e8c152`; `v4_validation/drive_inputs.json` `5200d0e6310913d2ca455f0ba54fdb0800ed45a0cf2f25eea06e764aa22d0944`.

## Authoritative final freeze refresh

| Success criterion | Scenario | Invocation | Binary observable | Captured artifact |
| --- | --- | --- | --- | --- |
| Embedded sources exactly match the authoritative V4 freeze | Compare each of the 19 source files against `.omo/evidence/v4-training/source-sha256.txt` before rebuilding | `uv run --with nbformat python -` receipt assertion | `FROZEN_SOURCE_RECEIPT_OK 19` | `.omo/evidence/v4-training/source-sha256.txt` |
| Delivered notebook contains exactly the frozen source bytes | Build, parse, validate, compare all `%%writefile` bodies byte-for-byte and SHA-256-for-SHA-256; also assert no code output | `uv run build_v4_notebook.py` then `uv run --with nbformat python -` semantic assertion | `FROZEN_NOTEBOOK_EXACT_OK 28 19 203053` | `DACON_236749_trainable_v4.ipynb` |
| Allocation shutdown contract remains present in the delivered notebook | Inspect generated code for safe deadline, 60-second runner stop margin, capture request, rank plot, fvcore dependency, watchdog-before-mount, and cancel-before-final-unassign ordering | Same semantic assertion | Assertions passed | `DACON_236749_trainable_v4.ipynb` |
| Frozen runner is importable and exposes its CLI inputs | Parse help and import `train_v4` under the declared portable dependency environment | `uv run train_v4.py --help`; `uv run --with soundfile python -c 'import train_v4; print("FROZEN_IMPORT_CLOSURE_OK")'` | Help exposes all six inputs; `FROZEN_IMPORT_CLOSURE_OK` | Current terminal captures |

Authoritative final SHA-256: `build_v4_notebook.py` `177fce3d1445f7d84c1d874ed580902719a3ea56d736706503f1df742656787f`; `DACON_236749_trainable_v4.ipynb` `24136738effbdaf990c0be0b5039422678fd9ddfe158f5def1f9f940b1222479`; `v4_validation/drive_inputs.json` `5200d0e6310913d2ca455f0ba54fdb0800ed45a0cf2f25eea06e764aa22d0944`.

## Final P4 control-delta freeze refresh

| Success criterion | Scenario | Invocation | Binary observable | Captured artifact |
| --- | --- | --- | --- | --- |
| Final P4 runner receipt is current before embedding | Load the authoritative 19-source receipt and compare each live source SHA-256, including P4 `train_v4.py` and `v4_training_trials.py` hashes | `uv run --with nbformat python -` receipt assertion | `FINAL_P4_FREEZE_RECEIPT_OK 19` | `.omo/evidence/v4-training/source-sha256.txt` |
| Delivered notebook contains exactly the final P4 source bytes | Rebuild, validate with nbformat, parse every `%%writefile` cell, and compare each body and SHA-256 with the live final source and authoritative receipt | `uv run build_v4_notebook.py` then `uv run --with nbformat python -` exact-source assertion | `FINAL_P4_NOTEBOOK_EXACT_OK 28 19 171339` | `DACON_236749_trainable_v4.ipynb` |
| Delivered notebook has no fabricated execution results | Inspect every generated code cell after the final P4 rebuild | Same exact-source assertion | All code-cell `outputs` lists are empty | `DACON_236749_trainable_v4.ipynb` |

Final P4 SHA-256: `build_v4_notebook.py` `177fce3d1445f7d84c1d874ed580902719a3ea56d736706503f1df742656787f`; `DACON_236749_trainable_v4.ipynb` `1e745cf5ac04b8bbe5a933ca8b43832d0de180ad516b774e55c65e60b35ed876`; `v4_validation/drive_inputs.json` `5200d0e6310913d2ca455f0ba54fdb0800ed45a0cf2f25eea06e764aa22d0944`.

## Retry allocation-clock builder smoke

| Success criterion | Scenario | Invocation | Binary observable | Captured artifact |
| --- | --- | --- | --- | --- |
| Retry preserves the original allocation deadline | Generate a temporary retry notebook with original A100 allocation start `1789283305` and a pre-existing run name | `uv run --with nbformat python -` importing the builder with `--allocation-start 1789283305 --resume-run colab_retry_001` | Embedded setup contains `SAFE_DEADLINE = ALLOCATION_START + 7080`, the pinned allocation start, and the run name | `/tmp/DACON_236749_trainable_v4_retry_builder_smoke.ipynb` |
| Retry retains prior runner budget state and only reuses verified local inputs | Inspect the generated retry setup and execution cells | Same invocation | The run requires an existing resume directory, writes `retry_budget.json` instead of replacing `budget.json`, and only reuses an archive after its full size and SHA-256 match the embedded receipt | `/tmp/DACON_236749_trainable_v4_retry_builder_smoke.ipynb` |
| Retry notebook remains a clean, valid source-embedded artifact | Validate the temporary notebook and inspect every generated code cell | Same invocation | `RETRY_BUILDER_SMOKE_OK 28 19`; all code-cell outputs are empty | `/tmp/DACON_236749_trainable_v4_retry_builder_smoke.ipynb` |

The tracked upload target was restored after the temporary smoke build and remains the verified P4 artifact: `DACON_236749_trainable_v4.ipynb` SHA-256 `1e745cf5ac04b8bbe5a933ca8b43832d0de180ad516b774e55c65e60b35ed876`. Retry-capable builder SHA-256: `25a3be622d9115f9a9d7318f5ee3cc5f3b02a596fb7ba8c6ce5d62958a572c2d`.

## Canonical retry budget-freeze delivery

| Success criterion | Scenario | Invocation | Binary observable | Captured artifact |
| --- | --- | --- | --- | --- |
| Retry notebook embeds the canonical adaptive-budget runner source | Compare every current runtime source to the atomically refreshed 19-source trainer receipt before building | `uv run --with nbformat python -` SHA-256 receipt assertion | `RETRY_FREEZE_RECEIPT_OK 19`; includes `train_v4.py` `497dadf…` and `v4_training_refit.py` `6ccd878…` | `.omo/evidence/v4-training/source-sha256.txt` |
| Retry retains the original global allocation boundary | Build with the original allocation start, then parse the setup cell | `uv run build_v4_notebook.py --allocation-start 1789283305` followed by `uv run --with nbformat python -` notebook assertion | Setup pins `1789283305.0` and computes `SAFE_DEADLINE = ALLOCATION_START + 7080` (`1789290385`) | `DACON_236749_trainable_v4.ipynb` |
| Retry notebook carries exact current source bytes and safe local archive reuse | Validate nbformat, compare every embedded source body and SHA-256 to the receipt, check no code outputs, and inspect reuse contract | Same build and assertion | `RETRY_NOTEBOOK_EXACT_OK 28 19 172108`; all code outputs empty and reuse requires full local archive size/SHA-256 validation | `DACON_236749_trainable_v4.ipynb` |

Canonical retry SHA-256: `build_v4_notebook.py` `25a3be622d9115f9a9d7318f5ee3cc5f3b02a596fb7ba8c6ce5d62958a572c2d`; `DACON_236749_trainable_v4.ipynb` `6637e4dd9ab9ba757c80dd6f2508518fd121eee16e7d5848a79a929dc9f53975`; `v4_validation/drive_inputs.json` `5200d0e6310913d2ca455f0ba54fdb0800ed45a0cf2f25eea06e764aa22d0944`.

## Corrected canonical retry receipt refresh

| Success criterion | Scenario | Invocation | Binary observable | Captured artifact |
| --- | --- | --- | --- | --- |
| Corrected runner source is the sole embedded delta | Compare the live 19-source closure to the refreshed canonical receipt, including corrected `train_v4.py` | `uv run --with nbformat python -` SHA-256 receipt assertion | `CORRECTED_RETRY_RECEIPT_OK 19`; `train_v4.py` SHA-256 `fc399dfa456f34f12ccf25851c6a4213f4fb2beaa6a5bc1e849f01c2a488bc3a` | `.omo/evidence/v4-training/source-sha256.txt` |
| Retry notebook preserves the immutable original allocation budget | Rebuild with no resume run and inspect the generated setup cell | `uv run build_v4_notebook.py --allocation-start 1789283305` then `uv run --with nbformat python -` notebook assertion | Pinned start `1789283305.0`, `PINNED_RESUME_RUN = None`, and `SAFE_DEADLINE = ALLOCATION_START + 7080` | `DACON_236749_trainable_v4.ipynb` |
| Delivered notebook contains exactly the corrected source closure | Validate notebook structure, compare every `%%writefile` body and SHA-256 to its live source and receipt, and require clean code outputs | Same build and assertion | `CORRECTED_RETRY_NOTEBOOK_EXACT_OK 28 19 172065` | `DACON_236749_trainable_v4.ipynb` |

Corrected canonical retry SHA-256: `build_v4_notebook.py` `25a3be622d9115f9a9d7318f5ee3cc5f3b02a596fb7ba8c6ce5d62958a572c2d`; `DACON_236749_trainable_v4.ipynb` `9e40c7c58bdc262f98699e0fef351ed15a01a8b8be7b296760bdfeead51a413c`; `v4_validation/drive_inputs.json` `5200d0e6310913d2ca455f0ba54fdb0800ed45a0cf2f25eea06e764aa22d0944`.

## Security-bound builder preparation

| Success criterion | Scenario | Invocation | Binary observable | Captured artifact |
| --- | --- | --- | --- | --- |
| Runtime dependency selections are immutable and recorded | Generate a temporary retry notebook and inspect its execution environment contract | `uv run --with nbformat python -` with builder `--allocation-start 1789283305` | `SECURITY_BUILDER_SMOKE_OK 28`; pins `imageio-ffmpeg==0.6.0` and `safetensors==0.8.0`, with requested and resolved package-version maps written to `environment.json` | `/tmp/DACON_236749_v4_security_builder_smoke.ipynb` |
| Outer baseline ZIP does not extract unused members | Inspect the generated extraction control flow | Same invocation | Exactly one safe, unique, size-bounded `baseline_submit.zip` member is streamed, checked against fixed bytes/SHA-256, then passed to nested `safe_extract`; no outer `open.zip` extraction occurs | `/tmp/DACON_236749_v4_security_builder_smoke.ipynb` |
| Prepared changes preserve a clean notebook contract | Validate the temporary notebook and inspect code outputs | Same invocation | nbformat validation passed and every code-cell output is empty | `/tmp/DACON_236749_v4_security_builder_smoke.ipynb` |

Prepared security-builder SHA-256: `5c68dcbe5b7a1addf428281fd476fed94bb573ca86a4b0c5a1029e6e01108273`. The tracked retry notebook remains unchanged pending trainer source freeze: `DACON_236749_trainable_v4.ipynb` SHA-256 `9e40c7c58bdc262f98699e0fef351ed15a01a8b8be7b296760bdfeead51a413c`.

## Matched P1/P3 recheck notebook

| Success criterion | Scenario | Invocation | Binary observable | Captured artifact |
| --- | --- | --- | --- | --- |
| Recheck embeds the frozen baseline-free closure | Compare the final 14-source receipt to each local source before generation, then compare every embedded `%%writefile` body and SHA-256 to that receipt | `uv run --with safetensors --with imageio-ffmpeg python -` receipt assertion; `uv run --with nbformat python -` generated-notebook assertion | `RECHECK_SOURCE_RECEIPT_OK 14`; `MATCHED_RECHECK_NOTEBOOK_EXACT_OK 22 14 116647`; wrapper SHA-256 `5c85f91add3fde1113d97087e1783e84b71a820e600e0fb479c75a2ff5dbe934` | `.omo/evidence/v4-training/recheck-source-sha256.txt`; `DACON_236749_v4_matched_recheck.ipynb` |
| Recheck preserves the original global deadline and creates a distinct run | Generate with the recheck builder mode and inspect the setup and runner command | `uv run build_v4_notebook.py --allocation-start 1789283305 --matched-recheck` | Setup computes `1789283305 + 7080 = 1789290385`; output root is `runs_trainable_v4_matched_recheck/<UTC timestamp>` with no resume input | `DACON_236749_v4_matched_recheck.ipynb` |
| Recheck stages only allowed inputs and binds raw G0 bytes | Parse embedded `INPUT_ARCHIVES` and execution controls | `uv run --with nbformat python -` scope assertion | `MATCHED_RECHECK_INPUT_SCOPE_OK 4 37`; inputs are train/dev/metadata/SONICS only; G0 raw SHA-256 is `bd1c27227829f36e9ae6191db7600920bf6649810f0896a791730b10555f0854` | `DACON_236749_v4_matched_recheck.ipynb`; `.omo/evidence/v4-retry-G0_predictions.raw.csv` |
| Recheck cannot silently perform final-selection work | Inspect execution cell and validate notebook output state | Same generated-notebook assertion | No AST or sealed archive staging, `open.zip`, baseline extraction, selection, refit, or sealed CLI route; all code-cell outputs are empty | `DACON_236749_v4_matched_recheck.ipynb` |

Matched recheck SHA-256: `build_v4_notebook.py` `d2d61aaaaa7654c7a287048274c5817d166ee3b488a817b0d73aa617ddcc6676`; `DACON_236749_v4_matched_recheck.ipynb` `6f3fce54ff87a430e5b493b53a565e91f4e4b414988ba947c8ed85f32dce64a2`; recheck source receipt `744edeb3ef5cf14ab950b337b7d185c25ec76b986cb3db069a0f720038f07676`.

## Guarded matched-recheck refresh

| Success criterion | Scenario | Invocation | Binary observable | Captured artifact |
| --- | --- | --- | --- | --- |
| Standalone wrapper rejects an altered or incomplete audited panel before model load | Verify the trainer’s final 14-source receipt and inspect the frozen wrapper’s pinned manifest/count guards | `uv run --with nbformat python -` source assertion | `GUARDED_RECHECK_RECEIPT_OK 14`; wrapper SHA-256 `811534002492c462e3411106919bb823e1ba924637cc9fd2fad74671da01f349`, train/dev SHA checks, and `1_116` eligible / `210` development-row guards are present | `.omo/evidence/v4-training/recheck-source-sha256.txt`; `recheck_v4_matched.py` |
| Distinct notebook embeds the guarded wrapper exactly | Build and compare every embedded source byte and SHA-256 against the final receipt | `uv run build_v4_notebook.py --allocation-start 1789283305 --matched-recheck`; `uv run --with nbformat python -` exact assertion | `GUARDED_RECHECK_NOTEBOOK_EXACT_OK 22 14 117647` | `DACON_236749_v4_matched_recheck.ipynb` |
| Recheck preserves data-role and reproducibility boundaries | Inspect generated inputs and runtime command | Same exact assertion | Raw G0 Drive SHA is pinned; sealed/AST archives are absent; wrapper records shared seed/init receipts and explicitly states CUDA bitwise equivalence is not claimed | `DACON_236749_v4_matched_recheck.ipynb` |

Guarded matched-recheck SHA-256: `build_v4_notebook.py` `d2d61aaaaa7654c7a287048274c5817d166ee3b488a817b0d73aa617ddcc6676`; `DACON_236749_v4_matched_recheck.ipynb` `c047a249f86f4fdbf03a131aa0feada413d5584cc6b0ef5fbf0082e60c2cf9e0`; final source receipt `6f5c965632bd4031a9d36bc8922bc00980555a2e9a0ae9cdc85ce28950b519cf`.

## Matched runtime write-target repair

| Success criterion | Scenario | Invocation | Binary observable | Captured artifact |
| --- | --- | --- | --- | --- |
| Setup-created workspace matches every embedded write target and runner working directory | Rebuild the same distinct recheck notebook, parse the setup/embedded/runner cells, and write every embedded source into an isolated setup-created workspace | `uv run build_v4_notebook.py --allocation-start 1789283305 --matched-recheck`; `uv run --with nbformat python -` integration assertion | `MATCHED_WRITEFILE_RUNTIME_PATH_OK 14 22`; all 14 `%%writefile` parents are `/content/dacon_trainable_v4`, setup creates that directory, every body writes byte-exactly and compiles, and the runner uses `cwd=WORK` | `DACON_236749_v4_matched_recheck.ipynb` |

Repaired matched-recheck SHA-256: `build_v4_notebook.py` `8c5d0706b1213c2711b8fdca4b589cc0e2e03889f7b4c9eb143e0499e0935bd7`; `DACON_236749_v4_matched_recheck.ipynb` `e9e306cffa68e37337e8fa4722576ba5a860c2e767b8e816ceeba40878b7074e`.

## Matched cleanup-path repair

| Success criterion | Scenario | Invocation | Binary observable | Captured artifact |
| --- | --- | --- | --- | --- |
| A failing recheck execution persists failure evidence and releases the runtime | Rebuild, execute the exact generated final cell in an isolated setup-global harness with a pre-existing `RUN/code` directory to force the earliest guarded failure | `uv run build_v4_notebook.py --allocation-start 1789283305 --matched-recheck`; `uv run --with nbformat python -` failure-path integration assertion | `MATCHED_FAILURE_CLEANUP_OK ['failure.txt'] 1 1`: `failure.txt`, `drive_receipt.json`, and `release_requested.json` exist; mocked `watchdog.cancel()` and `runtime.unassign()` were each called once | `DACON_236749_v4_matched_recheck.ipynb` |
| Cleanup repair preserves the frozen source closure | Compare every embedded source SHA-256 to the final recheck receipt after the failure-path scenario | Same assertion | `MATCHED_CLEANUP_NOTEBOOK_EXACT_OK 22 14`; all code outputs remain empty | `DACON_236749_v4_matched_recheck.ipynb`; `.omo/evidence/v4-training/recheck-source-sha256.txt` |

Cleanup-repaired matched-recheck SHA-256: `build_v4_notebook.py` `3c90e7d5bde01c9a8f32eb35882098015b68ad7884f1e8f43bbeb7a0b0e23610`; `DACON_236749_v4_matched_recheck.ipynb` `8868cff794f6ecfa724585068db0c1144088568c3d24460f5562e6ded7e8a855`.

## Ten-epoch matched-recheck freeze

| Success criterion | Scenario | Invocation | Binary observable | Captured artifact |
| --- | --- | --- | --- | --- |
| Both fresh matched arms use the revised bounded epoch count | Compare the new frozen 14-source receipt, rebuild the same target, and inspect wrapper plus notebook methods text | `uv run --with nbformat python -` receipt assertion; `uv run build_v4_notebook.py --allocation-start 1789283305 --matched-recheck`; generated-notebook assertion | `TEN_EPOCH_RECHECK_RECEIPT_OK 14`; `TEN_EPOCH_RECHECK_NOTEBOOK_EXACT_OK 22 14 117647`; wrapper has `EPOCHS = 10` and notebook states ten epochs per arm | `.omo/evidence/v4-training/recheck-source-sha256.txt`; `DACON_236749_v4_matched_recheck.ipynb` |
| Previous workspace and cleanup fixes remain in the revised artifact | Inspect generated execution cell after the epoch-only source refresh | Same assertion | Shared `/content/dacon_trainable_v4` workspace and single-brace failure receipt/release cleanup remain present; code outputs are empty | `DACON_236749_v4_matched_recheck.ipynb` |

Ten-epoch matched-recheck SHA-256: `build_v4_notebook.py` `1fd3f190b0db7730331036798e34468182e7f93ad9f552685bd7ba75cffd4982`; `DACON_236749_v4_matched_recheck.ipynb` `3d465f9285c887dbe16db10deaaef7d70e721a301380fab98cfd201a1cc661c1`.
