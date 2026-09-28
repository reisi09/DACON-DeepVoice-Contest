# Eight-H100 independent preflight QA

Overall source/data and CPU preflight verdict: **PASS_FOR_SOURCE_DATA_LAUNCH_GATE**. The final manifest is SHA-256 `fc21a54a7d9aeec69575ad1b6a9c0d0c35910f13ba317fde434587686309b537`. Native H100 execution is pending and was not launched by this QA lane.

## manualQa

### surfaceEvidence

| scenario id | criterion reference | surface | exact invocation | verdict | artifactRefs |
|---|---|---|---|---|---|
| data-identity | plan data contract | real dataset files | `.venv/bin/python` independent data recompute heredoc | PASS | A-data |
| provenance-roles | split-before-mix; 3/3/2/2 speakers | `manifest.csv` metadata | independent union-find over all `group_id` and `parent_ids` atoms | PASS | A-data |
| label-masks | unknown-label contract | manifest label columns | independent per-column empty-value count | PASS | A-data |
| E2-reference | E2 matched parent/duration/gain | real E2 WAVs and manifest | RMS of `(mix - raw parent_ids[0])`, grouped by parent and background | PASS | A-e2 |
| E5-controls | E5 matched controls | real partial manifest rows | group by `pair_id`; compare labels, split, duration, and parents | PASS | A-data |
| cpu-entrypoints | all fixed/probe entrypoints | Python CPU fixture | `.venv/bin/python -m pytest -q test_eight_integration.py` | PASS | A-fixture |
| targeted-regressions | numerical/data/cache/gradient invariants | Python test modules | `.venv/bin/python -m pytest -q test_eight_features.py test_eight_fixed.py test_eight_lora.py test_eight_integration.py` | PASS | A-tests |
| notebook-imports | notebook validity and order | `DACON_Eight_H100.ipynb` | `.venv/bin/python` import/nbformat/order heredoc | PASS | A-notebook |
| E7-head-only | E7 head-only control and selection lock | `eight_lora_train.py` | `.venv/bin/python` source contract audit heredoc | PASS | A-e7 |
| native-H100 | plan F3 measured H100 run | H100 notebook/runtime | not executed by instruction; root launch/download prerequisite missing | FAIL_BLOCKED | A-plan |

### adversarialCases

| scenario id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
|---|---|---|---|---|---|
| changed-audio-hash | manifest identity | data-integrity mutation | isolated changed bytes rejected | PASS | A-adv |
| same-group-cross-role | split-before-mix | split leakage | isolated contributor crossing roles rejected | PASS | A-adv |
| transitive-bridge | global connected components | provenance bridge | `a|b`, `b|c`, `c|d` resolve to one component | PASS | A-transitive |
| expired-deadline | immutable run boundary | deadline guard | reject before notebook/model work | PASS | A-adv |
| archive-corruption | export integrity | archive mutation | CRC/archive open fails | PASS | A-archive |
| replay-old-cache | feature identity | stale-cache replay | changed audio identity rejected | PASS | A-adv |

### artifactRefs

| id | kind | description | path |
|---|---|---|---|
| A-data | command log | independent manifest/source/provenance/E5 audit | `.omo/evidence/eight_h100_20260917/independent_data_recompute_final2.log` |
| A-e2 | command log | independent byte-level E2 gain audit | `.omo/evidence/eight_h100_20260917/e2_gain_corrected.log` |
| A-fixture | pytest log | CPU all-entrypoint fixture integration | `.omo/evidence/eight_h100_20260917/preflight_integration_fixture.log` |
| A-tests | pytest log | targeted independent regression checks | `.omo/evidence/eight_h100_20260917/preflight_targeted_final.log` |
| A-notebook | command log | imports, notebook validation, and order | `.omo/evidence/eight_h100_20260917/notebook_cpu_imports.log` |
| A-e7 | command log | E7 head-only control and test-lock audit | `.omo/evidence/eight_h100_20260917/e7_head_only_static_audit.log` |
| A-adv | command log | isolated mutations, expiry, and stale-cache replay | `.omo/evidence/eight_h100_20260917/adversarial_cases_final.log` |
| A-transitive | command log | independent transitive component probe | `.omo/evidence/eight_h100_20260917/provenance_transitive_probe.log` |
| A-archive | command log | truncated archive rejection | `.omo/evidence/eight_h100_20260917/archive_corruption_final.log` |
| A-plan | plan | H100 requirement and no-launch boundary | `.omo/plans/eight-h100-20260917.md` |

The earlier normalized-parent subtraction probe was invalid for the final artifact and is retained only as superseded diagnostic context. The final raw-parent formula above is the recorded E2 verdict.
