# Evaluation statistics audit — 2026-09-18

Scope: read-only production audit of redesign OOF, paired bootstrap, and eight fixed-feature selection. CPU inspection and recomputation only; no refits, GPU/cloud operations, production edits, or commits. Source and saved-run paths below are relative to `/Users/choiwuseck/Documents/ChatGPT/데이콘`. Paths beginning `evidence/` are relative to that repository's `results/precision_audit_20260918/` audit directory.

## Verified implementation defects

### S1 — The source-balanced training sampler permanently omits eligible parents (high priority)

`eight_heads.py:46-60` reuses a global `ordinal` to choose the source, generator, and parent. When a cell is revisited on a cadence sharing a divisor with its parent count, only some parent indices are reachable. The RNG at line 62 only selects a view within the already chosen parent; changing the seed cannot restore the omitted parents.

Actual 500-update, batch-32 schedules were reconstructed from the 7,101-row manifest and bound to the saved checkpoints by the exact ordered-ID draw SHA-256:

| Cohort and sampler | Eligible parents | Drawn parents | Specific omissions |
|---|---:|---:|---|
| Music, parent-uniform E1 | 831 | 831 | None |
| Music, source-generator-parent | 831 | 789 | 42 of 84 real FMA parents omitted |
| Voice, parent-uniform E1 | 212 | 212 | None |
| Voice, source-generator-parent | 212 | 129 | 53 real and 30 fake parents omitted |

Voice examples: only 2/4 real `MLAAD` parents and 51/102 `mlaad_real` parents are drawn; only 1/10 `Llasa-8B` and 1/10 `MeloTTS` parents are drawn. The selected real `MLAAD` parents each receive 2,000 draws, while the other two receive zero. The intended class and source totals can therefore be correct while parent exposure is wrong.

**Saved-run binding:** all 30 inspected E1 balanced, E4, and E6 checkpoints across seeds 41/42/43 exactly match these reconstructed schedules. E1 parent-uniform control seed41 hashes also match and cover every parent. Evidence: `evidence/evaluation_statistics_sampler.json` and `evidence/evaluation_statistics_sampler_binding.json`. Call paths: `eight_ablation.py:76-80`, `:235-236`, `:273`; saved schedule hashing: `eight_heads.py:169-183`.

**Consequence:** E1's intended balancing contrast also changes which training parents appear. E4/E6 still compare heads trained under the same recorded exposure, but their claimed complete parent-balanced training population is false. This defect does not itself prove the direction or size of a quality change. Preserve saved results as historical; before another run, advance a counter local to each source/generator cell (or sample parents independently) and check per-cell parent coverage and frequency, then retrain affected arms. No fix or retraining was performed in this audit.

### S2 — The generic eight bootstrap can declare improvement after discarding many invalid replicates

`eight_metrics.py:95-106` drops resamples lacking both truth classes, yet sets `uncertain` to false if even one valid delta remains. `comparison_status` at `:142-154` checks CI existence and five total components, but does not check the valid-resample fraction or per-class component support.

A concrete five-component fixture, with only one positive component, produced 645 valid and 355 invalid draws from 1,000 resamples, `ci95=[-1,-1]`, `uncertain=false`, and `comparison_status="measured_improvement"`. Evidence: `evidence/evaluation_statistics_populations_bootstrap.json`, `rare_class_bootstrap_reproducer`. This demonstrates an admission weakness; it is **not an observed failure of the saved E6 comparison**, whose stored intervals report 1,000/1,000 valid replicates. The redesign evaluator already records invalid fractions (`redesign_evaluation_common.py:167-175`) and requires `<0.05` (`redesign_evaluate.py:81-85`). Apply an explicit invalid-fraction and class-component policy before generic eight bootstrap output drives adoption. Five total components alone is not a precision guarantee.

## Verified population and interpretation limits

### S3 — Per-experiment denominators must be scoped to the evaluated split

`eight_ablation.py:329-336` computes top-level label and component counts from every known manifest label, although the accompanying `population` text names a modality/variant-specific population. This is a reporting defect already acknowledged and corrected in `results/eight_h100_20260917/postrun_e2/fixed_experiment_population_denominators.json:2-5`; consumers should use its branch/role counts and the saved prediction IDs.

| Evaluated population | Unique rows | Truth counts | Global connected components |
|---|---:|---|---:|
| E1/E4/E6/E8 core music test | 127 | 23 real / 104 fake | 24 |
| E3 primary Opus music test | 127 | 23 real / 104 fake | 24 |
| E1/E6 core voice test | 48 | 24 real / 24 fake | 1 |
| E3 primary Opus voice test | 48 | 24 real / 24 fake | 1 |
| E5 partial test | 144 | 72 real / 72 fake | 1 |
| E2 voice diagnostic test | 368 | 172 real / 184 fake / 12 unknown voice | 1 |

E2 has 2,208 CSV rows because each unique audio ID appears for six model/pooling combinations. E5's `component_count=4` describes all four splits: `eight_probes.py:172-173,240`; it must not become a test-component count. Direct CSV evidence is in `evidence/evaluation_statistics_probe_populations.json`. Existing corrected E6 counts are at `postrun_e2/fixed_experiment_population_denominators.json:250-289`.

The full manifest has 256 connected components and **zero components crossing splits**, verified by rebuilding `eight_fixed_io.connected_components` over all 7,101 rows. Music test has real labels in 15 components and fake labels in 18; these supports overlap, so they are not additive independent counts. Component sizes range from 1 to 21 rows. Evidence: `evidence/evaluation_statistics_populations_bootstrap.json`.

### S4 — Group bootstrap preserves dependency, but the point metric is row-weighted

Both implementations resample connected components and include every row belonging to the sampled component: `redesign_evaluation_common.py:111-160`, `eight_metrics.py:87-99`. The eight component IDs are resolved once over the complete manifest before subsetting: `eight_fixed_io.py:35-43,51-79`. This correctly preserves transitive contributor links. No current full-manifest cross-split component leak was found.

The EER itself uses ordinary unweighted ROC rows (`eight_metrics.py:27-38`), and each sampled cluster contributes all its rows. Thus bootstrap clustering does **not** convert the estimand into equal-source, equal-generator, or equal-parent performance. A component with 21 test rows has more within-class influence than a component with one. This is acceptable for the stated empirical row population; it cannot establish generalization to a different source mixture. All 23 real music test rows are FMA/FMA or FMA, while fake rows come from Echoes, suno, and udio. Report per-source/generator behavior and define any alternative target weighting explicitly; do not quietly replace the official row metric.

### S5 — Seeds and historical holdouts are not independent population replications

The redesigned OOF constructor uses `fold_groups(rows, folds, 41)` and inner early-stop seeds `41+fold`, regardless of training seed (`redesign_oof.py:95-103`). This makes the outer and inner group partitions shared across seeds. In eight, all seeds use the same stored train/dev/test rows (`eight_ablation.py:58-80`). Seed spread therefore measures optimization variability on those populations, not new-source sampling variability.

The eight dataset explicitly reuses former train/validation/sealed rows (`eight_data.py:44-59`) and earlier fresh-cohort manifests (`:60-82`). The saved receipt states that canonical and fresh-cohort records were exposed to prior project work (`results/eight_h100_20260917/data_receipt.json:53-54`). The current split can be contributor-disjoint while remaining historically exposed. A new split or a new filename does not turn those rows into an untouched confirmation population.

The code's 100-per-class and 30-per-stratum redesign gates (`redesign_evaluate.py:80,103-106`), five-component eight gate (`eight_metrics.py:89-91`), and receipt's 300-music-component acquisition target (`data_receipt.json:55-57`) are project policies. None alone establishes sufficient precision for every metric, effect size, dependency structure, or deployment distribution. More clips from an existing connected component do not add independent components.

### S6 — Selection-time and fixed-prediction intervals have limited scope

E4 selects among five mixture weights using dev EER and then reports a bootstrap on that same selected dev result (`eight_ablation.py:237-247`). That interval is a descriptive interval after selection, not independent confirmation of the chosen weight. The subsequent test score uses the dev-locked weight (`:248-258`) and is stronger within the historically exposed diagnostic scope.

E6 selects the layer using mean dev EER across seeds before test evaluation (`eight_ablation.py:273-289`). The saved lock chooses MERT12 and Anti24 with `test_used=false` (`output/selection_E6_locked.json:5-10`). No direct test-label-based layer choice was found in this code. It subsequently reports every layer on test, however; later choosing among those results is adaptive reuse. The post-run MERT12-vs-MERT24 intervals explicitly condition on fixed fitted heads and the existing split (`postrun_e2/E6_mert12_vs_l24_test_bootstrap.json:2`). They do not include retraining, model-selection uncertainty, or correction for a broad experiment search. Do not relabel the three seed CIs as three independent replications or an interval for the probability ensemble.

### S7 — Most saved thresholded metrics use a threshold fitted to the evaluated rows

`eight_fixed_io.evaluate` defaults to `threshold=None` (`:123-130`); `eight_metrics.binary_summary` then chooses the EER threshold from that input population (`:61-69`). E1/E3/E4/E6 test calls use that default. Their EER/AUC remain descriptive ranking metrics; the accompanying `fpr`, `recall`, and `decision_threshold` are test-label-fitted operating points, not evidence of a calibration-fixed deployed threshold. E5 (`eight_probes.py:207-218`) and E8 (`eight_ablation.py:304-315`) explicitly pass calibration-derived thresholds and avoid this particular limitation.

## OOF checks that held

The independent OOF sub-audit inspected all six saved A/B seed artifacts and 30 saved OOF checkpoints. Its executed, hash-bound results are saved in `evidence/evaluation_oof_verification.json`, with `all_measured_invariants_pass=true`; replay source is `evidence/evaluation_oof_replay.py`. Each artifact covers 1,469 unique training IDs across 97 atomic groups, with each ID held out exactly once. Every held/fitting/early-stop row and group intersection is empty and each fold partition union covers all 1,469 rows. Held/fitting/early row counts are 253/938/278, 306/924/239, 315/978/176, 293/958/218, and 302/768/399.

All 30 checkpoint hashes, fitting IDs, early-stop IDs, and selected epochs match their receipts. Ensemble members exactly equal the saved seed artifacts; prediction and window-score mean errors are 0.0. OOF and global validation overlap in zero IDs and zero groups. Raw parent IDs, artist IDs, speaker IDs, and audio SHA-256 values do not overlap across held/fitting/early partitions.

Relevant safeguards: `redesign_oof.py:118-131`, `redesign_oof_validation.py:28-57,88-96`; population construction: `redesign_train.py:96-110`. OOF covers the training population, not all 1,696 accepted inputs. Music truth is known on 989 rows (145 real / 844 fake); 480 speech rows have null music labels. Voice residual training was skipped (`results/redesign_h100_20260917/downloaded/unpacked/output/experiments/summary.json:64-67`). No demonstrated saved OOF leakage or omission was found. Current `redesign_fit.py` differs from the saved version; the above checkpoint provenance checks inspect the saved evidence directly.

## Reviewable next actions

1. Fix and verify sampler coverage before spending compute; preserve historical results and retrain affected arms when authorized.
2. Correct report consumers to use per-split unique IDs and connected-component counts, especially E5 test 1 instead of all-split 4.
3. Add invalid-bootstrap fraction and per-class component support to the generic eight adoption decision.
4. Keep fixed-model, historically exposed results diagnostic. For a new generalization claim, lock the candidate and target estimand before opening a new source/generator-disjoint population; choose acquisition size from desired precision and actual independent-component structure.

## Reproduction

From `/Users/choiwuseck/Documents/ChatGPT/데이콘`, run the existing CPU environment:

```sh
.venv/bin/python results/precision_audit_20260918/evidence/evaluation_statistics_replay.py
.venv/bin/python results/precision_audit_20260918/evidence/evaluation_oof_replay.py
```

The executed replay re-created `evaluation_statistics_sampler.json`, `evaluation_statistics_sampler_binding.json`, `evaluation_statistics_probe_populations.json`, and `evaluation_statistics_populations_bootstrap.json`. It verified the saved draw hashes and hyperparameters for the 30 balanced checkpoints plus the two E1 parent-control seed41 checkpoints. It read the actual prediction tables for split counts and re-ran the explicitly synthetic rare-class failure case. `evidence/evaluation_statistics_replay_manifest.json` records the replay command, input SHA-256 hashes, script hash, and Torch version. No fitting occurs.

The executed OOF replay returned exit code 0 and recreated `evidence/evaluation_oof_verification.json`: six seed artifacts, 30 checkpoint receipts, and two ensembles passed the recorded invariants. Its JSON hashes each consumed artifact/checkpoint, current source, and replay script. The source-version comparison is reported separately from the measured saved-checkpoint invariants.
