# A100 capacity decision — 2026-09-18

This is a local, read-only capacity study. No compute was started, inputs uploaded, executable code changed, or run deadline created. The latest user instruction is **finish acquisition/validation/reporting, then stop**; GPU work is deferred and requires a later instruction. Additional acquired original audio plus metadata must remain **strictly below 20 GB**. Earlier whole-bank acquisition targets are superseded; they are not retained counts or experiment commitments.

**Future experiment recommendation:** freeze an experiment containing approximately **2,000–2,500 eligible eight-second music training rows**, plus the existing **1,001 nontraining rows**. Keep **125 updates × effective file batch 128 × seeds 41/42/43** initially, and admit parents according to the actual source/generator quotas below. This is a capacity candidate, **not a measured four-hour completion guarantee**. The current AST path has no A100 end-to-end timing evidence. If the user later resumes GPU work, a representative A100 canary should establish that all 12 fits and final inference fit inside the original four-hour deadline before processing the full subset. This report does not authorize that canary or a launch.

## Four separate quantities

| Quantity | Definition / limit |
|---|---|
| Acquired bank | Retained original audio and metadata; additional collection must remain below 20,000,000,000 bytes under the conservative decimal-GB interpretation. Count temporary/archive duplication in the physical storage ledger as well. A 19 GB operational target leaves room for final metadata. |
| Admitted bank | Acquired parents with documented usage rights, usable audio, defensible labels and required split exclusions. Neither file count nor a successful decode proves musical quality. |
| Experiment manifest | Frozen, representative selection from admitted parents, existing core data and locked evaluation groups. Only these rows enter feature extraction. |
| Actually exposed population | Unique parents and rows present in each seed's generated draw schedule. Current code requires every training **parent** at least once; it records unseen row IDs but does not reject them. |

The bank-byte cap does not mean the derived GPU cache will fit in 20 GB. Do not put the whole retained bank into `manifest.csv`. Speech-only additions are not eligible music training rows in the current trainer; preserve the bounded presence evaluation and keep unused speech originals outside this run's manifest.

## Measured historical evidence and its limits

| Local receipt | Recorded measurement | Valid use |
|---|---|---|
| [H100 eight extraction](../eight_h100_20260917/downloaded/unpacked/output/extraction.json), [GPU identity](../eight_h100_20260917/downloaded/unpacked/output/gpu_preflight.json) | 7,095 newly extracted records + 6 cached; 29,231 new windows; 3,759.8296 s total; peak allocated 11,026,868,736 bytes / 10.270 GiB. Derived 7.7746 new windows/s. | Historical mixed extraction reference. No AST-prefix stage. Not an A100 throughput measurement. |
| Same receipt, `core_music_base` / `core_music_processed` | 3,963 / 9,402 windows in 351.1045 / 800.0685 s; derived 11.2872 / 11.7515 windows/s. | Workload-sensitive historical references; do not substitute these for the long confirmation or speech workloads. |
| [H100 redesign pilot](../redesign_h100_20260917/downloaded/unpacked/output/pilot.log) | 100 fresh records / 1,644 windows / 267.1914 s extraction; peak 7.770 GiB. Derived 6.1529 windows/s. | Real pilot timing. Its 4,351.8290 s full-stage forecast was an estimate, not a completed-stage measurement. |
| [H100 six extraction](../six_h100_20260917/downloaded/unpacked/output/extraction.json) | 730 fresh + 5 cached records / 3,339 total windows / 501.2528 s; peak 7.017 GiB. | The derived 6.6613 windows/s includes cached windows. |
| [A100 head benchmark](../mert12_a100_20260918/downloaded/unpacked/output/mert12_full/batch_benchmark.json) | Batch 32 / 128 / 256: 24,262.67 / 95,906.29 / 195,283.24 draws/s; peak allocated at most 76,996,096 bytes. | Explicitly throwaway frozen-head work. It cannot estimate MERT extraction, AST-prefix extraction or AST backward time. |
| [A100 original runtime](../mert12_a100_20260918/downloaded/evidence/output/final_smoke_original/runtime_receipt.json), [permuted runtime](../mert12_a100_20260918/downloaded/evidence/output/final_smoke_permuted/runtime_receipt.json) | Same 4 files / 12 windows: 52.3721 versus 32.5630 s; peak 6.106 GiB allocated / 6.527 GiB reserved. | Complete older inference pipeline with material cold/warm variation. Not a clean encoder benchmark. |
| [A100 parity diagnostic](../mert12_a100_20260918/attempt5/downloaded/unpacked/output/mert12_batch_parity_probe.json) | 4 files × 3 windows; 11.9627 s; 1.713 GiB allocated / 2.195 GiB reserved. | Both singleton and batched forwards are included. No independent large-dataset rate. |

The actual A100 was `NVIDIA A100-SXM4-80GB`, with CUDA-reported capacity 85,094,825,984 bytes. The earlier head receipt used Torch 2.9.1; the later parity/runtime evidence used the 3.11.15 Python / Torch 2.7.1 environment. None measures the newly pinned AST pipeline. The H100 full-refit receipt's 42.37 s is **all-cache loading**, not fresh extraction, and is excluded from forecasts.

## Exact code-derived cache budget

Current [features](../../localization_features.py) use 64,600 samples per window and a 32,300-sample hop at 16 kHz, with one deduplicated tail window:

```text
W(N_samples) = 1 + ceil(max(0, N_samples - 64600) / 32300)
AST prefix bytes = windows × 1214 × 768 × 2 = windows × 1,864,704
Saved stems bytes ≈ audio_seconds × 16000 × 2 stems × 4 bytes
Other aligned arrays ≈ windows × 9,232 bytes
```

Durations 4 / 8 / 16 / 30 / 60 seconds create 1 / 3 / 7 / 14 / 29 windows. Each AST window still uses native 1,214 tokens, including attention over padding; an eight-second row therefore occupies **5,594,112 bytes of AST prefix alone**. These are array payload calculations, excluding headers, input copies, models, runtime wheels, checkpoints and export archives. Feature completion hashes the entire prefix; training startup hashes it again.

The preserved pilot manifest is SHA-256 `daf9317856e63fba6cdba04f8fa28c48afa447fffe9046b246571ddecdbebff4`:

| Split | Rows | Windows | Audio seconds |
|---|---:|---:|---:|
| Train | 1,469 | 4,407 | 11,752 |
| Dev | 164 | 492 | 1,312 |
| Calibration | 200 | 600 | 1,600 |
| Historical test | 127 | 381 | 1,016 |
| Confirmation | 352 | 3,880 | 8,464 |
| External confirmation | 130 | 389 | 1,037 |
| Presence confirmation | 28 | 312 | 662.97375 |
| **Total** | **2,470** | **10,461** | **25,843.97375** |

Pilot payload: AST **19.5067 GB**, stems **3.3080 GB**, other arrays **0.0966 GB**; total approximately **22.9113 GB**. The 352 duration/partial confirmation rows account for 37.1% of all windows. Preserve this locked evaluation workload instead of discarding its long examples to enlarge a headline training count.

With the same 1,001 nontraining rows / 6,054 windows / 14,091.97375 audio seconds and all training rows eight seconds:

| Eligible train rows | Total rows | Total windows | AST prefix GB | Total cache payload GB |
|---:|---:|---:|---:|---:|
| 1,469 — pilot | 2,470 | 10,461 | 19.507 | 22.911 |
| 2,000 | 3,001 | 12,054 | 22.477 | 26.440 |
| 2,500 | 3,501 | 13,554 | 25.274 | 29.763 |
| 3,000 | 4,001 | 15,054 | 28.071 | 33.086 |
| 4,000 | 5,001 | 18,054 | 33.665 | 39.732 |
| 10,000 | 11,001 | 36,054 | 67.230 | 79.607 |
| 50,000 | 51,001 | 156,054 | 290.995 | 345.439 |

The present fixed **30 GB free-space check after unpacking is insufficient for an expanded 2,500-row training subset**, whose cache alone approaches 30 GB. For that candidate, require at least **40 GB free after input extraction** as an operational estimate: about 30 GB cache plus 10 GB for output/archive/temporary space. Replace the constant with the actual manifest forecast and measured output allowance. This is runtime working storage, separate from the user's additional acquisition cap.

## Exposure is constrained by source cells, not just total rows

[The corrected sampler](../../eight_heads.py) supplies 64 draws per class per update. With 125 updates it supplies 8,000 draws per class and 16,000 total draws per seed. It cycles source, then generator, then parent; it randomly chooses a view within the selected parent.

For a class with `S` sorted sources, source index `j`, `G` generators in that source, generator index `k`, and `U` updates, a cell receives:

```text
cell_draws = max(0, 1 + floor((64*U - 1 - (j + S*k)) / (S*G)))
updates_required_for_P_parents = ceil((j + S*(k + G*(P-1)) + 1) / 64)
```

Freeze the real manifest and run all three schedules locally before extraction; these formulas are planning bounds. Parents shared across cells need a conservative per-cell audit as well as the trainer's global parent check. If every transformed **row** must appear, the current random-view sampler does not guarantee that and requires a separate explicit decision.

The pilot has 21 fake source strings. If FakeMusicCaps and SONICS each add one source, retaining the existing strings gives 23 fake sources. At 125 updates this gives only **69–70 FakeMusicCaps parents per generator** for its five-generator source, and **173–174 HAIM/SONICS parents per generator** for each two-generator source. Adding thousands in those cells would fail the exposure guard, which currently runs only **after all expensive feature extraction**.

| Updates / seed | Draws / class | Guaranteed floor per FakeMusicCaps generator, S=23/G=5 | Floor per HAIM/SONICS generator, S=23/G=2 | AST training windows, all 3 seeds |
|---:|---:|---:|---:|---:|
| 125 | 8,000 | 69 | 173 | 144,000 |
| 200 | 12,800 | 111 | 278 | 230,400 |
| 250 | 16,000 | 139 | 347 | 288,000 |
| 300 | 19,200 | 166 | 417 | 345,600 |
| 500 | 32,000 | 278 | 695 | 576,000 |

At 125 updates, a concrete quota illustration is original831 + 128 existing partial/sham rows + 200 additional FMA + 400 total Jamendo training rows + 5×64 FakeMusicCaps + 2×160 HAIM + 2×160 SONICS = **2,519 training rows**, **3,520 total rows**, **13,611 windows**, approximately **29.889 GB cache**. Bank-derived training rows are 1,560, compared with 510 in the pilot. This is an illustrative capacity allocation, not a claim that those licensed, independent originals are available or fit the under-20 GB bank. Use the final retention/admission ledger to reduce quotas while preserving each generator and artist/parent diversity. Recompute source counts and exposure if names, grouping or generators differ. Existing Echoes generator/mode strings are separate sources, so silently merging them changes the experiment's exposure objective.

## Four-hour planning envelope and required measurement

Current code runs HTDemucs per file, then batched MERT, then AST prefix, then fixed PANNs/DF-Arena on nontraining rows. Demucs and fixed-branch calls remain per-row. Increasing the encoder microbatch does not accelerate every stage. Encoder candidates are 16/32/64/128 windows; training candidates are 32/64/128 files. Choose from actual throughput and the existing 80% allocated-VRAM guard; prior small head allocation does not justify choosing AST batch128 in advance.

The 2,500-train-row candidate has 13,554 windows. Merely dividing by historical H100 rates 6.1529–11.2872 gives **20.0–36.7 minutes**. This is a **nonportable historical reference**, missing the current AST work and changed implementation; it is not an A100 stage estimate. For 50,000 training rows, the same optimistic reference is already **230–423 minutes**, before new AST work, training or export. Whole-bank processing is incompatible with this experiment's practical four-hour plan.

The principal new cost is AST:

- 125 updates × 128 files × 3 windows × 3 seeds = **144,000 suffix training-window presentations**.
- Its per-seed 32/64/128 benchmark adds **10,080 suffix training-window presentations** across three seeds, if all candidates run. Repeated largest-item trials can benefit from CPU page cache and are not a random-parent I/O benchmark.
- Dev evaluation every25 updates adds **7,380 suffix forward-window presentations** across three seeds.
- Final AST inference currently predicts **every manifest row**, including train: another `3 × total_windows`, or **40,662** forward windows at the 2,500-row candidate.
- AST prefix encoder benchmarking adds 720 prefix windows if all four candidates run. MERT adds its own 720-window benchmark.
- Training alone reads/copies approximately **268.5 GB of float16 prefix payload** through repeated mmap/concatenate operations; OS page-cache reuse can reduce disk reads. It does not eliminate CPU-to-GPU transfer or backward computation.

A proposed allocation, explicitly an operational target rather than a measured duration, is:

| Stage allowance | Seconds | Minutes |
|---|---:|---:|
| Startup, mounted-input readback, unpack and preflight | 900 | 15 |
| All feature stages, their benchmarks, hashing and canary | 5,400 | 90 |
| All 12 fits, dev selection, locking and final inference | 6,000 | 100 |
| Contingency for measured I/O/runtime overhead | 1,500 | 25 |
| Final export/download reserve | 600 | 10 |
| **Original absolute maximum** | **14,400** | **240** |

For a 2,500-row / 125-update plan, a useful conservative trainer admission condition is:

```text
154080 / measured_AST_training_windows_per_second
+ 48042 / measured_AST_forward_windows_per_second
+ measured_or_reserved_other_9_fits_and_IO_seconds
<= 6000 seconds
```

Training and inference have different costs; do not divide their sum by one unqualified throughput. If both paths managed only 40 windows/s, the AST terms alone would take about 5,053 seconds, leaving roughly 947 seconds for the nine small-head fits, their evaluation, model loading, hashing and reporting. At 250 updates the corresponding AST workloads become 298,080 training and 55,422 forward windows: doubling exposure can consume the entire remaining budget even when dataset size is unchanged. These are arithmetic scenarios, **not observed A100 performance**.

For features, use measured per-stage canary timings on representative 4/8/16/30/60-second audio, both train and fixed-branch evaluation rows, plus measured prefix-write/hash bandwidth. Forecast HTDemucs by duration/workload, MERT and AST by windows, and fixed branches by their evaluated rows/durations. Include loading and benchmark overhead. Do not infer all of this from the current encoder warm-up: it occurs only after the full Demucs loop.

## Minimum preparation changes for root to consider

1. Produce a frozen, byte-audited experiment subset independent of the retained bank. Preserve locked artist/parent/prompt/speaker exclusions and unknown labels; record admitted, selected and exposed counts separately.
2. Move or duplicate the **actual three-seed sampler exposure audit before feature extraction**. Fail locally if any parent is unseen; use explicit per-source/generator caps. Keep 125 updates until a measured runtime forecast justifies more.
3. Calculate total windows, audio seconds and cache bytes from the final manifest; use that forecast for free-space admission. The 30 GB constant must not approve a near-30 GB cache without headroom.
4. Add an early representative canary and an explicit feature/training stage budget within the immutable original deadline. The current common `deadline-600` stops runaway compute but can leave no time for training. If the full plan does not fit, reduce the frozen subset before training or export an explicit partial result; never reset the deadline or silently omit an arm/seed.
5. Retain the measured 32/64/128 training-batch decision and distinguish microbatch from effective batch128. Larger microbatch does not increase sample exposure. Any change to source hierarchy, steps, held-out duration coverage or final prediction scope must be frozen and reported before scoring.

No implementation or launch decision is made by this report. Actual current-path completion and VRAM remain unmeasured. If a later instruction resumes this experiment, retain its four-hour hard stop and 600-second export reserve. For the current task, stop after collection, validation and reporting.

## Inspected code identities

These identities prevent a later code edit from being mistaken for the version budgeted here:

```text
localization_features.py  a39bcce9a177af8ed185ac581394c89bd4e72de658090d74a3acb9a17e93e0cc
localization_train.py     d23676f9e0792b32f1abb844985f2e32e25054437bd658297aaba3740f106c9a
eight_heads.py            c3e040ad607e0f40046358a7bd6069d0fe389f6c6c6201c39992060838eeba09
localization_bootstrap.sh fb4d8e2fa5dae481fd23908bd199e75c8b8ca6a0974b8e4d8a1974780f9ba41c
```
