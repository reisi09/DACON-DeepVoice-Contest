# Skeptic review: INTEGRATION lane (work3/integration)

Reviewed: every file in this folder (`fit_ft_fusion.py`, staged `full_bank_submission.py` and
`package_datav4_submission.py` diffed against the project copies, `check_mert_fp16.py`, `ft_runtime_parity.py`,
`smoke_runtime.py`, `standin/`, `out/**` including the three package receipts). Cross-read: `work3/DESIGN.md`,
`brief.md`, `antideepfake_runtime.py`, `component_fusion.py`, `decision_heads.py`, `data_v4/extract_backbones.py`,
`verify_datav4_submission.py`, `work3/trainer/{finetune_ad.py, score_ft.py, run_ft.sh}`, `work3/lessons/critique.md`,
`submissions/DataV4_S4_MLPHeads/submit.json`.

**Verdict: fix-needed.** Nothing is fundamentally broken: splits are clean, polarity is right, the gate can pass
and chooses on val only, and the packager reproduces S4. But:
- the one script documented to run on VESSL crashes at import there;
- the staged runtime has never run end to end in any package;
- nothing ties the gate result, the ft CSVs and the packaged ft checkpoint together.

## CPU checks run (2; scripts and outputs in scratchpad `integ_skeptic/`)

1. **check1.py (CSV only, a few seconds).**
   - Question: do train/val/test share anything?
     - Answer: no. ID, group_id, component_id and audio_sha256 overlaps are 0 for train&val, train&test and val&test.
     - `bank_trainval.csv` holds only train (30,658) and val (4,931) rows; `bank_test.csv` holds only test (6,450).
   - Question: does the fp32 window max saturate?
     - The stand-in zero-shot values never reach 1-1e-6.
     - If every logit gap is doubled (a plausible fine-tuned model), 23-31% of rows come out exactly 1.0 in fp32.
2. **check2.py (three real `fit_ft_fusion.py db` runs, 200 reps, 373 MB RSS).** Input is one combined val+test CSV
   with a `voice_silent` column, the exact shape `score_ft.py` writes.
   - The db mode accepts it.
   - **Oracle positive control:** ft = label on voice rows.
     - Gate **passes**: held-out RT dADS +0.02609, P(>0) = 1.0. The val choice is w=1.
     - So polarity is correct and the gate can be passed at all.
   - **Saturation cost:** same ranks, fp32 sigmoid(2·logit) with 23.5% ties at 1.0, vs the float64 version with no
     ties.
     - Held-out deltas are identical to 5 decimals for R/A under RT (-0.01489/-0.00841).
     - A under AC differs by 0.00009.
     - The saturation concern is **refuted** for this fusion.

## Findings (most severe first)

**F1. `fit_ft_fusion.py` crashes at import on VESSL. [high, confirmed]**
- Line 31, `ROOT = Path(__file__).resolve().parents[4]`, raises `IndexError: 4` for `/root/ft/code/fit_ft_fusion.py`
  and `/root/ft/fit_ft_fusion.py` (checked).
- It runs before argparse, so passing `--code/--bank/--stacking` does not help.
- This breaks the "plain paths/CLI args" requirement for the one VESSL command in how_to_run.
- `ft_runtime_parity.py` and `check_mert_fp16.py` use the same pattern, but they are Mac-only by design.
- Fix: `P = Path(__file__).resolve(); ROOT = P.parents[4] if len(P.parents) > 4 else P.parent`.
  - Simpler: run `db` on the Mac only. It needs just the two ft CSVs (a few MB) and takes about 30 s.

**F2. The staged runtime has never executed `Engine.predict` in a package. [high, process gap]**
- `out/receipts/S4_regression_submit.json` carries `model/runtime/full_bank_submission.py` sha `1cf4a25…`. That is
  the *project* runtime.
  - All 63 member hashes equal the shipped S4 package. This proves the staged **packager**; it does not prove the
    staged **runtime**.
- The stand-in R/A packages do contain the staged runtime (`56e48bc…`, `95f27d6…`), but they were never run
  (`verify_datav4_submission.py` was skipped).
- The staged Engine was exercised only through `Engine.__new__` + `load_fusion`/`fused_voice`, on 2-/3-layer cut
  detectors.
- So "for an S4 fusion the operations are identical to the current runtime" is true by inspection only.
- A runtime error costs a daily submission.
- Fix: the orchestrator runs step 4, `verify_datav4_submission.py --out <pkg_R dir>`, on the existing stand-in R
  package **now**, before the GPU results arrive. Then run the no-`--ft-parity` rehearsal of `ft_runtime_parity.py`.
  Do the same for A if A stays in play.

**F3. Nothing ties gate, fit and packaged checkpoint together. [medium]**
- `p1` takes `--option/--w` by hand. It never reads `ft_fusion_db.json`, and it will fit and emit an R/A heads dir
  even if `gate.pass` is false or the option/w differ from the db choice.
- No step records or checks the sha256 of the `model.safetensors` that produced the ft CSVs:
  - `run_identity.json` stores only the CSV paths;
  - the packager streams whatever `--antideepfake ft=DIR` points at.
- Step 5 does not catch this. Its runtime side and its offline side both use the packaged FT_DIR. Any of these would
  pass every check and ship tables fitted to a different model's scores:
  - a rescored run;
  - `last` instead of `best`;
  - `score_ft.py --bf16`.
- Fix:
  - `score_ft.py` writes a sidecar: ckpt sha256, precision, row count.
  - `p1` takes `--db ft_fusion_db.json`, refuses on `gate.pass == false` or an option/w mismatch, and copies the ckpt
    sha into `run_identity.json`.
  - The packager refuses when `sha256(DETECTORS['ft']/model.safetensors)` differs from it.
  - This is about 10 lines total.

**F4. Held-out is read twice, and option A departs from the fixed DESIGN. [medium, protocol]**
- `db` reports both options on held-out, and how_to_run tells the reader to "use boot_vs_S4['R<w>|RT']" for an
  R-only reading. That is a second look at test: if A fails and R passes, choosing R is selection on held-out.
- A departs from DESIGN.md in three ways:
  - package size is not unchanged: +1.31 GB, 0.404 GB under 10 GB;
  - it adds a second fp32 1B pass per file on L4;
  - it needs a new fp16 MERT artifact.
- The val rule picked A in both runs I made (stand-in and oracle) because 'ad' is strong on val (EER 0.0257). Expect
  it to pick A with the real ft too.
- The rule is also biased: the ft checkpoint is early-stopped on the same val, so val over-rates ft. That pushes the
  choice toward higher w and toward R. The held-out gate stays honest.
- Fix: pre-declare one primary configuration before the ft CSVs exist and gate only that one. DESIGN says R. Treat
  the other option's held-out numbers as diagnostics that cannot change the ship decision. The orchestrator must
  sign off on A explicitly.

**F5. The ft parity step barely tests the numerics the tables depend on. [medium]**
- Step 5 computes offline ft with `extract_backbones.py --device cpu` from float32 stems and compares it with the
  runtime on CPU. Same code, same device, so it proves plumbing only.
- The fusion tables are fitted on `score_ft.py` values: H100, from **float16-stored** VESSL stems.
- That gap is what could bias the ft CDF at runtime, and it is never measured.
- Fix:
  - Score the 10 parity files through the `score_ft.window_probs` path on the H100 stems, fp32 compute. One extra
    minute inside the scoring job.
  - Compare against runtime ft on the member itself (|Δ ft_max|, or better |Δ log-odds| ≤ 0.05; see lessons F3), not
    only against fused VOICE ≤ 0.02.

**F6. The gate population differs from the LB population. [low-medium, report only]**
- (a) Held-out includes rows that AntiDeepfake saw in pretraining. For S3 the gain on unseen rows was about half the
  gain on seen rows (`work/orchestrator/s3_unseen_check.py`). Ft is AD-derived, so report the gate delta on the
  unseen subset as well.
- (b) The trainer tiles sub-window clips (`Crops`), but `score_ft` and the runtime score them as one raw short clip.
  The bank has many sub-4 s rows, while the DACON test is 4–60 s. Also report dADS on rows ≥ 64,600 samples.
- Neither changes the gate rule, but both tell you how well a pass will transfer.

**F7. Cross-lane: `run_ft.sh` can stop before scoring. [low]**
- `run_ft.sh` asserts `roundtrip.ok` (Δprob ≤ 1e-3 on 4 files) under `set -e`.
- A correct fp16 export can exceed that near p ≈ 0.5 (lessons F3). If it does, `score_ft.py` never runs and
  integration gets no CSVs.
- If it fires: inspect `result.json`, then run `score_ft.py` by hand. Do not re-train.

**F8. Small items. [low]**
- Option A leaves 404 MB under 10 GB, so it cannot be combined with any other lane's weights (e.g. work2 EAT
  +0.62 GB). R leaves about 1.7 GB.
- how_to_run's `--split-note 'train 30,658 + held-out 6,450 / val 4,931'` describes the heads, not ft. Ft is train
  only, selected on val. Say so in `--heads-note`; the round-2 document review reads the manifest.
- Reusing a `p1 --out` dir leaves stale `voice_ad_*.pt`. The packager then refuses, which is loud and safe. Use a
  fresh dir, or have p1 refuse a non-empty one.
- `db --ft-col ft_top30` can print `gate.pass: true` for a member the runtime cannot compute. p1 refuses it later,
  so this is harmless but misleading.

## Checked and OK

- **Labels/splits:**
  - The trainer reads only `split == 'train'` rows of `bank_trainval.csv` for training and `val` for selection, and
    it asserts no train/val component overlap. `bank_test.csv` reaches only `score_ft.py`.
  - `db` fixes w and option on val before any test metric is computed.
  - `p1` uses val only. Check 1 found zero overlap on every key.
- **Fake-index polarity:** index 0 is fake everywhere:
  - trainer targets (`label 1 → class 0`);
  - `score_ft`/`window_probs` `softmax[:, 0]`;
  - runtime `windows()[1]` and the `voice_ad` heads (asserted).
  - The oracle control confirms the direction through the whole db gate.
- **Window/pooling:** `window_probs` matches `AntiDeepfake.windows`:
  - hop 32,300, plus a tail window;
  - a sub-window file is one unpadded clip, padded to at least 400 samples;
  - no batch padding.
  - Only differences: float16 stem storage, and cross-file batching (float noise).
- **Key layout:** `save_native` enforces the full native key set and exact shapes and writes F16. The runtime loads it
  through the unchanged `AntiDeepfake`. The trainer smoke test asserts the round trip.
- **DDP:** there is no DistributedSampler to pad or duplicate.
  - Each rank samples independently with replacement, seeded `seed*1000+rank`.
  - Worker seeds differ per rank.
  - Val shards are `val[rank::world]`, gathered and sorted.
  - `score_ft` parts are merged by ID.
- **bf16:** it is used only in training and trainer eval. `score_ft` defaults to fp32, and the runtime is fp32 with
  TF32 off (set in `AntiDeepfake.__init__`). The one gap is that nothing records CSV precision (F3).
- **Frozen params:** `requires_grad` comes from `trainable()` (layers ≥ 32, final encoder LN, classifier) plus
  `freeze_feature_encoder`, followed by a fingerprint diff. `run_ft.sh` asserts `changed_outside_trainable == []`.
  Layerdrop stays off because `encoder.training` is False.
- **Sizes:**
  - R: 8,288,341,878 B.
  - A: 9,595,957,456 B.
  - The packager enforces 1e10 compressed and 3.2e10 expanded.
  - Forgetting `--mert-fp16` for A makes the packager refuse (about 10.23 GB).
- **L4:**
  - R has the same compute and memory as S4 (S3 measured 24 min).
  - A adds about 6-8 min (lane4 estimate, not measured) and about 3.9 GB GPU. Two fp32 AD models + DF-Arena + MERT
    come to about 16 GB of 22.4.
  - CPU load peak is about 6 GB per detector, transient.
  - fp16 MERT loads bitwise-identical fp32 weights under the DACON transformers 4.57.6 (`out/check_mert_fp16.log`).
  - Beyond F2, I found no crash path.
- **Runtime guards:**
  - An unknown fusion member raises `SubmissionError`.
  - A missing `--antideepfake` mapping raises KeyError at package time.
  - Silent voice skips fusion, the same in fit and runtime.
