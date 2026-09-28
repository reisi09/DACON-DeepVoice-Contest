# Round 17 — final decision (2026-09-28 17:20 KST). Both reviewers get the same file.

Read codex_round16.md and fresh16_r2.md (your two answers). You agreed: reuse existing P1 heads (0 retraining, 0 runtime
changes); clean selection = {s7_new, mix voice s7_new / music W_new} on newval; adopt the mix iff newval dADS >= +0.002 with
paired CI low > 0; E1 >= 0, E2-unlinked >= 0, E2-open >= -0.002 as vetoes; no sequential search after a failure.

## 1. Results of the pre-registered selection (paired group bootstrap 2,000, vs s7_new, lineage-linked rows removed)
| set | mix voice s7_new / music W_new | rule |
|---|---|---|
| newval unlinked (2,281 rows with presence) | +0.0013 [-0.0020, +0.0057], p>0 0.819 | FAIL (needs >= +0.002, CI low > 0) |
| E1 (4,822) | +0.0045 [-0.0003, +0.0085], p>0 0.970 | veto passed |
| E2 unlinked (12,351) | +0.0085 [+0.0058, +0.0118], p>0 1.000 | veto passed |
| E2 open unlinked (2,361) | +0.0011 [-0.0032, +0.0031] | veto passed (>= -0.002) |
(mix voice s7_new / music W_newB, not eligible for the clean test: E1 +0.0039 [-0.0008, +0.0090], E2 unlinked +0.0098.)
Coverage defect: 1,521 of 3,802 unlinked newval rows lack presence (AT-ADD train-pool rows dumped before the presence key;
back-filling needs the HTDemucs pass on a GPU), so FILE/ADS could only use 2,281 rows.
Supplement without presence, all 1,545 newval rows with a music label: MUSIC EER s7_new 0.0265 vs W_new 0.0265 (gain +0.0000,
CI [-0.0033, +0.0082]); AT-ADD music 0.0260 vs 0.0190, OpenBMAT 0.2910 vs 0.2910. newval AT-ADD music comes from the
TRAINING generators (easy, EER ~0.026); E2 has the unseen generators, where the W music head is better (0.0774 -> 0.0579).

## 2. Verification fixes done since round 16
w_runtime_parity.py: RECEIPT is required, verification device asserted, offline references asserted finite; pkg_chain.sh
passes RECEIPT. Re-run on s7_new / W_new / W_newB: all PASS (V/M/F <= 0.00064 / 0.00102 / 0.00021).

## 3. Budget
VESSL $19.00 minus the CPU workspace (~$0.3 so far); GPUs all paused. No GPU was used for rounds 15-17.

## 4. Final questions (Korean, VERIFIED / INFERRED / UNKNOWN, short)
(a) Under the pre-registered rule the mix is rejected. Confirm, or name a principled exception. Specifically: is completing
    the presence of the missing newval rows (~0.1-0.2 H100-hours, ~$0.5) and applying the SAME rule once a legitimate
    completion of the pre-registered test, or a forbidden second look?
(b) Final submission list for 2026-09-29 (3 slots) from the verified packages {s7_new, W_new, W_newB} (+ s7_new_ob whose CUDA
    check is pending, ~$0.5), with the order and a one-line reason each.
(c) The integrated design to record for after the competition (component by component) and the experiment plan that would
    have been run with more time and data (e.g. a larger clean selection split, Korean fake speech), with GPU-hour estimates.
(d) Any remaining BLOCKING issue.
