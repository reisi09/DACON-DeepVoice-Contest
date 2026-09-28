# Round 20 — check the report to the user (2026-09-28 17:37 KST). Both reviewers get the same file.

Round 19: both chose slot 3 = mix (if it passes) > s7_new_ob (if CUDA ok) > W_new, the same order in every LB band, and
withdrew the causal reading of section 4.0. FINAL_REPORT_DRAFT.md (Korean) is what goes to the user.

(a) Check every number and claim in FINAL_REPORT_DRAFT.md against the files in this folder (virtual_configs_eval numbers are
    in ROUND16_CROSS.md, the selection results in ROUND17_FINAL.md, composition in code_r14/new_data_summary.csv).
    List only errors or misleading statements, each with the fix.
(b) Order of slots 1-2. The order no longer depends on LB results, so it matters only if an upload or scoring fails or runs
    past 10:00 KST (each zip is 9.75 GB; scoring about 30-40 min). Top-15 needs about ADS +0.022 over S7; offline only
    E2-type gains reach that (W_newB +0.047, s7_new +0.0285; E1 gains are <= +0.0064). Counterpoint: W_newB has the higher
    Korean real FPR (E3 0.040 vs 0.027) and lower E1. Keep s7_new first, or put W_newB first? One line reason.
(c) Anything BLOCKING before the report is sent. Korean, VERIFIED / INFERRED / UNKNOWN, short.
