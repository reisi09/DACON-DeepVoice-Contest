# Round 3 synthesis (main Claude) — sign off or raise BLOCKING objections only
Inputs: codex_round10/11 (long-running Codex), fresh_r1/r2 (fresh Codex), claude_fresh_verifier_r1/r2, dataset_scout_r1,
claude_main_r1*/r1b (actual weights). Every reviewer agrees: the current (label, source, generator) cell weighting plus
global duration balancing concentrates training mass (voice: singing 15% of fake vs 41% of real, Korean 5.7% of fake;
music: one legacy lineage component 22-29%, Kish-equivalent groups 12-19) and is fixable without new features.

## 1. Primary reweighting W (pre-registered, one shot)
- Heads retrained: DF-Arena voice MLP, AntiDeepfake voice ResMLP (both with rule L as in S6/S7), WavLM-8 member (original labels),
  music WindowMIL on S7 features (MERT 6+12 | EAT 16 | SSLAM-pt 5, backing stem). Encoders, inputs, seeds 41/42/43, epochs/
  patience, presence, voice fusion weights 1/0.5/1 FIXED; FILE weights FIXED at S7's 0/9/1 (no refit: removes the FILE
  confound); per-member CDF tables refit on val with the unchanged procedure (heads changed).
- Voice strata (mapping fixed from source + parent provenance before any result): KS = Korean speech, XS = other/unknown-language
  speech, SM = speech mixed onto music (constructed_mix with speech parents), SG = singing (a cappella + songs).
  Target mass per class: q = KS 0.25, XS 0.45, SM 0.05, SG 0.25 (the same q in both classes = within-stratum 50:50).
  A stratum missing in one class keeps its q in the other class only; masses renormalised per class.
- Music: global class 50:50 only (no strata targets; the pathology is lineage concentration, handled by the caps).
- Base row weight inside (class, stratum): b_i proportional to max(n_c, 20)^-1/2, cell c = (class, stratum, source-family,
  generator); source-family = source with obvious legacy aliases merged (existing:SONICS=sonics, existing:HAIM=haim, ...).
- Duration: within each (stratum) for voice / globally for music, log-duration 10-quantile bins, class histogram matched to the
  pooled one, multiplier clipped to [0.5, 2]; then (class, stratum) sums restored to their targets.
- Final caps applied LAST and simultaneously by iterative capped allocation (<= 50 iterations, until no violation): per class,
  row weight <= 5x the class mean, lineage component (group_id) sum <= 2% of the class mass; no renormalisation after.
  Feasibility checked on the admitted rows of every head before training; if infeasible, report and fall back to C
  (no cap relaxation after seeing results).
- Logged before training: per head and class, mass by stratum/source/language/top groups, Kish effective rows and groups.

## 2. Alternative C (cheap control, trained in the same batch)
S7's actual post-duration weights + the same final caps only (isolates concentration relief from stratification).

## 3. Gates (before an LB slot)
- Validity (must pass): package integrity, training-vs-runtime parity (VOICE <= 0.02, MUSIC <= 0.01, FILE <= 0.02), CUDA offline
  run, peak GPU allocated <= S7 + 0.3 GiB, L4 time estimate <= 45 min.
- Catastrophe (must pass), DB RT held-out vs S7 with identical protocol: natural-source subset (drop constructed_*,
  existing:constructed_music, HAIM crossfade/concat edits) dADS >= -0.005; no component EER worse by > 0.02 on the natural subset;
  natural real-source FPR (val TPR-95 threshold) not up by > 5 pp on any source with >= 50 rows.
- Ordering (not pass/fail): W before C unless C beats W by >= 0.003 on the natural subset. Reported only: full DB dADS,
  group bootstrap, source jackknife, composition-shift scenarios. Rationale: best-ever scoring and a DB that cannot see the LB
  domains make superiority gates on the DB the wrong filter; they are kept as diagnostics.

## 4. Data additions (prepared in parallel, trained only on top of the chosen weighting)
- D1 (certain, no download): 2x2 vocal/accompaniment remixes from cached fp16 stems, donors assigned to splits by lineage BEFORE
  mixing; realV+fakeM (real vocal stems from vocalset/csd/openmic-voice + AI accompaniment stems from SONICS/HAIM/Echoes fakes),
  fakeV(sung)+realM (AI vocal stems from SONICS/HAIM + real accompaniment from openmic/musan/FMA), plus matched controls from the
  same process (realV+realM, fakeV+fakeM); vocal/accompaniment active-RMS ratio -12/-6/0/+6 dB; ~150 per combo; every mix goes
  through the real runtime separation + feature path (Engine-based feature dump).
- Conditional: CtrSVDD UNMIXED only (ND licence; ACE-Studio bona fide rows removed) if the user downloads by 13:00;
  GTSinger-ko as real vocal donors for D1; MUSDB18-HQ unmixed songs; AI-Hub 465/473/71544 only on deneb-kr (kr-west) if the
  user approves by 12:00 and in-Korea extraction starts by 14:00 (else excluded); only head weights leave Korea.

## 5. Slots 2026-09-29
1 (00:00): W (fallback C, then D1 on S7 weights). 2: if slot-1 LB >= S7 + 0.001 -> W + D1; if within +-0.003 -> the other of
W/C + D1; if worse -> S7 weights + D1. 3: extension of the best (or AI-Hub variant if ready), last resort S8 with FILE 0/9/1.
Never S9. All packages verified by 08:00.

## Questions to each reviewer
(a) Any BLOCKING objection to sections 1-5 (a flaw that would invalidate the result or break the rules)? Give the minimal fix.
(b) Anything essential missing? Keep it short; non-blocking preferences are not needed.
