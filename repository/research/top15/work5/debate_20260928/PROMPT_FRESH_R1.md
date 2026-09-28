You are an independent reviewer with NO prior involvement. Read FULL_BRIEFING_20260928.md (sections 1-10), then verify its data
claims yourself from data/bank_trainval.csv and data/bank_test.csv (run read-only Python; recompute the numbers you rely on),
and read code/split.py, code/full_bank_batches.py, code/label_rules.py, code/sources.py as needed.
Do NOT read codex_round*.md or PROMPT_ROUND*.md (another reviewer's debate) in this round: form your own view first.
Questions:
1. Is data imbalance / coverage (missing cells: fake singing, AI covers, Korean songs, commercial real music; single-class
   sources) the most likely main cause of the 0.07 DB-vs-LB gap and of S8's negative transfer? Evidence for and against; at
   least two competing explanations, each with a test that would distinguish it.
2. Which imbalances matter for THIS metric (0.5 FILE + 0.2 VOICE + 0.3 MUSIC EER) and which do not (e.g. handled by the
   (label, source, generator) cell weights)? Quantify from the CSVs.
3. Remedies ranked by expected LB gain per hour with ~15 h left and 3 submissions (09-29 00:00-10:00): new data (which kinds,
   how much; include AI-Hub under the in-Korea processing constraint of section 10, and gated/token/terms datasets the user can
   unlock), constructed mixtures (real vocal + AI accompaniment, AI vocal + real accompaniment, SNR range), rebalancing, and
   evaluation changes. Be concrete about what can realistically be built, featurised and retrained in time (section 8).
4. How should any data addition be validated when the DB already failed to predict S8? Propose a validation that would have
   caught S8's failure.
Answer in Korean. Mark every claim VERIFIED / INFERRED / ESTIMATE / UNKNOWN. Concise but complete.
