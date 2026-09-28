# Official constraints reread 2026-09-17

Primary rules, description, evaluation pages were opened by root. Rules lines105–115 prohibit updating on private evaluation data and cross-file prediction/statistical correction. Within-file segment aggregation is permitted. Description lines121–126 define generated components as FAKE and component-preserving quality/noise/gain processing as REAL. Neural codec/vocoder reconstruction is not specifically named: blanket claims that every reconstruction is REAL or FAKE are unsupported.

Description lines108–144 confirms1,200 files,4–60s,16kHz,mono/stereo,MP3/WAV/FLAC,telephone examples,L4 22.4GiB,60min total inference and separate10min installation.

Consequences: training-only reference statistics and fixed calibration are viable; test-batch NMF/adaptation/pseudo-labeling is excluded. No code/enhancement operation receives an automatic label merely from the presence of a neural network. Voice conversion/new synthesis must remain separate from component-preserving processing.

These are primary-source-only exceptions in the claim graph: the competition organizer is authoritative. This research does not establish current leaderboard position or account score.
