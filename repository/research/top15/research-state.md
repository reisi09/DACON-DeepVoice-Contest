# Research state (2026-09-27 15:10 KST)

## Completed
- Workflow 1 `wf_136c0253-8ff` (12 agents): lanes 1-4, research R1-R4, reviews v1-v4 (reviews re-run with S3 LB and
  time-only GPU directive). Result: `work/orchestrator/wf1_result.json`.
- Orchestrator check: `work/orchestrator/s3_unseen_check.py` (S3-style gain on member-unseen rows).
- S3 LB ADS 0.7853015873 (user, ~14:00 KST).

## Source revisions
- Project working tree (no local commit; HEAD f81e24c). Mirror `reisi09/DACON-DeepVoice-Contest` main at 0a890d1.
- Code changed after S3: `voice_heads.py` (new), `full_bank_submission.py` (arch-built heads), `full_bank_evaluate.py`,
  `component_fusion.py` (clip), `data_v4/train_backbone_heads.py` (--voice-arch/--ad-arch/--voice-epochs/--voice-patience).

## Leading hypotheses
1. Representation diversity through trained heads on new frozen backbones transfers (S3 +0.0153 LB).
2. Head capacity limits voice (R2 MLP +0.012 seed-avg held-out).
3. Music representation is the weakest component (flat learning curve; music-context voice EER 0.206).

## Rejected (reason)
- AI-Hub phone reals (S2a −0.0012 LB); zero-shot AntiDeepfake/SONICS fusion (held-out harm); AntiDeepfake-2B (no val
  support, too large); SLS layer weighting on pooled MERT (worse); inference-only pooling change (hurts ADS); 50→150 epochs.

## Unresolved
- S3 L4 runtime / 48-min fallback firing; daily reset time; queue latency; official test composition; DF-Arena/MERT fp16
  parity; candidate encoder offline loading; AI-Hub round-2 delivery; transfer ratio per change type.

## Outstanding dependencies / running
- VESSL: backbone extraction for remaining 26% (wsp-9683432r72m7, ends ~15:30 KST), local pull `pull_bb.sh`.
- Workflow 2 `wf_0a21e0e7-a05`: depth, diversity, backbones, stacking (+ skeptic per study) in `work2/`.

## Next action
Train S4 (`--final --voice-arch mlp --ad-arch resmlp --voice-epochs 150 --voice-patience 10`) on 100% rows once the
cache is complete; package; offline + runtime parity verification; hand to user for 09-28 slot 1.

## 15:20 update — fp16 storage check (work/orchestrator/fp16)
- MERT-v1-330M weights: 100% exactly fp16-representable -> fp16 storage is lossless (saves 0.63 GB).
- DF-Arena 1B weights: 0.007% fp16-exact, 0.002% bf16-exact -> lossy; the first parity run was INVALID because
  baseline/script.py was symlinked and resolves BASE_DIR to the original asset folder (DF-Arena loaded from fp32 original).
  Headroom with MERT fp16: ~2.3 GB (enough for one ~0.3B encoder in fp16). DF-Arena fp16 only after a real-copy parity test.

## 16:00 KST update
- S3 L4 runtime 24 min (user) -> fallback never fired; S3 compliant. Headroom ~36 min.
- Backbone cache complete (42,039). S4 trained on 100% rows (P1 protocol), packaged `submissions/DataV4_S4_MLPHeads`
  (sha beeda9a6), time fallback removed, CPU verification + runtime parity PASS (max |d| 3.4e-4). Ready for 09-28 slot 1.
- VESSL paused, credit $5.57.

## 16:00 KST — workflow 2 done (wf_0a21e0e7-a05; result work2/wf2_result.json)
- depth: no ship (held-out -0.0002); all-row DF MLP +0.0021 (already in S4). diversity: not the limiter; no data adds.
  stacking: S5 = S4. backbones: MuQ-large-msd-iter (+EAT-large) chosen; extraction running on VESSL from /root/newbb
  (FUSE mount served a stale script; local copy used), 95-min deadline, ~2.6 files/s, manifest = R2 31,270 rows ordered
  val+test -> music train -> rest.
- Next: when extraction ends (~17:25 KST) pull /root/newbackbone_cache, run work2/backbones/run.py + ads.py gate
  (locked val rule in work2/backbones/choice_lock.json), build S5 if the gate passes.
