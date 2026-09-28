# Trainable V4 execution ledger

User authorization 2026-09-13: make the notebook and run the designed experiment now, using Aside Colab and Drive. A100 allocation ceiling remains 7200 seconds. No GPU allocated yet.

Plan: .omo/plans/trainable-recovery-v4.md at commit f69f4bf.
No tool goal is created: user requested execution, not a separate goal. This ledger tracks the work.

## Task state
- [x] 1. Data/model protocol pinned — actual source and metadata receipts saved
- [x] 2. Full original acquisition, disjoint panels and archives — independently verified and uploaded
- [x] 3. Full-FT adapters and meaningful CPU gradient/feature checks — independently verified; A100 profiling remains task 4
- [x] 4. Actual Colab run 20260913_074847 — all five pipelines completed at source a01a50e0c774f181c20238109127c5b81cc95dc1; legacy P3 paired RNG proof is absent
- [x] 5. Final development decision — G0 retained; no candidate passed, so refit and sealed were not entered
- [x] 6. Comparison results and executed capture saved to Drive; release 1789288090.7249749; UI confirmed 0 sessions and 49.02 units
- [x] 7. Matched P1/P3 recheck 20260913_084740 — both 10 epochs/350 updates, matched_complete true; identical seed/RNG/initial-weight receipts; release requested 1789290093.9671278
- [x] F1. Plan compliance audit — current goal/context reviews PASS at d2397ad; v4-current-review-ledger.json
- [x] F2. Code quality review — current code/security reviews PASS at d2397ad
- [x] F3. Actual notebook/Drive/runtime QA — current QA PASS; both matched arms completed; 8 final Drive artifacts byte-verified; UI 0 active sessions
- [x] F4. Evidence-bound outcome report — v4-results-report.md and fully executed CPU analysis notebook saved; G0 retained, no final refit/sealed/submission

## Shared implementation contract
- v4_validation/train.csv, dev.csv, sealed.csv; paths relative to v4_validation.
- CSV columns retain ID,path,group_id,split,source,FILE_FAKE,VOICE_FAKE,MUSIC_FAKE,VOICE_PRESENT,MUSIC_PRESENT; add panel.
- split=train/dev/sealed; panel=music_real/music_seen/music_ood/speech/mixture/splice. Real music participates in both seen and OOD metrics.
- Unknown labels are empty strings; all mixtures carry parent group union.
- Root-level protocol.json, provenance.json, integrity_report.json record identity, original/evaluation durations, licenses and exclusions. Audio is ignored. Archives contain v4_validation/ paths.
- Model module v4_model.py exposes load_model(kind,assets,device), prepare_windows(model,audios), window_logits(model,inputs), full-file prediction and bag pooling. Kind=sonics/ast. Owners may refine signatures immediately through direct coordination before implementation.
- Baseline remains graph_features.Extractor; metrics reuse graph_scoring/v2_scoring. Outputs keep protected voice/presence exact.
- GPU runner CLI train_v4.py --data PATH --baseline PATH --sonics PATH --ast PATH --output PATH --deadline UNIX; deadline is final safe shutdown, internally reserve refit/sealed/export.
- Root owns orchestration, UI/Drive dispatch and .omo state. Workers own code, tests and substantive execution verification. No worker may allocate GPU or control Aside before explicit coordination.

## Readiness and ownership receipts
- Aside readiness agent verified 53.34 compute units, approximately0/hour,0active sessions, disconnected runtime. No GPU/cell execution. Evidence .omo/evidence/v4-colab-ui/v4-colab-ui-manual-qa.md and v4-readiness-resources.png.
- Drive project root metadata verified id1jKgz89YyiNRFIAsmM7canCtm0QODanNT. Existing open.zip, SONICS archive and five AST asset parts present. Content SHA verification remains in notebook startup.
- Workers: v4_data_execution owns acquisition/provenance; v4_model_execution owns trainable adapters; v4_training_execution owns runner; v4_eval_execution owns gates; v4_notebook_execution owns builder. v4_colab_ui exclusively owns Aside UI and is waiting for launch.
- Data worker reports metadata assignment10dev/10sealed disjoint, training60old+10new. Actual download/decode/quotas still pending; metadata is not a completed dataset.
- Model worker completed real-weight CPU verification: SONICS 17,172,493 and AST 86,188,033 trainable parameters, every learned parameter has a nonzero gradient, early/middle/late/head updates measured. Quiet/loud batch-serial maximum differences 2.38e-7 and 5.96e-8. Independent model review underway; these checks are not performance results.
- Initial independent selection review identified recipe-winner binding, expected full-refit corpus binding, and sealed-inference provenance gaps. Evaluation worker is fixing them with trainer coordination. Actual selection/refit/sealed artifacts remain pending the real run.
- 2026-09-13 06:15 UTC: no A100 allocated. Data acquisition reports 68/90 TTA sets and 68/70 ATA artist sets downloaded; archive construction pending. Notebook generation correctly waits for the real runner and verified Drive receipt.
- Model independent re-review APPROVE after real AST sub-400-sample tail fix: v4_model SHA256 510a36e121e49707c5cd8d42408c9ba4394eda9c547fb94f25f7608d64051a7b; report .omo/evidence/v4-model-review.md. Not a final same-commit execution gate.
- Evaluation independent re-review APPROVE for implementation readiness: v4_evaluation SHA256 a3a8e307d1024f089748a1777265c1830ae22609c9df81013679499500172144 and v4_selection 602f46332acdab20453e4f3e9272a21c15f6edcb5487465e39137a879d64e72c; report .omo/evidence/v4-eval-review.md. Live selection/refit/sealed acceptance still pending.
- Data audit found additional TTA versions per reference/provider: 256 exact members being acquired beyond the first keyed-prefetch pass. Actual speaker identity exclusion is being checked separately from utterance-group disjointness. No data-completion claim until those audits finish.
- Data acquisition and independent final review completed. Actual rows: train1516 (1116 music-eligible,400 speech-only),dev210,sealed210. Full-waveform1936-source and archive audits passed; actual prior English speakers4 and new German speakers2 are disjoint. Report .omo/evidence/v4-data-final-review.md; data commit 3f146acaf3a45ec6b1bc9fab04883735d3f29e46.
- Drive uploads completed40/40; connector metadata readback verified IDs, names and sizes; bytes3,729,127,756. Durable actual receipts .omo/evidence/v4-drive-uploads.json. Colab will verify content hashes again.
- Notebook built from actual receipts,28cells/19embedded runtime sources, no outputs. Final refresh is pending training orchestration fixes; no GPU launch yet.
- Read-only training integration review found fixed-epoch patience, P4 matching, selected-best refit, terminal-resume, stale G0 cache and incomplete dependency-hash defects. Worker fixing before launch; also checking full-original dev refit paths, best metadata binding, synchronized CUDA timing and P4 auxiliary weighting. Real CPU SONICS preprocessing/update/checkpoint/resume/reload/prediction integration already passed; this is not performance evidence.
