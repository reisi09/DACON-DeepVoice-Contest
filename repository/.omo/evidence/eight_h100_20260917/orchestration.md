# Eight H100 execution evidence ledger

User explicitly authorized all eight proposed research experiments on one H100, notebook creation, environment setup and execution. L4 validation is deferred to submission; no actual submission is requested here. Implementation authorization supersedes planning skill approval pauses.

Initial workspace API read (2026-09-17 ~10:36 UTC): wsp-9683432r72m7 phase standby, H100 SXM count1, compute2.98credits/hour,200GB RAM/ephemeral. Org qwdkw, team Default. Whole API response saved only under private ignored workspace file; no endpoints/token output. One initial org guess choiwuseck failed read-only and was corrected from CLI error to existing canonical qwdkw.

Previous latest run voice_music_h100_20260917_retry1 completed6; archive599members verified and paused18:01KST per storedreceipt. Root read latest run metadata and FINAL_REPORT. Earlier explorer runner report anchored the original failed run and recommended expireddeadline reuse; that stale claim is rejected. Newrun gets distinct max4hdeadline afteruploads, oldones preserved.

Scope: .omo/plans/eight-h100-20260917.md;5independent implementation lanes and1design reviewer. Existing tracked dirty .omo/evidence/v4-training/source-sha256.txt and unrelated untracked files preserved. Goal tool not created: no explicit user request to create a goal; objective recorded here instead.

Data/assets audit: canonical prepared full training1696ish rows plus freshsourcecohorts available locally. Historicalheldoutexposuremeansdiagnosticnotsealedexternalvalidation. Allfittingcontrols retrain onlydeclaredtrain; M330fullhistoricaltraining cannotbeunbiasednewsplitbaseline. Exactcountsjointprovenance pendingdatareceipt. Dataworkerresponsible sourcehashdedup/componentgroups/derivativebinding; noGPUyet.

Dispatched ownership:
- eight_data_build: contracts, preparation, actualdataset, datareceipts.
- eight_features_build: one shared extraction, spectrum features, E2/E5 pairedprobes.
- eight_heads_build: E1/E3/E4/E6/E8, metrics/selection/checkpoints.
- eight_lora_build: E7 adapters and actualtraining loop.
- eight_runner_build: notebook, dependency/preflight, bundle/upload/watch/stop; GPUstarthelduntilrootrelease.
- eight_design_review: independentmetisdesignaudit.

Verification plan: actualCPUdataset integrity and matching; numericalEER/sampler/consistency/REAL-only invariants; nativeAnti parity andrealgradientchecks onH100beforelongtraining; per-filemetrics/checkpoints; ZIP+memberhashverified thenstandby+bootstraprestore. Partialresultsnevercalledcomplete. Five-minutequietwatchafterlaunch.

Preflight progress ~10:50UTC: data core preparation completed but integrity checker rejected duplicate e2_voice_only_minus6dB (same unmodified vocal encoded as separate SNR control). Data/features owners correcting shared-reference construction, not disabling source duplicate checks. Potential explanations distinguished by owners: intentional repeatedbaseline vs acousticzero/decoderfailure vs reusedwrongID. Preserve generatedcore data to avoid redo. E2reference uses contributingvocalparent ID; exactgainmustmatch.

Independent design review found25bounded amendments, communicatedtoeachowner. Exact10speakercomponents=>3train/3cal/2dev/2test andallhistoricallyexposed diagnostic. Momus returnedOKAY aftercontract/QA amendments, evidence momus.md. IndependentCLIattempt isolatedauth401 inconclusive; boundedsecondattemptreuseonlyexpectedlocalCodexauth authorized withnoartifactsecretcopy. NativepreflightQA separateworkeractive.

Clarificationslocked: E7 addshead-onlyCEcontrol200steps3seeds toattributeadaptersvsclassifier separately;CE/KLLoRA1200updatesplus600head-onlyupdates. E3 reportsbase/telephone/MP3/heldoutOpus+telephone. E4 spectral448d butprimaryresidual384subsetchosenfixed. E1actualgenerator-excludedtrainingrequired,notplaceholderstring. H100admission firstrealcoremusic+corespeech+60spartial extractionandLoRAgradient beforefullfeaturepass. StillnoGPUstart.
# Release preparation, 2026-09-17 21:28 KST

All implementation owners reported their modules frozen. Independent fixture integration exercised E1/E3/E4/E6/E8 and E2/E5 through their public entrypoints, using explicitly synthetic features and reduced steps; these are runtime checks, not quality results. The real 7,101-record dataset is frozen at manifest SHA-256 `fc21a54a7d9aeec69575ad1b6a9c0d0c35910f13ba317fde434587686309b537`.

An independent E2 check initially subtracted an RMS-normalized parent and reported incorrect background gain ratios. That assumption was invalid for the final raw-parent additive construction. Rechecking `mixture - raw parent_ids[0]` confirmed all 80 pairs: target ratio `10**(-6/20)`, maximum absolute error `8.61e-8`. The corrected check supersedes the initial E2 finding; evidence is `e2_gain_corrected.log`. Pair duration must match its own parent/control, not every other speaker.

Root authorized source packaging, upload, full readback verification, and a scoped local implementation commit. GPU release is still withheld pending the independent final CPU receipt and verified upload. Both stored original-bootstrap backups have SHA-256 `d027558a7774b4581c6ed1485ed9e94f8bec5b0b8fc88687aa5366e520b07a94`.
# H100 released, 2026-09-17 21:42:40 KST

Independent source/data launch gate passed. The 3,246,461,219-byte source archive passed full remote upload/download SHA-256 verification: `2606dbc7fbdb98b024d781467c17528b95f94e58486d1be77d0be7efff12d363`. Executable code is committed at `b3bf54dce2778bba3270df803fdd5db7077ac578`; the notebook was restored from the archive after a generator-only random-cell-ID change, with no semantic difference.

Workspace `wsp-9683432r72m7` is live `active/running`, H100 SXM x1. Start `1789648960`, immutable deadline `1789663360` (2026-09-18 01:42:40 KST). Local watcher PID `33955` identity was independently confirmed; existing heartbeat `h100-15` is ACTIVE for this run at five-minute intervals. Native forward/gradient admission and all eight measured results remain pending. No completion or performance claim is made by the CPU gate.

E2 post-run Music/File analysis is preregistered before native results: reuse this run's E1 heads and cached features on CPU; reconstruct train-only normalizers, preserve both sampler arms without selection, lock thresholds from calibration before heldout scoring, distinguish genuine File-label changes from already-fake vocal mixtures, and retain source/provenance limitations. This supplements Voice separator diagnostics without another GPU run or editing the uploaded bundle.

# Recovery state, 2026-09-17 22:31 KST

No experiment result or native feature/LoRA admission has passed yet. Four failure causes were observed in order: missing remote imageio_ffmpeg; offline wheel binary mode before discovery; wrong remote assets path; H100 AntiDeepfake native-reference parity error (~0.006535 maximum on first real case versus rtol=1e-4/atol=3e-4). The remote 10-check environment audit passed before the fourth failure, which does not prove model execution correctness. CPU parity on the same three reference waveforms passed (maximum ~2.53e-5); TF32 is a hypothesis pending a controlled H100 comparison, not a confirmed cause.

Attempt4 partial archive SHA-256 `9cd3a5f8db2098dbc914db311709db63b5f451a0a8b7b1f44a7ec8ba1c184a78`, 12,344 bytes / 8 exported members, was verified by runner. A monitor defect was also isolated: recursive download brought nested retry/wheel files into a top-level state download. Minimal eight-wrapper exclusion `*/*` passed a real CLI download with exactly one target file. Old watcher45159 was actually alive but unable to read the state; cleanup/verification/restoration is delegated to eight_watch_fix, and runner must wait for RESTORED_AND_STOPPED before next bootstrap upload.

Immutable original start1789648960/deadline1789663360 retained throughout. No new workspace, no L4/submission, no deadline extension. Existing unrelated dirty files remain preserved.

# Native admission outcome and urgent recovery, 2026-09-17 22:50 KST

Attempt5 established actual H100 model inference: three real records (music, speech,60s partial) completed in30.477815s with peak allocated19,757,025,792bytes. Strict Anti parity passed all3realwaveforms with maximumlogiterror2.956390380859375e-05. On the same H100/current flags, maximumerror0.0065348 and2/3casesfailed; strictCUDA/cuDNN TF32false restores the previous successfulrun policy. Rawnotebookkernel argv/sys.executable/sys.prefix were verified to use `/root/fullft8/.venv`.

LoRA admission then failed before any optimizerupdate because newA/B Linear layers were allocated onCPU while the pretrainedprojection/input were onCUDA. Minimalfix inheritsbase.weight.device anddtype; actualfailuretrace is in attempt5archive. Regressiontest first failed onwrongdtype andpassedafterfix, plus existinggradient/statechecks. Nativegradient/rollback is stillpending and must rerun; featureadmission doesnotprove training. Corrected eight_lora.pySHA `be8828a1eb51dc7a8c8059abfc86ffbd4dda25d27e6b98a0932867a816c1f579`.

Attempt5archiveSHA `c321b23164d6334a70df4f60fa19a83a56c6346867ac1f7b046415613f99a46d`,1,037,774bytes/12members. The diagnosticreceipt used unsupportedstatus `admission_exported`; the normalwatcher rejectedthatreceipt and safely stoppedcompute. Independently verify andpreservearchive before next run. Watcherconfirmedstandby1789652891.656714 and restoredoriginalbootstrap. No repeatpause isrequired.

User explicitly requested immediateGPUrecovery. Nextstart goes directly through the fullnotebook's corrected six-profile featureadmission andLoRAgradientadmission into all8; no furtherseparateadmission-onlyrestart cycle. Originaldeadline unchanged. Featuresowner is freezing rolepruning, headsowner hasfixedactualmusicselection andvalidated4,457music-viewrows, independentQA verifiesactualmanifest/consumercoverage.
