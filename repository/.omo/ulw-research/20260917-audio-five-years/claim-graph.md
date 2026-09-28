# Claim graph

## Verified claims allowlist

| id | statement | scope / status | synthesis location |
|---|---|---|---|
| C01 | Five-year relevant public evidence has broad coverage; not every paper/notebook | supported with bounded scope | Research scope |
| C02 | Latest160 speech records do not distinguish temporalmax/top30 | supported | Current facts |
| C03 | 106atomicIDs yield93connectedcomponents;each8mixturepanelonecomponent | supported | Current facts |
| C04 | Additional sampling experiment is distinct from moredata/headgrid | supported as hypothesis | E1 |
| C05 | Accompaniment swap and roundtrip have diagnostic decision value | supported as hypothesis | E2 |
| C06 | Paired consistency adds an objective beyond plain augmentation | supported as hypothesis | E3 |
| C07 | Lowband spectral arm can be implemented for16k without restoredhighfreq | supported, gainunproven | E4 |
| C08 | Partialoccupancy needs sameprocessingREALcontrols and unit-correctmetrics | supported as hypothesis | E5 |
| C09 | MERT/Anti layerprobe andLoRA newcontrolled options;DF alreadyalllayer | supported as hypothesis | E6/E7 |
| C10 | MusicDET implementation cannot be copied as strictfake-free selection | supported within pinnedcode | E8/codeconstraints |
| C11 | Reportedcompetitionranks are track/metric-specific | supported;primaryorganizerexception | Competitiontable |
| C12 | Current notebook sources are not certified originalwinningversions | supported;primary API exception | Notebookscope |
| C13 | Private testdata cross-file statistics and updating excluded | supported;organizer-onlyauthoritativeexception | Exclusions |
| C14 | REAL/FAKE processing label is conditional on componentgeneration | supported;organizer-onlyauthoritativeexception | Labelboundary |
| C15 | No proposed method has measured currentDACONgain/runtime | supported with scope | Scope/resource |
| C16 | Fullband/targetcollectionNMFnotdropin16k independentprediction | supported | Exclusions |
| C17 | Compositeweights0.45/0.18/0.27 are0.9timesADSweights | supported;officialsourceexception | Adoptioncriteria |

## Nodes

### C01
- statement: Five-year relevant public evidence has broad coverage; not every paper/notebook
- type/risk: method / normal
- scope: current project decisions; papers remain author-reported; supported with bounded scope
- intent_ids: I1
- supporting_observations: O-A01..W12
- contradicting_observations: Partial/full read boundaries;access gaps
- independent_observation_groups: allresearchlanes
- convergence/status: supported with bounded scope
- counter-search: 167 inventory entries; rank not popularity
- primary_source_backing: sources.csv
- dependencies: exactprotocol,version,labelmap; no localgaininference
- synthesis_location: Research scope

### C02
- statement: Latest160 speech records do not distinguish temporalmax/top30
- type/risk: numerical / high
- scope: current project decisions; papers remain author-reported; supported
- intent_ids: I3
- supporting_observations: O-L1
- contradicting_observations: Longmusic hasmultiplewindowsbutunknownvoicetruth
- independent_observation_groups: root+K
- convergence/status: supported
- counter-search: Executed count and exactscoreequality
- primary_source_backing: verify-single-window.json
- dependencies: exactprotocol,version,labelmap; no localgaininference
- synthesis_location: Current facts

### C03
- statement: 106atomicIDs yield93connectedcomponents;each8mixturepanelonecomponent
- type/risk: integrity / high
- scope: current project decisions; papers remain author-reported; supported
- intent_ids: I3
- supporting_observations: O-L2
- contradicting_observations: No crosssplitconnectedcomponent;trueindependenceunknown
- independent_observation_groups: root+K+J
- convergence/status: supported
- counter-search: Existing union-find rerun and provenance qualification
- primary_source_backing: verify-connected-groups.json
- dependencies: exactprotocol,version,labelmap; no localgaininference
- synthesis_location: Current facts

### C04
- statement: Additional sampling experiment is distinct from moredata/headgrid
- type/risk: recommendation / normal
- scope: current project decisions; papers remain author-reported; supported as hypothesis
- intent_ids: I3
- supporting_observations: O-J01;O-A08;localhistory
- contradicting_observations: Some selection policies fail;musictransferunproved
- independent_observation_groups: J+A+L
- convergence/status: supported as hypothesis
- counter-search: Random/control/equalupdate proposed
- primary_source_backing: DOSS and localaudit
- dependencies: exactprotocol,version,labelmap; no localgaininference
- synthesis_location: E1

### C05
- statement: Accompaniment swap and roundtrip have diagnostic decision value
- type/risk: recommendation / normal
- scope: current project decisions; papers remain author-reported; supported as hypothesis
- intent_ids: I3
- supporting_observations: O-C09;O-C15;O-L2
- contradicting_observations: Voiceinvariance notFileinvariance;T01only
- independent_observation_groups: C+E+root
- convergence/status: supported as hypothesis
- counter-search: Samevoiceholding and split-beforemixing
- primary_source_backing: SingGraphfollowup/MixFake
- dependencies: exactprotocol,version,labelmap; no localgaininference
- synthesis_location: E2

### C06
- statement: Paired consistency adds an objective beyond plain augmentation
- type/risk: recommendation / normal
- scope: current project decisions; papers remain author-reported; supported as hypothesis
- intent_ids: I3
- supporting_observations: O-C02;O-C13;O-W01
- contradicting_observations: Somecleanregression;publiccodedefaultissues
- independent_observation_groups: C+G
- convergence/status: supported as hypothesis
- counter-search: Equaldata/steps andbothlabelscontrols
- primary_source_backing: CLAD RTCFake TFCL
- dependencies: exactprotocol,version,labelmap; no localgaininference
- synthesis_location: E3

### C07
- statement: Lowband spectral arm can be implemented for16k without restoredhighfreq
- type/risk: compatibility / high
- scope: current project decisions; papers remain author-reported; supported, gainunproven
- intent_ids: I3
- supporting_observations: O-D04;O-D13;Mcode
- contradicting_observations: Unseengeneratorfailures;parentgroupcodeissues
- independent_observation_groups: D+M+root
- convergence/status: supported, gainunproven
- counter-search: Explicit16karmprimary pluscode inspected
- primary_source_backing: Deezer2025/2026
- dependencies: exactprotocol,version,labelmap; no localgaininference
- synthesis_location: E4

### C08
- statement: Partialoccupancy needs sameprocessingREALcontrols and unit-correctmetrics
- type/risk: recommendation / normal
- scope: current project decisions; papers remain author-reported; supported as hypothesis
- intent_ids: I3
- supporting_observations: O-C01;O-C18;O-W03;O-W06
- contradicting_observations: Maxsometimeslosestomean
- independent_observation_groups: C+K
- convergence/status: supported as hypothesis
- counter-search: Currentonewindowproof andcounterliterature
- primary_source_backing: PartialSpoof/RealFraud
- dependencies: exactprotocol,version,labelmap; no localgaininference
- synthesis_location: E5

### C09
- statement: MERT/Anti layerprobe andLoRA newcontrolled options;DF alreadyalllayer
- type/risk: code/recommendation / normal
- scope: current project decisions; papers remain author-reported; supported as hypothesis
- intent_ids: I3
- supporting_observations: B16;B08;localaudit
- contradicting_observations: Two papersselectontests;cachemissinglayers
- independent_observation_groups: B+K+L
- convergence/status: supported as hypothesis
- counter-search: Dev-onlylayerchoice;no gainclaim
- primary_source_backing: Currentcode/NAACL/MLDG
- dependencies: exactprotocol,version,labelmap; no localgaininference
- synthesis_location: E6/E7

### C10
- statement: MusicDET implementation cannot be copied as strictfake-free selection
- type/risk: code / high
- scope: current project decisions; papers remain author-reported; supported within pinnedcode
- intent_ids: I3
- supporting_observations: O-D10;O-M1
- contradicting_observations: Constant omittedtermnotinvalidatingpaperEER
- independent_observation_groups: D+M
- convergence/status: supported within pinnedcode
- counter-search: ActualActNormCPUcheck;train/dev polarity read
- primary_source_backing: MusicDET pinned source
- dependencies: exactprotocol,version,labelmap; no localgaininference
- synthesis_location: E8/codeconstraints

### C11
- statement: Reportedcompetitionranks are track/metric-specific
- type/risk: dated/rank / high
- scope: current project decisions; papers remain author-reported; supported;primaryorganizerexception
- intent_ids: I2
- supporting_observations: F01;F03;G07;E04;I01;I04;H01;H13
- contradicting_observations: 2022rankresolvedwave2;publicprivate distinctions
- independent_observation_groups: F+G+E+H+I
- convergence/status: supported;primaryorganizerexception
- counter-search: Organizer+authormethodwhereavailable
- primary_source_backing: Officialrankdocuments
- dependencies: exactprotocol,version,labelmap; no localgaininference
- synthesis_location: Competitiontable

### C12
- statement: Current notebook sources are not certified originalwinningversions
- type/risk: provenance / high
- scope: current project decisions; papers remain author-reported; supported;primary API exception
- intent_ids: I2
- supporting_observations: H04;H07;H11;H14;H20
- contradicting_observations: Unrankedtutorialnotwinner
- independent_observation_groups: H
- convergence/status: supported;primary API exception
- counter-search: Version/hashsourceinspection notexecution
- primary_source_backing: wave-1-kaggle.md
- dependencies: exactprotocol,version,labelmap; no localgaininference
- synthesis_location: Notebookscope

### C13
- statement: Private testdata cross-file statistics and updating excluded
- type/risk: rule / high
- scope: current project decisions; papers remain author-reported; supported;organizer-onlyauthoritativeexception
- intent_ids: I5
- supporting_observations: O-R01
- contradicting_observations: Train-fixedstats/perfilesegmentsallowed
- independent_observation_groups: root
- convergence/status: supported;organizer-onlyauthoritativeexception
- counter-search: Officialrulesfreshread
- primary_source_backing: DACONrules
- dependencies: exactprotocol,version,labelmap; no localgaininference
- synthesis_location: Exclusions

### C14
- statement: REAL/FAKE processing label is conditional on componentgeneration
- type/risk: rule / high
- scope: current project decisions; papers remain author-reported; supported;organizer-onlyauthoritativeexception
- intent_ids: I3
- supporting_observations: O-R02;O-C03;O-C17
- contradicting_observations: Neuralcodecnotexplicitlynamedinrule
- independent_observation_groups: root+C+Ncorrection
- convergence/status: supported;organizer-onlyauthoritativeexception
- counter-search: BlanketallcodecREAL/FAKErejected
- primary_source_backing: DACONoverview
- dependencies: exactprotocol,version,labelmap; no localgaininference
- synthesis_location: Labelboundary

### C15
- statement: No proposed method has measured currentDACONgain/runtime
- type/risk: evidenceboundary / high
- scope: current project decisions; papers remain author-reported; supported with scope
- intent_ids: I4
- supporting_observations: Currentresearchtoolactivity;FINAL_REPORT
- contradicting_observations: HistoricalGPUparitynotnewinference
- independent_observation_groups: root+all
- convergence/status: supported with scope
- counter-search: NoGPU/modelrunstartedthisresearch
- primary_source_backing: BRIEF and researchreceipt
- dependencies: exactprotocol,version,labelmap; no localgaininference
- synthesis_location: Scope/resource

### C16
- statement: Fullband/targetcollectionNMFnotdropin16k independentprediction
- type/risk: compatibility/rule / high
- scope: current project decisions; papers remain author-reported; supported
- intent_ids: I3;I5
- supporting_observations: O-D09;O-D12;O-R01;O-R02
- contradicting_observations: LowbandFourierdifferent viablearm
- independent_observation_groups: D+M+root
- convergence/status: supported
- counter-search: PrimarySR/fitprotocolread plusofficialrule
- primary_source_backing: ArtifactNet/NMF/DACON
- dependencies: exactprotocol,version,labelmap; no localgaininference
- synthesis_location: Exclusions

### C17
- statement: Compositeweights0.45/0.18/0.27 are0.9timesADSweights
- type/risk: numerical / high
- scope: current project decisions; papers remain author-reported; supported;officialsourceexception
- intent_ids: I4
- supporting_observations: O-R03;COMPETITION_CONTRACT.md
- contradicting_observations: None;partiallabels stillpreventtotals
- independent_observation_groups: root+L
- convergence/status: supported;officialsourceexception
- counter-search: Arithmetic andcontractchecked
- primary_source_backing: Officialevaluation
- dependencies: exactprotocol,version,labelmap; no localgaininference
- synthesis_location: Adoptioncriteria

## Inventory claim nodes
All source summaries below are scoped to inspected source/revision, not independently reproduced effects. Primary-only exception applies to attribution of what an author reports; it does not certify truth or transfer. Quantitative claims are promoted only through nodes above and cited report context.

- S-A01: AASIST — Small waveform graph detector; historical baseline, not untouched mixed-audio proof. | type=source-observation; risk=normal; scope=full paper+author code; intent=I1/I3; support=O-A01; counter=limitations column; group=speech_early; status=partial for external validity, supported as source reading; primary=https://arxiv.org/abs/2110.01200; dependencies=source revision; synthesis=sources.csv.
- S-A02: RawBoost — Separate corruption policies; training only; gains are task dependent. | type=source-observation; risk=normal; scope=full paper+author code; intent=I1/I3; support=O-A02; counter=limitations column; group=speech_early; status=partial for external validity, supported as source reading; primary=https://arxiv.org/abs/2111.04433; dependencies=source revision; synthesis=sources.csv.
- S-A03: SSL-AASIST — DF EER2.85 versus6.64 using different augmentation; author report. | type=source-observation; risk=normal; scope=full paper+author code; intent=I1/I3; support=O-A03; counter=limitations column; group=speech_early; status=partial for external validity, supported as source reading; primary=https://arxiv.org/abs/2202.12233; dependencies=source revision; synthesis=sources.csv.
- S-A04: Investigating SSL front ends — Band-stop probes and shallow backend controls. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-A04; counter=limitations column; group=speech_early; status=partial for external validity, supported as source reading; primary=https://arxiv.org/abs/2111.07725; dependencies=source revision; synthesis=sources.csv.
- S-A05: Vocoded training and contrastive loss — Paired-content benefit on some corpora; LA regression; resynthesis label compatibility unresolved. | type=source-observation; risk=normal; scope=full paper+pinned code; intent=I1/I3; support=O-A05; counter=limitations column; group=speech_early; status=partial for external validity, supported as source reading; primary=https://arxiv.org/abs/2210.10570; dependencies=source revision; synthesis=sources.csv.
- S-A06: SAMO — Target-only/enrollment protocol; no test enrollment permitted in proposed adaptation. | type=source-observation; risk=normal; scope=full paper+pinned code; intent=I1/I3; support=O-A06; counter=limitations column; group=speech_early; status=partial for external validity, supported as source reading; primary=https://arxiv.org/abs/2211.02718; dependencies=source revision; synthesis=sources.csv.
- S-A07: W2V-ASDG — Domain-adversarial/triplet objective; official split reuse requires audit. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-A07; counter=limitations column; group=speech_early; status=partial for external validity, supported as source reading; primary=https://www.isca-archive.org/interspeech_2023/xie23c_interspeech.pdf; dependencies=source revision; synthesis=sources.csv.
- S-A08: Active-learning training selection — Equal-budget selection; simple random competitive; labeled pool assumed. | type=source-observation; risk=normal; scope=full paper+author project; intent=I1/I3; support=O-A08; counter=limitations column; group=speech_early; status=partial for external validity, supported as source reading; primary=https://arxiv.org/abs/2203.14553; dependencies=source revision; synthesis=sources.csv.
- S-A09: Spoofing attack augmentation — Within-generator configuration variation, not universal family transfer. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-A09; counter=limitations column; group=speech_early; status=partial for external validity, supported as source reading; primary=https://arxiv.org/abs/2309.09586; dependencies=source revision; synthesis=sources.csv.
- S-A10: Large-scale vocoded SSL and distillation — Large pretraining disproportionate; some codec-heavy tests regress. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-A10; counter=limitations column; group=speech_early; status=partial for external validity, supported as source reading; primary=https://arxiv.org/abs/2309.06014; dependencies=source revision; synthesis=sources.csv.
- S-A11: Variational information bottleneck — Small regularization hypothesis; official code not located. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-A11; counter=limitations column; group=speech_early; status=partial for external validity, supported as source reading; primary=https://arxiv.org/abs/2204.01387; dependencies=source revision; synthesis=sources.csv.
- S-A12: Feature Pyramid Conformer and Elastic Margin — Loss ablation lower priority; cross-corpus proof absent. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-A12; counter=limitations column; group=speech_early; status=partial for external validity, supported as source reading; primary=https://www.isca-archive.org/interspeech_2023/gong23b_interspeech.pdf; dependencies=source revision; synthesis=sources.csv.
- S-R01: DACON236749 official rules — File-independent prediction; no private-data training/pseudolabel or cross-file calibration. | type=source-observation; risk=normal; scope=full official page; intent=I1/I3; support=O-R01; counter=limitations column; group=root; status=partial for external validity, supported as source reading; primary=https://dacon.io/competitions/official/236749/overview/rules; dependencies=source revision; synthesis=sources.csv.
- S-R02: DACON236749 task definition — 16kHz; component-generating vs preserving distinction; blanket neuralcodec label not explicit. | type=source-observation; risk=normal; scope=full official page; intent=I1/I3; support=O-R02; counter=limitations column; group=root; status=partial for external validity, supported as source reading; primary=https://dacon.io/competitions/official/236749/overview/description; dependencies=source revision; synthesis=sources.csv.
- S-R03: DACON236749 scoring/runtime — ADS and total-score weights must not be confused; L4 end-to-end runtime. | type=source-observation; risk=normal; scope=official page; intent=I1/I3; support=O-R03; counter=limitations column; group=root; status=partial for external validity, supported as source reading; primary=https://dacon.io/competitions/official/236749/overview/evaluation; dependencies=source revision; synthesis=sources.csv.
- S-F01: ASVspoof5 organizer evaluation — Official Track1 ranks by minDCF, open not equivalent to open-source. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-F01; counter=limitations column; group=asvspoof_winners; status=partial for external validity, supported as source reading; primary=https://www.isca-archive.org/asvspoof_2024/wang24_asvspoof.pdf; dependencies=source revision; synthesis=sources.csv.
- S-F02: ASVspoof5 expanded organizer report — Cross-dataset top-system limits; USTC absent from cross-dataset comparison. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-F02; counter=limitations column; group=asvspoof_winners; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2601.03944v1; dependencies=source revision; synthesis=sources.csv.
- S-F03: USTC-KXDIGIT ASV5 — Open Track1 rank1; private data;12-vs6 subsystem discrepancy unresolved. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-F03; counter=limitations column; group=asvspoof_winners; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2409.01695v1; dependencies=source revision; synthesis=sources.csv.
- S-F04: Intema ASV5 — Rank2; simple WavLM-base mean+linear but finetuning and5modelensemble. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-F04; counter=limitations column; group=asvspoof_winners; status=partial for external validity, supported as source reading; primary=https://www.isca-archive.org/asvspoof_2024/aliyev24_asvspoof.pdf; dependencies=source revision; synthesis=sources.csv.
- S-F05: WavLM/MHFA ASV5 T27 — Rank3;augmentation counterexample and anchoring/layerwiseLR. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-F05; counter=limitations column; group=asvspoof_winners; status=partial for external validity, supported as source reading; primary=https://www.isca-archive.org/asvspoof_2024/stourbe24_asvspoof.pdf; dependencies=source revision; synthesis=sources.csv.
- S-F06: Whispeak ASV5 — Rank4;checkpointaveraging;in-domain gain OODregression. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-F06; counter=limitations column; group=asvspoof_winners; status=partial for external validity, supported as source reading; primary=https://www.isca-archive.org/asvspoof_2024/falez24_asvspoof.pdf; dependencies=source revision; synthesis=sources.csv.
- S-F07: SZU-AFS ASV5 — Rank5;gradient-awareoptimization adds roughly3xtrain cost. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-F07; counter=limitations column; group=asvspoof_winners; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2408.09933v1; dependencies=source revision; synthesis=sources.csv.
- S-F08: ParallelChain ASV5 — Closedrank2;learnedfusion degraded;largeensemble. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-F08; counter=limitations column; group=asvspoof_winners; status=partial for external validity, supported as source reading; primary=https://www.isca-archive.org/asvspoof_2024/tran24_asvspoof.pdf; dependencies=source revision; synthesis=sources.csv.
- S-F09: Inria Storyzy ASV5 — Closedrank3;calibration useful;permutedREAL positive labels incompatible unqualified. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-F09; counter=limitations column; group=asvspoof_winners; status=partial for external validity, supported as source reading; primary=https://www.isca-archive.org/asvspoof_2024/duroselle24_asvspoof.pdf; dependencies=source revision; synthesis=sources.csv.
- S-F10: ASVspoof2021 retrospective — In-window retrospective;original2021-09-16papers outsidewindow. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-F10; counter=limitations column; group=asvspoof_winners; status=partial for external validity, supported as source reading; primary=https://arxiv.org/pdf/2210.02437; dependencies=source revision; synthesis=sources.csv.
- S-G01: ADD2022 organizer report — Separate LF/PF/FG tracks;currentleaderboard differshistorical. | type=source-observation; risk=normal; scope=full PDF; intent=I1/I3; support=O-G01; counter=limitations column; group=add_challenges; status=partial for external validity, supported as source reading; primary=https://arxiv.org/abs/2202.08433; dependencies=source revision; synthesis=sources.csv.
- S-G02: Vicomtech ADD2022 — LFwinner;frozenmultilingualalllayeradaptation. | type=source-observation; risk=normal; scope=full PDF; intent=I1/I3; support=O-G02; counter=limitations column; group=add_challenges; status=partial for external validity, supported as source reading; primary=https://arxiv.org/abs/2203.01573; dependencies=source revision; synthesis=sources.csv.
- S-G03: Tencent ADD2022 PFwinner — Historicalrank corroborated;architecture notfullverified. | type=source-observation; risk=normal; scope=primary abstract only; intent=I1/I3; support=O-G03; counter=limitations column; group=add_challenges; status=partial for external validity, supported as source reading; primary=https://resourcecenter.ieee.org/conferences/icassp-2022/spsicassp22vid0026; dependencies=source revision; synthesis=sources.csv.
- S-G04: Fake-span discovery — PFsecond;auxboundary supervision. | type=source-observation; risk=normal; scope=full PDF; intent=I1/I3; support=O-G04; counter=limitations column; group=add_challenges; status=partial for external validity, supported as source reading; primary=https://arxiv.org/abs/2202.06684; dependencies=source revision; synthesis=sources.csv.
- S-G05: Neural stitching — FGwinner;inference layerdeletion model-specific. | type=source-observation; risk=normal; scope=full PDF; intent=I1/I3; support=O-G05; counter=limitations column; group=add_challenges; status=partial for external validity, supported as source reading; primary=https://arxiv.org/abs/2204.08720; dependencies=source revision; synthesis=sources.csv.
- S-G06: CAU frequency masking — LFthird;moremasking notalwaysbetter. | type=source-observation; risk=normal; scope=full PDF; intent=I1/I3; support=O-G06; counter=limitations column; group=add_challenges; status=partial for external validity, supported as source reading; primary=https://ikwak2.github.io/publications/ddam004-kwak.pdf; dependencies=source revision; synthesis=sources.csv.
- S-G07: ADD2023 organizer analysis — Historicalofficialranks. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-G07; counter=limitations column; group=add_challenges; status=partial for external validity, supported as source reading; primary=https://arxiv.org/abs/2408.04967; dependencies=source revision; synthesis=sources.csv.
- S-G08: USTC-NERCSLIP ADD2023 — FG-Dwinner;pairedcopy synthesis;labeldefinitionmustreconcile. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-G08; counter=limitations column; group=add_challenges; status=partial for external validity, supported as source reading; primary=https://ceur-ws.org/Vol-3597/paper20.pdf; dependencies=source revision; synthesis=sources.csv.
- S-G09: Team Spirit confidence calibration — FG-Dsecond;EOWtraining notPlatt. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-G09; counter=limitations column; group=add_challenges; status=partial for external validity, supported as source reading; primary=https://ceur-ws.org/Vol-3597/paper12.pdf; dependencies=source revision; synthesis=sources.csv.
- S-G10: CAU_KU ADD2023 — FG-Dthird;complementaryerrors. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-G10; counter=limitations column; group=add_challenges; status=partial for external validity, supported as source reading; primary=https://ceur-ws.org/Vol-3597/paper5.pdf; dependencies=source revision; synthesis=sources.csv.
- S-G11: SM-ASDG — FG-Dfifth;speech-only ROIcanhurt. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-G11; counter=limitations column; group=add_challenges; status=partial for external validity, supported as source reading; primary=https://ceur-ws.org/Vol-3597/paper10.pdf; dependencies=source revision; synthesis=sources.csv.
- S-G12: DKU-DUKEECE localization — Localizationwinner;dataset-specific middle/shorterfake priors. | type=source-observation; risk=normal; scope=full PDF; intent=I1/I3; support=O-G12; counter=limitations column; group=add_challenges; status=partial for external validity, supported as source reading; primary=https://arxiv.org/abs/2308.10281; dependencies=source revision; synthesis=sources.csv.
- S-G13: TranssionADD localization — Localizationsecond;multiframesmattermorethan isolatedframepenalty. | type=source-observation; risk=normal; scope=full PDF; intent=I1/I3; support=O-G13; counter=limitations column; group=add_challenges; status=partial for external validity, supported as source reading; primary=https://ceur-ws.org/Vol-3597/paper19.pdf; dependencies=source revision; synthesis=sources.csv.
- S-G14: HarmoNet — Postchallenge method notwinner. | type=source-observation; risk=normal; scope=full PDF; intent=I1/I3; support=O-G14; counter=limitations column; group=add_challenges; status=partial for external validity, supported as source reading; primary=https://www.isca-archive.org/interspeech_2024/liu24g_interspeech.pdf; dependencies=source revision; synthesis=sources.csv.
- S-G15: AT-ADD2026 organizer report — RobustspeechWaveShield andalltypestarfirewinners;MacroF1notEER. | type=source-observation; risk=normal; scope=full HTML+officialevent; intent=I1/I3; support=O-G15; counter=limitations column; group=add_challenges; status=partial for external validity, supported as source reading; primary=https://arxiv.org/abs/2608.23437; dependencies=source revision; synthesis=sources.csv.
- S-G16: Hidden-domain routing starfire — Track2winner;exclusivecliptype doesn'tsupportsimultaneouscomponents. | type=source-observation; risk=normal; scope=full HTML; intent=I1/I3; support=O-G16; counter=limitations column; group=add_challenges; status=partial for external validity, supported as source reading; primary=https://arxiv.org/abs/2608.00493; dependencies=source revision; synthesis=sources.csv.
- S-G17: NaCL naturalness curriculum — Naturalnessproxy notuniversaldifficulty;DNSMOSregresses. | type=source-observation; risk=normal; scope=full paper+code; intent=I1/I3; support=O-G17; counter=limitations column; group=add_challenges; status=partial for external validity, supported as source reading; primary=https://arxiv.org/abs/2505.13976; dependencies=source revision; synthesis=sources.csv.
- S-G18: AUDETER — Curriculum/realdiversity lead;selection caveats;repo READMEonly. | type=source-observation; risk=normal; scope=full paper+repo; intent=I1/I3; support=O-G18; counter=limitations column; group=add_challenges;dataset_validity; status=partial for external validity, supported as source reading; primary=https://arxiv.org/abs/2509.04345; dependencies=source revision; synthesis=sources.csv.
- S-C01: PartialSpoof — Full+code;utterance andframe metricsdistinct. | type=source-observation; risk=normal; scope=full methods/results; intent=I1/I3; support=O-C01; counter=limitations column; group=robust_partial; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2204.05177v3; dependencies=source revision; synthesis=sources.csv.
- S-C02: CLAD — Full+code;ASV2019only;pairedinvariance hypothesis. | type=source-observation; risk=normal; scope=full methods/results; intent=I1/I3; support=O-C02; counter=limitations column; group=robust_partial; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2404.15854v1; dependencies=source revision; synthesis=sources.csv.
- S-C03: Codecfake Xie — Full;reconstructedREALlabeledFAKE. | type=source-observation; risk=normal; scope=full methods/results; intent=I1/I3; support=O-C03; counter=limitations column; group=robust_partial; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2405.04880v1; dependencies=source revision; synthesis=sources.csv.
- S-C04: CodecFake Wu — Full;differentpaperfromXie;fewVCTKtestspeakers. | type=source-observation; risk=normal; scope=full methods/results; intent=I1/I3; support=O-C04; counter=limitations column; group=robust_partial; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2406.07237v1; dependencies=source revision; synthesis=sources.csv.
- S-C05: CodecFake+ — Full+card;codecpositiveontology. | type=source-observation; risk=normal; scope=full methods/results; intent=I1/I3; support=O-C05; counter=limitations column; group=robust_partial; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2501.08238v3; dependencies=source revision; synthesis=sources.csv.
- S-C06: MeasuringRobustness — Full;corruption diversity notsourceindependence. | type=source-observation; risk=normal; scope=full methods/results; intent=I1/I3; support=O-C06; counter=limitations column; group=robust_partial; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2503.17577v1; dependencies=source revision; synthesis=sources.csv.
- S-C07: ADD-C — Full;packetloss andcodec conditions shareparents. | type=source-observation; risk=normal; scope=full methods/results; intent=I1/I3; support=O-C07; counter=limitations column; group=robust_partial; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2504.12423v1; dependencies=source revision; synthesis=sources.csv.
- S-C08: Pindrop it — Full;temporalmaxpositiveprecedentnotuniversal. | type=source-observation; risk=normal; scope=full methods/results; intent=I1/I3; support=O-C08; counter=limitations column; group=robust_partial; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2508.08141v1; dependencies=source revision; synthesis=sources.csv.
- S-C09: How Does Instrumental Music Help SingFake Detection — Full;unpairedmusic/noise control,T01only. | type=source-observation; risk=normal; scope=full methods/results; intent=I1/I3; support=O-C09; counter=limitations column; group=robust_partial; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2509.14675v2; dependencies=source revision; synthesis=sources.csv.
- S-C10: S-MGAA greetings — Full;shortutterance notbriefsplice. | type=source-observation; risk=normal; scope=full methods/results; intent=I1/I3; support=O-C10; counter=limitations column; group=robust_partial; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2601.19573v1; dependencies=source revision; synthesis=sources.csv.
- S-C11: SAL segment position — Full;boundaryshortcutcontrols,OODstillpoor. | type=source-observation; risk=normal; scope=full methods/results; intent=I1/I3; support=O-C11; counter=limitations column; group=robust_partial; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2601.21925v1; dependencies=source revision; synthesis=sources.csv.
- S-C12: Speech enhancement noisy detection — Full;denoisingcanhurt. | type=source-observation; risk=normal; scope=full methods/results; intent=I1/I3; support=O-C12; counter=limitations column; group=robust_partial; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2603.14767v1; dependencies=source revision; synthesis=sources.csv.
- S-C13: RTCFake — Full;pairedplatformconsistency. | type=source-observation; risk=normal; scope=full methods/results; intent=I1/I3; support=O-C13; counter=limitations column; group=robust_partial; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2604.23742v1; dependencies=source revision; synthesis=sources.csv.
- S-C14: MIST — Full;internalinconsistencies,headlineexcluded. | type=source-observation; risk=normal; scope=full methods/results; intent=I1/I3; support=O-C14; counter=limitations column; group=robust_partial; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2605.02223v1; dependencies=source revision; synthesis=sources.csv.
- S-C15: MixFake — Full;fourauthenticitycells,SNR;sourceconfound;promptHHTnotraw. | type=source-observation; risk=normal; scope=full methods/results; intent=I1/I3; support=O-C15; counter=limitations column; group=robust_partial; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2605.23201v1; dependencies=source revision; synthesis=sources.csv.
- S-C16: Proteus — Full;transformchains/privateproductiondetector. | type=source-observation; risk=normal; scope=full methods/results; intent=I1/I3; support=O-C16; counter=limitations column; group=robust_partial; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2606.29544v1; dependencies=source revision; synthesis=sources.csv.
- S-C17: CD-ADD — Label-preservingEnCodec vsCodecFakeconflict;unequaltrainingepochs. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-C17; counter=limitations column; group=robust_partial;dataset_validity; status=partial for external validity, supported as source reading; primary=https://aclanthology.org/2024.emnlp-main.286/; dependencies=source revision; synthesis=sources.csv.
- S-C18: AudioDetectors vsRealFraud — 34partialincidents;maxworsethanmean;privatedata. | type=source-observation; risk=normal; scope=full PDF; intent=I1/I3; support=O-C18; counter=limitations column; group=robust_partial; status=partial for external validity, supported as source reading; primary=https://openaccess.thecvf.com/content/WACV2026W/SAFE-2026/papers/Gajewska_Audio_Deepfake_Detectors_vs._Real_Fraud_-_The_Fall_of_WACVW_2026_paper.pdf; dependencies=source revision; synthesis=sources.csv.
- S-C19: Amplifying Artifacts with Speech Enhancement — Trainednoise/enhancement/residualpipeline notcurrentresidualhead. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-C19; counter=limitations column; group=robust_partial; status=partial for external validity, supported as source reading; primary=https://www.isca-archive.org/interspeech_2025/trachu25_interspeech.pdf; dependencies=source revision; synthesis=sources.csv.
- S-C20: M2S — Learnedbinauralconversionnotnativechannelinvariance. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-C20; counter=limitations column; group=robust_partial; status=partial for external validity, supported as source reading; primary=https://www.isca-archive.org/interspeech_2023/liu23v_interspeech.pdf; dependencies=source revision; synthesis=sources.csv.
- S-C21: RADARchallenge — Transformationpolicyonly,noverifiedwinner. | type=source-observation; risk=normal; scope=officialpage; intent=I1/I3; support=O-C21; counter=limitations column; group=robust_partial; status=partial for external validity, supported as source reading; primary=https://radar-challenge.github.io/; dependencies=source revision; synthesis=sources.csv.
- S-J01: DOSS data-centric detection — Source×generator weighting;in-domainsetsincludedaverage;synthetic-onlyaccuracy notbalanced. | type=source-observation; risk=normal; scope=full20page paper; intent=I1/I3; support=O-J01; counter=limitations column; group=dataset_validity; status=partial for external validity, supported as source reading; primary=https://aclanthology.org/2026.acl-long.796.pdf; dependencies=source revision; synthesis=sources.csv.
- S-J02: Echoes — Reference-conditioned tracks;parentgroupingrebuildneeded;notnewlocaldataset. | type=source-observation; risk=normal; scope=full+card; intent=I1/I3; support=O-J02; counter=limitations column; group=dataset_validity; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2603.23667v3; dependencies=source revision; synthesis=sources.csv.
- S-J03: AI4T pruning — 128datasetcombinationselectionexposure;notcleanOODgain. | type=source-observation; risk=normal; scope=fullpaper; intent=I1/I3; support=O-J03; counter=limitations column; group=dataset_validity; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2506.09606; dependencies=source revision; synthesis=sources.csv.
- S-J04: Circumventing shortcuts — Leading silence shortcut,specificAVsets;methodneedsvideo. | type=source-observation; risk=normal; scope=fullpaper+code; intent=I1/I3; support=O-J04; counter=limitations column; group=dataset_validity; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2412.00175; dependencies=source revision; synthesis=sources.csv.
- S-J05: FairSSD — REALsubgroupFPR;notproofcurrentmodelbias. | type=source-observation; risk=normal; scope=fullpaper; intent=I1/I3; support=O-J05; counter=limitations column; group=dataset_validity; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2404.10989; dependencies=source revision; synthesis=sources.csv.
- S-J06: M6 — Musiccoveragelabeltaxonomy/redistributionunresolved. | type=source-observation; risk=normal; scope=fullpaper+card; intent=I1/I3; support=O-J06; counter=limitations column; group=dataset_validity; status=partial for external validity, supported as source reading; primary=https://www.nature.com/articles/s41598-026-36044-w; dependencies=source revision; synthesis=sources.csv.
- S-J07: AIMEcard — IndependentREALsource;tagmatchingnotwaveformmatched;dateunverifiedsource-notpaper. | type=source-observation; risk=normal; scope=first-partycard; intent=I1/I3; support=O-J07; counter=limitations column; group=dataset_validity; status=partial for external validity, supported as source reading; primary=https://huggingface.co/datasets/disco-eth/AIME/blob/main/README.md; dependencies=source revision; synthesis=sources.csv.
- S-J08: Does Audio Deepfake Detection Generalize — Backgroundonly,notfullreaudit. | type=source-observation; risk=normal; scope=abstract/date; intent=I1/I3; support=O-J08; counter=limitations column; group=dataset_validity; status=partial for external validity, supported as source reading; primary=https://arxiv.org/abs/2203.16263; dependencies=source revision; synthesis=sources.csv.
- S-J09: Why detectors won't generalize — Positionpaper notimpossibilityproof. | type=source-observation; risk=normal; scope=fullHTML qualitative; intent=I1/I3; support=O-J09; counter=limitations column; group=dataset_validity; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2509.20405v1; dependencies=source revision; synthesis=sources.csv.
- S-D01: Music deepfakes easy but hard — Codecartifactontology;crossdecodergaps. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-D01; counter=limitations column; group=music_papers; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2405.04181v2; dependencies=source revision; synthesis=sources.csv.
- S-D02: SONICS SpecTTTra — Longcontextin-domainvsunseenfamilycollapse. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-D02; counter=limitations column; group=music_papers; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2408.14080; dependencies=source revision; synthesis=sources.csv.
- S-D03: FakeMusicCaps — ClosedaccuracyhidesunseenSunofailure. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-D03; counter=limitations column; group=music_papers; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2409.10684v2; dependencies=source revision; synthesis=sources.csv.
- S-D04: Fourier AI music artifacts — Explicit16k1–8kHzarm;crossgenfails;FMAreplacementunfaircomparison. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-D04; counter=limitations column; group=music_papers; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2506.19108v1; dependencies=source revision; synthesis=sources.csv.
- S-D05: Music detection augmentations — Meanprobabilitynotaccuracy;smallstresssample. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-D05; counter=limitations column; group=music_papers; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2507.10447v1; dependencies=source revision; synthesis=sources.csv.
- S-D06: SegmentTransformer — Beatgroupsstructurecontrol;native24kMERT. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-D06; counter=limitations column; group=music_papers; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2509.08283v1; dependencies=source revision; synthesis=sources.csv.
- S-D07: Melody or Machine CLAM — Sametracktworepresentationsnotseparatedstems;versiondiscrepancies. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-D07; counter=limitations column; group=music_papers; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2512.00621v1; dependencies=source revision; synthesis=sources.csv.
- S-D08: FusionSegmentTransformer — Smallincrementpapprox.09;nearceilingSONICS. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-D08; counter=limitations column; group=music_papers; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2601.13647v1; dependencies=source revision; synthesis=sources.csv.
- S-D09: ArtifactNet — 44.1khighfreq dependency not16kdropin. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-D09; counter=limitations column; group=music_papers; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2604.16254v2; dependencies=source revision; synthesis=sources.csv.
- S-D10: MusicDET — Real-onlytrainmixeddev;implementationauditrequired. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-D10; counter=limitations column; group=music_papers; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2605.18072v1; dependencies=source revision; synthesis=sources.csv.
- S-D11: CoMoE — Crossgenrepresentation;READMEonlycode. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-D11; counter=limitations column; group=music_papers; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2606.08663v1; dependencies=source revision; synthesis=sources.csv.
- S-D12: Finding the Noise — TransductiveNMF3–15kHz,nottestfitallowed. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-D12; counter=limitations column; group=music_papers; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2607.25530v1; dependencies=source revision; synthesis=sources.csv.
- S-D13: ImprovedMusicRobustness — 16k1–7kHzarm;robustcodegroupingsplitrisk. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-D13; counter=limitations column; group=music_papers; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2607.27454v2; dependencies=source revision; synthesis=sources.csv.
- S-D14: Generated music versus edited audio — Anchor-heldouteditedREAL labelsstillimperfect. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-D14; counter=limitations column; group=music_papers; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2608.14916; dependencies=source revision; synthesis=sources.csv.
- S-D15: AI Music Arms Race — CLAPdifferentrepresentation;BoomyOODpoor;paper32k vsreleasedwrapper48k. | type=source-observation; risk=normal; scope=fullPDF; intent=I1/I3; support=O-D15; counter=limitations column; group=music_papers; status=partial for external validity, supported as source reading; primary=https://storage.googleapis.com/jnl-up-j-tismir-files/journals/1/articles/254/68774364a6605.pdf; dependencies=source revision; synthesis=sources.csv.
- S-B01: AntiDeepfake — CorrectedEERv4;posttrainingcorpusoverlap;artifactlabelnotprovenance. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-B01; counter=limitations column; group=speech_recent; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2506.21090v4; dependencies=source revision; synthesis=sources.csv.
- S-B02: GRPO fine-tuning generalization — NoXLSR1Bpublishedresult;retentionnotsuperiority;codeequationdifference. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-B02; counter=limitations column; group=speech_recent; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2603.02914; dependencies=source revision; synthesis=sources.csv.
- S-B03: Speech Deepfake Arena — Macrovs pooled;historicalnotliverank. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-B03; counter=limitations column; group=speech_recent; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2509.02859; dependencies=source revision; synthesis=sources.csv.
- S-B04: Nes2Net — BackendMACsnotwholeencoder;differentprotocols. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-B04; counter=limitations column; group=speech_recent; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2504.05657; dependencies=source revision; synthesis=sources.csv.
- S-B05: XLSR-Mamba — Customdependencies;noL4proof. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-B05; counter=limitations column; group=speech_recent; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2411.10027; dependencies=source revision; synthesis=sources.csv.
- S-B06: LayerSelectionFusion — ExplicitoracleOODlayerchoice. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-B06; counter=limitations column; group=speech_recent; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2509.12003; dependencies=source revision; synthesis=sources.csv.
- S-B07: Layer probing2026 — SixTESTsetsusedforselection;notuntouchedevidence. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-B07; counter=limitations column; group=speech_recent; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2606.30791; dependencies=source revision; synthesis=sources.csv.
- S-B08: MLDG-LoRA — OrdinaryLoRAcontrol;metastepsmuchslower;codeunavailable. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-B08; counter=limitations column; group=speech_recent; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2502.10838; dependencies=source revision; synthesis=sources.csv.
- S-B09: MoLEx — Sparseexpertsdomain-specific,orthogonalitycritical. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-B09; counter=limitations column; group=speech_recent; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2509.09175; dependencies=source revision; synthesis=sources.csv.
- S-B10: Multi-levelSSL gating — Fullutteranceconfoundsheadcomparison. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-B10; counter=limitations column; group=speech_recent; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2509.03409; dependencies=source revision; synthesis=sources.csv.
- S-B11: Spoof-SUPERB — FrozenprotocolnotposttrainedAnti. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-B11; counter=limitations column; group=speech_recent; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2603.01482; dependencies=source revision; synthesis=sources.csv.
- S-B12: One-ClassKnowledgeDistillation — Teacherandstudentbothat inference. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-B12; counter=limitations column; group=speech_recent; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2309.08285; dependencies=source revision; synthesis=sources.csv.
- S-B13: ArtifactSpecificExperts — Silence shortcut;gateworseinsomeconditions. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-B13; counter=limitations column; group=speech_recent; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2607.21127; dependencies=source revision; synthesis=sources.csv.
- S-B14: ContinualUAPdistillation — Retentiontargettradeoff;notcurrentmissingdatarequirement. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-B14; counter=limitations column; group=speech_recent; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2511.19974; dependencies=source revision; synthesis=sources.csv.
- S-B15: MLDGgeometry — Explanationnotnewaccuracygain. | type=source-observation; risk=normal; scope=full paper; intent=I1/I3; support=O-B15; counter=limitations column; group=speech_recent; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2607.22010; dependencies=source revision; synthesis=sources.csv.
- S-B16: ComprehensiveSSLlayeranalysis — Controlledlayerprobe;no universalbestlayer. | type=source-observation; risk=normal; scope=fullpaper; intent=I1/I3; support=O-B16; counter=limitations column; group=speech_recent; status=partial for external validity, supported as source reading; primary=https://aclanthology.org/2025.findings-naacl.227.pdf; dependencies=source revision; synthesis=sources.csv.
- S-B17: All-Type SSL frontend — Bestaggregate dualmodelworseonmixtures;distillationregresses. | type=source-observation; risk=normal; scope=fullpaper; intent=I1/I3; support=O-B17; counter=limitations column; group=speech_recent; status=partial for external validity, supported as source reading; primary=https://www.isca-archive.org/odyssey_2026/das26_odyssey.pdf; dependencies=source revision; synthesis=sources.csv.
- S-B18: SLS — Twosslforwardsfixedshapehead;codeisnotcheaplinearlayerprobe. | type=source-observation; risk=normal; scope=abstract+actualcode;fullpaperblocked; intent=I1/I3; support=O-B18; counter=limitations column; group=speech_recent; status=partial for external validity, supported as source reading; primary=https://github.com/QiShanZhang/SLSforASVspoof-2021-DF/tree/89a09ac4404c5687d96d6c123fcf6db20e4e4b38; dependencies=source revision; synthesis=sources.csv.
- S-E01: SingFake — Paper/project/code;singingOODsevere. | type=source-observation; risk=normal; scope=fullpaper+selectedrepo; intent=I1/I3; support=O-E01; counter=limitations column; group=singing_svdd; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2309.07525; dependencies=source revision; synthesis=sources.csv.
- S-E02: FSD — ThreeSoVITSvariantsnotthreedistinctfamilies;redistributionrestricted. | type=source-observation; risk=normal; scope=fullpaper+selectedrepo; intent=I1/I3; support=O-E02; counter=limitations column; group=singing_svdd; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2309.02232v2; dependencies=source revision; synthesis=sources.csv.
- S-E03: CtrSVDD — Unaccompanied16ksinging;officialunseenA09–A14. | type=source-observation; risk=normal; scope=fullpaper+selectedrepo; intent=I1/I3; support=O-E03; counter=limitations column; group=singing_svdd; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2406.02438v2; dependencies=source revision; synthesis=sources.csv.
- S-E04: SVDDorganizer — OfficialA14andmatchingREALexclusion;WildSLTnosubmissions. | type=source-observation; risk=normal; scope=fullpaper+selectedrepo; intent=I1/I3; support=O-E04; counter=limitations column; group=singing_svdd; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2408.16132v2; dependencies=source revision; synthesis=sources.csv.
- S-E05: I2R-ASTAR — Official3rd2.22vsresearchensemble1.79different. | type=source-observation; risk=normal; scope=fullpaper+selectedrepo; intent=I1/I3; support=O-E05; counter=limitations column; group=singing_svdd; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2409.02302v1; dependencies=source revision; synthesis=sources.csv.
- S-E06: Qishan XWSB — Official4th;polaritycaveat. | type=source-observation; risk=normal; scope=abstract+code; intent=I1/I3; support=O-E06; counter=limitations column; group=singing_svdd; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2409.18558; dependencies=source revision; synthesis=sources.csv.
- S-E07: FIONA — Developmentsetnotofficialtest. | type=source-observation; risk=normal; scope=fullpaper+selectedrepo; intent=I1/I3; support=O-E07; counter=limitations column; group=singing_svdd; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2409.14131v1; dependencies=source revision; synthesis=sources.csv.
- S-E08: Whisper singing detector — AveragefourtestEERnotpooled. | type=source-observation; risk=normal; scope=fullpaper+selectedrepo; intent=I1/I3; support=O-E08; counter=limitations column; group=singing_svdd; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2501.18919v1; dependencies=source revision; synthesis=sources.csv.
- S-E09: Sing-HiResNet — 44.1k;conferenceacceptanceunverified;notdirect16k. | type=source-observation; risk=normal; scope=fullpaper+selectedrepo; intent=I1/I3; support=O-E09; counter=limitations column; group=singing_svdd; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2604.04841v1; dependencies=source revision; synthesis=sources.csv.
- S-E10: SingFox — SingerIDswithheld/provenancegaps;Koreancoveragepotential. | type=source-observation; risk=normal; scope=fullpaper+selectedrepo; intent=I1/I3; support=O-E10; counter=limitations column; group=singing_svdd; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2606.18985v1; dependencies=source revision; synthesis=sources.csv.
- S-E11: COFFE — Fake-onlyattribution/dev-as-testnotbinaryunseendetector. | type=source-observation; risk=normal; scope=fullpaper+selectedrepo; intent=I1/I3; support=O-E11; counter=limitations column; group=singing_svdd; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2506.03364v1; dependencies=source revision; synthesis=sources.csv.
- S-E12: SingGraph — MERTloaderSRmismatch;recoveredsubsetcomparability. | type=source-observation; risk=normal; scope=fullpaper+code; intent=I1/I3; support=O-E12; counter=limitations column; group=singing_svdd; status=partial for external validity, supported as source reading; primary=https://www.isca-archive.org/interspeech_2024/chen24o_interspeech.pdf; dependencies=source revision; synthesis=sources.csv.
- S-E13: MIREXWildSVDDresults — UNIBSlowestboth;notSLTcontrolledranking. | type=source-observation; risk=normal; scope=officialtable+systemdescription; intent=I1/I3; support=O-E13; counter=limitations column; group=singing_svdd; status=partial for external validity, supported as source reading; primary=https://music-ir.org/mirex/wiki/2024:Singing_Voice_Deepfake_Detection_Results; dependencies=source revision; synthesis=sources.csv.
- S-E14: MIREXSongResults — Baselinesonlynotcompetitorranks;fullsongvsvoicegap. | type=source-observation; risk=normal; scope=officialtable; intent=I1/I3; support=O-E14; counter=limitations column; group=singing_svdd; status=partial for external validity, supported as source reading; primary=https://music-ir.org/mirex/wiki/2025:Song_Deepfake_Detection_Results; dependencies=source revision; synthesis=sources.csv.
- S-E15: GenuVoice — Officialdev mappedtest;1.82notchallengecomparable. | type=source-observation; risk=normal; scope=fullpaper+README; intent=I1/I3; support=O-E15; counter=limitations column; group=singing_svdd; status=partial for external validity, supported as source reading; primary=https://aclanthology.org/2026.findings-acl.1245/; dependencies=source revision; synthesis=sources.csv.
- S-E16: MIREX2026task — Fullygeneratedonly;noresultsverified. | type=source-observation; risk=normal; scope=officialpage; intent=I1/I3; support=O-E16; counter=limitations column; group=singing_svdd; status=partial for external validity, supported as source reading; primary=https://music-ir.org/mirex/wiki/2026:AI-Generated_Music_Detection; dependencies=source revision; synthesis=sources.csv.
- S-H01: BirdCLEF2022 organizer — KDLrank1confirmed;repoofficialredirectmatched. | type=source-observation; risk=normal; scope=officialPDF+figure; intent=I1/I3; support=O-H01; counter=limitations column; group=kaggle_audio; status=partial for external validity, supported as source reading; primary=https://ceur-ws.org/Vol-3180/paper-154.pdf; dependencies=source revision; synthesis=sources.csv.
- S-H02: BirdCLEF2023 results — Ranks1–4verified;adjacenttask. | type=source-observation; risk=normal; scope=officialrows; intent=I1/I3; support=O-H02; counter=limitations column; group=kaggle_audio; status=partial for external validity, supported as source reading; primary=https://www.kaggle.com/c/birdclef-2023/leaderboard; dependencies=source revision; synthesis=sources.csv.
- S-H03: BirdCLEF2023rank1 — Dataquality+frontenddiversity;notuniversalSEDgain. | type=source-observation; risk=normal; scope=fullindexedwriteup+selectedcode; intent=I1/I3; support=O-H03; counter=limitations column; group=kaggle_audio; status=partial for external validity, supported as source reading; primary=https://www.kaggle.com/competitions/birdclef-2023/writeups/volodymyr-1st-place-solution-correct-data-is-all-y; dependencies=source revision; synthesis=sources.csv.
- S-H04: BirdCLEF2023rank2 — Sevenmodels;PL/EMAdidnothelp. | type=source-observation; risk=normal; scope=fullindexedwriteup+notebooksource; intent=I1/I3; support=O-H04; counter=limitations column; group=kaggle_audio; status=partial for external validity, supported as source reading; primary=https://www.kaggle.com/competitions/birdclef-2023/writeups/griffith-2nd-place-solution-sed-cnn-with-7-models-; dependencies=source revision; synthesis=sources.csv.
- S-H05: BirdCLEF2023rank4KD — Train-onlycachedteacherKD;costnontrivial. | type=source-observation; risk=normal; scope=fullindexedwriteup; intent=I1/I3; support=O-H05; counter=limitations column; group=kaggle_audio; status=partial for external validity, supported as source reading; primary=https://www.kaggle.com/competitions/birdclef-2023/writeups/atfujita-4th-place-solution-knowledge-distillation; dependencies=source revision; synthesis=sources.csv.
- S-H06: BirdCLEF2024 organizer — Competitionranksverifiednotnotebookvotes. | type=source-observation; risk=normal; scope=primaryrankingtable; intent=I1/I3; support=O-H06; counter=limitations column; group=kaggle_audio; status=partial for external validity, supported as source reading; primary=https://ceur-ws.org/Vol-3740/paper-206.pdf; dependencies=source revision; synthesis=sources.csv.
- S-H07: BirdCLEF2024rank1 — SimpleCNNbetterSED;SWA/EMAnegligible. | type=source-observation; risk=normal; scope=fullindexedwriteup+notebooksource; intent=I1/I3; support=O-H07; counter=limitations column; group=kaggle_audio; status=partial for external validity, supported as source reading; primary=https://www.kaggle.com/competitions/birdclef-2024/writeups/team-kefir-1st-place-solution; dependencies=source revision; synthesis=sources.csv.
- S-H08: BirdCLEF2024rank3 — Maskuncertaincroplabels;postprocessprivatehurt. | type=source-observation; risk=normal; scope=fullindexedwriteup+repo; intent=I1/I3; support=O-H08; counter=limitations column; group=kaggle_audio; status=partial for external validity, supported as source reading; primary=https://www.kaggle.com/competitions/birdclef-2024/writeups/nvbird-3rd-solution; dependencies=source revision; synthesis=sources.csv.
- S-H09: BirdCLEF2024rank4 — Raw+melcomplementarity;PLfailed. | type=source-observation; risk=normal; scope=fullindexedwriteup; intent=I1/I3; support=O-H09; counter=limitations column; group=kaggle_audio; status=partial for external validity, supported as source reading; primary=https://www.kaggle.com/competitions/birdclef-2024/discussion/511845; dependencies=source revision; synthesis=sources.csv.
- S-H10: BirdCLEF2025 leaderboard — Rank1bodyunavailable,notfullread. | type=source-observation; risk=normal; scope=officialrows; intent=I1/I3; support=O-H10; counter=limitations column; group=kaggle_audio; status=partial for external validity, supported as source reading; primary=https://www.kaggle.com/c/birdclef-2025/leaderboard; dependencies=source revision; synthesis=sources.csv.
- S-H11: BirdCLEF2025rank2 — AuthorgroupCVstillfailsdomainshift;126GBRAMrecipe. | type=source-observation; risk=normal; scope=fullpaper+notebooksource; intent=I1/I3; support=O-H11; counter=limitations column; group=kaggle_audio; status=partial for external validity, supported as source reading; primary=https://ceur-ws.org/Vol-4038/paper_256.pdf; dependencies=source revision; synthesis=sources.csv.
- S-H12: BirdCLEF2025rank4 — KDdidnothelp;softAUCconfoundedwithsemisupervision. | type=source-observation; risk=normal; scope=fullindexedwriteup; intent=I1/I3; support=O-H12; counter=limitations column; group=kaggle_audio; status=partial for external validity, supported as source reading; primary=https://www.kaggle.com/competitions/birdclef-2025/writeups/dylan-liu-4th-place-solution; dependencies=source revision; synthesis=sources.csv.
- S-H13: BirdCLEF2026 leaderboard — Ranks1–6verified. | type=source-observation; risk=normal; scope=officialrows; intent=I1/I3; support=O-H13; counter=limitations column; group=kaggle_audio; status=partial for external validity, supported as source reading; primary=https://www.kaggle.com/competitions/birdclef-2026/leaderboard; dependencies=source revision; synthesis=sources.csv.
- S-H14: BirdCLEF2026rank1 — EmbeddingKD;studentiteration3regresses. | type=source-observation; risk=normal; scope=substantialindexedwriteup+notebooksource; intent=I1/I3; support=O-H14; counter=limitations column; group=kaggle_audio; status=partial for external validity, supported as source reading; primary=https://www.kaggle.com/competitions/birdclef-2026/writeups/1st-place-solution-noisy-student-meets-distillati; dependencies=source revision; synthesis=sources.csv.
- S-H15: BirdCLEF2026rank2 — Defaultattentionpathconcern,alternatetrainingvalid;notwinningbugproof. | type=source-observation; risk=normal; scope=writeup+pinnedtrainingcode; intent=I1/I3; support=O-H15; counter=limitations column; group=kaggle_audio; status=partial for external validity, supported as source reading; primary=https://www.kaggle.com/competitions/birdclef-2026/writeups/2nd-place-diverse-ensemble-with-pseudo-labeling-a; dependencies=source revision; synthesis=sources.csv.
- S-H16: BirdCLEF2026rank3 — Teacher/frontenddiversity;notebookunread. | type=source-observation; risk=normal; scope=substantialindexedwriteup; intent=I1/I3; support=O-H16; counter=limitations column; group=kaggle_audio; status=partial for external validity, supported as source reading; primary=https://www.kaggle.com/competitions/birdclef-2026/writeups/3rd-place-solution; dependencies=source revision; synthesis=sources.csv.
- S-H17: BirdCLEF2026rank4 — DuplicateKD/SED/PL;codeunread. | type=source-observation; risk=normal; scope=substantialincompletewriteup; intent=I1/I3; support=O-H17; counter=limitations column; group=kaggle_audio; status=partial for external validity, supported as source reading; primary=https://www.kaggle.com/competitions/birdclef-2026/writeups/4-th-place-solution; dependencies=source revision; synthesis=sources.csv.
- S-H18: BirdCLEF2026rank5 — UnreliableCV/luckyselectionacknowledged. | type=source-observation; risk=normal; scope=substantialindexedwriteup; intent=I1/I3; support=O-H18; counter=limitations column; group=kaggle_audio; status=partial for external validity, supported as source reading; primary=https://www.kaggle.com/competitions/birdclef-2026/writeups/5th-place-solution-both-are-all-you-need; dependencies=source revision; synthesis=sources.csv.
- S-H19: BirdCLEF2026rank6 — SharedfrontendCPU;longerwindow/topkfailed. | type=source-observation; risk=normal; scope=substantialindexedwriteup; intent=I1/I3; support=O-H19; counter=limitations column; group=kaggle_audio; status=partial for external validity, supported as source reading; primary=https://www.kaggle.com/competitions/birdclef-2026/writeups/6th-place-solution; dependencies=source revision; synthesis=sources.csv.
- S-H20: MLAAD tutorial — Unrankededucationalunfinishedexercises,randomcrop/evalmodeconcerns. | type=source-observation; risk=normal; scope=notebooksource; intent=I1/I3; support=O-H20; counter=limitations column; group=kaggle_audio; status=partial for external validity, supported as source reading; primary=https://www.kaggle.com/code/muellermnicolas/audio-deepfake-detection-with-w2v2-and-mlaad-tiny; dependencies=source revision; synthesis=sources.csv.
- S-H21: BirdCLEF2026rank1 sourcepackage — 19sourcefiles172016bytes;contentsnotfullread,nodownloadweights. | type=source-observation; risk=normal; scope=APIinventoryonly; intent=I1/I3; support=O-H21; counter=limitations column; group=kaggle_audio; status=partial for external validity, supported as source reading; primary=https://www.kaggle.com/datasets/nikitababich/birdclef-2026-1st-place-src-inference; dependencies=source revision; synthesis=sources.csv.
- S-I01: SAFE2025 organizer — ISPLprivatewinner;fake-onlycorruptions. | type=source-observation; risk=normal; scope=fullpaper; intent=I1/I3; support=O-I01; counter=limitations column; group=safe_other; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2510.03387v1; dependencies=source revision; synthesis=sources.csv.
- S-I02: SAFE ISPL winner — Heterogeneousspectralexperts;publicT3fourth;codeunavailable. | type=source-observation; risk=normal; scope=fullpaper; intent=I1/I3; support=O-I02; counter=limitations column; group=safe_other; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2509.17585v1; dependencies=source revision; synthesis=sources.csv.
- S-I03: SAFE ISSF — Public2/private3bodyvsabstract;12sITWregression. | type=source-observation; risk=normal; scope=fullpaper+code; intent=I1/I3; support=O-I03; counter=limitations column; group=safe_other; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2508.20983v1; dependencies=source revision; synthesis=sources.csv.
- S-I04: DACON236253rank — SKKU1leaderboard;finaljudgeprocessdistinct. | type=source-observation; risk=normal; scope=officialtable; intent=I1/I3; support=O-I04; counter=limitations column; group=safe_other; status=partial for external validity, supported as source reading; primary=https://dacon.io/competitions/official/236253/leaderboard; dependencies=source revision; synthesis=sources.csv.
- S-I05: SKKUwinningcode — Real/fakespeechpresence,notcurrentmusic;multiresolutionandcountaux. | type=source-observation; risk=normal; scope=officialpost+pinnedactualcode; intent=I1/I3; support=O-I05; counter=limitations column; group=safe_other; status=partial for external validity, supported as source reading; primary=https://dacon.io/competitions/official/236253/codeshare/11562; dependencies=source revision; synthesis=sources.csv.
- S-I06: DACON2024rank4 — Denoisingnotprovenuniversalgain. | type=source-observation; risk=normal; scope=officialpost+README; intent=I1/I3; support=O-I06; counter=limitations column; group=safe_other; status=partial for external validity, supported as source reading; primary=https://dacon.io/competitions/official/236253/codeshare/11566; dependencies=source revision; synthesis=sources.csv.
- S-I07: DACON2024rank5 — Whispersmallenhance;notAFWhispernorcurrentproof. | type=source-observation; risk=normal; scope=officialpost+README; intent=I1/I3; support=O-I07; counter=limitations column; group=safe_other; status=partial for external validity, supported as source reading; primary=https://dacon.io/competitions/official/236253/codeshare/11574; dependencies=source revision; synthesis=sources.csv.
- S-I08: DACON2024MOTA — DANNrank10notisolatedgain. | type=source-observation; risk=normal; scope=README; intent=I1/I3; support=O-I08; counter=limitations column; group=safe_other; status=partial for external validity, supported as source reading; primary=https://github.com/Orca0917/fake-audio-detection; dependencies=source revision; synthesis=sources.csv.
- S-I09: Syn-Att SPCup2022 — GeneratorattributionREALunknown;testPLincompatible. | type=source-observation; risk=normal; scope=fullpaper+code+winnerannouncement; intent=I1/I3; support=O-I09; counter=limitations column; group=safe_other; status=partial for external validity, supported as source reading; primary=https://arxiv.org/abs/2309.08146; dependencies=source revision; synthesis=sources.csv.
- S-I10: Inclusion2024report — Top2multimodal,rank3MFCCunknownfakepoor. | type=source-observation; risk=normal; scope=fullreport; intent=I1/I3; support=O-I10; counter=limitations column; group=safe_other; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2412.20833v2; dependencies=source revision; synthesis=sources.csv.
- S-I11: 1MDeepfakesresults — Pindropfinalclassification2/localization1notaudio-only. | type=source-observation; risk=normal; scope=officialresults; intent=I1/I3; support=O-I11; counter=limitations column; group=safe_other; status=partial for external validity, supported as source reading; primary=https://deepfakes1m.github.io/2025/evaluation; dependencies=source revision; synthesis=sources.csv.
- S-I12: DeepFox2 — Winnernamesnotreproduciblemethods. | type=source-observation; risk=normal; scope=officialpages+indexedclosureexcerpt; intent=I1/I3; support=O-I12; counter=limitations column; group=safe_other; status=partial for external validity, supported as source reading; primary=https://sites.google.com/view/s4p2025/deepfake-challenge; dependencies=source revision; synthesis=sources.csv.
- S-W01: TFCL — Sameaugmentationobjectiveablation;lossweight/evalmodeconcerns. | type=source-observation; risk=normal; scope=fullpaper+pinnedcode; intent=I1/I3; support=O-W01; counter=limitations column; group=add_challenges; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2607.17761; dependencies=source revision; synthesis=sources.csv.
- S-W02: Dataset MT and GRL — Macro vs pooledoppositedirections;licenseunverified. | type=source-observation; risk=normal; scope=fullpaper+inferencecode; intent=I1/I3; support=O-W02; counter=limitations column; group=add_challenges; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2607.23961; dependencies=source revision; synthesis=sources.csv.
- S-W03: TDL revised — v1numberssuperseded;160msframe7.04/11.23. | type=source-observation; risk=normal; scope=fullpaper+scoringcode; intent=I1/I3; support=O-W03; counter=limitations column; group=robust_partial; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2309.03036v2; dependencies=source revision; synthesis=sources.csv.
- S-W04: TemporalPyramid — Utterancescore notframelocalization;OODregression. | type=source-observation; risk=normal; scope=fullpaper; intent=I1/I3; support=O-W04; counter=limitations column; group=robust_partial; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2606.16837v1; dependencies=source revision; synthesis=sources.csv.
- S-W05: DeePen — AccuracygainnotEER;selectedaugvsall. | type=source-observation; risk=normal; scope=fullpaper; intent=I1/I3; support=O-W05; counter=limitations column; group=robust_partial; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2502.20427v1; dependencies=source revision; synthesis=sources.csv.
- S-W06: RobustLocalization — Framevsutterance andpositiveclassmetriccontract. | type=source-observation; risk=normal; scope=fullpaper; intent=I1/I3; support=O-W06; counter=limitations column; group=robust_partial; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2507.03468v3; dependencies=source revision; synthesis=sources.csv.
- S-W07: TRACE — Devlabelselection;fixedthresholdBERnotEER;diagnostic-only. | type=source-observation; risk=normal; scope=fullpaper; intent=I1/I3; support=O-W07; counter=limitations column; group=robust_partial; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2604.01083v1; dependencies=source revision; synthesis=sources.csv.
- S-W08: LlamaPartialSpoof — Matchedcrossfades;parentdependencies. | type=source-observation; risk=normal; scope=fullpaper; intent=I1/I3; support=O-W08; counter=limitations column; group=robust_partial; status=partial for external validity, supported as source reading; primary=https://arxiv.org/html/2409.14743v2; dependencies=source revision; synthesis=sources.csv.
- S-W09: AttackAgnosticDataset — OfficialASVpartitionspooledandresplit;notuntouchedtest. | type=source-observation; risk=normal; scope=fullpaper+pinnedcode; intent=I1/I3; support=O-W09; counter=limitations column; group=dataset_validity; status=partial for external validity, supported as source reading; primary=https://arxiv.org/pdf/2206.13979; dependencies=source revision; synthesis=sources.csv.
- S-W10: DeepFakeVoxHQ — REALcountsourcesinconsistent;grouping/releasegap. | type=source-observation; risk=normal; scope=fullICLRpaper+authorpage; intent=I1/I3; support=O-W10; counter=limitations column; group=dataset_validity; status=partial for external validity, supported as source reading; primary=https://arxiv.org/abs/2411.00121; dependencies=source revision; synthesis=sources.csv.
- S-W11: ReliabilityEstimation — Inferencewindowselectionnotdataprune;fullpaperblocked. | type=source-observation; risk=normal; scope=abstract+partialindexedthesis; intent=I1/I3; support=O-W11; counter=limitations column; group=dataset_validity; status=partial for external validity, supported as source reading; primary=https://resourcecenter.ieee.org/conferences/icassp-2023/spsicassp23vid1808; dependencies=source revision; synthesis=sources.csv.
- S-W12: Layer-conditionedembeddingfusion — Supervised+SSLcrossrepresentation;fulltext/codeunavailable. | type=source-observation; risk=normal; scope=publisherabstract; intent=I1/I3; support=O-W12; counter=limitations column; group=speech_early; status=partial for external validity, supported as source reading; primary=https://www.sciencedirect.com/science/article/pii/S0885230823001183; dependencies=source revision; synthesis=sources.csv.

## Refuted / unresolved
- Reused N response confused ADS and totalweights andstalevoiceresidualgate; rejected in favoroffreshL/root.
- Universalcoverage,officialcurrentrank/improvement,targetGPUtiming:notestablished.
- T32identity,USTCensemble6vs12,unreleasedwinnercode:primarypublicgapsremain.
- Datasetlicenses/weights/dataredistributionnotfullyverified;nofutureuseauthorizedbythisresearch.
