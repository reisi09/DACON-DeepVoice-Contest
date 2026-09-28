# ASVspoof and ADD evidence

F:36 varied searches, organizer reports and sevenASV5solutionpapers. G:43queries,organizer andsolutionfullpapers,threepublicrepos. Source inventory F01–F10/G01–G18.

ASV5 Track1-open official order by minDCF: T45 USTC, T36 Intema, T27 EPITA, T23 Whispeak, T43 SZU. EER ordering differs. USTC's2.59%EER is speechchallenge evidence only; privatebusinessdata and12-versus6subsystem reporting disagreement block exact replication and cost claims. Intema's WavLM-base mean+linear is simpler but still fine-tuned and five-model averaged. Some ASV5 leaders exceed10%EER on other datasets; the organizer cross-dataset table excludes USTC. Open condition does not mean public implementation.

ADD2022 historical winners: LF Vicomtech21.71%EER;PF Tencent4.80%;FG-D neuralstitching10.1%WEER. ADD2023 FG-D USTC-NERCSLIP12.45%WEER,TeamSpirit17.93,CAU_KU22.13. Localization DKU-DUKEECE67.13 vsTranssion62.49; these are combined localization scores, not EER. Currentleaderboard andpostchallenge improvements are separate evidence states.

Most transferable ideas: equal-data paired content training; multiple temporal views; checkpoint weight averaging; modest fixed-score logistic fusion controls; source/condition balancedsampling. Counterexamples: RawBoost policies can regress drastically; augmentation improvesdev whilehurtingITW; learnedfusion canworsen progress; localizationwinner uses artificial middle/shortersegment priors. All are bounded hypotheses, not transferable gains.

AT-ADD2026 exists and includes all-type audio. Its exclusive routing does not implement DACON's simultaneous voice/music outputs. ADD2025 notverified afterexactqueries. OriginalASV2021workshop2021-09-16 outsidewindow;2022/2023retrospective admitted.

Pinned code:

- https://github.com/asvspoof-challenge/asvspoof5/blob/fe23d3053a0889b305d98c9e4cadf418c213d972/Tool-score-fusion/score-fusion.py#L130 — calibration code; SASVAND semantics must notbecome FileOR.
- https://github.com/caizexin/speechbrain_PartialFake/tree/3cefabaf2689d64baddcec42b6113800819bcc31 — laterADD2023-trained scores nothistoricalADD2022winner.
- https://github.com/xieyuankun/AT-ADD-Baseline/tree/c7faab14d2f6467b02cba0f0e436f647316371dc — baseline notwinner,upstreamresamplingrequired.
- https://github.com/rlataewoo/nacl_sdd/tree/a47ed51f1bc3d525f7e174661e78a1f269e683a0 — curriculumcode,notexecuted.

## EXPAND

- T32 closedwinner exactcode/identity; USTC6vs12 count; exactauthorreleases: primary search exhausted withinlane, unresolvedreproduction details,no localreplicationproposal.
- Lower-ranked systems: duplicate recent-speech/source-selection lanes; furtherbenchmark-only methods notprioritizedwithoutdistinctmechanism.
- Tencentfullpaper unavailable: primaryabstract+officialrank only,closedaccessgap.
- AT-ADDrunnerup pairedconsistency/GroupDRO: expansion2 withB/J; exactwinner code/timing unavailable remainsgap.
- Algorithmrecognition89.63%: closedasdiagnostic-only,notauthenticity outcome.
- CFADdataset: datasetvaliditylane;license/transfer gap recorded.
