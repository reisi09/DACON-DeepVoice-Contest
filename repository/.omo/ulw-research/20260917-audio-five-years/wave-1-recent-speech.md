# Recent speech findings

Observer B:>30queries,fullpapers/code. Sources B01–B18. Anti v4 corrects earlier EER; its artifact-positive supervision includes restored/codec speech, not identical to provenance labels. Explicit posttraining overlap includes ASV19/21/5,MLAAD,WaveFake,SpoofCeleb,DFADD,DECRO,DSD. ITW1.35 is not DFEval4s26.76; neither predicts singing/mixture performance.

Layer probes are distinct for current MERT/Anti, but DF already aggregateslayers. NAACL controlledlayerstudy supports hypothesis;2509.12003 explicitlyoracleand2606.30791uses6testsetsforselection. Nofixedlayerrecipepromoted.

OrdinaryLoRA beforeMLDG/GRPO. Proposedrank8q/v count1966080 excludeshead andisnotVRAM/timing. ReferenceKLonlyontrustedlabels/fixedviews;donotpreserveteacher'sknownlabelconflicts. GRPO paper omitsXLSR1B;officialconfighasthreefullmodels;64binarysamplesnot64encoderpasses. Pinned0dea622bde8f064c8ee5a557f2598643123fc6b6 code hasbatchrewardnormalization/currentpolicysamplingvsperinput/oldpolicy description: unresolvedreproductioncontract.

All-typeSSL2026bestaggregate dualXLSR+EATloses toEATonESDD2mixtures. One-ClassKD needsbothteacher/student at inference,so notcompression. Nes2NetbackendMACreductionisnotfullencoderreduction. Newbackbonegridsdeferred.

## EXPAND

GRPOvariant,referenceanchorinventory,license/pretrainoverlap areimplementationprerequisites,notunreadrecommendationclaims. FTDKD fulltext inaccessible;DK-CAST/Frequency-mixKD/MSLAonlypartialleads:notpromoted;duplicatecompressionfamilycoveredKaggle/OCKD. Fake-Mambaidentityunverifiedexcluded. LayerconditionedfusionclosedbyA. No additionalunverifiedperformanceused.
