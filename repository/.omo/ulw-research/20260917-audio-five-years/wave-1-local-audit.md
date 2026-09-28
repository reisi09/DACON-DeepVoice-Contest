# Local flow/history audit and root verification

K local_flow and L evaluation_audit used filename,symbol/caller,semanticresult searches. Root re-executed the two decisive numerical/data-integrity checks. No GPU or model inference was required.

## Corrections to previously retained descriptions

- `verify-single-window.json`:160 speech-prefixed entries,160 with one actual Anti window,0 max/top30 disagreements.32parents×5variants;4.0sinputs versus4.0375swindow. This does not test temporalpooling for sparsefake speech.
- `verify-connected-groups.json`:568rows,106atomicgroupIDs,119compositegroupstrings,93provenance-connectedcomponents. Both dev/test−6dB8-filepanels each collapse toONEconnectedcomponent;sharedspeaker/backgrounds connect the samples. No recorded component crosses splits.93 is not proof of completeindependence;unrecordedprovenance/pretrainingoverlapunknown.
- Existing `parent_groups` reports count distinct composite strings, not connected independence (`six_h100_experiments.py:39`). Existing preparation's106 count is atomicgroupassignmentcount (`prepare_voice_music_h100.py:127`). Correct before nextpromotiondecision.
- Retry residual checkpoint actually exists:504rows/7speakers,seed41;no reportedEERbenefit. Stale skipped-gate descriptions apply to earlier path,notretry.
- Calibration8voicefiles representonespeaker,plus21musicfiles. Single positive-slopePlatt preservesrank exceptties;separatePlattthenensemble canchange ranking. ADSweights andtotalweights differ by0.9,notcontradiction.

## Already executed; do not market as new

48headfits;MLP/cosine/prototypes/attention/convmean/logmeanexp/labelsmoothing/scheduler;1vs8temporalbins;AASIST-XLSR300M;AFWhisper;MERT/EATrealfullFT;teacher-remixseparatoradaptation;V4coarsemusicpartialbags;centered3spartialvoice/music;alignedpresence×fakefusion;max/noisyOR;Anti top30implementedbutlatestvoicedpaneldegenerate.

Historical S0 frozen26.67vsMERTfullFT31.94vsEAT30.00 MusicEER on150music60real/90fake. S2EAT16.76 isdifferent190music12real/178fakepanel;cannotrankagainstS0. Source:results/fullft8_h100x1_20260916/RESULTS.md and perrun evaluationJSON.

## Actual feature paths

- MERT `redesign_features_runtime.py:66–80`:last_hidden_state temporalmean;cache doesnotretainwithin-windowtimeorintermediatelayers. Newlayerprobe requiresnewextraction.
- DF-Arena `v5_experts.py:71–91`:alreadyalllayerhiddenstates andlearnedattention,notlastlayeronly.
- Anti `antideepfake_runtime.py:66–83`:halfwindowoverlap;DF/MERTnonoverlap+tail,shortaudioDFtileswhileAntidoesnot. Comparealignedschedulesbeforeattributinggainstobackbone.
- Music armB `redesign_models.py:56–82`:gatedraw/stemresidual,notjuststemMLP.
- Existinggroupbootstrap seam `redesign_evaluation_common.py:111–133`;fullID/label/groupagreement requiredby `redesign_evaluate.py:36–42`.

No heldout cause demonstrated for zeroaudiblestemmasks or clippedDFlimits:568cachefiles/1486windowsscreenedwithzero suchcases. Do notturnhypotheticalfailuremodesintodiagnosedcauses.

## EXPAND

- MERT/Anti layers→B;forensicbranch→D/M;framelevelpartial→C/E;pairedinvariance/LoRA→B/C/J. Coveredelsewhere.
- V5multiexperttrainingexecutionnotfullyreaudited: do notsayjointtrainingneverattempted.
- Overallhistoricaltestreuseandmissingvoicelabels remainlimits,noofficialgaininferred.
