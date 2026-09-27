# DACON 236749 — Top-15 research plan (all branches), 2026-09-27

Written 2026-09-27 06:40 KST. Leaderboard submission closes **2026-09-29 10:00 KST (~51 h)**. Scope: read-only investigation of
every git ref (main `b6dfcb4`, `cpu/lb-improvement-20260924` `1c08b1b`, `data-v4` and `pull/1` — both already contained in main),
official rules, and targeted literature. No training, GPU, data acquisition or DACON submission was performed.

Full evidence (every slice, verdict, experiment, defect and source): [RESEARCH_RESULTS_2026-09-27.md](RESEARCH_RESULTS_2026-09-27.md).

Labels used below: **[V]** verified in this investigation against repo code/JSON; **[R]** recorded in the repo but resting on a
user screenshot or markdown not in the repo; **[I]** inferred; **[U]** unverified external claim (search snippet only).

## Decision summary

1. **Incumbent is 0.7912534339, not 0.7396.** Submission 102411 (09-24 02:09:50): total 0.7912534339, ADS 0.7683174603,
   CPS 0.9976771958, 17m32s [R]. It is bound to `DataV4_DurationBalanced` (zip `05ae7868…`) only by timing (upload 4m31s after
   packaging commit `c157c2f`) [I]. Two later attempts regressed: E08 ADS 0.7064761905 [R], PANNs 104224 total 0.7901426243 [R].
   main's docs still show 0.7396 as best; they are stale.
2. **No validated improvement over the incumbent exists on any branch.** Every post-09-24 idea was either confounded (E08
   changed 6 things), touched the saturated presence path (PANNs), or was rejected on held-out data (AntiDeepfake/SONICS fusion).
3. **The measured bottleneck is component separability on music-containing files, not calibration or presence.** On the
   held-out proxy, 538/705 FILE errors are on files with music; Voice EER is 0.206 when music is present vs 0.044 voice-only;
   Music EER is 0.200; oracle music labels would cut FILE EER 0.109→0.045 [V: recomputed by branch audit, confirmed by verifier].
   Presence is saturated (CPS headroom +0.00023 total).
4. **Highest-value next action: submit the already-built `DataV4_P1Refit` after a judge-stack preflight.** It is the incumbent
   runtime byte-for-byte (`c157c2f` + candidate string [V]) with presence assets identical [V], so its LB result isolates one
   lever: training on the 6,450 held-out rows, which are exactly the domains that cost the incumbent most (Lyria-3 Pro,
   fx_encoder, OpenVoice, VoxCPM, FLEURS-ko, MUSAN, all 400 constructed real songs). Paired val gain +0.00548 ADS
   (selection-biased on both sides) [V].
5. **Then run two cheap, single-factor screens in parallel on CPU caches** and spend later slots only on those that pass a
   held-out gate: (a) complete the missing `VOICE_FAKE=1` labels on ~780 fully-AI songs' vocals and retrain voice heads only;
   (b) select the MERT layer for the music head (all layers come from the same forward pass, so runtime cost is zero).
6. **Do not submit `DataV4_S2a_AIHubPhone`.** No val gain over P1, and AI-Hub terms ("no export abroad") conflict with the
   round-2 obligation to hand over actual training data. Because the leaderboard keeps the best score, a higher S2a score would
   lock the team into that compliance risk.
7. **Top 15 cannot be estimated from here.** dacon.io is blocked by this environment's network policy and no leaderboard
   snapshot is in the repo. The current rank-15 cutoff is the most important missing number.

---

## A. Top-15 target assessment

| Item | Value | Status |
|---|---|---|
| Competition | DACON 236749, deepfake voice/music detection (행안부·NIA·국과수) | [V] `repository/COMPETITION_CONTRACT.md` (rules checked 09-13) |
| Metric | Score = 0.9·ADS + 0.1·CPS; ADS = 0.5(1−EER_file)+0.2(1−EER_voice)+0.3(1−EER_music); component EERs only on files whose component is truly present | [V] contract |
| Ranking | Public = 100% of the 1,200 files; Private = final Public; best score kept; ties → earlier submission | [V] contract + FAQ |
| "Top 15" | Interpreted as final leaderboard rank ≤ 15, then passing verification into round 2 (30 pts performance, 70 pts reports) | contract |
| Best verified-ish result | 0.7912534339 (ADS 0.7683174603, CPS 0.9976771958) | [R]/[I] |
| Current rank / rank-15 cutoff | **unknown** — dacon.io blocked (egress 403), not in repo | gap |
| Slots left | ≤ 9 if none used on 09-27 (3/day; reset semantics unverified — 4 uploads are recorded on 09-13) | [I] |
| Scoring determinism | ADS lies on a 1/126000 grid; 88418 = 88809 to 10 dp | [R] |

Score sensitivity: 0.01 EER is worth 0.0045 (FILE), 0.0027 (music), 0.0018 (voice) of total score; +1.0 presence AUC headroom is
already ≤ 0.00023. Because Private = Public on the same files, an LB gain is the final-ranking gain; the only cost of a
regression is the slot.

**What cannot be estimated:** the probability of top 15 (cutoff unknown), and the LB transfer of any internal gain. The only
proxy→LB anchor is one pair (Balanced7144→DataV4: proxy ΔADS 0.0611, LB ΔADS 0.0383, ratio 0.63, n=1) [R]. Internal numbers
overstate LB error-free performance heavily (DataV4: val ADS 0.918 → held-out 0.863 → LB 0.768).

## B. Repository and experiment reconstruction

### B.1 Branch map

| Ref | Head | Relation | Content |
|---|---|---|---|
| main / claude/amazing-archimedes-a92agr | b6dfcb4 (09-27 04:56) | — | DataV4 pipeline, P1Refit/S2a packages, zero-shot fusion (rejected), learning curve, CPU/MPS trainer, AI-Hub/Jamendo extras |
| cpu/lb-improvement-20260924 | 1c08b1b (09-26 21:48) | 59 ahead, 1 behind main (base c157c2f) | 87-finding audit, data admission (1,477 rows, never trained), A100 partial E00/E08, local head experiments, PANNs adaptation + LB 104224 |
| data-v4 | 435f153 | ancestor of main | nothing unique |
| pull/1 | 13a957e | ancestor of main | nothing unique |

**Do not merge the branch into main before the deadline.** `git merge-tree` gives 3 textual conflicts (full_bank_train.py,
full_bank_submission.py, package_datav4_submission.py) plus semantic breaks: branch `score_rows` rejects heads without
`feature_contract` (DataV4/P1/S2a), and the merged `full_bank_features.py` hash invalidates every existing cache [V by workflow
verifier]. Cherry-pick fixes only if needed.

### B.2 Official submission ledger

| ID | Time (KST, displayed) | Package | Total | ADS | CPS | Runtime | Identity |
|---|---|---|---:|---:|---:|---|---|
| 88418 / 88809 | 09-13 | official baseline G0 | 0.6909110688 | 0.65775 | 0.98932 | 21m59s | [R] |
| — | 09-13…16 | AST, V5 B1, V6, C3, S2 EAT | 0.6175–0.6844 | | | | [R] all below baseline |
| — | 09-16 | C08 (music Conv1D swap) | 0.7314822646 | | | | [R] |
| — | 09-16 | M330 MLP (music swap) | 0.7396465503 | | | | [R] |
| 91610 | ~09-17 | B_full1696 (noisy-OR) | 0.6838 | | | | [I] mtime only |
| 92294 | 09-17…19 | MERT12 full1322 | 0.7479394074 | | | | [I] |
| 92644 | 09-17…19 | Expanded | 0.7579036931 | 0.73218 | | | [I] |
| 93093 | 09-17…19 | Balanced7144 | 0.7559347884 | 0.7299920635 | 0.9894193122 | | [I] |
| **102411** | **09-24 02:09:50** | **DataV4_DurationBalanced** | **0.7912534339** | **0.7683174603** | **0.9976771958** | 17m32s | [R]/[I] timing |
| (no ID) | 09-24 ~23:00 | E08 partial-data top-30% voice | — | 0.7064761905 | — | — | [R] user-reported |
| error | 09-26 20:37:55 | PANNs R3 (NumPy 1.26 crash) | — | | | | [R] slot deducted |
| 104224 | 09-26 21:13:14 | PANNs R3 fix1 | 0.7901426243 | 0.7674603175 | 0.9942833862 | 26m16s | [R]/[I] |

Packaged, never submitted: `DataV4_P1Refit` (1b9ba441…), `DataV4_S2a_AIHubPhone` (c7c8b1c8…), Existing9504, B_full1696 variants.

### B.3 What is trustworthy

| Evidence | Verdict |
|---|---|
| DataV4 held-out proxy (6,444 rows, heads never trained on them): File 0.10940, Voice 0.11071 (n=5,139), Music 0.20005 (n=2,849), ADS 0.86316 | **Valid for relative comparison** (named candidate: phase-C duration-balanced, file_ensemble 3e0056d1). A second "DataV4 proxy" (phase-B, File 0.113911) exists in branch diagnostics — never mix them. |
| DataV4 val (4,931) | Biased: used for checkpoint selection and CDF fit. Paired comparisons with shared bias are usable for direction only. |
| Branch local panels (selection 421 / calibration 335 / music 35–43) | **Invalid for decisions**: all rows were DataV4's own val; music panels too small (one error ≈ 2.3–5.7 pp). |
| P1Refit test metrics (e.g. Music 0.03191) | **Invalid**: trained on those rows. |
| E08 LB drop | Confounded (partial 5,196-row data, 1 seed, no duration balance, music head retrained on 373 rows, legacy decision layer, top-k pooling). Says nothing about pooling. |
| PANNs 104224 | Clean single-path change (presence only; fake heads and CDF tables byte-frozen). Regressed: CPS −0.0034, ADS −0.0009 (FILE via stale CDF tables). |
| Zero-shot AntiDeepfake/SONICS fusion | Rejected on held-out all-rows (ADS 0.86316 → 0.85465); SONICS never selected. |
| Learning curve (1 seed) | Voice held-out improves with training (0.129→0.110) but is confounded with update count; music held-out flat 50%→100% (0.198→0.206). |

### B.4 Pipeline (incumbent runtime, `c157c2f`) — traced, no defect in the scored path

16 kHz mono, peak cap 0.6 → HTDemucs (vocals vs sum of other stems) → **voice**: DF-Arena-1B on non-overlapping 4.04 s vocal-stem
segments, fc5-input 1280-d embeddings → `Linear(1280,2)` initialised from fc5, file score = max over segments of softmax[fake]
(= sigmoid of max odds, same as training) → mean of 3 seeds; **music**: MERT-v1-330M layer 12 time-mean of 4.04 s backing-stem
windows (hop 2.02 s, 16→24 kHz) → WindowMIL (MLP per window, log-mean-exp) → mean of 3 seeds; **presence**: logistic heads on
PANNs group stats + stem energy; **FILE**: equal-weight mean of val-CDFs of max(Pv·Fv, Pm·Fm), noisy-OR, max(Fv,Fm).
Train/inference consistency holds for DataV4 rows (all peak-capped; pooling identical; bf16 vs fp32 heads |ΔEER|=0) [V].

Verified defects (none in the scored inference path):

| Defect | Location | Impact |
|---|---|---|
| Echoes generator parsed as `Echoes/TTA`/`Echoes/ATA` (12 generators collapsed; 59/273 reference songs span splits) | main `data_v4/build_manifest.py:135-143` (fixed on branch, not re-run) | Wrong per-generator caps/weights for 594 rows; mild leakage in DB val/test |
| Construction closure moved all 400 constructed real songs and 276/300 enhanced-real rows to test | main `build_manifest.py` | DB trained on almost no real songs → **P1 fixes this by training on test** |
| Legacy fully-AI songs have `VOICE_FAKE=null` (SONICS 751, suno 18, udio 14, …) | manifest labels; `full_bank_train.py:37-40` | Fake singing ≈ 1.5% of voice fake-class weight; 0 fake a cappella rows |
| `decision_heads.py` hash is part of the feature-cache identity | `full_bank_features.py:117-119`, `full_bank_train.py:50-52` | Editing it blocks retraining/resuming on every cache → put new decision logic in a new module |
| FILE_FAKE_mask drops whole rows from reporting (6 HAIM rows) | main `data_v4/file_ensemble.py:71-72` (fixed on branch) | Reporting only |
| Fusion time budget makes output depend on processing order | main `full_bank_submission.py:127-133` | Only if fusion is packaged — it is not |
| PANNs workflow ships a trained arm even when the baseline wins | branch `panns_submission_workflow.py:155` | Selection optimism |

## C. Competition requirements matrix

| Requirement | Official source | Current implementation | Evidence | Gap |
|---|---|---|---|---|
| 5 probabilities in [0,1], ID order, UTF-8 CSV | data / evaluation pages | `validate_predictions`, verifier | [V] 10-file offline QA | none |
| Per-file independence (rule 2-4) | talk 417344 | CDF tables are fixed per file; no cross-file stats | [V] reversed-subset check | fusion time-budget path (unused) |
| Offline, ≤60 min on L4, NumPy 1.26.4 / torch 2.7.1+cu128 | evaluation page | c157c2f runtime ran 17m32s on LB | [R] | **P1 not run on L4 or under NumPy 1.26.4**; PANNs crash precedent |
| ZIP ≤10 GB, `model/ script.py requirements.txt` | evaluation page | 6.35 GB | [V] receipts | none |
| ≤3 submissions/day; post-execution errors deducted | rules/FAQ | — | 3 of 20 recorded uploads errored | slot fragility |
| Round 2: reproducible training code, actual training data files, 2 HWP reports, due 09-30 12:00–10-05 10:00 | rules; talk 417280/417198/417336 | manifests with source/revision/hash; generation scripts | [R] | **rights register**: MLAAD (NC, research-only reading), JamendoMaxCaps 400 rows in DataV4, AI-Hub in S2a, 900 regenerated MUSAN noise files + 8 missing HAIM files |
| External data must be publicly accessible, ≥ non-commercial use | rules; 417193/417212 | per-row license | [R] | Jamendo TDM terms; AI-Hub export clause |
| No LB probing | talk 417702 | none executed | [R] | do not run constant-column probes |

## D. Bottleneck analysis

| # | Observed weakness | Evidence | Alternative explanation | Discriminating test | Priority |
|---|---|---|---|---|---|
| D1 | Music-fake separability under unseen generators | Held-out Music EER 0.200; Lyria instrumental FNR ≈0.75 (n=77); real-music FPs (MUSAN/OpenMIC/constructed) inherited from parents (P(FP\|parent FP) 0.607 vs 0.149); learning curve flat 50→100% | Held-out set is 2 fake generators vs 1 real corpus, so part of the EER may be MUSAN-vs-rest corpus shift | Layer screen on the same split + second real corpus check (M-LAYER) | **1** |
| D2 | Singing voice inside songs | Voice EER 0.206 with music vs 0.044 without; Lyria vocal FNR 0.631 (n=222); within-singing AUC 0.75–0.78 for every head | Speech-trained frozen embedding lacks a singing-fake direction (feature limit) rather than a labeling/weighting limit | Label completion + voice-only retrain, evaluated on singing vs speech subsets (V-SING) | **2** |
| D3 | Training data never covered held-out domains | DataV4 ADS val 0.918 vs held-out 0.863; closure put all real songs in test | Generator families in the private set may still be unseen | LB A/B of P1Refit (only heads change) | **1 (ready now)** |
| D4 | FILE fusion alignment | FILE variants all within ±0.004 on proxy; one exposed-panel Platt signal (−0.011 on 335 rows, no CI) | Panel noise / exposure | Fit on val, test on 6,444 held-out rows with component bootstrap (F-PLATT) | 3 |
| D5 | Cross-condition score shift (speech vs singing, duration) inflating pooled EER | Oracle singing offset: proxy Voice 0.1107→0.1059 at best, **val worsens** 0.0510→0.0542 | The shift direction differs between splits, so it will not transfer | Already tested in repo — **deprioritised** | 5 |
| D6 | Voice head under-converged at 50-epoch cap | 2/3 seeds stop at cap; val headroom ≈0.0014–0.0019 | Linear probe is convex; more seeds identical (Spearman ≈1) | Fold into V-SING (max-epochs 100) | 4 |
| D7 | Presence | CPS 0.9977; oracle presence moves FILE −0.0037 (CI includes 0); every presence change lowered CPS on LB | — | none needed | reject |

## E. Research synthesis

Egress blocked arxiv/isca/ieee for the research agents; GitHub READMEs/code were opened. Paper-level numbers marked [U].

| Method | Primary source | Relevance | Transfer limitations | Repository integration | Verdict |
|---|---|---|---|---|---|
| In-domain singing training fixes speech-CM failure on songs | SingFake, ICASSP 2024 (arXiv 2309.07525) [U]; official repo `dataset/separate.py` [V: Demucs vocals + VAD] | Direct | Their CMs trained only on ASVspoof19; ours already sees some singing | V-SING label completion | pilot |
| Speaker/music FMs complement speech CMs for singing | Phukan et al., ICASSP 2025 (arXiv 2409.14131) [U]; SingGraph repo README [V] | Domain difference (a cappella) | Needs vocal-stem MERT/x-vector extraction (GPU) | not before deadline | speculative |
| Lower/middle SSL layers most discriminative for deepfakes | Kheir et al., NAACL Findings 2025 (`Yaselley/SSL_Layerwise_Deepfake` README) [V] | Domain difference (speech SSL, not MERT) | No MERT-on-music evidence; lower layers more channel/corpus sensitive | M-LAYER on existing multi-layer cache | pilot |
| Single task-appropriate layer often beats weighted fusion in music SSL | Layer-wise music SSL (arXiv 2505.16306) [U] | Plausible | No deepfake task | M-LAYER design (select 1–2 layers) | supports pilot |
| Short context hurts AI-song detection; unseen generators harder | SONICS, ICLR 2025 (`awsaf49/sonics` README: 5 s F1 0.76–0.78 vs 120 s 0.88–0.97) [V] | Direct | Zero-shot SONICS ≈ chance on our data | Our windows are 4 s but aggregated by LME over the file | informs D1 |
| Decoder upsampling artifacts ("fakeprints") | Afchar et al., ISMIR 2025 (`deezer/ismir25-ai-music-detector` code) [V] | Domain difference | Default band 5–16 kHz is above our 8 kHz Nyquist; mp3 and unseen decoders break peaks | CPU feature + LR | speculative |
| Separation does not recover AI-stem artifacts; detect on mix | Deezer, ISMIR 2026 (arXiv 2607.26874) [U] | Direct question to our stem-only music input | Codec-reconstruction fakes; our own raw-vs-stem runs were inconclusive (EXPERIMENTS §13) | needs mixture MERT extraction (GPU) | speculative |
| Real-only density / OOD scoring | MusicDET, ICML 2026 (repo README) [V]; numbers [U] | Plausible | Real/fake corpus confound in its benchmarks | fits on cached MERT windows | speculative |
| Frozen SSL + logistic head generalises and is calibrated | Pascu et al., Interspeech 2024 (arXiv 2309.05384) [U]; `danoneata/aletheia` [V] | Direct (speech) | No singing | Matches our voice head design | supports current design |
| Condition-dependent calibration improves pooled EER | DCA-PLDA (`luferrer/DCA-PLDA`) [V]; WeSpeaker AS-Norm+QMF README (vox1-O 0.867→0.659%) [V] | Domain difference (speaker verification) | Our oracle test caps the gain at ~0.005 Voice EER with a sign flip on val | none | rejected for now |
| Global monotone transforms cannot change EER; unlabeled-cohort norm can collapse | arXiv 2606.21584 [U] | Direct caution | — | never normalise on test statistics (also rule 2-4) | constraint |
| Codec-aware training reduces cross-codec drift | ArtifactNet (arXiv 2604.16254) [U] | Plausible | Needs re-extraction | DataV4 already applies a label-independent codec table | already covered |

## F. Ranked improvement portfolio

Expected gains are qualitative; do not add them.

| Rank | Proposal | Mechanism | Evidence strength | Cost | Inference impact | Main risk | Next experiment |
|---|---|---|---|---|---|---|---|
| 1 | **P1Refit submission** | Heads trained on +6,450 rows from the exact domains the incumbent misses | Medium: paired val +0.0055 ADS (File −0.0045, Voice −0.0008, Music −0.0103), AD-unseen val +0.0064; ~2–3.5 SE; one-anchor transfer ~0.63 | 1 slot, 0 compute | none (same runtime) | New rows may shift real-music calibration; untested on L4/NumPy 1.26.4 | Judge-stack preflight, then submit |
| 2 | **V-SING**: `VOICE_FAKE=1` for fully-AI songs' vocals + voice-only retrain (max-epochs 100) | Gives the linear probe a fake-singing direction it currently lacks (fake singing ≈1.5% of fake weight; 0 fake a cappella) | Low–medium: weighting imbalance confirmed; within-singing AUC 0.75–0.78 suggests features limit it | CPU minutes | none | Relearning "music context ⇒ fake" (old head had MUSAN FPR 0.668) | DB-protocol screen, gates in G |
| 3 | **M-LAYER**: MERT layer choice for music head | Lower/middle layers may keep decoder artifacts that L12 time-mean discards | Low–medium: speech-SSL layer evidence only; flat learning curve implies representation limit | CPU head training; needs multi-layer cache (VESSL extraction started 09-27, status unknown) | none (hidden states already computed) | Lower layers learn corpus/codec shortcuts; only 2 held-out fake generators | Layer screen, gates in G |
| 4 | F-PLATT: monotone Platt on voice logit before FILE CDFs | Re-aligns voice vs music evidence scales inside FILE | Low: one exposed 335-row panel, no CI | CPU minutes + new module | none | Proxy noise; past FILE variants ±0.004 | Val-fit / held-out test with component bootstrap |
| — | S2a (AI-Hub phone reals) | Real phone channel | None on val (voice EER = DB's) | 1 slot | none | Compliance lock-in; "real call ⇒ real" shortcut | **reject** unless DACON confirms data handover |
| — | Top-k / LME voice pooling | Less max-pooling length drift | None valid; calibration worse (0.0236→0.0355/0.0374) | — | — | — | reject |
| — | PANNs / presence changes | — | LB regressed; CPS cap +0.00023 | — | +8m44s runtime | — | reject |
| — | AntiDeepfake / SONICS fusion | — | Held-out rejected | — | +runtime | — | reject |
| — | Branch-admitted 1,477 rows (LibriSpeech, MLAAD, MusicArena) | More generators | Single-class sources; metadata-only AUC 1.0; never extracted | GPU/MPS extraction | — | Shortcut learning | reject for this deadline |
| — | Mixture-branch MERT, vocal-stem MERT, fakeprints, real-only density | See E | Speculative | GPU extraction (hours) | +1 MERT pass | Deadline | only if GPU approved and ranks 2–3 finish early |

**Simpler alternatives considered:** for rank 2, raising voice max-epochs alone (≤ +0.0009 total, dominated); for rank 3,
retraining L12 with more data (learning curve says no); for rank 4, swapping to noisy-OR alone (±0.001, CI includes 0).

## G. Executable experiment roadmap

All commands run from `repository/` on the machine holding `_workspace/cache_v4` (main `b6dfcb4` code, `--device cpu`).
Paths below are the ones used by main's scripts; adjust only the local prefix.

### Stage 0 — before any upload (every package)
1. Load every `model/trained/*.pt` from the zip with `torch.load(..., weights_only=True)` in a venv pinned to
   `numpy==1.26.4 torch==2.7.1` (CPU wheel is enough). This is the check that would have caught the PANNs crash.
2. Run `verify_datav4_submission.py --out submissions/<Pkg>` (10 real files, offline guard).
3. Record: upload time, zip sha256, DACON ID, total/ADS/CPS, runtime, into `repository/submissions/<Pkg>/lb.json`.

### Stage 1 — incumbent challenge (slot 1, 09-27)
- **Candidate:** `submissions/DataV4_P1Refit/submit.zip` (sha `1b9ba441…`), unchanged.
- **Fixed:** runtime, presence assets (hashes identical to DB [V]).
- **Accept:** LB total > 0.7912534339. CPS must equal 0.9976771958 ± 1e-9 if 102411 is DB; a CPS change means the incumbent
  identity assumption is wrong — record it.
- **If accepted:** P1 becomes the base for ranks 2–3 (refit style: train + held-out rows). **If rejected:** keep DB as base,
  and screen ranks 2–3 as DB-style (train split only) packages.

### Stage 2 — CPU screens (09-27, in parallel; no slot)
Common protocol: DB recipe (duration balance, 3 seeds, val early stopping, patience 5 unless stated), train on the DataV4
**train** split only, evaluate once on the **6,444 held-out rows** with official masks (`graph_scoring.official_metrics`),
paired component bootstrap (1,000 resamples, unit `component_id`) against the DB heads' held-out predictions.

**V-SING (voice only)**
- Label rule (in memory, manifest file untouched so cache identity holds): rows with `FILE_FAKE=1`, `MUSIC_FAKE=1`,
  `VOICE_PRESENT=1`, `VOICE_FAKE` unknown, source ∈ {`existing:SONICS`, `existing:suno`, `existing:udio`} → `VOICE_FAKE=1`
  (a fully generated song's vocal is generated). Exclude HAIM (partial/edited), FakeMusicCaps and Echoes (vocal presence
  uncertain) unless their stem vocal ratio ≥ 0.35.
- Implemented: `repository/data_v4/sing_relabel_screen.py`. It wraps `full_bank_train.head_labels` during
  `read_items`, so every cache identity check still applies and the manifests are untouched. It reuses
  `full_bank_train.step/predict` (voice route, duration balance, 3 seeds, `--max-epochs 100 --patience 10`), copies the
  incumbent music heads, refits `file_ensemble.json` on val, and runs `full_bank_evaluate.evaluate`. The output dir
  therefore packages with `package_datav4_submission.py --heads OUT`. Held-out labels are never completed.
- Run three arms (CPU):

  ```sh
  B=<bank dir>; TV=<cache_tv>; TE=<cache_test>; INC=<DB heads dir, e.g. _keep_best_20260926/train_duration_balanced>
  P=results/data_v4_20260923/data_v4_results_20260923T1639Z/train/presence_head.json
  A="--primary $B/bank_trainval.csv $TV --heldout $B/bank_test_as_val.csv $TE --incumbent $INC --presence $P"
  python data_v4/sing_relabel_screen.py $A --out _workspace/vsing/control --no-completion
  python data_v4/sing_relabel_screen.py $A --out _workspace/vsing/completion --control _workspace/vsing/control/sing_screen_report.json
  # only if the completion gate accepts, and with the P1 heads dir as --incumbent if P1 won on the LB:
  python data_v4/sing_relabel_screen.py $A --out _workspace/vsing/refit --refit
  ```

  `bank_test_as_val.csv` is the held-out manifest the test cache was built from (split `val`; recorded in the S2a run
  identity). Pass the manifest whose sha matches `cache_test/index.json`.
- Report (`sing_screen_report.json`): official held-out metrics for both head sets; pooled, voice-only and with-music
  Voice EER; real-with-music FPR at the pooled EER operating point; paired component bootstrap for ΔVoice and ΔFILE EER.
- **Accept:** pooled held-out Voice EER −0.010 or better with bootstrap CI excluding 0, **and** pooled below the control
  arm, **and** voice-only EER not worse than incumbent or control by > 0.003, **and** real-with-music FPR not higher.
  **Reject** otherwise and stop V-SING.
- Verified in this session only on a synthetic cache in the exact on-disk schema, under torch 2.7.1 / NumPy 1.26.4 /
  transformers 4.57.6 / sklearn 1.8.0. That run exercised identity checks, training, `score_rows`, `evaluate`, the
  ensemble refit, the bootstrap and refit output against the packager's checks. It is wiring evidence, not performance
  evidence. In that run the control-arm check caught a completion arm that bought singing accuracy with speech accuracy.
- Cost: minutes on CPU (the whole 8-run learning-curve audit, including the 24,623-row voice run, finished at 1.3 min on
  its cumulative timer; `results/audit_20260927/learning_curve.json`).

**M-LAYER (music only)**
- Input: multi-layer backbone cache from `results/vessl_runs_20260927/backbones/run.sh` (`/root/backbone_cache`,
  layers 3,6,9,12,15,18,21,24, fp16). First confirm coverage (train/val/test rows) and that the extraction finished.
- Run `python data_v4/train_backbone_heads.py --cache <cache> --out <dir> --layers 3 6 9 12 15 18 21 24` (single seed).
- **Sanity gate:** the L12 row must reproduce DB's held-out Music EER (0.200) within ±0.01; otherwise the cache or windowing
  differs and results are void.
- **Selection:** choose ≤ 2 layers by **val** EER only; rerun those plus L12 with seeds 41/42/43.
- **Accept:** 3-seed-ensemble held-out Music EER −0.02 or better vs L12 (≈57 files of 2,849) with CI excluding 0, and
  MUSAN/OpenMIC real-music FPR not higher. Report Lyria instrumental FNR separately.
- Packaging change: `full_bank_submission.py` uses `hidden_states[12]` → selected index; heads' `mean/scale` width follows
  the layer(s). Needs a new packager option; re-run Stage 0.

**F-PLATT (FILE only)**
- Fit `Fv' = σ(a·logit Fv + b)` on DB val (4,927), refit `file_ensemble` CDF tables on Fv' (val), apply to held-out rows.
- Implement in a new module called from the runtime, **not** in `decision_heads.py` (cache identity).
- **Accept:** held-out FILE EER −0.004 or better with CI excluding 0; VOICE column unchanged by construction.

### Stage 3 — LB single-factor A/Bs (09-27 evening → 09-28)
Only for screens that passed. Refit each on train + held-out rows (P1 style, `--extra-train bank_test.csv cache_test`) if P1
was accepted; package with `package_datav4_submission.py --heads <dir> --candidate <name> --output submissions/<name>/submit.zip`
(main packager; the branch packager rejects heads without `feature_contract`). One change per upload versus the current best.

### Stage 4 — combination (09-28 late → 09-29 before ~08:00)
Combine only factors that each raised the LB. Submit the combination once; if it does not beat its best part, keep the part.
Keep one slot on 09-29 as reserve for a failed upload. Do not upload after ~08:30 on 09-29 (26 min runtime + queue).

### Stage 5 — round-2 readiness (in parallel, no slots)
Rights register for every training source of the final model (MLAAD scope, JamendoMaxCaps, regenerated MUSAN noise,
missing HAIM files; exclude AI-Hub unless DACON answers the handover question in writing). Generation receipts (model,
revision, seed, prompt) for CosyVoice2/Melo/Seed-VC/ACE-Step rows. Exact training command + code commit per submitted model.

### Slot plan (≤ 9)

| Day | Slot | Upload | Condition |
|---|---|---|---|
| 09-27 | 1 | P1Refit | after Stage 0 |
| 09-27 | 2 | base + V-SING voice heads | V-SING gate passed |
| 09-27 | 3 | reserve | — |
| 09-28 | 4 | base + M-LAYER music heads | M-LAYER gate passed |
| 09-28 | 5 | base + F-PLATT | F-PLATT gate passed |
| 09-28 | 6 | reserve / second-best layer | — |
| 09-29 | 7 | combination of LB-confirmed factors | ≥ 2 factors confirmed |
| 09-29 | 8–9 | reserve | — |

## H. Immediate next action

**Run the Stage 0 preflight on `DataV4_P1Refit` and upload it.** It is the only candidate that needs no compute, changes a
single lever against the incumbent's exact runtime, and targets the measured gap (held-out domains the incumbent never
trained on). Its result also decides the base for every later screen: if P1 loses, refit-style candidates are
discounted and later slots go to DB-style changes. Start the V-SING and M-LAYER CPU screens while it scores.

## I. Evidence limitations

- dacon.io is blocked here (egress 403): no live leaderboard, rank, cutoff, or remaining-slot count.
- Every post-09-17 LB number comes from transcribed user screenshots not in the repo; 102411→DataV4 and 104224→PANNs fix1
  bindings are timing inferences. E08's ID, total and CPS were never recorded.
- No caches, prediction CSVs or audio are in the repo, so no metric here was recomputed from row-level predictions in this
  session; recomputations cited come from repo JSON (fusion reports, learning curve, receipts).
- Status of the 09-27 VESSL multi-layer backbone extraction and the local MPS S2 feature chain is unknown (both were running
  when main was committed).
- P1Refit: L4 runtime, NumPy 1.26.4 compatibility, and its per-seed best epochs are unmeasured; its local QA took 606.9 s for
  10 files under memory pressure.
- Research agents could not open arxiv/ISCA/IEEE; paper numbers marked [U] come from search snippets. GitHub code/READMEs
  were opened.
- Round-2 handover: DACON has not answered whether AI-Hub audio can be handed over; MLAAD admission rests on the project's own
  reading of the rules.
