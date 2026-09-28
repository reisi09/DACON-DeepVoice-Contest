# DACON 236749 deepfake audio detection — full state for an independent review (2026-09-28 ~08:45 KST)

Everything below is VERIFIED from files/logs unless marked ESTIMATE/HYPOTHESIS. Files in this folder: data/bank_trainval.csv,
data/bank_test.csv (all 42,039 data v4 rows, metadata + labels; no audio, no AI-Hub rows), code/*.py (pipeline code),
results/*.json (study outputs), PUBLIC_DATASET_SURVEY_20260927.md (earlier dataset/rights survey), codex_round*.md (earlier
debate with one Codex session), MODEL_ARCHITECTURE_ENSEMBLE_DATA_AUDIT_20260928.md (earlier audit).

## 1. Task, metric, rules, constraints
- Per test file predict FILE_FAKE, VOICE_FAKE, MUSIC_FAKE (+ presence probs). Score = 0.9*ADS + 0.1*CPS;
  ADS = 0.5(1-FileEER) + 0.2(1-VoiceEER) + 0.3(1-MusicEER). VoiceEER only on files with voice, MusicEER only with music.
  CPS (presence AUC) ~0.998, saturated. LB = 100% of the test set and the final (private) score = best submission ever.
- Deadline 2026-09-29 10:00 KST. 3 submissions/day. ALL 3 of 09-28 are used. Remaining: 3 on 09-29 between 00:00 and 10:00.
- Runtime: offline, L4 22.4 GiB, 60 min for the whole test set, zip <= 10 GB. Current best package: 30 min, 9.74 GB (STORED).
- External data: publicly accessible + at least non-commercial use allowed; round 2 requires sources and all training data files.
  The user will personally handle downloads that need tokens, gated access or terms acceptance.
- AI-Hub (Korean government) data must NEVER leave Korea: it cannot go to the cloud GPUs (VESSL, EU) and may only be processed
  on the local Mac, which is currently unavailable for heavy training (user). Codex/cloud sessions must never receive AI-Hub data.
- Top-15 cut on 09-27 was about 0.86 ADS (ESTIMATE, may rise).

## 2. Leaderboard history (ADS)
| package | change | DB RT held-out ADS | LB ADS |
|---|---|---|---|
| FullFT8 (old) | MERT/EAT full fine-tuning | - | 0.638 (-0.10 vs frozen) |
| P1 (old) | legacy heads | - | ~0.770 |
| S3 | frozen AntiDeepfake-1B voice-embedding heads + MERT 6+12 music | (AC) 0.8727 | 0.7853 |
| S6 | + EAT-large b16 music, voice heads with label completion L, 100% rows | 0.89877 | 0.8220 |
| S7 (BEST) | + SSLAM-pretrain b5 (backing stem) music, + WavLM-L h8 voice member | 0.90921 | **0.8374** |
| S8 | + SSLAM-pretrain b5 on the RAW MIX appended to music features; FILE weights refit (equal vs S7's 0/9/1) | 0.91352 | 0.8340 (-0.0034) |
DB->LB: S6->S7 DB +0.0104 -> LB +0.0154; S7->S8 DB +0.0043 (unseen-generator subset +0.013) -> LB -0.0034. DB is ~0.07 above LB.

## 3. Current best model (S7)
- 16 kHz mono, peak cap 0.6 -> HTDemucs -> voice stem / backing stem. Windows 64600 samples, hop 32300.
- VOICE = val-CDF fusion (weights 1 / 0.5 / 1) of: DF-Arena-1B fc5-input embeddings -> MLP (segment max); AntiDeepfake XLS-R-1B
  window embeddings -> ResMLP (window max); WavLM-Large hidden 8 (fp16-rounded stem) -> MLP (window max). Silent voice stem -> 0.
- MUSIC = WindowMIL (log-mean-exp) on concat[MERT-v1-330M h6,h12 | EAT-large b16 | SSLAM-pretrain b5] of the backing stem.
- Presence = PANNs Cnn14 tag statistics + stem energy -> logistic heads. FILE = CDF ensemble of max(Pv*Fv, Pm*Fm), noisy-OR,
  max-without-presence (weights 0/9/1 chosen on val).
- All encoders frozen; only small heads trained (3 seeds). Encoders tried and rejected: HuBERT-L, data2vec-audio-L, UniSpeech-SAT-L,
  WavLM on raw mix (voice, all < +0.002 val ADS); MuQ, SSLAM-finetuned, EnCodec latents (music); SONICS zero-shot (AUC 0.49 on
  unseen Suno/Udio); partial fine-tuning of AntiDeepfake (val best, held-out worst, -0.004); SingMOS singing data (+0.0002);
  learned FILE combiners (GBM -0.0057 held-out).

## 4. Evaluation protocol (DB)
- data v4 split by lineage components (data_v4/split.py: connected components over speaker/track/prompt/text/parent keys);
  0 groups span splits. R2 frozen rows: val 3,680, held-out test 4,822 (of 4,931 / 6,450). Train on train, early-stop on val,
  held-out reported (reused many times; not an independent confirmation). Held-out contains 6 fake generators unseen in train
  (1,135 rows) and some unseen sources. No leave-one-dataset-out retraining has been run (a purged design exists, not executed).

## 5. Training balance already applied (code/full_bank_batches.py)
- loss_weights: equal total weight per class, and within a class equal total weight per (source, generator) cell (a 5-row cell
  weighs as much as an 8,000-row cell). duration_balance: each class re-weighted to the pooled log-duration histogram.
- Codec chains (11) are applied equally to real and fake (shares identical, e.g. pcm_wav 0.29/0.30, mp3_128 0.12/0.12).

## 6. Data distribution (computed today from the bank CSVs)
Splits: train 30,658 rows / 44 sources / 13,304 lineage groups (FILE fake 21,492 / real 9,147); val 4,931 / 37 / 2,818;
held-out 6,450 / 18 / 2,730 (fake 2,624 / real 3,820). Languages: unknown 18,117, ko 12,263, en 9,196, others ~2,000.
Component combos (train): fakeV only 15,243 | realV only 6,052 | fakeM only 2,234 | ?V+fakeM 2,082 | realV+realM 1,395 |
realM only 1,362 | fakeV+fakeM 1,287 | fakeV+realM 641 | ?V+realM 357 | realV+fakeM 5. Held-out has realV+fakeM 107.

Voice-present rows by type (P1 train+held-out / val):
| type | real | fake | fake by rule L | vocal label unknown |
|---|---|---|---|---|
| speech | 7,405 / 916 | 16,551 / 2,197 | - | - |
| speech mixed onto music (constructed) | 193 / 19 | 1,149 / 132 | - | - |
| a cappella singing (vocalset, csd) | 934 / 66 | **0** | - | - |
| singing in songs | 1,935 / 237 | 1,595 / 218 | 1,364 / 237 | 462 / 77 |
- Fake singing exists ONLY as fully AI-generated songs (Suno/Udio/HAIM-acestep/Echoes/FakeMusicCaps/own generated_music).
  No SVS/SVC singing deepfakes, no voice-converted AI covers (RVC-style), no CtrSVDD. "fakeV+realM" = TTS SPEECH over real music.
- Fake voice = 71.3% MLAAD + ASVspoof (TTS/VC speech).
- Korean: speech real 5,100 (zeroth 3,000, fleurs 1,500, speechfake 600) vs fake 6,830 (mlaad 3,830, speechfake 3,000);
  Korean a cappella real 100 (csd, one singer), fake 0; Korean songs with accompaniment: real 0, fake 222 (own generated_music).
- Label purity: 56.5% of VOICE_FAKE labels, 74.8% of MUSIC_FAKE, 61.8% of FILE_FAKE come from sources that are >= 99% one
  class (e.g. mlaad all fake; zeroth, fleurs, vocalset, csd, openmic voice all real) -> source-identity shortcut risk.
- Real music: 6,266 rows, 3,605 lineage groups; top-10 groups (a few MUSAN composers, one FMA artist) hold 19%. Sources:
  openmic 2,404, constructed_mix 1,098, musan_music 645, FMA 385, Jamendo 340, haim 328, constructed_song 316. No commercial /
  professionally mastered releases (MUSDB-like) and no K-pop. Real singing: 3,139 rows, 1,588 groups, top-10 hold 32%
  (csd one singer 200 rows). Fake music: 24 generators (acestep 948, udio 901, suno 900, musicgen 892, audioldm2, stable_audio,
  musicldm ...). Fake generators overall 352 strings.
- Unknown vocal authenticity: 2,140 rows; 1,601 completed by rule L (vocal of a fully generated track is AI); ~520 remain
  (HAIM crossfade/concat edits 385, Echoes audio-to-audio 43, constructed 24, FakeMusicCaps without vocal caption ~30).
- No row has a silent separated vocal while VOICE_PRESENT=1; vocal-to-accompaniment ratio is not recorded.

## 7. S8 failure diagnostics (results/diag_s8.json)
- FILE weights in DB: S7 equal 0.90921 vs 0/9/1 0.90982; S8 equal 0.91352 vs 0/9/1 0.91478 (weights explain ~0.001, not 0.0077).
- Real-music FPR by source at val TPR-95: S8 LOWER than S7 on every DB real-music source (musan_music, openmic, constructed,
  haim) -> the DB contains no domain where raw-mix features fail; hypothesis: mastered/commercial real music on the LB.

## 8. Operational facts
- Cloud: VESSL H100x4 and H100x1 (paused). Caches on the cloud volume: stems (fp16) and frozen features for all 42,039 rows;
  feature extraction for a new encoder over all rows ~15-20 min on H100x4; head screening ~30 min; P1 retrain ~15 min;
  packaging + verification ~1 h. VESSL->Mac download ~2.4 MB/s, upload ~3.3 MB/s. Credit ~$70.
- Adding NEW ROWS requires the whole feature pipeline for them (stems + DF-Arena/AD/MERT/EAT/SSLAM/WavLM/PANNs features);
  the packaged runtime Engine itself can be used as the extractor (exact runtime features).

## 9. The user's position
"The data imbalance looks too large." The user wants an hours-long, independent, evidence-based debate and a conclusion,
including datasets that need tokens / gated access / terms (the user will handle access).

## 10. AI-Hub (Korean government data portal) — the user asks to include it
- Competition use is ALLOWED: DACON organisers answered 2026-09-21 (talkboard 417844). AI-Hub policy restricts use/export
  abroad and third-party provision -> AI-Hub audio may only be processed in Korea: on the local Mac (currently not available for
  heavy jobs; availability later = UNKNOWN, ask the user) or on an in-Korea GPU (VESSL lists a Korea-region cluster "deneb-kr"
  with H100, never used; whether it is physically in Korea and acceptable = UNKNOWN, must be confirmed by the user).
  AI-Hub audio/features must never be sent to cloud LLM sessions or the EU GPUs. Package weights trained on it are allowed.
- Candidate AI-Hub sets from the earlier survey (catalog facts only; access needs the user's AI-Hub approval):
  465 multi-speaker singing (18 singers, 4,681 items: ballad/R&B, rock/pop, trot; 256 train + 55 dev files already local) |
  473 multi-timbre guide vocals (92 singers, 4,000 songs, 157 h; Korean real singing with timbre/age/gender) |
  71767 popular-music raw material (120,000 WAV instrument/loop sources) | 71470 Korean traditional music (11,069 WAV) |
  71314 broadcast conversational speech with background music/noise (7,000 h) | 123 KsponSpeech (spontaneous speech) |
  542 multi-speaker TTS-training speech | 71405 noisy-environment commands | 644 environmental sounds (3,928 instruments).
- Possible uses: Korean REAL singing (465/473) as negatives and as sources for SYNTHETIC fakes made locally (SVC/RVC AI covers,
  SVS), Korean real speech over real music (71314), real popular-music material (71767) mixed under Korean vocals; all
  processing (stems, features, generation) must run in Korea.
- UPDATE 08:45 KST (VERIFIED, `vesslctl cluster list`): cluster deneb-kr has REGION kr-west (Korea) with H100 SXM and CPU-only
  specs; capella-emea (the current GPUs and caches) is eu-north1; betelgeuse-na us-west-2. An in-Korea GPU therefore exists;
  whether AI-Hub processing there is acceptable is the user's decision (pending). deneb-kr has its own storage: non-AI-Hub
  caches would have to be copied EU->KR (tens of GB; transfer speed UNKNOWN), AI-Hub-derived data must stay in KR.
