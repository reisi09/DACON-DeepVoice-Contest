# Round 24 — more high-quality data: strengthen our offline grader and help training (2026-09-28 18:15 KST)

Both reviewers get the same file. Web search is enabled for this round: verify every dataset/model claim (license, access,
size, language, generators) with a link. Answer in Korean, VERIFIED / INFERRED / UNKNOWN.

## 1. The user's request
"Is there more high-quality data we can use?" Two goals, discuss both:
(G1) strengthen our offline grader: evaluation sets that predict the DACON leaderboard better than E1/E2/E3;
(G2) training data that raises the LB score.

## 2. What the real grader contains (official_text.txt in this folder)
- Theme: "신종 보이스피싱 범죄 대응"; background: "다양한 생성기술과 실제 통신환경에서 발생하는 음성 변형".
- 1,200 files, 4 s - 60 s, standardized to 16 kHz, mono and stereo, MP3/WAV/FLAC, "일부 샘플에는 전화채널 오디오".
- Types: speech (speech or vocals only), music (no vocals), mixed (speech and music "동시에 또는 순차적으로"; songs are mixed).
- FAKE if any component is AI-generated; real audio with enhancement / denoise / loudness only stays REAL.
- Top 15 on the private LB go to the 2nd round (reports + all training data files, including generated audio); top 7 win.

## 3. LB history (ADS) and what it taught
P1 0.7700 | S2a 0.7689 (P1 + 800 AI-Hub REAL phone calls: -0.0012) | S3 0.7853 (new encoders, held-out +0.0083 -> LB +0.0153) |
S6 0.8220 (heads + label completion, held-out ~+0.027 -> LB +0.0367) | S7 0.8374 best (DB +0.0104 -> LB +0.0154) |
S8 0.8340 (raw-mix SSLAM + FILE rule change: DB +0.0043 -> LB -0.0034). DB held-out predicted the LB direction 3 of 4 times.
DB held-out ADS ~0.909 vs LB 0.837: a large shift. Tomorrow: s7_new -> W_newB -> slot 3 (see SUBMISSION_PLAN_20260929.md).

## 4. Current training data (DATA_INVENTORY_20260928.txt has per-source counts and labels)
- S7 bank: 35,589 train/val + 6,450 test rows, 44 sources (MLAAD, ASVspoof 2019/2021 LA/DF, SpeechFake, Zeroth, FLEURS-ko,
  OpenMIC, FakeMusicCaps, SONICS, HAIM, Echoes, MUSAN, VocalSet, CSD, FMA, JamendoMaxCaps, LibriSpeech, generated_music,
  constructed mixes / splices / songs).
- Correction of an earlier claim: Korean fake SPEECH is NOT zero. Korean voice rows: fake MLAAD-ko 3,830, SpeechFake-ko 3,000,
  generated_music-ko 222; real Zeroth 3,000, FLEURS-ko 1,500, SpeechFake-ko 600, CSD 100. Unknown: which generator families these
  Korean fakes cover, and whether any fake passed through a phone channel.
- New data today (new_data_summary.csv): AT-ADD T2 speech/singing/music 34,000 (fake 21,500), HAIM B3/B4/B6 2,400, AI-OpenBMAT 600
  (+400 extra), GTSinger-ko 1,640 (real), SD-RR 496, D1 remixes 636.
- Constructed sequential speech->music files: constructed_splice 121 rows only. Phone channel: only the 800 REAL AI-Hub calls in S2a
  (not in S7). No codec/phone augmentation of FAKE audio anywhere (features are cached from frozen encoders; augmentation means new
  rows and new feature dumps).

## 5. Constraints
- Rules: public resource that anyone can access, at least non-profit use allowed (DACON allowed gated AI-Hub, NC models, CC BY-NC-SA
  data); all training files incl. generated audio go to DACON in round 2; cite sources. No private-test use, file-independent inference.
- AI-Hub audio/features never leave the Mac (no VESSL EU/US, no cloud LLM). Mac is slow (MPS dump ~1-5 rows/min).
- Normal models only; no LB probing; selection on offline data only.
- Time: now 18:15 KST. Slot 3 upload should start by ~07:30 on 09-29; a replacement package needs P1 retrain (two stages, 3 seeds)
  + packaging + CPU/parity/CUDA checks. Feature dump speed ~18 audio-s/s per H100 (2 workers). Budget: VESSL $18.79
  (H100 x1 $2.98/h); every GPU use needs the user's approval. Package is 9.745/10 GB: no new encoders.

## 6. Questions
(a) G2 external data: a ranked list (max 8) of datasets not yet used that fill a real gap (Korean or multilingual fake speech from
    NEW generator families, voice conversion / cloning, partial fakes, phone/codec channel, sequential mixes, fake singing, commercial
    AI music, real Korean songs / broadcast). For each: link, license and access (verified), size to download, what gap, risk.
(b) G2 self-made data: recipes we can make from data we already hold (e.g. codec/phone augmentation of BOTH real and fake Korean
    speech, sequential speech->music splices, partial fakes), and open Korean-capable TTS/VC models with licenses (verified) if
    generation is worth it. Which of these is most likely to move the LB, and why?
(c) G1 grader: which evaluation set would predict the LB better, how to build it (sources, held-out generator families, channel
    conditions, sizes), and how to check its predictiveness with the LB history we have (6 LB points above).
(d) Tonight vs later: is anything worth doing before 07:30 given the time, the $ and the rule that a new package must pass a
    pre-registered offline test? Give GPU-hours, the adoption rule, and the stop condition. If nothing is worth it tonight, say so.
(e) What not to do (known traps: label errors, lineage leakage, generator overlap between train and test, AI-Hub export).
