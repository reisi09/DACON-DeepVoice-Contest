# R4: data acquisition and submission strategy (2026-09-27, KST)

Read-only. Everything written here lives in `research/top15/work/r4/`:
- `aihub_docs/`: AI-Hub view.do HTML pages, file trees (`api.aihub.or.kr/info/<key>.do`), extracted text, and `usagepolicy.*`. All fetched 2026-09-27 at about 13:52-14:10 KST. Documentation only: no login, no download, no terms acceptance.
- `long_bag_check.py`, `long_bag_check_bygen.py` with outputs `long_bag_{test,val}.json`, `long_bag_{test,val}_bygen.json` and `long_bag_seeds.txt`. CPU only, 5 s per run, 0.36 GB RSS.

## 1. Measured gaps (inputs to candidate ranking)

Held-out test, DB heads (`_keep_best_20260926/train_duration_balanced/test_predictions_decision.csv` joined to `bank_test.csv`), inline CPU check:

| Slice | Voice EER | n |
|---|---|---|
| All voice-present rows | 0.1107 (threshold 0.1913) | 5,139 |
| Rows with music (songs, mixes) | 0.2060 | 1,544 |
| Rows without music | 0.0443 | 3,595 |

Voice error cells:

| Cell | Label | Errors / rows | Error rate |
|---|---|---|---|
| haim song vocals | fake | 174 / 438 | 40% miss |
| fleurs_ko | real | 149 / 1,500 | 9.9% |
| constructed_enhanced | real | 70 / 276 | 25% |
| musan_music vocals | real | 57 / 187 | 30% |

FILE EER is 0.1094. Across the 15 largest error cells there are 703 errors:

| Group | Errors | Breakdown |
|---|---|---|
| Real music | 323 | musan_music 130, constructed_song 77, openmic 70, constructed_mix-real 34, haim-real 12 |
| AI music | 214 | haim 115, constructed_mix-fake 55, constructed_song-fake 21, echoes 13, generated_music 10 |
| Speech | 166 | mlaad 71, fleurs 41, enhanced 34, vocalset 19, speechfake 1 |

So about 76% of held-out FILE errors are music-domain.

Caveat, from lanes 2 and 3: P1 trained heads on exactly these held-out rows and moved LB ADS by only +0.0017. The held-out error mass is therefore not proven to be the LB error mass.

Duration coverage, from bank rows joined to cache_v4 `samples`:

| Split / route | Label | ≥15 s share | Rows ≥15 s | Rows 30-60 s |
|---|---|---|---|---|
| Train voice | fake | 1.3% | 201 | 4 |
| Train voice | real | 6.5% | 395 | 172 |
| Held-out voice | both | | | 0 |

Only 77 of the ≥15 s real voice rows in train are Korean. So the official 4-60 s range is unmeasured for voice.

Virtual long-bag check (`long_bag_check*.py`). Long files are built by concatenating cached DF-Arena window embeddings of rows with the same source (or the same source and generator) and the same label. Scoring uses the DB heads with window max and seed mean, as in `full_bank_evaluate.py:58`.

| Split | Bag design | Natural EER | k=1 | k=15 | Uniform 1-15 windows, 5-6 seeds |
|---|---|---|---|---|---|
| Held-out test | same source+label | 0.111 | 0.16 | 0.07-0.08 | 0.103-0.113 |
| Held-out test | same generator | 0.111 | 0.16 | 0.07-0.08 | 0.108-0.129 |
| Val | | 0.051 | | | 0.041 / 0.048 |

Per-length EER falls with length. Mixed lengths add a calibration penalty of about +0.005 to +0.015. Real-bag mean scores rise from 0.035 to 0.22 across k.

Verdict: at official lengths voice EER changes by only about ±0.01, roughly ±0.002 ADS direct. Long-duration voice data is not a large lever. This rests on an approximation: windows from different rows are treated as one file.

## 2. AI-Hub candidates (documentation inspected 2026-09-27 unless noted)

Policy page (`aihub_docs/usagepolicy.txt`, https://www.aihub.or.kr/intrcn/guid/usagepolicy.do, accessed 2026-09-27) says:
- Data may not be shown, provided, transferred, rented or sold to any legal entity, group or individual not approved by the builders and NIA.
- Export abroad needs a separate agreement.
- Training use only.
- Attribution is required.

The competition's hosts are 행정안전부 and 한국지능정보사회진흥원 (NIA); 국립과학수사연구원 runs it (`lane1/official_text.txt` lines 48-50). DACON 417844 says "AI HUB 데이터는 사용 가능합니다". DACON 417280-1 says verification submission differs from redistribution, but data whose license restricts third-party provision cannot be used. Round-2 delivery of AI-Hub files therefore stays UNRESOLVED (it needs NIA approval or a DACON ruling). Processing is local-only, per the project's own policy (`repository/data/aihub_usage/README.md`): VESSL is abroad.

| key | title | modality / format | size unit (smallest source part) | speakers / sources | label semantics (proposed) | gap addressed | stage |
|---|---|---|---|---|---|---|---|
| 71314 | 방송콘텐츠 대화체 음성인식 | wav, TV/radio (KBS, KBSN, EBS, TBS, OBS), 7,002.7 h; env tags: 배경음악/잡음 × 적음/보통/많음 (BGM in about 85% of 13,009 programs) | VS per genre 4-10 GB (VS_예술 4 GB) | 61,209 speaker appearances, 2-11+ speakers per program | FILE/VOICE REAL; BGM real; MUSIC_PRESENT only at program level, so mask unless verified | real speech + real music, long conversational real | documentation inspected |
| 71544 | 음악 유사성 판별 | wav 44.1 kHz 16-bit stereo, 20,000 crowd-sourced songs: ballad 30%, dance 20%, trot 20%, hiphop/RnB/rock 10% each; originals + covers | VS.zip 23 GB (single unit) | crowd-sourced composers (count UNVERIFIED) | MUSIC REAL; vocal presence UNVERIFIED | real K-genre music false positives | documentation inspected |
| 473 | 다음색 가이드보컬 | wav, 4,000 songs, 92 singers in 16 timbre/age/genre groups | VS1 3 GB | 92 singers | VOICE REAL (singing); a cappella vs backed UNVERIFIED | real sung vocals (musan vocals 30% FP) | documentation inspected |
| 465 | 다화자 가창 | wav, 4,681 recordings, 18 pro singers, ballad/R&B, rock/pop, trot | Validation 5.1 GB, local | 18 singers | VOICE REAL (singing), no music | real Korean singing | downloaded; S2-A features 256/55 rows; not admitted |
| 71349 | 감성 및 발화스타일 동시 고려 음성합성 | wav 44.1 kHz mono studio, 1,012 h, 7 styles × 4 emotions; about 89 voice actors + 16 laypeople; 52,107 downloads | 353 VS parts, 10-200 MB each | about 105 | VOICE REAL, studio expressive; hard negatives for Korean TTS | expressive studio Korean reals (unmeasured) | documentation inspected |
| 71557 | 뉴스 대본 및 앵커 음성 | wav 44.1 kHz, 1,132 h, 381,456 utterances (mean 10.7 s) | VS.zip 25 GB | 86 anchors | VOICE REAL | clean Korean read speech (weak gap) | documentation inspected |
| 568 | 소음 환경 음성인식 | wav, 8,070 h in 11 noise classes; utterances <20 s 42%, 20-40 s 34%, ≥40 s 24% | VS_02 교통수단 11 GB | 20-50s ages (count UNVERIFIED) | VOICE REAL + natural noise; music masked | noisy / long real speech | documentation inspected |
| 71417 | 극한 소음 음성인식 | wav, 2,083 h, 438 speakers; includes 공연장 (concert/orchestra/국악) noise | VS 2-28 GB (수상운송 2 GB) | 438 | VOICE REAL; music masked | extreme-noise real | documentation inspected |
| 71768 | 119 신고접수 음성 | wav, 3,064 h / 158,973 calls (mean about 69 s), mobile channel | VS parts 0.01-0.9 GB | callers + operators | VOICE REAL, phone channel; sensitive content | phone channel (refuted by S2a) | documentation inspected, not recommended |
| 71767 / 71470 / 644 / 123 / 87 / 100 / 571 | loops, 국악, sounds, Kspon, calls | | local | | | | downloaded; S2 features partial; 571 admitted in S2a (LB -0.0012 ADS) |

Overlap risks (all UNVERIFIED):
- 71314 with 463 (the same broadcasters).
- 465 with 473 (singers).
- 71349 with 466 (voice actors).
- Inside each set, split by program, singer or speaker.
- None of these datasets is in the data v4 bank.

## 3. Non-AI-Hub FAKE-diversity status

In the bank:
- MLAAD: 13 large recent systems for ko plus about 70 small ones.
- SpeechFake MD ko: CosyVoice, EdgeTTS, MeloTTS, OpenVoice, SeedVC.
- ASVspoof 2019 LA and 2021 LA/DF.
- SONICS, FakeMusicCaps (5).
- Echoes: 12 generators, with the generator field collapsed.
- HAIM: acestep, musicgen, Lyria-3 Pro, fx, C1/C2.
- generated_music: ACE-Step, 233 ko.

On disk but not in the bank:

| Source | Contents | License / rights | Status |
|---|---|---|---|
| Music Arena (`lb_collect_20260924/payload/music-arena-v1`) | 1,000 files, 13 generators: acestep-1.5, elevenlabs-music-v1, riffusion-fuzz-1.0/1.1, sonauto-v2.2/v3, magenta-rt, musicgen, 2 anonymous previews | CC BY 4.0; rights review closed "training_allowed true" (`lb_admission_20260924/rights_decisions.json`) | S2-B features: dev complete, train partial |
| Jamendo | 1,000 tracks | CC BY/BY-SA; rights review says "training_allowed false: jamendo_tdm_terms_unresolved". The bank already holds 400 JamendoMaxCaps rows and HAIM B1 (from MTG-Jamendo) | |
| MLAAD-long | 400 English files, 80 systems incl. ElevenLabs v3, Gemini-3.1-Flash-TTS, VibeVoice; 15-58 s | CC BY-NC; the 09-24 review excluded new MLAAD ("research_scope_unresolved") | |
| LibriSpeech-long | 400 files | CC BY 4.0 | |
| mlaad_ko | 8,170 unused Korean fakes from the same 12 generators (no new family) | | |
| Zeroth parquet | about 19k unused utterances | | |
| PartialSpoof | train/dev tar.gz 2 GB each | README: CC BY 4.0 + ODC-By | not extracted |
| ASVspoof5 | incomplete .part files | ODC-By, 142 GB total | |
| CtrSVDD | 5 GB | ND | excluded |

Not on disk, eligible per `DATA_PLAN_236749_2026-09-19.md` §5 (checked 09-19, not re-verified): HAIM B6 (Korean vocals from K-pop lyrics, ACE-Step) and B2-B4/B7-B9 (CC BY-NC 4.0), SpeechFake full (license ⚠).

Excluded in the same plan: CtrSVDD and Codecfake (ND, plus a label conflict), SingFake and WildSVDD (links only), DEEP-VOICE, FoR.

## 4. VESSL cost for the missing 26% of backbone features
- 10,769 rows are missing (42,039 − 31,270).
- The earlier run extracted 31,270 in a 150-minute deadline window, i.e. 3.47 files/s. The brief says 3.8.
- Billed cost was $7.86 = 17.71 − 9.85 (memory file), i.e. 2.64 H100-h at $2.98/h, so about 8 min of overhead.
- For the rest: 47-52 min of extraction plus about 8 min overhead is about 1.0 h, so $3.0 (about $3.6 with a 20% margin). About $6.2 would be left.
- Expected benefit is small:
  - Lane 4: music heads are flat beyond 50% of the data; voice might gain.
  - Lane 3: the subset is representative.
  - The S3 AD heads were trained on 22,101 rows, against about 29.7k available.
- Recommendation: only piggyback it on a VESSL session that is justified by something else, and only after S3's LB result confirms the backbone line.

## 5. Slot strategy facts
- Limit is 3 per day. Deadline is 2026-09-29 09:59:59 KST. Best-ever score counts; the private LB equals the public one.
- Errors after script.py starts count against the limit.
- DACON 417856-2 allows submissions made to check the environment.
- DACON 417702-4 discourages LB probing (inferring subgroup statistics and feeding them into constants).
- P1 ran 17m32s on L4 (memory file: DurBal). The S3 estimate is 24-35 min (lane 4).
- The queue was reported "congested" at 01:40 on 09-27 (memory file).
- The reset time is UNVERIFIED (FAQ 93 body inaccessible, lane 1).
