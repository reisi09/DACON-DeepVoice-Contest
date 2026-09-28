# DACON 236749 데이터 불균형 진단: 독립 검증 보고서 (claude_fresh_verifier_r1)

- 작성: 2026-09-28 KST. 검토자는 이전 토론에 참여하지 않았고 codex_round*/PROMPT_ROUND*/fresh_r* 파일도 읽지 않았다.
- 근거 자료: `FULL_BRIEFING_20260928.md`, `data/bank_trainval.csv` + `data/bank_test.csv`(42,039행), `code/full_bank_batches.py`, `split.py`, `label_rules.py`, `sources.py`, `train_backbone_heads.py`(가중치가 쓰이는 경로 확인), `file_ensemble.py`(FILE 가중치 규칙), `results/diag_s8.json`, `s8_heldout.json`, `rt_s6enc_report.json`, `unseen_report.json`, `s8_run_identity.json`, `rt_check.json`. `experiment-ledger.csv`는 키워드 검색에만 썼다.
- 이 자료에는 **오디오도 행 단위 모델 예측도 없다.** 그래서 모델 점수에 관한 주장은 모두 INFERRED 또는 ESTIMATE이다.
- 표기 규칙
  - **VERIFIED**: CSV, 코드, 결과 JSON으로 직접 재계산하거나 확인했다.
  - **INFERRED**: 검증된 사실에서 논리적으로 끌어냈다.
  - **ESTIMATE**: 가정을 두고 낸 수치이다.
  - **UNKNOWN**: 이 자료로는 판단할 수 없다.
- 가중치 분석 조건: `loss_weights` 공식을 그대로 재현했다(cell = (label, source, generator)). items는 head label이 있는 행으로 잡았다. voice head는 VP=1이고 rule L 적용 후 VF가 알려진 행, music head는 MP=1이고 MF가 알려진 행이다. **`duration_balance`(클래스별 log-duration 히스토그램 정합, clip 10)는 CSV에 길이 정보가 없어 반영하지 못했다.** silent-stem 제외도 반영하지 않았다. 그래서 가중치에서 파생한 수치는 모두 ESTIMATE이다. 산술 자체는 VERIFIED이다.

---

## 0. 결론 요약

1. **브리핑 §6의 수치는 대부분 재현된다 (VERIFIED).** 불일치는 작고 주로 정의 차이이다(§2.8 목록). 그중 의미가 있는 것은 다음과 같다.
   - 조합표는 rule L 적용 **전** 원라벨 기준이다. 학습에는 L이 적용되므로 train의 fakeV+fakeM은 1,287이 아니라 2,354~2,688이다.
   - L 적용 후에도 보컬 진위가 미상인 행은 "~520"이 아니라 **539**이다. Echoes ATA가 43이 아니라 66이다.
   - "real music top-10 그룹 19%"는 held-out 그룹 9개가 끌어올린 수치이다. **train만 보면 4.5%**이다.
   - "held-out 미관측 생성기 1,135행"은 R2 동결 평가 부분집합의 수치이다. bank 전체 held-out에는 **1,550행**이 있다.
   - "S8이 모든 DB 실음악 소스에서 FPR 낮음"은 정확하지 않다. **haim은 0.089로 동일**하다.
2. **브리핑이 놓친 핵심은 원시 행수가 아니라 현재 가중치가 만든 숨은 불균형이다** (산술은 VERIFIED, 수치는 ESTIMATE).
   - 단일 계보 그룹 `artist:FMA:1041`(레거시 `existing:*` 행)이 DB train 가중치를 크게 차지한다.
     - music FAKE 클래스 가중치의 **37.6%**
     - music REAL의 **19.4%**
     - voice REAL의 **10.5%**
   - Kish 유효 계보 그룹 수는 music fake **6~7개**, music real 25개, voice real 52개이다.
   - 층(stratum)별 사전확률이 역전된다.
     - 음악 위 음성(constructed_mix)의 fake 비율은 행 기준 92.9%에서 가중 **4.3%**로 떨어진다.
     - 노래 속 보컬은 66%에서 **16%**가 된다.
     - 한국어 음성은 62%에서 **26%**가 된다.
     - 한국어 fake 음성은 fake 행의 27%이지만 가중치로는 **4.4%**이다.
   - fake voice 가중치의 **91.5%**가 MLAAD와 ASVspoof에서 온다. 두 소스는 generator 문자열이 많기 때문이다.
3. **"불균형과 coverage가 0.07 격차와 S8 음전이를 설명한다"는 가설은 부분적으로만 맞다.**
   - 원시 행수 불균형은 코드가 이미 상쇄한다. 게다가 EER은 클래스 사전확률에 불변이므로 설명력이 거의 없다.
   - 더 강한 설명은 세 가지이다.
     - (i) coverage gap: 존재하지 않는 셀이 있다. AI cover, fake 무반주 가창, 한국어 실제 노래, 상업 음원이 그렇다.
     - (ii) **DB held-out이 LB를 대표하지 못한다.** DB 실음악 평가행의 39%가 우리가 만든 constructed 합성 믹스이다. S8의 음악 FP 감소 중 약 **78%**가 이 합성 소스에서 나왔다. S8의 미관측 생성기 이득은 HAIM의 두 생성기(Lyria-3 Pro, fx_encoder_style_transfer)에 집중되어 있다.
     - (iii) 잡음(약 2σ)과 FILE 가중치 변경(약 -0.001)이다.
   - 가중치 병리는 기여 요인일 수 있다. 그러나 **한 번도 절제(ablation)된 적이 없어** 크기는 UNKNOWN이다.
4. **처방 (기대 LB 이득/시간 순, 모두 ESTIMATE)**
   - ① 층화 재가중: 층 안에서 클래스를 균형 맞추고, sqrt 셀 가중을 쓰고, 행 상한 5×와 계보 그룹 상한 2%를 둔다. 특징 재추출이 없고 약 3~3.5 h 걸린다.
   - ② 캐시된 stem으로 불일치(discordant) 노래와 AI-cover 대리 리믹스를 만든다. 같은 공정의 일치 리믹스를 대조군으로 넣는다. 약 5 h 걸리는 고분산 티켓이다.
   - ③ 기존 구성요소를 "자연 소스 전용" 지표로 재선택한다. 약 2.5 h 걸리는 저분산 대비책이다.
   - AI-Hub는 한국 내 처리 제약 때문에 이번 마감(09-29 10:00) 안에는 비현실적이다.
   - 최종 점수는 최고 제출 점수이므로, 남은 3장은 서로 다른 가설을 시험하는 고분산 조합으로 쓰는 편이 유리하다.
5. **S8을 걸렀을 검증 방법**은 네 가지 조합이다.
   - 자연 소스 전용 ADS 게이트
   - 소스 잭나이프(이득이 한 소스에 몰렸는지 확인)
   - 계보 그룹 단위 부트스트랩
   - 층 구성을 바꿔 가중한 강건성 평가
   - 앞의 두 가지는 S8을 "비강건"으로 표시했을 가능성이 높다(INFERRED). 행 단위 예측으로 확인해야 한다.

---

## 1. 정의와 방법

- **라벨.** `X_mask=1`이면 라벨이 알려진 것이다. 이 보고서의 VP/MP/VF/MF/FF는 알려진 값(0/1)이고 미상은 `?`로 쓴다.
- **rule L 재현.** `label_rules.completes`의 조건을 그대로 썼다.
  - FakeMusicCaps는 `fmc_caption_vocal.json`이 폴더에 없어 판정할 수 없다. VP=1인 FMC 400행은 후보로만 둔다.
  - 비-FMC 완성 수는 **1,249행 (VERIFIED)**이다. 따라서 브리핑의 1,601은 FMC 완성 352행을 포함한다는 뜻이다. 이 352행은 UNKNOWN이지만 다른 수치와는 정합한다.
- **내가 정한 보컬 범주** (VP=1 행, 소스 기반으로 분류)

| 범주 | 소스 |
|---|---|
| A 음성(speech) | mlaad, asvspoof2021_la, asvspoof2021_df, existing:asvspoof2019_la, speechfake, zeroth, fleurs_ko, musan_speech, existing:librispeech_test-clean, constructed_enhanced(zeroth/fleurs + musan_noise), constructed_splice(speechfake 실제+SeedVC 접합) |
| B 무반주 가창 | vocalset, csd |
| C 음악 위 음성(합성) | constructed_mix (부모: mlaad/speechfake/fleurs_ko/asvspoof + openmic/haim/echoes/musan/sonics…) |
| D 음악 위 무반주 가창(합성) | constructed_song (vocalset + openmic/haim/echoes/…, held-out 전용) |
| E 음악 트랙 속 보컬 | 나머지 music route 소스 전부 (openmic, FMA, Jamendo, haim, SONICS, Echoes, FMC, generated_music …) |

---

## 2. Task 1: 분포 재계산과 브리핑 대조

### 2.1 split 규모 (VERIFIED)

| split | 행 | 소스 | 계보 그룹 | FILE fake / real / 미상 |
|---|---|---|---|---|
| train | 30,658 | 44 | 13,304 | 21,492 / 9,147 / 19 |
| val | 4,931 | 37 | 2,818 | 3,385 / 1,542 / 4 |
| held-out | 6,450 | 18 | 2,730 | 2,624 / 3,820 / 6 |

- `group_id == component_id`는 모든 행에서 성립한다. split을 넘나드는 그룹은 0개이다 (VERIFIED).
- 언어 분포는 unknown 18,117 / ko 12,263 / en 9,196 / 기타 **2,463**이다 (VERIFIED). 브리핑은 기타를 "~2,000"으로 적었다.
- **held-out 전용 소스** (VERIFIED)
  - fleurs_ko 1,500
  - musan_music 645
  - constructed_song 400
  - constructed_enhanced 276/300
- **R2 동결 평가 집합.** file 4,819행(fake 1,928, real 2,891)이다. `diag_s8.json`의 소스별 n을 보면 소스마다 약 75%씩 균일하게 부분추출한 것으로 보인다(INFERRED, ledger의 "74% cache rows, coverage uniform"과 일치). 평가 real 2,891 중 **fleurs_ko가 1,143행(39.5%)**이다 (VERIFIED).

### 2.2 성분 조합 (train, VERIFIED)

| 조합 | 원라벨(브리핑과 동일) | rule L 후 (FMC 미완성~전부완성) |
|---|---|---|
| fakeV only | 15,243 | 15,243 |
| realV only | 6,052 | 6,052 |
| fakeM only | 2,234 | 2,234 |
| "?V"+fakeM | 2,082 = VF미상 1,727 + VP미상 355 | 660~326 + 355 |
| realV+realM | 1,395 | 1,395 |
| realM only | 1,362 | 1,362 |
| fakeV+fakeM | 1,287 | **2,354~2,688** |
| fakeV+realM | 641 (전부 constructed_mix의 TTS 음성) | 641 |
| "?V"+realM | 357 = 12 + 345 | 동일 |
| realV+fakeM | **5** (held-out 107, val 5) | 5 |

- 브리핑의 수치는 정확하다. 다만 **원라벨 기준**이다.
- "?V"에는 성격이 다른 두 집합이 섞여 있다. "보컬 존재 미상 700행"과 "보컬 진위 미상 1,739행"이다.

### 2.3 보컬 존재 행의 범주 (P1 = train+held-out / val)

| 범주 | real | fake | fake(L, 비FMC) | FMC 후보 | 진위 미상 | 판정 |
|---|---|---|---|---|---|---|
| A 음성 | 7,405 / 916 | 16,551 / 2,197 | - | - | - | VERIFIED(일치) |
| C 음악 위 음성 | 199 / 20 | 1,149 / 132 | - | - | - | 브리핑 193/19. 6행/1행은 경계 정의 차이 |
| B 무반주 가창 | 934 / 66 | **0** | - | - | - | VERIFIED(일치) |
| D 음악 위 무반주 가창 | 400 / 0 | 0 | - | - | - | 브리핑은 "singing in songs"에 포함 |
| E 노래 속 보컬 | 1,529 / 236 | 1,595 / 218 | 1,067 / 182 | 334 / 66 | 425 / 66 | D+E real = 1,929+6 = 1,935 (브리핑과 일치) |

- 브리핑의 "fake by rule L 1,364 / 237"은 비-FMC 1,067 / 182에 FMC 297 / 55를 더한 값으로 해석하면 정합한다. FMC 부분은 UNKNOWN이다.
- 브리핑의 "unknown 462 / 77"은 425 + 37과 66 + 11로 일치한다 (INFERRED).
- "fake 가창은 완전 생성 노래에만 존재한다"는 서술은 대체로 맞다. 예외가 하나 있다. held-out의 `haim / fx_encoder_style_transfer` 216행은 VF=1이지만 실제 음원을 AI로 처리한 것이다 (VERIFIED, 사소함).

### 2.4 한국어 커버리지 (VERIFIED)

| 항목 | train | val | held-out |
|---|---|---|---|
| 실제 음성 | zeroth 2,465, speechfake bonafide(CommonVoice) 493 | 436, 92 | fleurs_ko 1,500, zeroth 99, bonafide 15 |
| fake 음성 | mlaad-ko 2,889, speechfake 2,009 | 524, 315 | mlaad-ko 417, speechfake 676(OpenVoice 600 포함) |
| 실제 무반주 가창 | csd 100 (계보 그룹 1개) | 0 | 0 |
| 실제 노래(반주 포함) | **0** | 0 | 0 |
| fake 노래 | generated_music(acestep) 177 | 45 | 0 (VF 미상 11행 별도) |

- 브리핑 수치(real 5,100 / fake 6,830 / 무반주 100 / 한국어 노래 real 0, fake 222)는 모두 재현된다.
- **추가 관찰**
  - fleurs_ko는 held-out에만 있다. 그래서 DB 모델(train만 학습)이 본 한국어 실제 음성은 zeroth(114개 화자 그룹)와 CommonVoice 493행이 전부이다.
  - speechfake EdgeTTS 600행은 **2개 화자 그룹**에 몰려 있다.
- music 소스의 language가 비어 있어서 "한국어 실제 노래 0"은 메타데이터 기준의 판단이다 (INFERRED).

### 2.5 라벨 순도 (소스가 99% 이상 한 클래스인 비율)

| 라벨 | 전체 행 기준 (브리핑 정의) | train 행 기준 | **DB train 가중치 기준** |
|---|---|---|---|
| VOICE_FAKE | 0.565 (VERIFIED, 일치) | 0.536 | real 0.706 / fake 0.612 |
| MUSIC_FAKE | 0.748 (VERIFIED, 일치) | 0.836 | real 0.444 / **fake 0.868** |
| FILE_FAKE | 0.618 (VERIFIED, 일치) | 0.605 | - |

- 브리핑의 정의는 42,039행 전체, 원라벨, 두 클래스 합산이다. 이 정의로 세 수치 모두 정확히 재현된다.
- 학습에 실제로 걸리는 가중치 기준으로 보면 fake music 가중치의 87%가 한 클래스만 가진 소스에서 온다.
- "Fake voice = 71.3% MLAAD+ASVspoof"는 **P1 행, 원라벨** 기준일 때만 재현된다(13,763/19,295). **가중치 기준으로는 91.5%**이다.

### 2.6 계보 그룹 집중

| 집합 | 행 | 그룹 | top-10 비중 | 비고 |
|---|---|---|---|---|
| real music(전 split) | 6,266 | 3,605 | 19.1% (VERIFIED) | top-10 중 9개가 held-out 그룹이다. MUSAN 작곡가 6명 외에 `commonvoice_ko` 화자 2명과 Echoes ATA item 1개가 들어 있다. constructed 믹스가 부모 계보를 합쳐 만든 거대 성분이다. 브리핑의 "MUSAN 작곡가 몇 명과 FMA 아티스트 1명"이라는 설명은 부정확하다. |
| real music(train) | 3,755 | 2,834 | **4.5%** | 최대 그룹은 artist:FMA:1041(140행) |
| real singing(B+D+E, 전 split) | 3,165 | 1,590 | 32.1% | 브리핑 3,139 / 1,588 / 32%와 정의 차이 26행 |
| real singing(train) | 1,890 | 1,301 | 31.4% | csd 1명(200행)과 vocalset 4개 그룹 |

- **가중치 기준 집중** (DB train, ESTIMATE): `artist:FMA:1041` 한 그룹이 아래를 차지한다.
  - music fake 가중치 **37.6%**(P1은 34.9%)
  - music real 19.4%
  - voice real 10.5%
  - 이 그룹은 `existing:Echoes/*`(sources.py가 말하는 "8 s legacy crops"), `existing:suno`, `existing:udio`, `existing:constructed_music`, `existing:FMA`로 이루어져 있고 행 수로는 575행뿐이다.

### 2.7 기타 검증

- **코덱.** real/fake 간 코덱 분포가 동일하다 (VERIFIED). pcm_wav 0.295/0.299, mp3_128 0.122/0.122이다. 유형별 총변동거리는 voice 0.013, music 0.050, mixed 0.027이다.
- **보컬 진위 미상 2,140행** (VERIFIED)
  - rule L 비-FMC 완성 1,249행
  - FMC 후보 400행
  - 완성 불가 491행: haim 401(event interval 편집 385 + 실음악 위 미상 보컬 16), Echoes ATA **66**(songgen 32, musicgen 11, audioldm 10, diffrhythm 9, producer 4), existing:constructed_music 24
  - 브리핑 1,601이 맞다면 남는 행은 **539**이고, FMC 미완성은 48이다.
- **미관측 생성기** (VERIFIED)
  - bank held-out 기준 1,550행: haim Lyria-3 Pro 300, fx_encoder_style_transfer 300, mlaad VoxCPM2 300 + VoxCPM-0.5B 25 + VoxCPM-1.5 25, speechfake OpenVoice 600
  - 동결 평가 집합 기준 1,135행
  - **음악 쪽 미관측 생성기는 HAIM의 2개뿐이다.**
- **교차 소스 중복** (VERIFIED, 영향 미미)
  - `existing:SONICS`와 `sonics`는 같은 곡 키 29개(곡 번호 기준 65개)를 공유하지만 계보로 연결되지 않았다. 이 중 3개(곡 번호 기준 8개)가 train/val에 걸쳐 있다.
  - openmic과 FMA는 동일 트랙 6개를 공유하고 그중 4개가 연결되지 않았다.
- **라벨 정책 노출** (VERIFIED)
  - HAIM `sonicmaster`(실제 음원의 AI 마스터링)는 MUSIC_FAKE=0이다.
  - `fx_encoder_style_transfer`(AI 스타일 변환)는 fake이다.
  - DB train에서 haim의 real cell 3개(sonicmaster 200행, crossfade 23행, concat 30행)가 **real-music 가중치의 33%**를 차지한다. 주최 측이 AI 처리 음원을 어떻게 라벨링하는지는 UNKNOWN이다.
- "VOICE_PRESENT=1인데 분리 보컬이 무음인 행이 없다"는 주장은 CSV로 검증할 수 없다 (UNKNOWN).

### 2.8 브리핑 불일치 목록

| # | 브리핑 | 재계산 | 중요도 |
|---|---|---|---|
| 1 | 조합표(암묵적으로 학습 기준) | 원라벨 기준. L 후 train fakeV+fakeM 2,354~2,688 | 중 |
| 2 | "?V" | VP 미상(존재 자체 미상) 700행과 VF 미상 1,739행이 섞임 | 하 |
| 3 | 잔여 미상 ~520 (Echoes ATA 43, FMC ~30) | 539 (Echoes ATA 66, haim 401 = 385+16, FMC 48 추정) | 하 |
| 4 | real music top-10 = 19% | 전 split 기준. train은 4.5%. 설명(작곡가/FMA)도 부정확 | 중 |
| 5 | 미관측 생성기 1,135행 | 동결 부분집합 기준. bank held-out은 1,550 | 하 |
| 6 | 71.3% MLAAD+ASVspoof | P1 원라벨 행 기준. 가중치로는 91.5% | **상** |
| 7 | S8 FPR이 모든 DB 실음악 소스에서 낮음 | haim은 동일(0.089) | 하 |
| 8 | Real music 6,266과 소스 목록(openmic 2,404 …) | 총계는 전 split, 소스 목록은 P1(합 5,580). 기준이 섞임 | 하 |
| 9 | 기타 언어 ~2,000 | 2,463 | 하 |
| 10 | "a 5-row cell weighs as much as an 8,000-row cell" | 사실(VERIFIED). 그러나 그 결과(§3)는 보고되지 않음 | **상** |

---

## 3. Task 2: 어떤 불균형이 ADS에 중요한가

### 3.1 ADS 손실의 구성 (DB, S7, VERIFIED 산술)

`1-ADS = 0.5·FileEER + 0.2·VoiceEER + 0.3·MusicEER = 0.0377(41%) + 0.0122(13%) + 0.0409(45%) = 0.0908`

- DB에서는 **음악 head가 직접(0.3) 그리고 FILE을 통해 간접적으로 손실의 대부분**을 만든다.
- FILE EER은 모든 파일 유형을 **하나의 임계값**으로 묶어서 계산한다. 따라서 head 간·층 간 보정(calibration)의 차이가 그대로 손실이 된다.
- LB의 성분별 EER은 UNKNOWN이다. S7 LB의 가중 오류합은 0.1626이다. 예를 들어 FE 0.13, VE 0.10, ME 0.25 정도면 맞아떨어진다 (ESTIMATE, 하나의 가능한 분해일 뿐이다).

### 3.2 현재 가중치가 실제로 만드는 분포 (DB train, duration_balance 전, ESTIMATE)

**클래스 안 소스 비중 (행 비중과 가중 비중)**

| head / 클래스 | 셀 수 | 주요 가중 비중 |
|---|---|---|
| voice fake | 379 | mlaad 205셀 **54.1%**(행 36.7%), asvspoof2021_df 29.0%, 2019_la 5.0%, 2021_la 3.4%, speechfake **1.1%**(행 10.8%), fake 가창 소스 전체 26셀 **6.9%**(행 13.8%), constructed_mix 0.3%(행 4.2%) |
| voice real | 17 | 셀마다 5.9%. zeroth 2,465행 = FMA/FMA 9행 = existing:constructed_music 13행. **행당 가중치 최대/평균 49배** |
| music real | 9 | haim 3셀 33%(253행), openmic 11%(2,018행, 행 54%), FMA/FMA 11%(45행), existing:constructed_music 11%(64행, 그룹 1개) |
| music fake | 38 | FakeMusicCaps 13.2%. 레거시 existing:Echoes/*·suno·udio·constructed_music 21셀 약 55% |

**층별 가중 사전확률 P(fake | 층)**

| 층 | 행 기준 | 현재 가중 | 제안 가중(§5.2 W1) |
|---|---|---|---|
| voice A 음성, 한국어 | 0.623 | **0.262** | 0.493 |
| voice A 음성, 기타 언어 | 0.803 | 0.715 | 0.497 |
| voice B 무반주 가창 | 0 | 0 (fake 없음) | 0 (층 질량 ×0.5) |
| voice C 음악 위 음성 | 0.929 | **0.043** | 0.793 (행 상한 적용 결과) |
| voice E 노래 보컬, 기타 언어 | 0.640 | **0.159** | 0.505 |
| voice E 노래 보컬, 한국어 | 1.0 | 1.0 (real 없음) | 1.0 (층 질량 ×0.5) |
| music 반주만 | 0.621 | 0.487 | 0.498 |
| music 보컬 포함 | 0.681 | 0.616 | 0.512 |
| music constructed | 0.163 | 0.191 | 0.472 |

- 한국어 비중: 한국어는 voice real 행의 41%, fake 행의 27%이다. 그런데 가중치 기준으로는 **real 14.7%, fake 4.4%**이다.
- 해석(INFERRED)
  - 얼린 1B 인코더 위의 작은 MLP는 언어, 가창 여부, 배경음악 여부를 쉽게 식별한다.
  - 그러면 가중 BCE를 최소화하려고 층마다 logit 편향을 학습한다. 가중 사전확률로 환산하면 한국어 음성은 기타 언어 음성보다 약 -2 nat(logit -1.04 대 +0.92) "real 쪽"으로 치우친다. 음악 위 음성은 클래스 균형 prior(logit 0) 대비 약 -3.1 nat이다.
  - 층 하나만 놓고 보면 EER은 평행이동에 불변이다. 그러나 VOICE(음성+가창)와 FILE(모든 유형)은 **층을 섞어 하나의 임계값**으로 계산하므로 손해를 본다.
  - 특히 LB에 TTS 내레이션+BGM 같은 "음악 위 fake 음성"이 있으면 Fv가 낮게 나온다. Fm은 실제 음악이라 낮다. 결국 FILE이 놓친다.

**유효 표본 (Kish)**

| head / 클래스 | 행 | 현재: 유효 행 / 유효 계보 그룹 | sqrt 셀 | 층화 sqrt + 행상한 5× + 그룹상한 2% |
|---|---|---|---|---|
| voice real | 7,452 | 909 / **52** | 5,151 / 108 | 6,133 / 186 |
| voice fake | 18,572 | 7,988 / 152 | 13,221 / 174 | 12,825 / 429 |
| music real | 3,755 | 635 / **25** | 2,410 / 96 | 2,710 / 1,090 |
| music fake | 5,608 | 1,330 / **6~7** | 4,237 / 26 | 2,699 / 559 |

- 계보 그룹 기준의 유효 표본은 bootstrap이나 일반화에서 "독립 증거의 수"와 비슷한 의미를 가진다.
- music fake 클래스가 사실상 **그룹 6~7개**로 학습된다는 점은 음악 head가 LB로 일반화하는 데 가장 의심스러운 지점이다 (INFERRED).

### 3.3 불균형의 분류

| 항목 | 종류 | 현재 처리 | ADS 영향 경로 | 가중치로 고칠 수 있는가 | 중요도 |
|---|---|---|---|---|---|
| FILE 클래스 비 70/30 | 개수 | 클래스 동일가중 | EER이 사전확률에 불변 | 이미 해결 | 낮음 (VERIFIED 코드) |
| 소스·생성기 행수 편차 | 개수 | cell 동일가중 | 과보정으로 유효표본 급감 | **예** (sqrt, 상한) | 중~상 (ESTIMATE) |
| 층별 사전확률(가창, 음악 위 음성, 한국어) | 조건부 개수 | 처리 안 됨, 가중치가 역전시킴 | FILE·VOICE 풀링 EER | **예** (층 내 클래스 균형) | 중~상 (INFERRED) |
| 계보 집중(FMA:1041 37.6%) | 가중 집중 | 없음 | MUSIC(0.3) + FILE | **예** (그룹 상한) | 상 (INFERRED) |
| fake 무반주 가창(SVS/SVC) 0행 | coverage | 없음 (SingMOS 시도) | VOICE, FILE | 아니오 | U (DB에 해당 셀 0행이라 **DB로는 측정 불가**) |
| AI cover(fake 가창 + real 반주) 0행 | coverage | 없음 | VOICE, MUSIC, FILE | 아니오 (합성 대리는 가능) | U, 잠재적으로 큼 |
| real 가창 + fake 반주 train 5행 | coverage | 없음 | MUSIC, VOICE | 아니오 | U |
| 한국어 실제 노래 0, 한국어 노래는 전부 fake | coverage + shortcut | 없음 | VOICE, FILE | 아니오 (AI-Hub는 한국 내 처리만 가능) | U |
| 상업·마스터링 음원, K-pop 0 | coverage | 없음 | MUSIC, FILE | 아니오 | U (S8 가설과 연결) |
| 단일 클래스 소스(mlaad 전부 fake, zeroth·fleurs·openmic 전부 real 등) | shortcut | 코덱만 균등화 | 모두 | 부분적 | 중 (가중 기준 music fake 87%) |
| DB held-out의 층 구성(real의 40%가 fleurs_ko) | 평가 구성 | 없음 | DB 과대평가 | 평가 가중으로 가능 | 중 (ESTIMATE) |

---

## 4. Task 3: 0.07 격차와 S8 음전이 가설 검증

### 4.1 격차의 이력 (VERIFIED)

| 패키지 | DB | LB | 격차 | LB 이득 / DB 이득 |
|---|---|---|---|---|
| S3 | 0.8727 (AC 프로토콜) | 0.7853 | 0.0874 | - |
| S6 | 0.89877 | 0.8220 | 0.0768 | 1.41 (프로토콜 차이 있음) |
| S7 | 0.90921 | 0.8374 | 0.0718 | 1.48 |
| S8 | 0.91352 | 0.8340 | 0.0795 | **-0.79** |

- 격차는 S3부터 0.07~0.09로 **구조적이다.** 반복 선택 때문에 커지는 추세는 보이지 않는다 (INFERRED).
- 따라서 적응적 과적합은 격차의 주원인이 아니다.

### 4.2 S8 분해 (diag_s8.json, val TPR-95 임계값 기준 FP 수 = n × FPR)

| 실음악 소스 (평가 n) | S7 FPR → S8 | FP 변화 |
|---|---|---|
| openmic (278) | 0.068 → 0.054 | -3.9 |
| musan_music (503, 미관측 자연 소스) | 0.109 → 0.097 | -6.0 |
| haim (56) | 0.089 → 0.089 | 0 |
| **constructed_mix** (309) | 0.126 → 0.071 | **-17.0** |
| **constructed_song** (233) | 0.129 → 0.047 | **-19.1** |
| 합계 (1,379) | 147.7 → 101.7 FP | -46.0, **constructed 비중 78%** |

- FILE real FP는 105.1에서 73.6으로 줄었다. 감소분의 구성은 musan 38%, constructed 35%, fleurs·openmic·vocalset 27%이다.
- 여기서 나오는 사실들
  - 산술은 VERIFIED이다. FPR이 소수점 3자리로 반올림되어 있어 근사치이다. 두 모델의 임계값은 각자 val TPR-95 지점이다.
  - DB 실음악 평가행의 **39%(542/1,379)가 우리가 만든 합성 믹스**이다.
  - 원믹스 SSLAM 특징은 "합성 믹스의 흔적"에 특히 잘 반응했다 (INFERRED). stem 분리·재합성이나 오버레이 가공처럼 bank 고유의 신호로 보인다.
  - 미관측 생성기 부분집합에서 음악 EER은 0.1430에서 0.1169로(-0.026), 전체에서는 0.1364에서 0.1275로(-0.009) 내려갔다. 미관측 음악 fake는 HAIM Lyria-3 Pro와 fx_encoder_style_transfer뿐이다. 따라서 **음악 이득이 HAIM 두 생성기에 집중**되었다 (INFERRED. seen-gen 부분집합의 수치가 보고되지 않았다).
  - 게다가 P1 패키지는 이 두 생성기를 이미 학습에 포함한다. LB에서 이 이득이 의미가 있는지 자체가 의문이다 (INFERRED).
- **FILE 가중치**
  - P1 실행에서 val grid 이득이 0.0008로 `min_gain` 0.002보다 작아 equal이 선택되었다 (VERIFIED `s8_run_identity.json`).
  - DB에서 equal은 0/9/1보다 0.0013 나쁘다 (VERIFIED).
  - equal을 쓰면 **presence 게이트가 없는 `max_nopresence`의 가중이 1/3**이 된다. 그런데 음악 head는 MP=0 행으로 학습된 적이 없다. 음성 전용 파일에서 원믹스 특징으로 나온 Fm은 외삽이다.
  - DB 음성 소스의 FPR은 오히려 좋아졌다. 그래서 이 메커니즘은 **추측 단계**이다 (INFERRED, 검증 가능).
- **잡음.** DB 쌍 부트스트랩 CI [0.001, 0.011]이므로 SE는 약 0.0026이다. LB 쌍 SE도 비슷하다고 가정하면 DB 대 LB 불일치 0.0077은 약 **2.1σ**이다 (ESTIMATE, LB 크기는 UNKNOWN).

### 4.3 경쟁 설명 비교

| # | 설명 | 지지 근거 | 반대 근거 | 판정 |
|---|---|---|---|---|
| H1 | 원시 개수 불균형(클래스·소스 행수) | 행수 편차가 큼 | 코드가 클래스·셀 동일가중으로 상쇄(VERIFIED). EER은 사전확률에 불변. S3→S7 LB 이득은 특징에서 나옴 | **약함** |
| H2 | 가중치가 만든 숨은 불균형(유효표본, 층 사전확률 역전, 한국어 저가중, FMA:1041 집중) | §3.2 수치(ESTIMATE). 작은 셀에 과적합하면 DB는 같은 소스 계열이라 벌점이 없고 LB에서만 드러남 | duration_balance가 일부 상쇄할 수 있음(UNKNOWN). **ablation이 한 번도 없음** | **그럴듯한 기여 요인. 가장 싸게 검정 가능** |
| H3 | coverage gap(존재하지 않는 셀) | 결핍은 VERIFIED. S8이 DB 모든 층에서 개선됐는데 LB에서 하락 → LB에 DB 밖 도메인이 있음. ledger R3의 불일치 믹스 FPR 0.105→0.337 | LB 구성 정보가 없음. CPS 0.998이라 presence는 정상 | **격차의 주원인 후보 (H4와 함께)** |
| H4 | DB가 LB를 대표하지 못함(합성 constructed 비중, HAIM 편집, fleurs 40%) | §4.2: 음악 FP 감소의 78%가 constructed. 미관측 음악 이득이 HAIM 2개에 집중 | S8은 자연 미관측 소스 musan에서도 개선(FILE FPR 7.4→5.0%) | **S8에 대해 강함. 격차에 대해 중간** |
| H5 | held-out 반복 재사용에 의한 선택 편향 | 후보를 많이 선별함. +0.0043은 다중비교 잡음 범위 | 격차가 안정적(0.072~0.087) | 격차에 대해 약함. S8에 대해 중간 |
| H6 | 잡음 | 약 2σ | 방향이 처음으로 반대 | 부분적 |
| H7 | FILE 결합 가중 변경 | DB 기준 -0.0013 | 불일치 0.0077의 약 15% | 작음 |
| H8 | 평가 집합 층 구성 차이(풀링 EER) | DB real의 40%가 쉬운 fleurs(FPR 1.2%). 음악 real FPR은 7~13% | LB 구성 UNKNOWN | 중간 (ESTIMATE로 약 +0.01~0.02 FE) |
| H9 | 런타임·포맷·길이 불일치 | CSV에 길이 정보 없음. window-max pooling은 길이에 민감 | parity 검사 존재. CPS 0.998 | UNKNOWN |
| H10 | 라벨 정책 불일치(AI 마스터링·편집) | sonicmaster=real이 real-music 가중 11% | LB 정보 없음 | UNKNOWN, 작을 것 |
| H11 | 게이트 없는 max_nopresence × 원믹스 Fm | equal 가중에서 1/3. 음악 head는 음성 전용 원믹스를 학습하지 않음 | DB 음성 FPR은 개선 | 추측. 검정 가능 |

### 4.4 판정

- **0.07 격차**
  - 원시 불균형 때문이 아니다.
  - coverage와 소스 도메인 격차(H3), 그리고 **DB의 비대표성**(H4, H8)이 주된 설명이다.
  - 가중치 병리(H2)는 크기가 UNKNOWN인 기여 요인이다. 새 데이터 없이 고칠 수 있는 유일한 부분이다.
- **S8 음전이**
  - "불균형" 설명은 거의 지지되지 않는다. 가중치는 S7과 동일했다.
  - 가장 강한 증거는 **DB 이득이 bank 고유 합성 소스와 HAIM 생성기에 집중**되었다는 점이다.
  - 그 위에 잡음(약 2σ)과 FILE 가중치 변경(약 0.001)이 더해진 것으로 본다.
  - 브리핑의 "상업·마스터링 음원" 가설은 이와 양립한다. 다만 자료에 직접 증거는 없다 (UNKNOWN).
- **DB 사각지대.** DB에는 fake 무반주 가창, AI cover, 한국어 실제 노래가 0행이다. 그래서 이 셀을 겨냥한 처방(예: SingMOS의 +0.0002)은 **구조적으로 DB에서 중립으로 보인다.** SingMOS를 기각한 것은 LB에 대한 반증이 아니었다 (INFERRED).

---

## 5. Task 4: 처방, 제출 계획, 검증 방법

### 5.1 순위 (기대 LB ΔADS/시간, 모두 ESTIMATE, 주관적 범위)

| 순위 | 처방 | 소요 | GPU | 기대 ΔLB (중앙값, 범위) | 이득/시간 | 분산 |
|---|---|---|---|---|---|---|
| 1 | **W1 층화 재가중** (S7 특징 그대로, head만 재학습) | 3~3.5 h | 약 1 h H100x1 또는 CPU | +0.003 (-0.006~+0.015) | 약 +0.0009/h | 중 |
| 2 | **W2 불일치 노래·AI-cover 대리 리믹스 + 일치 리믹스 대조군** (+W1 가중) | 4.5~5.5 h | 약 1~1.5 h H100x4 | +0.002 (-0.008~+0.025) | 약 +0.0004/h, **상방이 가장 큼** | 고 |
| 3 | **C1 자연 소스 전용 재선택** (S7/S8 음악 또는 평균, FILE 0/9/1·equal·presence-gated) | 2~2.5 h | 없음 | +0.001 (-0.003~+0.004) | 약 +0.0004/h | 저 |
| 4 | P-norm 층별 사후 정규화 (presence로 층을 판정하고 val real CDF 정합) | 2.5~3 h | 없음 | +0.002 (-0.005~+0.008) | 약 +0.0007/h | 중. W1과 겹침 |
| 5 | 마스터링·음량·코덱 스트레스 테스트 | 2 h | 약 0.5 h | 직접 이득 0 (의사결정용) | - | - |
| 6 | AI-Hub (한국 내 처리) | 6~8 h 이상, 승인·위치 확인 필요 | deneb-kr (UNKNOWN) | U | 이번 마감에는 약 0 | - |
| - | LB probe (예: VOICE 상수 제출) | 1장 소모 | - | 정보만 얻음 | 비추천 | - |

- 최상위권 컷(~0.86)에 가려면 +0.023이 필요하다. 어떤 단일 처방의 중앙값보다 크다.
- 현실적인 경로는 고분산 티켓(W1+W2)을 조합하는 것이다 (ESTIMATE).

### 5.2 상위 3개 처방의 구체 내용

**W1: 층화 재가중** (`full_bank_batches.loss_weights` 한 곳만 교체하고 호출부는 그대로 둔다)

1. **층을 정의한다.**
   - voice head의 층은 보컬 유형 × 한국어 여부이다. 보컬 유형은 §1의 A/B/C/D/E이고 source로 매핑한다. 한국어 여부는 `language == 'ko'`로 판정한다.
   - music head의 층은 반주만 / 보컬 포함 / 보컬 존재 미상 / constructed이다.
2. **층 안에서 클래스를 균형 맞춘다.** 층의 질량은 √(층 행수)에 비례시킨다. 한 클래스만 있는 층(B, 한국어 노래)은 ×0.5로 줄인다.
3. **(층, 클래스) 안의 셀 가중은 1/√n_cell로 둔다.**
4. **상한을 둔다.** 행 가중은 평균의 5배 이하로 제한한다. 계보 그룹 합은 클래스 가중의 2% 이하로 제한한다. 이어서 전역 클래스 합을 같게 다시 맞춘다. 그 다음 기존 `duration_balance`를 그대로 적용한다.
5. **먼저 로그를 찍는다.** duration_balance를 적용한 **실제** 층별 P(fake), 한국어 가중 비중, Kish 유효 행·그룹 수, 최대 그룹 비중을 출력한다. 약 10줄이면 되고, 이것으로 이 보고서의 ESTIMATE가 확정된다.
6. **DB에서 선별한다.** 비교 대상은 현재 가중, sqrt만, W1이다. 3 seed로 돌리고 §5.4 게이트를 통과한 것만 P1 재학습과 패키징(약 1.25 h)으로 넘긴다.
7. 위험: 기존 cell 동일가중이 미관측 생성기 일반화를 도왔을 수도 있다. DB의 unseen-gen 부분집합으로 확인한다.

**W2: 불일치·일치 리믹스** (브리핑 §8에 따르면 42,039행 전체의 fp16 stem이 이미 캐시에 있다. 새로 내려받을 것은 없다)

1. **네 조합을 같은 수로 만든다** (예: 각 750행, 가창).
   - 보컬 stem(real: vocalset/csd/openmic/FMA/Jamendo 보컬, fake: SONICS/sonics/Echoes/HAIM/generated_music 보컬) × 반주 stem(real: openmic/FMA/Jamendo/musan/haim real, fake: FMC/Echoes/SONICS/HAIM/acestep)
   - 여기에 real 음성 + fake 음악 500행을 더한다. train에는 이 조합이 5행밖에 없다.
2. **일치 조합(real+real, fake+fake)을 같은 공정으로 반드시 포함한다.** 그래야 "리믹스 흔적"이 라벨과 무관해진다. 이 조건이 빠지면 새 shortcut이 생긴다.
3. **믹싱 조건.** 보컬 대 반주 비율(VAR)은 -3~+9 dB에서 무작위로, 오프셋도 무작위로 준다. 목표 음량을 공통으로 맞추고, bank와 같은 코덱 체인 분포를 real/fake에 균등하게 적용한다.
4. **계보를 지킨다.** 같은 split의 부모끼리만 섞고, 그룹 키는 부모 키의 합집합으로 둔다. held-out 리믹스는 held-out 부모로만 만든다.
5. **특징은 패키지 런타임 Engine으로 추출한다.** HTDemucs 재분리와 전 인코더를 거치므로 런타임과 정확히 같은 특징이 나온다.
6. **평가.** 자연 부분집합이 나빠지지 않는지 보고, 불일치 리믹스의 VOICE FPR과 MUSIC FPR을 확인한다. 불일치 리믹스는 자체 합성이라 낙관적으로 나온다는 점을 명시한다.
7. 2라운드 규정 때문에 리믹스 파일과 생성 스크립트를 보관해야 한다.
8. **근거가 약하다는 점은 분명히 해 둔다.** 파일 단위로 VOICE/MUSIC 라벨이 따로 있다는 사실이 불일치 파일의 존재를 **약하게** 시사할 뿐이다 (INFERRED, 약함).

**C1: 자연 소스 전용 재선택** (학습 없음)

1. S7과 S8의 행 단위 DB 예측으로 다음 지표를 계산한다.
   - constructed_*를 뺀 자연 부분집합 ADS
   - HAIM 편집 생성기를 따로 보고한 지표
   - 소스 잭나이프
2. 결정할 것
   - 음악: S7 / S8 / 두 CDF의 평균
   - FILE: 0/9/1 / equal / `max`와 `noisy_or`만 쓰는 presence-gated 조합
3. S7이 이미 최고점이므로 저분산 대비 티켓으로만 쓴다.

### 5.3 제출 계획 (09-29 00:00~10:00, 3장, 최고점만 반영)

1. **빌드 순서**
   - W1은 가장 빠르다. 약 15시 전후 준비를 목표로 한다.
   - W2는 음원 합성과 특징 추출을 W1 코딩과 병렬로 진행한다. 약 21~22시 준비를 목표로 한다.
   - C1은 여유 시간에 준비한다.
   - 유휴 GPU는 즉시 일시정지한다.
2. **00:00 1장.** W1+W2가 DB 자연 부분집합 게이트를 통과하면 이것을 낸다. 아니면 W1만 낸다. 상방을 먼저 확인한다.
3. **결과를 보고 2장을 고른다** (패키지 실행 약 1 h 이후).
   - 1장 > 0.8374이면: 같은 방향의 변형(sqrt 강도 0.75, 그룹 상한 1%)을 낸다.
   - 1장 < 0.8374이면: W1과 W2를 분리해서 낸다(원인 분리).
4. **3장.** 남은 것 중 기대값이 가장 높은 것을 낸다. 모든 것이 실패하면 C1을 낸다.
5. **대안 설계.** FILE 열을 S7에 고정하고 VOICE/MUSIC만 바꾸면 성분별 LB 효과를 정확히 분리할 수 있다. 다만 FILE 이득(가중 0.5)을 포기하는 것이므로 2장까지로 제한하는 것을 권한다.

### 5.4 S8을 걸렀을 검증 방법 (앞으로의 게이트)

1. **자연 부분집합 게이트.** constructed_*를 제외한 held-out에서 ΔADS ≥ +0.002이고 P(Δ>0) ≥ 0.9여야 한다. 부트스트랩은 **계보 그룹 단위**로 한다. held-out에는 musan 작곡가나 commonvoice 화자 같은 거대 성분이 있어 행 단위 부트스트랩은 분산을 과소평가한다.
2. **소스 잭나이프.** held-out 소스를 하나씩 빼고 ΔADS를 다시 계산한다. 최솟값이 0보다 커야 하고, 한 소스(또는 한 생성기군)가 이득의 50% 이상을 차지하면 안 된다.
3. **층 구성 강건성.** held-out 층 가중치를 음성 중심, 노래 중심, 음악 중심, fleurs 축소의 3~4가지로 바꿔 평가한다. 모든 경우에 Δ ≥ 0이어야 한다.
4. **입력 도메인 변경(stem에서 원믹스로)** 에는 스트레스 테스트가 필요하다. 자연 real 음악에 음량 정규화(-9/-14 LUFS), 리미팅, 스테레오→모노, AAC/Opus를 적용하고 FPR 이동을 비교한다.
5. **LB 1장에는 가능한 한 변경 하나만 담는다.** S8은 원믹스 특징과 FILE 가중이라는 두 변경이 섞여 원인을 분리할 수 없었다.

- S8에 적용했다면 1번과 2번이 음악 쪽 이득의 집중(constructed 78%, HAIM 2생성기)을 표시했을 가능성이 높다. FILE 쪽은 musan 개선이 있어 통과했을 수도 있다 (INFERRED. 행 단위 예측으로 확인 필요).
- 이 검증으로도 **bank에 없는 LB 도메인은 잡을 수 없다.** 그것은 새 데이터나 LB 티켓으로만 확인할 수 있다.

### 5.5 AI-Hub

- 대회 사용은 허용되었다 (브리핑 §10). 그러나 한국 밖 반출과 클라우드 LLM·EU GPU 전송은 금지이다.
- 처리 가능한 곳이 둘 다 막혀 있다.
  - Mac: 사용자가 고부하 작업을 하지 말라고 했다.
  - deneb-kr: 물리적 위치와 수용 가능 여부가 UNKNOWN이다.
- 설령 사용자가 deneb-kr을 확인하더라도 다운로드, stem 분리, 6개 인코더, head 학습, 패키징을 모두 한국 안에서 해야 한다. 6~8 h 이상 걸리고 운영 위험이 크다.
- 기대 이득은 UNKNOWN이다. LB에 한국어 실제 노래가 많다면 크고, 아니면 0이다.
- **이번 마감에는 권하지 않는다.** 마감 후나 2라운드 문서화 단계에서 465/473(한국어 실제 가창)을 negative로, 71767(대중음악 원천)을 real 반주로 쓰는 계획은 유효하다.

### 5.6 하지 말 것

- 원시 행수를 맞추려는 over/undersampling은 하지 않는다. 이미 가중치로 상쇄되어 있고, 문제는 과보정 쪽이다.
- DB에서 +0.002~0.004 수준의 개선을 근거로 입력 도메인을 바꾸지 않는다. S8의 교훈이다.
- 테스트 오디오 통계로 적응하지 않는다(test-time 정규화나 pseudo-label). 규정 위반 여부가 UNKNOWN이다.

---

## 6. 한계와 미확인 사항

- duration_balance와 silent-stem 제외를 반영하지 못했다. 가중치 수치는 ESTIMATE이다. §5.2 W1의 5번 로그로 확정해야 한다.
- 행 단위 예측이 없다. §4.2 해석, §5.4 게이트의 사후 판정, per-stratum EER은 팀이 가진 예측으로 확인해야 한다.
- FMC rule L 352행은 UNKNOWN이다(`fmc_caption_vocal.json`이 없다).
- LB의 크기, 구성, 언어, 길이, 코덱 분포는 모두 UNKNOWN이다. 이 보고서의 LB 관련 서술은 모두 가설이다.
- 기대 이득 범위는 판단에 따른 ESTIMATE이다. 가중치 방식에 대한 과거 ablation은 ledger 키워드 검색으로 찾지 못했다(INFERRED, 전수 조사는 아님).
