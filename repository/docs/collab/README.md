# DACON 236749 딥페이크 오디오 탐지 — 협업 문서

기준 시각: 2026-09-28 (KST). 이 문서와 `docs/collab/tables/`만 읽어도 모델 구성, 학습 데이터, 검증 결과를 따라갈 수 있게
정리했다. 음원, 특징(npz), 모델 가중치(.pt), 제출 zip은 git에 넣지 않았다. 각 위치와 sha256은 아래에 적었다.

## 1. 대회와 평가

- 지표: `ADS = 0.5·(1 − FILE EER) + 0.2·(1 − VOICE EER) + 0.3·(1 − MUSIC EER)`. 테스트 1,200개 전체로 채점하며,
  Public과 Private가 같다. 최고 점수가 최종 점수다.
- 제출: 하루 3회, 마감 2026-09-29 10:00 KST. 제출 파일은 L4 GPU에서 오프라인으로 실행된다.
- 데이터 규정:
  - 규칙 2-1: 공개 데이터이면서 최소한 비상업 이용이 허용되어야 한다.
  - 규칙 3: 2차 평가 때 "학습에 활용한 데이터 파일 일체"를 제출해야 한다.
  - 토크보드 417198: 생성한 오디오도 제출 대상이다.
  - 토크보드 417844: AI-Hub 데이터 사용이 허용된다.
- AI-Hub 약관: 국외 반출과 제3자 제공이 금지된다. AI-Hub 음원과 특징은 국내 Mac에서만 처리하고, 외부로는 집계 지표만 내보낸다.
- AT-ADD 약관: 비상업 학술 연구와 교육 목적으로만 쓸 수 있다. 재배포할 때는 약정서를 함께 첨부해야 한다.

## 2. 리더보드 기록

| 제출 | 패키지 (`submissions/…/submit.json`) | sha256 앞 16자 | LB ADS | 변경 내용 |
|---|---|---|---|---|
| P1 | DataV4_P1Refit | 1b9ba4411b87c8fc | 0.7700 | data v4, 학습 + held-out 재학습 |
| S2a | DataV4_S2a_AIHubPhone | c7c8b1c83d242df9 | 0.7689 | P1에 AI-Hub 571 전화 음성 800개 추가 |
| S3 | DataV4_S3_Backbones | 062fc236bf0d1503 | 0.7853 | AntiDeepfake 음성 멤버, MERT 6+12층 음악 특징 |
| S6 | DataV4_S6_S5LabelCompletion | 7429b1830aaf0333 | 0.8220 | MLP/ResMLP head, EAT 음악 특징, 라벨 보완 규칙 L |
| **S7** | **DataV4_S7_SSLAMWavLM** | **5239c912ccbee8c2** | **0.8374 (최고)** | S6에 SSLAM b5 음악 특징과 WavLM h8 음성 멤버 추가 |
| S8 | DataV4_S8_MixSSLAM | b0e8744225e9252e | 0.8340 | S7에 원본 믹스의 SSLAM 특징 추가 (DB에서는 +0.0043, LB에서는 −0.0034) |
| W (후보) | DataV4_W_Reweight | a17442e6f3e2a830 | 미제출 | S7 구조 그대로, 모든 head를 W 가중치로 재학습 |

## 3. 모델 (S7 구조, W도 동일)

제출 zip의 구성 (`submissions/DataV4_W_Reweight/submit.zip`, 9.74 GB, 79개 파일, 압축 없이 저장):

```
script.py                  → model/runtime/full_bank_submission.run()
model/manifest.json        파일별 sha256 (실행 시작 시 검사)
model/runtime/             추론 코드, eat_core, bin/ffmpeg
model/assets/  (9.6 GB, 고정 인코더: HTDemucs, DF-Arena-1B, PANNs, AntiDeepfake XLS-R-1B, MERT-330M, EAT-large, SSLAM, WavLM-Large)
model/trained/ (29 MB, 학습한 head)
```

파일 하나의 처리 순서 (`full_bank_submission.py` `Engine.predict`):

1. 디코드 → 16 kHz → 피크를 0.6으로 제한 → HTDemucs로 보컬 stem과 반주 stem을 분리한다.
2. 보컬 stem에서 음성 멤버 3개를 계산한다.
   - DF-Arena fc5 입력 특징 → MLP head (`voice_41~43.pt`)
   - AntiDeepfake XLS-R-1B 특징 → ResMLP head (`voice_ad_*`)
   - WavLM h8 특징(fp16로 반올림한 stem 사용) → MLP head (`wavlm_*`)
   - 세 멤버를 val CDF로 융합한다. 가중치는 1 / 0.5 / 1이다 (`component_fusion.json`). 결과가 `VOICE_FAKE_PROB`다.
3. 반주 stem을 4.04초 창으로 나누고, 창마다 MERT 6·12층, EAT 16번 블록, SSLAM 5번 블록 특징을 이어 붙인다.
   이것을 WindowMIL head 3개에 넣어 `MUSIC_FAKE_PROB`를 얻는다.
4. PANNs와 presence head로 `VOICE_PRESENT_PROB`, `MUSIC_PRESENT_PROB`를 계산한다.
5. `FILE_FAKE_PROB`는 세 규칙의 CDF 가중 조합이다: max / noisy_or / max_nopresence = 0 / 9 / 1 (`file_ensemble.json`).

## 4. 학습

- 프로토콜:
  - **DB**: train으로 학습하고 val로 조기 종료한 뒤, held-out test(R2 frozen 4,822행)로 평가한다. 모델 선택용이다.
  - **P1**: train과 held-out test를 모두 학습에 쓰고 val로 조기 종료한다. 제출용이다.
- seed는 41, 42, 43이다. head는 GPU에서 fp32로 학습하며 TF32는 끈다.
- 라벨 보완 규칙 L (`research/top15/work4/voice_data.py` `completes`): 완전히 생성된 곡에 보컬이 있으면 그 보컬도 fake로 본다.
  DF-Arena와 AntiDeepfake 음성 head에만 적용한다.
- 학습 가중치 (`data_v4/weighting.py`, 환경 변수 `WEIGHTING`):
  - `s7`: 기존 방식. (라벨, 출처, 생성기) 셀마다 같은 질량을 주고 길이 분포를 맞춘다.
  - `C`: s7에 상한을 더한다. 행 하나는 클래스 평균의 5배 이하, 계보 그룹 하나는 클래스 질량의 2% 이하.
  - `W`: 음성 구간마다 클래스별 목표 질량을 둔다. 한국어 음성 0.25, 기타 음성 0.45, 음악 위 음성 0.05, 노래 0.25.
    셀 기본 가중치는 max(n, 20)^−1/2이고, 구간 안에서 길이 분포를 맞춘 뒤 C와 같은 상한을 적용한다.
- 가중치를 적용한 뒤 실제 학습 비중: `tables/train_weight_shares.csv`.
- P1 명령 (W 예시, x4 GPU):
  ```
  WEIGHTING=W HEAD_DEVICE=cuda:0 python data_v4/train_backbone_heads.py --cache _workspace/backbone_cache --final --voice-arch mlp \
    --ad-arch resmlp --voice-epochs 150 --voice-patience 10 --music-layers 6 12 --music-eat 16 \
    --eat-cache _workspace/newbackbone_cache _workspace/eat_extra_cache --out _workspace/final_WL_20260928 \
    --reuse-music _workspace/final_s6L_20260928 --label-completion            # A 실행: 규칙 L 음성 head
  WEIGHTING=W ... --out _workspace/final_W_20260928 --music-sslam 5 --sslam-cache _workspace/s6_cache --voice-wavlm 8 \
    --wavlm-cache _workspace/s6_cache --wavlm-weight 1 --reuse-voice _workspace/final_WL_20260928 --file-weights 0,9,1   # B 실행
  ```
- 패키징: `package_datav4_submission.py` (Mac)으로 만든다. 검증 순서:
  1. `verify_datav4_submission.py` (Mac CPU 오프라인)
  2. `research/top15/work5/w/w_runtime_parity.py` (학습 경로와 실행 경로의 출력 비교, 허용 오차 VOICE 0.02 / MUSIC 0.01)
  3. `research/top15/work5/s8/verify_s8_vessl.py` (VESSL CUDA 오프라인)

## 5. 데이터

- **data v4 bank** (모든 제출이 학습한 데이터): `tables/bank_v4_sources.csv`에 출처별 train/val/test의 real/fake 수와
  라이선스가 있다.
  - P1 학습 37,108행: real 12,967 / fake 24,116. val 4,931행.
  - 행 목록 원본: `results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank/*.csv` (git 제외, 14 MB).
- **새 데이터** (2026-09-28 추가): `tables/new_data_summary.csv`에 출처별 분할과 real/fake 수가 있다.
  - AT-ADD Track 2 (speech, 반주 없는 노래, 음악; train+dev에서 24,000, eval에서 10,000)
  - HAIM B6/B3/B4 (한국어·영어 가사로 만든 ACE-Step 노래, 마스터링한 AI 곡, 전문 믹스한 AI 곡)
  - MUSDB18-HQ, AI-OpenBMAT (방송 음성 위 AI 음악 또는 실제 음악), GTSinger 한국어 (노래와 말)
  - SD-RR (Jamendo 실제 곡)
  - D1 재혼합: real 또는 fake 보컬 × real 또는 fake 반주, 보컬/반주 비 −12 / −6 / 0 / +6 dB
- 분할 (`research/top15/work5/newdata/assign_splits.py`, 학습 전에 고정):
  - AT-ADD eval → `newtest` (train/dev에 없는 생성기)
  - AT-ADD train 쪽의 15% 그룹 → `newval`
  - 공개 데이터: 계보 그룹의 30% → `newtest`, 다음 15% → `newval`, 나머지 → `train`
  - 재혼합은 소스 음원이 모두 train 그룹인 것만 남겼다.
- **AI-Hub** (Mac 전용, 평가셋 E3):
  - 71544 음악 유사성 판별(Validation) 한국어 커버곡 107곡, 473 가이드보컬 75개, 465 다화자 가창 75개. 각 10초 발췌.
  - 원본, 발췌본, 특징, 행 목록은 `_workspace/aihub_prep`에만 있다. git과 VESSL에는 없다.
  - 주의: AI-Hub 도구(aihubshell 0.6)는 조각 파일을 문자열 순서로 합쳐 zip을 깨뜨린다. 71544 VS.zip은 조각 순서를 바로잡아 복구했다.
- 특징 추출 (`research/top15/work5/newdata/dump_features.py`):
  - 제출 패키지의 Engine을 heads 직전까지 그대로 실행한다.
  - 행 하나당 npz 하나에 학습 캐시의 모든 키와 presence를 저장한다.
  - 일치 검사 (`parity.py`): 기존 캐시와 cos ≥ 0.99999(H100, Mac MPS 모두). presence는 실행 결과와 차이 3e-8 수준이다.
- 저장 위치:
  - VESSL 볼륨: `/root/newdata/{원본, prepared, feats, rows}`
  - Mac: `_workspace/{backbone_cache, newbackbone_cache, s6_cache, cache_v4, final_*}`

## 6. 검증 결과

### 6.1 가중치 비교 (DB 프로토콜, 새 데이터 없음; `research/top15/work5/w/w_eval.json`)

| 구성 | val ADS | test ADS | natural test ADS | natural FILE / VOICE / MUSIC EER |
|---|---|---|---|---|
| S7_cpu (최고 제출과 같은 레시피) | 0.96151 | 0.90982 | 0.91754 | 0.0627 / 0.0549 / 0.1338 |
| s7 GPU 대조 | 0.96073 | 0.90899 | 0.91743 | 0.0629 / 0.0549 / 0.1338 |
| C | 0.96267 | 0.91088 | 0.91800 | 0.0621 / 0.0627 / 0.1280 |
| W | 0.96615 | 0.90615 | 0.91549 | 0.0716 / 0.0706 / 0.1154 |

C와 W 모두 사전 등록한 안전 기준을 통과했다: natural ΔADS ≥ −0.005, 성분 EER 악화 ≤ 0.02, real 오탐 증가 ≤ 5%p.
W는 FLEURS-ko real 오탐이 +4.99%p로 기준 경계에 있다.

### 6.2 새 데이터 비교 (같은 평가셋: E1 = DB held-out, E2 = 새 데이터 held-out, E3 = AI-Hub; `research/top15/work5/newdata/new_eval.py`)

모든 구성에 같은 조건을 적용했다: DB 프로토콜, 음성 융합 1/0.5/1, 기존 val에서 만든 CDF 표, FILE 0/9/1.
평가셋은 셋이다.
- E1: DB held-out 4,822행.
- E2: 새 데이터 held-out 12,426행. AT-ADD eval 쪽 생성기와 공개 데이터 계보 그룹 30%로 구성했다.
- E3: AI-Hub 한국어 real 257행. Mac에서만 채점했다.

| 구성 | E1 ADS (vs s7_ref) | E2 ADS (vs s7_ref) | E1+E3 ADS | E2 FILE / VOICE / MUSIC EER |
|---|---|---|---|---|
| s7_ref (최고 제출 레시피) | 0.90899 | 0.89334 | 0.91001 | 0.1155 / 0.0879 / 0.1045 |
| W_ref (W, 새 데이터 없음) | 0.90615 (−0.0028) | 0.88469 (−0.0087) | 0.90732 | 0.1248 / 0.1069 / 0.1051 |
| W_new (W + 새 데이터) | 0.91231 (+0.0033) | 0.93842 (+0.0451) | 0.91439 | 0.0694 / 0.0474 / 0.0579 |
| W_newB (W_new, 조기 종료 val에 newval 추가) | 0.91221 (+0.0032) | 0.94031 (+0.0470) | 0.91503 | 0.0669 / 0.0483 / 0.0553 |
| s7_new (S7 가중치 + 새 데이터) | 0.91540 (+0.0064, CI [0.0013, 0.0111]) | 0.92185 (+0.0285) | 0.91718 | 0.0843 / 0.0638 / 0.0774 |

E2 출처별 결과 (s7_ref / W_new / W_newB / s7_new):
- AT-ADD 음성: FILE EER 0.161 / 0.058 / 0.056 / 0.062, real 오탐 0.160 / 0.009 / 0.009 / 0.011
- AT-ADD 노래: VOICE EER 0.080 / 0.054 / 0.060 / 0.061, FILE EER 0.144 / 0.094 / 0.094 / 0.151
- AT-ADD 음악: MUSIC EER 0.111 / 0.057 / 0.055 / 0.083
- MUSDB real 오탐 0.063 / 0.027 / 0.018 / 0.018, SD-RR real 오탐 0.048 / 0.027 / 0.027 / 0.007
- AI-OpenBMAT: FILE EER 0.392 / 0.354 / 0.354 / 0.336, real 오탐 0.378 / 0.300 / 0.311 / 0.200. 모든 구성이 약하다.
- HAIM B3·B4·B6 fake 검출률은 모든 구성에서 0.995 이상이다.

E3 real 오탐 (s7_ref / W_ref / W_new / W_newB / s7_new):
- AI-Hub 465: 0.067 / 0.067 / 0 / 0 / 0
- AI-Hub 473: 0.053 / 0.107 / 0.040 / 0.040 / 0.027
- AI-Hub 71544: 모두 0

해석:
- W 가중치만으로는 새 도메인에서 오히려 나빠졌다.
- 새 데이터를 넣으면 모든 평가셋이 좋아졌다. 기존 영역(E1)과 한국어 real(E3)은 s7_new가, 새 생성기(E2)는 W 계열이 가장 좋다.
- E2의 공개 데이터 부분은 학습 데이터와 분포가 가까워서, E2 향상폭은 LB 향상의 상한으로 봐야 한다.
원본 결과: `research/top15/work5/newdata/new_eval_*.json`.


### 6.3 E2 세부 확인과 제출 계획 (2026-09-28 15:00)

E2를 AT-ADD 포함 여부로 나눈 결과 (`research/top15/work5/newdata/e2_subsets.txt`, s7_ref 대비 변화):

| 구성 | E2 공개 데이터만 (2,426행) | E2 AT-ADD만 (10,000행) |
|---|---|---|
| W_new | 0.9720 (+0.0072) | 0.9340 (+0.0561) |
| W_newB | 0.9720 (+0.0072) | 0.9361 (+0.0582) |
| s7_new | 0.9734 (+0.0085) | 0.9127 (+0.0349) |

제출 계획 (2026-09-29, 3번, 모두 정상 학습 모델):
1. ① s7_new (`submissions/DataV4_s7_new`): ① − S7 = 새 데이터 전체의 실제 효과
2. ② W_new: ② − ① = 가중치 효과
3. ③ 이긴 방향을 한 단계 더 민 모델: s7_new + AI-OpenBMAT 400행, 또는 W_newB

라이선스는 `LICENSES.md`에 정리했다. AT-ADD의 2차 평가 제출 위험은 감수하기로 결정했다.

### 6.4 통합 모델 심의와 사전 등록 선택 검정 (2026-09-28 16:44–17:45 KST, 심의 중 추가 GPU 사용 없음)

질문: 인코더와 실행 코드는 그대로 두고, 이미 학습한 head를 성분별로 조합해 하나의 모델로 만들면 나아지는가.
Codex 두 세션(장기 세션, 새 세션)과 15–22라운드를 심의했다. 기록은 `research/top15/work5/debate_20260928/`에 있다.

가상 조합: DB 프로토콜 head의 행별 예측으로 만들었다(`research/top15/work5/newdata/virtual_configs.py`, 결과 `virtual_configs_eval.txt`).
ens는 두 구성의 seed 평균이고, mix는 음성 성분과 음악 성분을 서로 다른 구성에서 가져온 것이다. 값은 s7_ref 대비 ADS 차이다.

| 구성 | E1 | E2 전체 (12,426행) | E1+E2 |
|---|---|---|---|
| s7_new | +0.0064 [+0.0013, +0.0111] | +0.0285 | +0.0210 |
| W_newB | +0.0032 | +0.0470 | +0.0334 |
| ens s7_new + W_newB (6 seeds) | +0.0061 | +0.0417 | +0.0305 |
| ens s7_new + W_new | +0.0057 | +0.0398 | +0.0286 |
| mix 음성 s7_new / 음악 W_newB | +0.0103 [+0.0036, +0.0162] | +0.0389 | +0.0298 |
| mix 음성 W_newB / 음악 s7_new | +0.0006 | +0.0366 | +0.0244 |
| mix 음성 s7_new / 음악 s7_new_ob | +0.0050 | +0.0376 | +0.0285 |

- 기존 val(4,931행)은 조기 종료와 CDF 적합에 반복 사용됐고, 후보 순위를 E1과 거의 반대로 매겼다. 그래서 이번 선택에는 newval을 썼다.
- W 계열은 E1 VOICE EER을 나쁘게 만든다(s7_new 0.0622, W_new 0.0791, W_newB 0.0745). 음성은 s7 레시피, 음악은 W 레시피가 낫다는 것이 mix의 가설이다.

사전 등록 선택 검정 (`select_newval.py`, 결과 `select_newval.txt`, 쌍체 그룹 bootstrap 2,000회):
- 후보: mix 음성 s7_new / 음악 W_new. DB 모델 기준으로 두 성분 모두 newval을 학습과 조기 종료에 쓰지 않았다.
  (W_newB는 newval로 조기 종료했으므로 검정 대상에서 뺐다.)
- 규칙: newval ΔADS ≥ +0.002이고 CI 하한 > 0일 때만 채택. 거부 조건: E1 < 0, E2(누출 제외) < 0, E2 공개 데이터 < −0.002.
- 실패하면 다른 조합을 이어서 찾지 않는다.

| 평가셋 (계보 공유 행 제외) | mix − s7_new |
|---|---|
| newval (2,281행) | +0.0013 [−0.0020, +0.0057], p>0 0.819 → **기준 미달, 채택 안 함** |
| E1 (4,822행) | +0.0045 [−0.0003, +0.0085] |
| E2 누출 제외 (12,351행) | +0.0085 [+0.0058, +0.0118] |
| E2 공개 데이터 누출 제외 (2,361행) | +0.0011 [−0.0032, +0.0031] |

- 한계: 계보 비공유 newval 3,802행 중 1,521행(40%)은 presence 특징이 없어 FILE/ADS 계산에서 빠졌다.
  AT-ADD 학습 풀 행을 presence 키가 생기기 전에 추출했기 때문이다.
- 음악 성분만 비교하면(presence 불필요, `music_newval.py`, 1,545행) MUSIC EER이 s7_new 0.0265, W_new 0.0265로 같다.
  newval의 AT-ADD 음악은 학습 생성기 계열이라 쉽다. 새 생성기가 있는 E2에서는 W 음악 head가 낫다(MUSIC EER 0.0774 → 0.0579).
- 선택 사항: 빠진 1,521행의 presence를 보완하고(0.1–0.2 H100시간) 같은 규칙을 한 번만 적용한다.
  부분 결과를 본 뒤의 보완이므로, 처음부터 완전했던 독립 검정과 같지는 않다.

대회 후 통합 설계 (가설이며 확인된 최적 구성이 아니다):
- 음성: s7 레시피 head(DF와 AD만 label completion L 적용, WavLM은 미적용), 융합 가중치 1/0.5/1.
- 음악: WindowMIL. s7 레시피, W 레시피, 두 레시피 확률 평균(50:50)을 비교한다. 음악 head만 따로 읽는 로더가 필요하다.
- FILE: head를 고른 뒤 0/9/1, 1/1/1, 0/1/0을 비교한다. presence는 방송 혼합 음원의 presence 오류가 측정되기 전까지 고정한다.
- 데이터: 한국어 fake 음성(real 2,000 + fake 2,000, TTS·VC 4계열 이상), 방송 혼합 음원, 상용 음악.
- 분할: 조기 종료 val, 새 생성기를 포함한 선택용 셋(5,000행 이상), 한 번만 여는 최종 test(5,000행 이상).
  화자, 곡, 생성기, 원본 트랙 계보는 한 분할 안에만 둔다. 안정성은 그룹 분할 3개 × seed 3개로 본다.
- 예상 비용(추정): 학습·평가 6–10 H100시간. 데이터 수집과 특징 추출 시간은 포함하지 않았다(미정).

## 7. 코드 지도

- `data_v4/`: bank 구축, 학습기(`train_backbone_heads.py`), 가중치(`weighting.py`), FILE 조합(`file_ensemble.py`)
- `research/top15/work2/backbones/run.py`, `research/top15/work4/voice_data.py`: DB 프로토콜 학습
- `research/top15/work5/`
  - `FINAL_PLAN_20260928.md`: 토론 결론과 제출 계획
  - `debate_20260928/`: 여러 세션의 토론 기록
  - `w/`: 가중치 실험
  - `s8/`, `s9/`: 인코더 추가 실험
  - `newdata/`: 새 데이터 행 생성, 특징 추출, 분할, 평가
- `full_bank_submission.py`, `package_datav4_submission.py`, `verify_datav4_submission.py`: 실행, 패키징, 검증
