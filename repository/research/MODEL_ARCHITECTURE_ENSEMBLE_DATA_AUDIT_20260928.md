# DACON 236749: 모델·앙상블·데이터 감사와 연구 보고서

작성: 2026-09-28 01:30 KST (02:00 정정: 4.2절 FILE 척도 오류 철회 / 02:20 Codex GPT-6-Astra 검토 반영: S7 런타임 미구현, 단위, HuBERT, E1·E3 성격, fuse 누락, L 민감도). 대상: DACON 236749 딥보이스 탐지(236753 아님).

증거 표시:
- **VERIFIED**: 이 저장소에서 이번 세션에 실행한 기록이 있음(파일 경로 명시)
- **REPORTED**: 문서·로그·사용자 보고로 확인했으나 지금 다시 실행하지는 않음
- **HYPOTHESIS**: 가설
- **ESTIMATE**: 추정
- **UNKNOWN**: 모름
- **NOT RUN**: 실행하지 않음

이 보고서 작성 중에는 학습·추론·클라우드 작업을 새로 실행하지 않았다. 캐시와 기록된 예측을 읽는 작은 집계 스크립트만 돌렸다. 인용한 실험 결과는 모두 이번 세션 앞부분에 실제로 실행한 것이며, 경로를 함께 적었다.

---

## 1. 우선순위 행동 3가지

1. **S7 완성**
   - 구성: S6 + SSLAM(음악) + WavLM(음성 멤버)
   - 근거: 같은 DB 프로토콜과 런타임 방식(RT)으로 held-out +0.0104 [+0.0043, +0.0152], P(>0)=1.0 (VERIFIED, `research/top15/work4/rt_s6enc_report.json`).
   - 처음 보는 생성기 부분집합에서도 개선된다(0.9017 → 0.9094).
   - 한계: 사전 잠금 규칙의 "seed 3개 모두 개선"은 음악 seed 42가 실패했다. 이 규칙의 예외로 채택하는 것이므로 기록을 남긴다.
   - 필요한 작업: **런타임 구현**(SSLAM EAT 계열 두 번째 모델, WavLM 멤버, `fuse` 누락 멤버 assert), P1 학습, 패키징, CPU 검증, 일치 검사.
   - (정정, Codex 지적) 초판은 "학습기 수정 완료"만 적어 런타임이 준비된 것처럼 읽혔다. **`full_bank_submission.py`는 아직 MERT+EAT와 ours/ad만 처리한다.**
2. **SSLAM 음악 이득의 seed 안정성 확인(E3, CPU 약 10분)**
   - 음악 seed별 held-out 변화는 −0.014 / **+0.008** / −0.024다.
   - group bootstrap CI는 모델을 고정한 상태에서 계산한 것이라, 학습 seed 분산은 반영되지 않는다.
   - S7을 패키징하기 전에 seed 5개를 추가로 돌려 확인한다.
   - (정정) 초판의 우선순위 2 "FILE 결합 재설계"는 근거가 틀렸다. 4.2절 정정을 참고한다.
3. **앞으로의 모든 판정을 RT 방식으로**
   - AC 방식(점수 없는 행을 0으로 채움)은 FILE EER을 약 0.004 낙관한다(VERIFIED).
   - 오늘 결론은 RT로 다시 계산해도 모두 유지됐다. 다만 다음 비교부터는 RT를 기본으로 한다.

---

## 2. 검증된 기준선과 대회 계약

### 2.1 저장소 상태 (VERIFIED)
- 프로젝트: `/Users/choiwuseck/Documents/ChatGPT/데이콘`, 브랜치 `main`, HEAD `26c2525a`
  - 수정 파일 10개, 추적 안 된 파일 929개(대부분 결과·캐시). 원격 저장소는 없다.
- 공개 미러: `reisi09/DACON-DeepVoice-Contest` main `e4987e1`
  - 이후의 `data_v4/label_rules.py`, trainer 확장, work4 분석은 아직 커밋하지 않았다.

### 2.2 제출물 (VERIFIED: 파일·해시·CPU 검증 / REPORTED: LB)
| 패키지 | 구성 | sha256 앞 8자리 | 크기 | LB ADS |
|---|---|---|---|---|
| S3 `DataV4_S3_Backbones` | DF-Arena linear + AD linear, MERT 6+12 | 062fc236 | 8.29 GB | **0.7853** (최고 기록) |
| S4 `DataV4_S4_MLPHeads` | DF MLP + AD ResMLP, MERT 6+12 | beeda9a6 | 8.30 GB | 대기 또는 미제출 (UNKNOWN) |
| S5 `DataV4_S5_EATMusic` | S4 + 음악 MERT+EAT16 | cf17eec9 | 8.92 GB | 미제출 (REPORTED) |
| S6 `DataV4_S6_S5LabelCompletion` | S5 + 음성 라벨 보완 L | 7429b183 | 8.92 GB | 미제출 (REPORTED) |

- 공식 채점 대기 중인 제출이 무엇인지는 대화 기록만으로 확정할 수 없다(UNKNOWN).
- 과거 LB: P1 0.7700, S2a 0.7689, S3 0.7853 (REPORTED, 사용자 보고).

### 2.3 로컬 최고 구성과 선택 프로토콜 (VERIFIED)
- **DB 프로토콜**: train split으로 학습하고 val로 early stopping, held-out(data v4 test 6,450행)은 한 번만 보고한다.
  - held-out은 출처를 분리한 세트다. 학습 fake에 없던 **생성기 6종의 fake 1,135행**과, 학습에 없던 출처의 행 1,944개를 포함한다.
  - DB 비교는 held-out 6,450행 전체가 아니라 R2 고정 행 **4,822행**에서 했다(정정, Codex 지적).
- RT held-out ADS (`rt_s6enc_report.json`):
  - S4 0.88457
  - S5 0.89673
  - S6 0.89877
  - S6 + SSLAM + WavLM 0.90921
- held-out은 LB보다 약 0.09 높다(REPORTED). 원인이 "LB가 더 낯선 데이터"라는 것은 **가설**이다(HYPOTHESIS, 정정).
- **P1**: 제출 패키지는 train과 held-out을 합쳐 다시 학습한다. 그래서 P1 모델에 대한 held-out 추정치는 없다.

### 2.4 코드에는 있지만 제출 경로에는 없는 것 (VERIFIED)
- SONICS 멤버, zero-shot AD 멤버, 시간 기반 폴백: S4부터 제거됐다.
- MuQ, HuBERT, SSLAM-AS2M: 판정에서 탈락했다.
- `--extra-voice`(SingMOS): 판정에서 탈락했다(+0.0002 AC).

### 2.5 대회 계약 (REPORTED: `research/PUBLIC_DATASET_SURVEY_20260927.md`가 인용한 공식 규칙·운영진 답변. 이번에 다시 가져오지 않음)
- **점수**: 0.9·ADS + 0.1·CPS, ADS = 0.5(1−FileEER) + 0.2(1−VoiceEER) + 0.3(1−MusicEER).
  - 우리 채점 코드 `graph_scoring.official_metrics`는 0.45/0.18/0.27에 presence AUC 0.05씩을 더한다.
  - ADS는 1/126000의 배수로 보고된다.
- **라벨**: 성분 하나라도 AI가 생성했으면 파일 fake. 음성·음악이 모두 없으면 FILE=0.
- **외부 자원**:
  - AI-Hub 사용 허용(417844). 단 해외 반출 금지는 우리가 따로 고정한 제약이다.
  - NC 사전학습 모델 허용(417857), CC BY-NC-SA 데이터 허용(417212).
  - 자체 생성 데이터는 파일 제출 의무가 있다(417198).
  - 모델·설정 선택에 쓴 검증 데이터는 제출 대상이 될 수 있다(417811).
- **파일 독립성**: 평가 세트 전체 통계를 이용한 보정은 금지(417885). LB 프로빙은 권장하지 않음(417702).
- **실행 환경**: L4 22.4 GiB, 1,200파일 60분 이내, zip ≤10 GB, 오프라인, 설치 ≤10분, 하루 3회, Private=Public(최고점 유지).
- **마감**: 2026-09-29 10:00 KST(09:59:59).
- **확인하지 못한 것(UNKNOWN)**:
  - 가중치 공개일 조건이 있는지
  - SSLAM의 HF 공개일은 2025-09-29, DF-Arena는 2025-10-20이다(VERIFIED, HF API). 이미 제출한 S3가 DF-Arena를 쓰므로 공개일 조건이 문제된 적은 없다(REPORTED).

---

## 3. 실제 추론 계산 그래프 (S6 기준. ▲는 S7용 **계획으로, 런타임에는 아직 구현되지 않음**)

진입점은 `script.py`가 `full_bank_submission.run(ROOT)`을 부르는 것이다(VERIFIED: zip에서 꺼내 확인).

1. **`run`**
   - `manifest.json`의 모든 멤버 sha256을 검사하고, ffmpeg PATH를 설정한다.
   - `torch.set_num_threads(4)`, TF32와 cudnn.benchmark는 끈다.
   - `order_audio_files`로 sample_submission 순서를 따른다.
2. **`Engine.predict`** (파일마다 독립)
   1. `base.load_audio`: 16 kHz mono로 디코딩한다(baseline `script.py`).
   2. 피크를 0.6으로 제한한다(`preserve_gain=False`).
   3. float WAV 임시파일을 만들고 HTDemucs로 음성·반주를 분리한다. 각 스템은 원래 길이에 맞게 잘라내거나 0으로 채운다.
   4. 창 배치 `window_layout(n)`: WINDOW 64600, HOP 32300, 끝부분 창 하나 추가. `windows()`는 뒤를 0으로 채운다.
   5. **음악**:
      - 반주 스템의 창마다 MERT(24 kHz soxr 리샘플) hidden 6·12의 시간 평균 = 2048차원
      - EAT-large block 16의 patch 평균(Kaldi fbank 128, 400 frame) = 1024차원
      - ▲ SSLAM block 5 = 768차원
      - 이것들을 이어 붙인 뒤 `WindowMIL` head(seed 3개)를 거친다.
      - 파일 점수 = σ(logsumexp_w z_w − log T)을 seed 평균한 값
   6. **음성 DF-Arena**:
      - `get_segment_starts`: 겹치지 않는 64600 세그먼트에 끝부분 하나를 더한다. 짧으면 **반복해서 채운다(tiling)**.
      - fc5 입력(1280차원)을 hook으로 받아 MLP head를 거친다.
      - softmax[fake]의 세그먼트 최대값을 seed 평균한다.
      - 음성 스템이 무음(RMS<1e-5)이면 0.
   7. **음성 AD**:
      - `AntiDeepfake.windows`: 절반씩 겹치는 창. 짧은 파일은 **채우지 않는다**.
      - layer_norm → XLS-R-1B last hidden의 평균(1280차원) → ResMLP head
      - softmax[0]의 창 최대값을 seed 평균한다.
   8. ▲ **음성 WavLM**: 창마다 평균 0·분산 1로 정규화 → hidden 8의 시간 평균(1024차원) → MLP head → 창 최대값 → seed 평균.
   9. **VOICE**: `component_fusion.fuse`로 val CDF 가중 평균(ours 1, ad 0.5, ▲wavlm 1)을 구하고, ours의 역CDF로 되돌린 뒤 [0,1]로 자른다.
   10. **Presence**: PANNs tag 통계와 스템 에너지를 쓰는 로지스틱 head(`decision_heads.presence`).
   11. **FILE**: `decision_heads.file_score`. 후보 3개의 val CDF를 가중 평균한다(4절).
3. `validate_predictions`로 열·ID 순서·범위[0,1]를 검사하고 CSV로 쓴다.

**학습과 런타임의 일치 (VERIFIED)**
- 파일 10개로 CPU 오프라인 실행 결과를 학습 경로 출력과 비교했다. S5는 VOICE 2.6e-4 / MUSIC 7.8e-4, S6는 7.9e-4 / 7.8e-4 차이였다.
- 런타임 EAT 특징은 추출기 특징과 fp16 기준으로 차이가 0이다.
- **CUDA 런타임 실행과 L4 소요 시간은 측정하지 않았다(REQUIRES EXECUTION)**. EAT의 CUDA 연산은 H100 추출에서만 돌렸다.

---

## 4. 앙상블과 점수 결합 분석

### 4.1 정확한 식 (VERIFIED, 코드 기준)
- **음성 멤버 m, seed k**: p_{m,k,w} = softmax(h_{m,k}(x_w))[fake]
  - s_m = mean_k max_w p_{m,k,w}
  - 확률 평균이다(logit 평균이 아님). 창에 max, seed에 mean을 쓴다.
- **VOICE** = clip(F_ours⁻¹(Σ_m w_m F_m(s_m) / Σ w_m), 0, 1)
  - F_m은 val의 무음 아닌 행에서 만든 1001점 분위표이고, 파일 하나의 값만 쓰는 고정 함수다.
  - 순위 기반이라 멤버 점수의 척도와 무관하다.
- **MUSIC** = mean_k σ(logsumexp_w z_{k,w} − log T): 창 logit의 log-mean-exp, 즉 부드러운 max다.
- **FILE**:
  - 후보 c₁ = max(Pv·Fv, Pm·Fm)
  - 후보 c₂ = 1 − (1 − Pv·Fv)(1 − Pm·Fm)
  - 후보 c₃ = max(Fv, Fm)
  - FILE = Σ_j w_j G_j(c_j) / Σ_j w_j, 여기서 G_j는 후보별 val CDF다.
  - 가중치: S4·S6는 동일 가중(격자 이득 0.0006 < 0.002), S5는 격자 결과(max 2, max_nopresence 1, noisy_or 7; 이득 0.00201)다.

### 4.2 가정과 한계
- c₁과 c₂는 Fv·Fm을 확률로, presence와 서로 독립으로 취급한다. 그런데 Fv는 DF head 척도로 되돌린 값이라 보정된 확률이 아니다(HYPOTHESIS: 결합이 최적이 아님).
- 예시(측정값 아님): Pv=Pm=1, Fv=Fm=0.5이면 c₁=0.5, c₂=0.75, c₃=0.5. 후보가 같은 입력에 서로 다른 순위를 주므로, CDF 평균은 두 성분의 증거를 섞는 방식에 따라 FILE 순위를 바꾼다.
- **실측 관찰과 정정 (VERIFIED, RT held-out)**:
  - 초판에서는 실제 보컬 + AI 반주 86개의 평균을 근거로 FILE 결함을 주장했다(MUSIC 0.711 대 FILE 0.41~0.44).
  - **이 비교는 틀렸다.** FILE은 val CDF 척도(대략 균등)이고 MUSIC은 확률 척도라, 평균을 서로 비교할 수 없다.
  - 순위 기준으로 다시 보면(해당 부분집합 대 held-out의 모든 실제 파일 AUC):
    - 실제 보컬 + AI 반주: FILE **0.922**, MUSIC 0.904
    - 둘 다 가짜: FILE 0.987, MUSIC 0.966
    - AI 보컬 + 실제 반주: FILE 0.956, VOICE 0.960
  - 즉 FILE은 성분 증거의 순위를 유지하거나 약간 높인다. **결함 근거는 없다.**
- 선택 불안정: 0.002 기준 근처에서 가중치가 뚝 바뀐다(S5 0.00201). 규칙이 설계상 불연속이다.
- 음성은 창 max라서 긴 실제 파일의 점수가 올라간다(REPORTED: 출처 안 Spearman +0.22).

---

## 5. 감사 발견 사항 (중요도 순)

| # | 위치 | 관찰 | 결과 | 상태 | 최소 수정 | 필요한 검증 |
|---|---|---|---|---|---|---|
| 1 | `research/top15/work4/eval.py` val_completed | 규칙 L로 보완된 val 162행에 예측이 없어 0으로 채점 | L 판정의 한 조건이 무의미했음 | CONFIRMED STATIC DEFECT (수정함, `eval2.py`) | 보완 행 예측 따로 저장(`valc_*`), RT는 모든 행 예측 | 수정 후 base가 S5를 정확히 재현 (VERIFIED) |
| 2 | `ads.py` 계열(AC) 멤버 누락 시 0 | 라벨 없는 행을 0으로 채움(val: 음성 835행, 음악 2,394행) | FILE EER 약 0.004 낙관, 비교 대상 모두에 적용 | PLAUSIBLE RISK (크기 VERIFIED) | 판정은 RT 모든 행 예측으로 | S5·L·S7을 RT로 재계산해 결론 유지 (VERIFIED) |
| 3 | `work3/integration/fit_ft_fusion.py:bank_rows` | split이 'val'이 아니면 무조건 `bank_test.csv`를 읽어, 'train'이면 빈 목록 | 지금 호출하는 곳은 val·test뿐이라 판정 영향 없음. 제 분석 코드가 한 번 이 결함에 걸림(수정) | CONFIRMED STATIC DEFECT (잠재) | split → 파일을 명시적으로 매핑하고, 빈 결과면 예외 | 단위 검사 |
| 4 | `work4/unseen.py`(분석) | 반올림한 EER로 ADS 계산 | 4e-5 오차 | CONFIRMED STATIC DEFECT (수정함) | 반올림하지 않은 값으로 계산 | 재현 assert 통과 (VERIFIED) |
| 5 | `decision_heads.file_score` 결합 | (정정) 평균 비교는 척도 오류였다. 순위 기준으로 FILE AUC 0.922는 MUSIC 단독 0.904보다 높다. 다만 AUC 우위가 공식 EER 우위를 보장하지는 않는다(Codex 지적). fakeV+realM은 FILE 0.956 < VOICE 0.960 | 이 검사로는 결함 근거를 찾지 못함 | NO ISSUE FOUND WITHIN INSPECTED SCOPE (초판 주장 철회) | — | 같은 적격 음성군으로 공식 EER 비교 |
| 6 | `file_ensemble.fit` 선택 규칙 | 격자 이득 0.002 경계에서 불연속 전환 | 패키지마다 FILE 규칙이 달라짐 | PLAUSIBLE RISK | 경계를 없애거나 OOF 적합 | 가중치별 held-out 민감도 |
| 7 | FakeMusicCaps `VOICE_PRESENT` | HTDemucs vocal_ratio ≥0.35만으로 부여. 400개 중 약 48개가 캡션·PANNs 근거와 불일치 | presence head 라벨 잡음. CPS는 이미 포화(0.998) | PLAUSIBLE RISK | 캡션과 vocal_ratio가 모두 동의할 때만 1 | presence AUC 비교 |
| 8 | 음성 라벨 공백 | AI 노래 2,124행이 보컬은 있는데 VOICE_FAKE 미상이라 학습에 쓰이지 않음 | AI 가창 탐지 약화 | 해결함(S6 규칙 L) | `data_v4/label_rules.py` | RT held-out +0.0020, 처음 보는 생성기 +0.006 (VERIFIED) |
| 9 | 조합 커버리지 | 실제 보컬 + AI 반주가 train에 5행(held-out에 107행) | 해당 유형의 일반화를 확인할 방법이 적음 | PLAUSIBLE RISK | 7절 D1 | held-out 부분집합 |
| 10 | S7 특징 출처 | S6 특징은 fp16으로 저장된 스템(`dump_stems.py`)에서 뽑았고, 런타임은 fp32 스템 | 작은 차이 예상 | REQUIRES EXECUTION | — | S7 일치 검사(VOICE ≤0.02, MUSIC ≤0.01) |
| 11 | L4 실행 | S5·S6(EAT)의 L4 시간 미측정. S7은 추가로 약 10분 예상(ESTIMATE) | 60분 초과 위험은 낮음 | REQUIRES EXECUTION | — | 사용자가 측정 생략을 결정 |
| 12 | 인코더별 분할 차이 | DF는 반복 채움, AD는 채우지 않음, 나머지는 0 채움 | 학습·런타임은 일치. 제안 C에서 토큰 정렬이 필요 | NO ISSUE (현 경로) | — | — |
| 13 | mask 대 값 | mask=0이면 값도 비어 있음(42,039행 × 5종) | 미상 라벨이 채점에 섞이지 않음 | NO ISSUE FOUND | — | — |
| 14 | 행 정렬 | `score_rows` 1:1, `validate_predictions`와 `Boot.ads`가 ID 순서 검사 | — | NO ISSUE FOUND | — | — |
| 15 | 창 단위 정답(음악 MIL) | 부분 편집 행(HAIM 587)에만 있도록 설계. 창 개수 불일치 0건(13,970행) | — | NO ISSUE FOUND | — | — |
| 16 | 패키지 구성 | S5의 음성 융합 표가 S4와 바이트 동일. 수정 trainer가 S5·S6 파일을 바이트 단위로 재현 | — | NO ISSUE FOUND (VERIFIED) | — | — |
| 17 | held-out 재사용 | 여러 스터디가 같은 held-out을 봄 | 최고 성적 선택의 낙관 편향 | PLAUSIBLE RISK | 새 held-out은 없음. 판정마다 사전 잠금 | — |
| 17b | `component_fusion.fuse` 18행 | 가중치가 있는 멤버가 values에 없으면 **조용히 제외**한다(시간 폴백용으로 만든 동작이 남은 것) | 런타임이 멤버를 빠뜨려도 오류 없이 VOICE가 바뀐다 | CONFIRMED STATIC DEFECT (잠재, Codex 지적) | 가중치 멤버가 없으면 예외 | 누락 입력 단위 검사 |
| 17c | 규칙 L 이득의 일반화 | 처음 보는 생성기를 하나씩 뺀 민감도 검사에서 S6−S5가 lyria-3 pro를 빼면 −0.0008로 부호가 뒤집힘. S7 후보−S6은 모든 제외에서 +0.0074 이상 (VERIFIED) | L의 "일반화 개선" 주장은 기각. S7 이득은 생성기 제외에 강건 | PLAUSIBLE RISK | — | — |
| 18 | 운영 | VESSL이 GPU 유휴 시 워크스페이스를 자동 정지하는 것으로 보임(19:29 x1, 00:25 x4) | 전송 중단 | PLAUSIBLE RISK (운영) | 추출 직후 바로 받기 | — |

과거에 고친 결함(REPORTED):
- `fusion_fit`의 FILE 척도 오류로 생긴 가짜 +0.014
- 포화 head 때문에 VOICE가 1을 넘던 문제(clip 추가)
- S2a 점수 오독

---

## 6. 사전학습 가중치 후보 (VERIFIED: HF API 날짜·라이선스, DB 결과 / REPORTED: 워크플로 조사)

| 순위 | 체크포인트 | 공개(HF 생성일) | 라이선스 | 대상 | 상태·근거 |
|---|---|---|---|---|---|
| 채택 | `worstchan/EAT-large_epoch20_finetune_AS2M` @53b1251 | 2025-05-03 | MIT | 음악(block 16) | RT +0.0122 (S5) |
| 1 | `ta012/SSLAM_pretrain` @af71a4b | 2025-09-29 | MIT | 음악(block 5) | RT +0.0058/+0.0060, seed 42 악화. fp16 손실 없음. 0.18 GB |
| 2 | `microsoft/wavlm-large` @c1423ed | 2022-03-02 | HF 태그 없음. UniSpeech 저장소 CC BY-SA 3.0(REPORTED) | 음성(hidden 8) | RT +0.0034/+0.0037. fp16 손실 없음. 0.63 GB |
| 3 | `facebook/hubert-large-ll60k` @ff022d0 | 2022-03-02 | Apache-2.0 | 음성 | B 단계 val 비교(w 0.25/0.5/1)에서 WavLM에 밀림(정정) |
| 4 | `ta012/SSLAM_AS2M_Finetuned` @4500c5d | 2025-09-30 | MIT | 음악 | block 5 val 0.094로 pretrain(0.081)보다 약함 |
| 탈락 | `OpenMuQ/MuQ-large-msd-iter` | 2024-12-17 | CC BY-NC 4.0 | 음악·음성 | EAT보다 약함. 음성 멤버는 이득 없음 |
| 탈락 | `nii-yamagishilab/xls-r-1b-anti-deepfake` 미세조정(ftC1) | 2025-05-13 | CC BY-NC-SA 4.0 | 음성 | in-domain val 0.021이지만 held-out 0.128. 처음 보는 음성 생성기 EER 0.089 → 0.104 |
| 미시험 | BEATs iter3+ (공식 HF 없음, 미러만) | UNKNOWN | UNKNOWN | 음악 | 검증 에이전트: frozen ESDD에서 EAT보다 약함, 출처 확인 부담 |
| 미시험 | `mispeech/dasheng-base` | 2024-06-06 | Apache-2.0 | 음악 | 근거 부족 |
| 미시험 | `laion/larger_clap_music`, `openai/whisper-medium` | 2023·2022 | Apache-2.0 | — | 의미 중심 학습이라 위조 흔적이 억제될 가능성(HYPOTHESIS) |

**가장 싼 반증 실험**: 캐시된 특징으로 layer 선별 → 조합 → val 게이트 → held-out RT 순서로 한 번(인코더당 CPU 약 5분, 추출은 H100×4에서 20분). 오늘 적용한 절차 그대로다.

---

## 7. 아키텍처 제안 3가지

### A. 최소 변경: OOF 학습 FILE 결합기 (캐시만 사용)
- **다루는 약점 (HYPOTHESIS)**: 현재 FILE은 손으로 만든 후보 3개의 CDF 평균이라, 성분 점수와 presence의 상호작용을 학습하지 못한다.
  - 4.2절 정정 이후로는 부분집합 결함 근거가 없다.
  - 남는 근거는 선택 규칙이 0.002 경계에서 불연속이라는 점뿐이라, 우선순위를 낮춘다.
- **입력(파일당)**: 멤버 점수 s_df, s_ad, s_wavlm, MUSIC, Pv, Pm, voice_silent, log(T)
  - 모두 val CDF로 변환한다(척도 제거).
  - 출처를 드러낼 수 있는 특징(코덱, 길이 원값)은 넣지 않는다. log(T)는 길이 편향 진단용으로만 두고, ablation에서 빼 본다.
- **모델**: 로지스틱 회귀(상호작용 항 PvFv, PmFm 포함) 또는 깊이 3의 GBM. 출력은 FILE 확률.
- **학습 데이터**: train split을 group 기준 5-fold로 나눠 성분 head를 fold마다 다시 학습한다. 그렇게 얻은 **OOF 예측**으로 결합기를 학습한다(in-sample 스태킹 금지).
  - val은 결합기 선택용, held-out은 1회 보고용이다.
- **손실**: FILE_FAKE 가중 BCE. 클래스 × 출처 × 생성기 균형은 `loss_weights`와 같게 한다. FILE 라벨이 미상인 행은 제외한다.
- **더 단순한 기준선**: 현재 3후보 CDF 평균, 또는 c₃ 하나.
- **채택 기준**: held-out RT FILE EER 개선 → ADS +0.003 이상, P(>0) ≥ 0.9. 실제 보컬 + AI 반주 부분집합에서 FILE 순위 개선.
- **기각 기준**: OOF val에서만 개선되고 held-out은 평평. 또는 처음 보는 생성기 부분집합이 악화.
- **비용 (ESTIMATE)**: CPU 1~2시간(fold 재학습 5 × 멤버 3 × seed 3). 런타임 증가는 없다(로지스틱 계수만 패키징). NOT RUN.

### B. 중간 변경: 보완 표현(S7 = SSLAM 음악 + WavLM 음성 멤버)
- **계산 그래프**: 3절의 ▲ 부분.
  - 음악 특징 = [MERT h6, h12 | EAT b16 | SSLAM b5], 각 창의 시간 평균, 총 3840차원 → WindowMIL
  - 음성 = CDF 융합(DF 1, AD 0.5, WavLM 1)
- **학습 대상**: head만 학습하고 인코더는 고정한다. 기존 음성 head는 S6 것을 재사용하고, WavLM head는 원래 라벨로 학습한다(검증한 구성과 동일).
- **Ablation (VERIFIED)**:
  - SSLAM만: +0.0058
  - WavLM만: +0.0037
  - 둘 다: +0.0104 (RT, S6 기준)
  - S5 기준으로도 거의 같다(+0.0060 / +0.0034 / +0.0093).
- **위험**:
  - 음악 seed 불안정(seed 42 악화).
  - zip 9.73 GB로 10 GB 제한에 약 0.27 GB 여유(ESTIMATE, 가중치 fp16 무손실 저장 기준).
  - L4 시간 증가(ESTIMATE +10분).
- **남은 검증**: P1 학습 → 패키징 → CPU 검증 → 일치 검사. 가능하면 seed 5개로 안정성 확인(아래 E6). NOT RUN.

### C. 고위험: 스템 교차 창 토큰 Transformer (fallback = B)
- **다루는 약점**: 음성과 음악이 각자 따로 max·MIL로 모인 뒤 손 규칙으로 합쳐진다. 그래서 같은 시각의 음성·반주 증거(AI 커버, 부분 편집)와 긴 파일의 증거 누적을 함께 모델링하지 못한다.
- **토큰**: 절반씩 겹치는 창 w = 1..T (60초에서 T ≤ 29). 아래 인코더들은 같은 창 배치를 공유한다.
  - 음성 토큰 v_w = LN(W_v[AD_w(1280) ‖ WavLM_w(1024)]) ∈ ℝ²⁵⁶
  - 음악 토큰 m_w = LN(W_m[MERT_w(2048) ‖ EAT_w(1024) ‖ SSLAM_w(768)]) ∈ ℝ²⁵⁶
  - DF-Arena는 세그먼트 배치가 달라(겹치지 않음, 반복 채움) 토큰 대신 파일 수준 점수로만 붙인다.
- **위치 정보**: 창 시작 시각을 사인 임베딩으로 넣는다. 두 흐름이 같은 창 번호를 쓰므로 시각이 정확히 맞는다.
- **층 구성(2층)**:
  1. 흐름별 self-attention: d=256, head 4개, head당 64차원, dropout 0.1, pre-LN 잔차
  2. 교차 attention: 음악 쿼리가 음성 키·값을, 음성 쿼리가 음악 키·값을 본다. scaled dot-product 1/√64.
- **마스크**: 길이가 다른 파일은 `key_padding_mask`로 가린다(True = 패딩). 인과 마스크는 쓰지 않는다(오프라인 파일 분류라 불필요).
  - 음성 스템이 무음이면 음성 흐름 전체를 가리고, 교차 attention의 출력을 0 벡터로 대체한다. 전부 가려졌을 때 softmax가 NaN이 되는 것을 막기 위해서다.
- **Pooling**: 과제별 학습 쿼리 q_v, q_m, q_f 3개가 attention pooling(패딩 마스크 적용)을 한다.
  - 각 결과에 선형층을 붙여 VOICE·MUSIC·FILE logit 3개를 낸다.
  - FILE logit에는 DF-Arena 파일 점수(CDF)를 이어 붙인다.
- **손실**:
  - 마스크 적용 다중 과제 BCE: Σ_c 1[라벨 c 알려짐]·w_row·BCE(ℓ_c, y_c)
  - 행 가중치 = 기존 클래스 × 출처 × 생성기 × 길이 균형
  - 창 정답이 있는 587행(HAIM 부분 편집)에는 창 단위 보조 BCE ×0.25
- **학습 대상**: 투영층·transformer·쿼리·출력층, 약 1.5M 파라미터(ESTIMATE). 인코더는 고정하고 캐시를 입력으로 쓴다.
  - AdamW 1e-4, early stopping(val official ADS), seed 3개
- **비용 (ESTIMATE)**: CPU 또는 MPS에서 seed당 수십 분. 추론 증가는 무시할 수준(토큰 ≤ 29개).
- **위험**:
  - 출처 정체(코덱·길이)를 외울 수 있다. 처음 보는 생성기 부분집합과 출처별 slice로 감시한다.
  - attention 가중치를 설명으로 해석하면 안 된다.
- **Ablation**:
  - (i) 교차 attention을 빼고 흐름별 mean pooling으로 대체
  - (ii) 흐름 하나만 사용
  - (iii) B와 비교
- **기각 기준**: held-out RT ADS가 B + 0.005에 못 미치거나, 처음 보는 생성기 부분집합이 악화되면 기각한다.
- **상태**: 마감을 고려하면 이번 대회에서 실행할 가능성은 낮다. NOT RUN.

---

## 8. 데이터 커버리지와 확보 후보

### 8.1 커버리지 매트릭스 (VERIFIED, bank CSV)
| split | 행 | 그룹 | 출처 | fake 생성기 | 주요 조합 (음성+음악) |
|---|---|---|---|---|---|
| train | 30,658 | 13,304 | 44 | 346 | fakeV 15,243 · realV 6,052 · fakeM 2,234 · 미상V+fakeM 1,727 · realV+realM 1,395 · fakeV+fakeM 1,287 · fakeV+realM 641 · **realV+fakeM 5** |
| val | 4,931 | 2,818 | 37 | 241 | realV+fakeM 5 |
| test(held-out) | 6,450 | 2,730 | 18 | 134 | **realV+fakeM 107**, 처음 보는 fake 생성기 1,135 |

- 언어: 미상 18,117 · ko 12,263 · en 9,196 · 나머지는 모두 합쳐도 약 2,000.
- 가장 큰 빈칸은 **실제 보컬 + AI 반주(AI 커버)의 학습 데이터**와 **다국어 실제 음성**이다.

### 8.2 확보 후보 (상세 권리는 `PUBLIC_DATASET_SURVEY_20260927.md` 참고)
| 후보 | 채울 빈칸 | 상태 |
|---|---|---|
| **D1: AI 커버 합성**. 실제 보컬(VocalSet CC BY 4.0, CSD NC-SA, AI-Hub 465·473은 로컬 전용) + AI 반주(FakeMusicCaps·MusicGen·SONICS 반주 스템)를 4조합 모두 같은 처리로 믹스 | realV+fakeM 학습 | NOT RUN. 대회 규칙상 생성한 파일은 제출해야 한다(417198). 원본 단위로 split을 먼저 나누고 그 안에서만 믹스 |
| SingMOS-Pro (CC BY 4.0) | 가창 SVS·SVC | **실행함**: held-out +0.0002 (AC)로 탈락 (VERIFIED) |
| Music Arena 나머지 | 음악 생성기 다양성 | 과거 시범 +0.0004 (REPORTED) |
| MoM/CLAM (YuE, DiffRhythm 등) | 새 음악 생성기 | 권리 확인 전 보류 (REPORTED) |
| NEXDATA·Defined.ai 유료 | — | 공개 접근성 요건과 2차 전달 조건을 확인하지 못함. 마감 전 납품이 불확실해 권하지 않음 (UNKNOWN) |

---

## 9. 통제 실험 매트릭스와 연산 계획

| ID | 가설 | 개입 | 기준·통제 | 1차 지표 | slice | 채택 | 기각 | 자원 | 상태 |
|---|---|---|---|---|---|---|---|---|---|
| E1 | S7 패키지가 올바르게 만들어진다(성능 주장 아님, 정정) | 런타임 구현 + P1 + 패키징 | S6 | CPU 검증 + 일치 검사 | — | 일치 VOICE ≤0.02 / MUSIC ≤0.01, zip ≤10 GB | 불일치 | CPU 약 2시간 | NOT RUN (보류) |
| E2 | FILE 결합기가 FILE EER을 줄인다 | 제안 A | 3후보 CDF 평균 | held-out RT ADS | 실제 보컬 + AI 반주, 처음 보는 생성기 | +0.003, P(>0) ≥ 0.9 | held-out 평평 | CPU 1~2시간 | NOT RUN |
| E3 | SSLAM 이득이 seed 잡음이 아니다(진단 전용. 잠근 규칙 C를 대체하지 않음, 정정) | 음악 seed 44~48 추가 | 같은 seed의 대조군 | seed별 held-out 음악 EER | — | (판정용 아님) | — | CPU 약 10분 | NOT RUN |
| E4 | WavLM을 L 라벨로 학습하면 더 낫다 | 라벨만 변경 | E1의 WavLM | RT ADS | 처음 보는 생성기 | +0.002 | 그 외 | CPU 약 10분 | NOT RUN |
| E5 | 교차 스템 모델링이 이득이다 | 제안 C | B | RT ADS | 출처별 | B + 0.005 | 그 외 | CPU·MPS 수 시간 | NOT RUN |
| E6 | AI 커버 학습이 realV+fakeM을 개선한다 | D1 데이터 | S6 | 해당 부분집합 FILE·MUSIC EER | 4조합 모두 | 부분집합 개선, 나머지 비열등 | 다른 조합 악화 | 생성·추출 반나절 | NOT RUN |

실행을 마친 비교(VERIFIED):
- S5: RT +0.0122
- L: RT +0.0020
- S (SingMOS): +0.0006
- LS: +0.0026
- ftC1: −0.0040
- S6 + SSLAM + WavLM: +0.0104

**H100 계획 (ESTIMATE)**
- 특징 추출은 파일 단위로 독립이라, x4(샤드 12개)에서 거의 선형으로 빨라진다. 실측: S6 인코더 4종, 42,039파일에 약 20분.
- head 학습은 CPU로 수 분이면 끝나서 GPU가 필요 없다.
- 미세조정은 DDP x4(전역 배치 128창)로 772 step에 12분이 걸렸다(VERIFIED). 하지만 in-domain 과적합으로 기각했다. 더 큰 배치나 GPU를 늘리는 것은 일반화 문제를 해결하지 못한다(갱신 횟수만 줄어든다).

---

## 10. 기각한 대안과 남은 질문
- **기각 (VERIFIED/REPORTED)**:
  - 음성 부분 미세조정(ftC1), 전체 미세조정(FullFT8 −0.10), 음악 EAT 전체 미세조정(in-domain 0.007 / test 0.249)
  - 깊은 head(held-out −0.0002), SLS 층 가중
  - SONICS·zero-shot 멤버, AntiDeepfake-2B
  - 노래 인식 VOICE(HAIM에서만 효과)
  - AI-Hub 통화 실음성 추가(LB −0.0012), Music Arena 추가(+0.0004), SingMOS 추가(+0.0002)
  - 길이 강건 pooling(ADS 잡음 수준)
- **남은 질문 (UNKNOWN)**:
  - 공식 test의 구성(AI 커버 존재 여부, 언어·길이 분포)
  - 대기 중인 제출의 정체와 점수
  - S5·S6·S7의 L4 소요 시간
  - 가중치 공개일 조건이 있는지(SSLAM 2025-09-29)
  - 일일 초기화 시각
  - AI-Hub 2차 제출 절차

## 11. 1차 출처와 한계
- 공식: https://dacon.io/competitions/official/236749/overview/description , /overview/rules
  - 운영진 답변 417844, 417857, 417212, 417198, 417811, 417885, 417702. 링크는 조사 문서에 있고, 이번에 다시 열지 않았다.
- 가중치: HF API(`/api/models/<repo>`)로 생성일·sha·라이선스를 이번에 조회했다.
- 한계:
  - held-out은 여러 번 참조됐다.
  - 처음 보는 생성기 부분집합은 생성기 6종뿐이다.
  - L4·CUDA 런타임은 실행하지 않았다.
  - LB와 held-out 사이 간격(약 0.09)의 원인은 확인하지 못했다.

---

## 마무리
- **먼저 해결할 정확성 문제**: 판정 도구의 AC 기본값 0 문제. 앞으로의 판정을 RT 방식으로 바꾼다.
  - 오늘 결정은 재검증 결과 모두 유지됐다.
  - `fit_ft_fusion.bank_rows`의 잠재 결함(split 'train'이면 빈 목록)을 고친다.
  - 기준: 수정 후 기존 수치 6개가 정확히 재현될 것.
- **첫 가중치 실험**: E3(SSLAM seed 확장).
  - 기준: seed 5개 중 4개 이상에서 held-out 음악 EER이 개선되면 S7 음악을 유지한다.
- **첫 아키텍처·앙상블 실험**: E3(SSLAM seed 확장)을 먼저 하고, 그다음 E2(OOF FILE 결합기)를 한다.
  - E2 기준: RT held-out +0.003 이상, P(>0) ≥ 0.9.
  - E2는 4.2절 정정으로 결함 근거가 사라져 우선순위가 낮다.
- **첫 데이터 추가**: D1(AI 커버 4조합을 같은 처리로 합성).
  - 기준: 해당 부분집합이 개선되고 나머지 조합이 나빠지지 않을 것.
  - 생성한 파일은 제출 의무가 있다.
- **fallback**: 검증을 마친 최선 패키지 S6. 대회 최고점은 LB 기준 S3(0.7853)가 이미 확보돼 있다.

---

## 12. 외부 검토: Codex GPT-6-Astra (ultra) 2라운드 토론
- **방식**: 보고서, 결과 JSON, 잠금 규칙, 핵심 코드만 담은 읽기 전용 폴더에서 `codex exec -m gpt-6-astra -c model_reasoning_effort="ultra" -s read-only`로 실행했다. AI-Hub 데이터, 캐시, 키는 넣지 않았다.
  - 기록: `research/top15/work4/codex_debate/`
- **Codex가 찾은 보고서 오류(모두 반영)**:
  - S7 런타임 미구현
  - `fuse` 누락 멤버 조용한 제외
  - 생성기 수와 행 수 단위 혼동
  - DB 행 4,822
  - HuBERT 탈락 단계
  - E1·E3의 성격(각각 패키징 검사, 사후 기준 금지)
  - FILE 평균 비교의 척도 오류(Claude도 독립적으로 발견)
- **Codex 제안 검사 결과 (VERIFIED)**: 처음 보는 생성기를 하나씩 뺀 민감도 검사
  - S6−S5는 lyria-3 pro를 빼면 −0.0008로 부호가 뒤집혔다. L의 일반화 주장은 기각한다.
  - S6+both−S6은 모든 제외에서 +0.0074 이상이었다.
  - S6+both−S5+both는 모든 제외에서 +0.0013 이상이었다. bootstrap +0.0031 [+0.0003, +0.0059], P(>0)=0.988.
- **합의한 결론**:
  1. 09-28 제출 순서: **S5 → S6 → S4**. S5의 이득(+0.012)은 강건하고 L은 불안정하다. 남은 슬롯은 다음 날로 넘어가지 않는다.
  2. **S7은 탐색 제출로 만든다.** 사전 잠금 규칙 C는 통과하지 못했다는 사실을 유지한다. 범위는 현재 구성의 런타임 구현, P1, 패키징, 검증으로 고정하고, 추가 seed나 구조 탐색은 하지 않는다. 검증을 끝내지 못하면 제출하지 않는다.
  3. S7의 음성 쪽은 S6 head(L)를 쓴다(위 직접 비교).
  4. CPU 일치 검사로는 L4 메모리와 60분 제한 충족을 입증할 수 없다(측정은 사용자가 생략하기로 결정).
- **남은 이견**: 없음. Codex는 "AUC 우위는 결함 부재의 증거가 아니다"라는 표현을 요구했고, 반영했다.
