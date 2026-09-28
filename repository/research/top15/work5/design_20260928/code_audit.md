# DACON 236749 — S7 / s7_new 실행 경로 코드·수치 감사 (coding session 1, 읽기 전용)

- 작성 시각: 2026-09-28 21:40 KST 무렵. 대상: LB 최고 패키지 **S7** (`submissions/DataV4_S7_SSLAMWavLM`, sha256 `5239c912…`)와 계획 패키지 **s7_new** (`submissions/DataV4_s7_new`, sha256 `fcdd8691…`).
- 방법: 코드 정독, 실제 zip 멤버 해시 대조, 체크포인트 메타데이터(`train_ids`) 교집합 계산, 기존 JSON/CSV/npz/pt만 쓰는 CPU 재계산. 학습, GPU, 네트워크, 오디오 읽기는 없었다. 보조 스크립트 두 개(`_parity_ablation.py`, `_scale_check.py`)는 이 폴더에 두었고, 프로젝트 루트에서 `PYTHONDONTWRITEBYTECODE=1`로 실행했다. 실행 전후 `git status`는 같았다(기존 미추적 파일 `research/top15/work5/newdata/bcast_rows.py` 하나뿐).
- 증거 라벨: **VERIFIED**(이번에 코드나 수치로 직접 확인), **REPORTED**(패킷·영수증에 기록된 값을 옮김), **HYPOTHESIS**, **UNKNOWN**, **NOT RUN**. 발견 분류: **CONFIRMED defect / PLAUSIBLE risk / HYPOTHESIS**.
- 경로 약칭: `RT7` = `review_packet_v1/runtime_pkg/DataV4_S7_SSLAMWavLM/model/runtime/`, `RTN` = `review_packet_v1/runtime_pkg/DataV4_s7_new/model/runtime/`. 그 밖의 경로는 프로젝트 루트 기준이다. 패킷의 `code/` 사본 31개는 프로젝트 루트 파일과 바이트 단위로 같다(VERIFIED, `cmp`). 패킷 `runtime_pkg`의 40개 파일도 두 zip의 멤버와 해시가 같다(VERIFIED).

---

## 0. 영향 순 요약

| # | 분류 | 요지 | 영향 추정 |
|---|---|---|---|
| F1 | CONFIRMED (평가 타당성 결함, 코드 버그 아님) | 패키지 자체의 held-out 추정치가 없다. s7_new 패키지는 E1·E2·newval 행으로 **학습**했고, 로컬 세트는 모두 선택에 쓰였다. manifest의 “E2 +0.0285”는 newval/newtest를 학습하지 않은 다른 모델(DB 구성)의 수치다. | 슬롯 결정 근거가 전부 개발용 증거. 크기는 UNKNOWN |
| F2 | PLAUSIBLE risk (수치 VERIFIED) | 음악 헤드는 음악이 있는 행만으로 학습했는데 런타임에서는 **모든 파일**을 채점하고, 그 점수가 `max_nopresence`와 `noisy_or`로 FILE에 들어간다. E1에서 음악 없는 **진짜** 파일의 Fm q90은 0.469, 가짜는 0.035로 방향이 거꾸로다. | 오라클 상한: E1 FILE EER 0.0741→0.0716 (0/9/1 기준, 약 −0.0025) |
| F3 | CONFIRMED defect (평가 코드, 과거 결정) | `research/top15/work2/backbones/ads.py`는 빠진 멤버 점수를 0으로 채운다. S6 연구의 E1 4,822행 중 2,673행은 Fm=0, 981행은 음성 멤버 전부 0이었다. 런타임과 다른 의미론이다. | S6 시기 FILE 수치가 낙관적(F2의 오라클과 같은 효과). 현행 RT 평가기에서는 고쳐짐 |
| F4 | CONFIRMED (설계 성질) / 영향 PLAUSIBLE | FILE 규칙은 척도에 의존한다. Fv(‘ours’ 척도로 역사상)와 원시 Fm을 max/noisy_or로 섞는다. Fv에 단조변환만 가해도 VOICE EER은 그대로인데 FILE EER이 변한다. S8은 FILE 가중치까지 1/1/1로 바뀌어 LB의 S7↔S8 비교가 교란돼 있다. | E1 ±0.0005 수준, val은 최대 +0.012 |
| F5 | CONFIRMED gap → 이번 재실행으로 해소 | s7_new의 `runtime_parity.txt`는 FILE 검사와 영수증 결속이 없는 구버전 스크립트로 만들었다. 엄격판 로직으로 재실행한 결과 FILE 1.1e-4, VOICE 1.8e-4, MUSIC 8.9e-4로 PASS. CUDA 검증 영수증은 없음(UNKNOWN). | 낮음 |
| F6 | CONFIRMED gap (데이터 수용) | 새 데이터 40,222행 전부 `audio_sha256`/`source_sha256`가 빈 값이다. 내용 기준 중복·동일성 검사를 할 수 없고, 추출 실패 8행은 조용히 빠졌다. | 누수 검사가 메타데이터에만 의존 |
| F7 | CONFIRMED (평가 성질) | 단일 클래스 슬라이스(realV+fakeM: old val 5행, E2 0행)와 출처 교란. E2 데이터셋 9개 중 7개는 FILE이 단일 클래스다. `Boot`는 NaN을 “>0 아님”으로 세서 p 값이 0.00으로 찍힌다. | 비교 해석 오류 가능 |
| F8 | PLAUSIBLE risk (측정상 작음) | 학습 캐시는 fp16 특징(및 fp32 AD 가중치로 추출했을 가능성), 런타임은 fp32 특징에 fp16 저장 AD 가중치를 쓴다. SSLAM/WavLM의 stem fp16 반올림은 학습과 일치한다. | parity 오차 ≤ 2.3e-3 |
| F9 | PLAUSIBLE risk | s7_new는 도메인이 다른 40k행을 더했지만 조기종료는 old val(은행 데이터)만 본다. music seed 41은 epoch 6에서 멈췄다. | UNKNOWN |
| F10 | PLAUSIBLE latent risk | 런타임의 선택적 JSON 누락은 조용한 강등으로 이어진다(현 zip에서는 해시 게이트 때문에 발동 불가). 파일 하나만 실패해도 전체 제출이 실패한다. `ad`/`wavlm` 헤드는 유한성 검사가 없다. `_label`은 `*_mask`를 무시한다. | 현 패키지에는 낮음 |
| F11 | CONFIRMED (기록 결함, 낮음) | `run_identity.json`의 `label_completion: false`, `train_with_backbone: 37108`이 실제(재사용 L 헤드, 58,275/27,300행)와 다르다. `validation_metrics.json`은 in-sample인데 그렇게 표시돼 있지 않다. | 해석 오류 |
| F12 | CONFIRMED (QA 스크립트 결함, 낮음) | `research/top15/work5/newdata/parity.py`는 presence **헤드 출력**과 **legacy PANNs max**라는 서로 다른 양을 비교한다. | presence parity 줄은 의미 없음 |
| F13 | HYPOTHESIS (낮음) | WavLM 멤버 점수가 포화된다. s7_new val 표의 29.1%가 1−1e−6 이상이다. DB 실행에서 포화는 가짜에서만 나와 EER 영향은 측정되지 않았다. | 측정상 0 |

과거 우려 5가지의 현재 상태는 2절에 요약했다. 한 줄로 줄이면 이렇다. (1) 반복 held-out 선택은 여전하다(F1). (2) 단일 클래스 슬라이스 처리는 `s8_eval`/`new_eval`에서는 올바르지만 부트스트랩 보고에는 결함이 있다(F7). (3) 척도 혼합은 FILE 층에 남아 있다(F4). (4) 제로필은 런타임에 없고, 과거 평가기 `ads.py`에만 있다(F3). (5) SSLAM과 WavLM은 실제 출력에 들어간다(VERIFIED, 2.5절).

---

## 1. 실제 실행 경로 추적 (S7 = s7_new 공통 경로, 차이는 4절)

### 1.1 데이터 수용 (admission)
- 은행 행: `data_v4/train_backbone_heads.py:39`의 `BANK`(`results/data_v4_20260923/.../data/bank`). CSV를 곧바로 읽고(`:296-297`) `full_bank_features.load_rows`의 컴포넌트 누수 검사(`full_bank_features.py:66-69`)는 거치지 않는다. 다만 lineage 키 교집합을 직접 계산해 보니 old val ∩ (bank train, bank test, new rows)은 `component_id`/`group_id` 모두 0이었다(VERIFIED).
- P1 패키지 학습 행: `final()`에서 `rows['train'] = bank train + bank test` (`:349`). 새 행은 `NEW_VAL=1`이 아니면 split과 무관하게 **전부** train에 들어간다(`:369-372`). 여기에는 `newval`과 `newtest`도 포함된다. 특징 npz가 없는 새 행은 조용히 빠지고 개수만 출력된다(`:61-73`). 로그에는 `new rows 40214 without features 8`(REPORTED, `_workspace/final_s7_new_20260928.log`).
- 특징 파일이 없는 은행 행도 `items_for`에서 조용히 빠진다(`:182-185`). 그런데 체크포인트 `train_ids`가 라벨 규칙으로 계산한 기대 행 수와 정확히 같았다. 음악 12,218, L 음성 31,126, 원라벨 음성 29,762로 은행 행 누락은 0이다(VERIFIED).
- 새 행의 `audio_sha256`/`source_sha256`는 생성 단계에서 빈 문자열로 들어간다(`research/top15/work5/newdata/atadd_rows.py:56`, `open_rows.py:42`). 40,222행 전부 해당한다(VERIFIED, F6). 은행 행은 추출 전에 해시를 확인한다(`full_bank_features.py:170`).

### 1.2 라벨과 마스크
- `full_bank_train.head_labels` (`full_bank_train.py:34-41`): route 헤드의 라벨은 **마스크를 보지 않고** `float()`로 읽는다(`:37`). 다른 헤드는 `_known`(값 ∈ {0,1}이고 mask ≠ 0)이면서 해당 성분 PRESENT == '1'일 때만 쓴다(`:38-40`). bank trainval/test와 new_rows 전 행에서 route 라벨은 항상 알려져 있고, route 성분 PRESENT는 항상 1이었다. 마스크 0인데 값이 0/1로 채워진 셀도 없었다(VERIFIED). 따라서 현재 데이터에서는 마스크 우회가 일어나지 않는다.
- 비-route 음성 행은 음성 stem이 무음이면 제외된다(`train_backbone_heads.py:215`, `wavlm_items:149`). 무음 판정 기준은 `calculate_rms(voice) < 1e-5`로, 캐시와 런타임이 같다(`assets/baseline/script.py:42,336`).
- 라벨 완성 규칙 L (`data_v4/label_rules.py:16-26`): P1의 train+test 37,108행 중 1,364행의 `VOICE_FAKE`를 1로 채운다(SONICS 638, FakeMusicCaps 297, HAIM 262, Echoes TTA 60, suno 18. VERIFIED). val의 완성 가능 행 237개는 unknown으로 남는다. 새 행은 완성 이후에 붙기 때문에 완성 대상이 아니다(`:352-355` → `:369-372`). S7과 s7_new 모두 `ours`/`ad` 헤드는 L 실행에서 재사용했고, WavLM과 음악 헤드는 원라벨로 학습했다(체크포인트 행 수 차이 1,364로 VERIFIED).
- 손실 축약 (`full_bank_train.py:96-117`): 파일 BCE × 행 가중치의 합을 **step 배치 크기**(256)로 나눈다(`:108`). 음악 창 단위 보조 손실은 0.25 × 알려진 창의 BCE 합 / 배치 전체의 알려진 창 수다(`:101,109-110`). 음성은 창 단위 로짓 차이(odds)의 최대값을 파일 로짓으로 쓴다(`:89-92`). 비유한 손실은 예외를 던지고(`:111-112`), 기울기 비유한도 예외로 처리한다(`:115`). 행 가중치는 `weighting.weights` 's7' 체계를 쓴다. `loss_weights`로 (라벨, source, generator) 셀을 균등화하고 `duration_balance`를 곱하며(`data_v4/weighting.py:84-92`, `full_bank_batches.py:27-54`), 클래스별 총질량 N/2를 유지한다.

### 1.3 전처리 (런타임 `RT7/full_bank_submission.py`)
- 디코드: `librosa.load(sr=16000, mono=True)`로 스테레오를 평균하고, 비유한 값이나 빈 오디오면 예외를 던진다(`assets/baseline/script.py:144-150`). 패키지에 든 `bin/ffmpeg`를 PATH 맨 앞에 둔다(`RT7:229-231`).
- 피크 캡: `peak > .6`이면 0.6으로 스케일한다(`RT7:132-134`). 학습 추출기도 모두 같은 캡을 쓴다(`data_v4/extract_backbones.py:61-63`, `extract_new_backbones.py:152-154`, `data_v4/dump_stems.py:48-50`, `dump_features.py:40-42`). `full_bank_features.py:174-176`의 `input_kind` 예외는 은행 CSV에 그 열이 없어서 발동하지 않는다(VERIFIED: 열 목록).
- HTDemucs: 캡을 씌운 raw를 FLOAT wav로 쓴 뒤 `torchaudio.load`와 `convert_audio`(16k→44.1k, 모노를 2채널로 복제)를 거친다(`mert12_submission.py:109-120`). 파일 단위 mean/std로 정규화하고 `apply_model(shifts=0, split=True, overlap=.25)`을 돌린 다음, vocals는 채널 평균, 나머지 합은 backing으로 삼아 16k로 재표본화한다(`script.py:264-308`). std < 1e-8이면 두 stem 모두 0이다(`:272-275`). 이후 `len(raw)`에 맞춰 자르거나 패딩한다(`RT7:139-140`).
- 창 배치: MERT/EAT/SSLAM/WavLM/AD는 64,600 샘플 창, hop 32,300, 정확한 꼬리 창을 쓰고 짧으면 오른쪽을 0으로 채운다(`mert12_submission.py:47-64`, `antideepfake_runtime.py:92-107`). DF-Arena와 PANNs는 비중첩 64,600 창과 꼬리 창을 쓰고, 짧으면 타일링한다(`script.py:157-175`).
- fp16 반올림: SSLAM 입력 backing과 WavLM 입력 voice에만 `fp16_round`를 적용한다(`RT7:25-27,151,179`). 학습 쪽 SSLAM/WavLM은 fp16으로 저장된 stem에서 추출했으므로(`dump_stems.py:58` → `research/top15/work2/s6/extract_s6.py:104-108`) 일치한다. EAT, MERT, AD, DF-Arena는 양쪽 모두 fp32 stem이다.

### 1.4 특징 추출 (학습 캐시 ↔ 런타임)
| 멤버 | 학습 특징 출처 | 저장 정밀도 | 런타임 | 일치 여부 |
|---|---|---|---|---|
| MERT h6,h12 (backing) | `extract_backbones.py:75-79` (배치 8, soxr_hq 24k) | fp16 | `RT7:141-149` (배치 8, fp32) | 연산 경로 같음, 특징 정밀도만 다름 |
| EAT-large b16 | `extract_new_backbones.py:91-110` | fp16 | `RT7:186-207` | 같은 fbank와 정규화, 특징 정밀도만 다름 |
| SSLAM-pt b5 | `extract_s6.py:35-54` (fp16 stem) | fp16 | `RT7:150-152` (fp16_round stem) | 같음, 특징 정밀도만 다름 |
| WavLM-L h8 (voice) | `extract_s6.py:57-66` (fp16 stem) | fp16 | `RT7:209-217` | 같음, 특징 정밀도만 다름 |
| AD-1B pooled (voice) | `extract_backbones.py:48,70-71` | fp16 | `RT7:176` (`detector.windows`) | 공용 함수. 가중치는 패키지가 fp16 저장본(F8) |
| DF-Arena fc5 입력 | `full_bank_features.py:198-202` | fp32 | `RT7:157-161` | 같음 |
| presence 특징 | `full_bank_features.py:204-206` | JSON | `RT7:164-168` | 같음(캡 적용 raw와 voice) |
| 새 행 전부 | `dump_features.py:36-68` (S7 패키지 Engine 자체) | fp16 (DF는 fp32) | — | 런타임과 같은 코드 |

### 1.5 학습 (`data_v4/train_backbone_heads.py final()`)
- 헤드만 학습하고 인코더는 캐시 특징으로만 들어온다. 인코더가 그래프에 없으므로 동결이 구조적으로 보장된다. 옵티마이저는 헤드 파라미터만 받는다(`:238`).
- 음악: `WindowMIL(mean, scale)` (`:229-231`). `FrozenHead(3840)`의 은닉 128, 창 로짓의 log-mean-exp. 입력 폭 3,840 = MERT 2×1,024 + EAT 1,024 + SSLAM 768 (VERIFIED, 체크포인트 shape).
- 음성: `voice_heads.build` (`:232-234`). `ours`는 mlp 1280→256→2, `ad`는 resmlp(사전학습 Linear에 0으로 초기화한 분기를 더함), `wavlm`은 mlp 1024. ResMLP는 마지막 층 가중치가 0이라 step 0에서 분기 첫 층의 기울기가 0이지만, 마지막 층이 먼저 갱신되면 풀리는 정상 동작이다.
- seed 41/42/43의 학습 행 집합은 같다(VERIFIED). s7_new는 `head_device: "cuda:0"`로 fp32 학습했다(`full_bank_train.py:104`의 autocast 조건이 `device == 'cuda'`라서 'cuda:0'에서는 꺼진다). S7의 `run_identity`에는 `head_device` 기록이 없다(구버전 코드이고 기본값은 'cpu', UNKNOWN). 어느 쪽이든 bf16 autocast는 적용되지 않는다.

### 1.6 체크포인트 선택
- 매 epoch old val에서 (EER, BCE)를 사전식으로 비교해 최선 상태를 복원한다(`train_backbone_heads.py:241-258`). patience는 음성 10(150 epoch), 음악 5(50 epoch)다. val 행은 옵티마이저에 들어갈 수 없다(`full_bank_train.py:98`, `full_bank_batches.py:22`).
- 선택된 epoch(VERIFIED, 체크포인트): S7 music 15/17/14, wavlm 33/54/81. s7_new music 6/16/12, voice 13/19/48, wavlm 43/41/53.

### 1.7 보정과 융합
- VOICE: `component_fusion.fuse` (`RT7/component_fusion.py:14-28`). 멤버별 val CDF의 가중평균을 구하고(1 / 0.5 / 1), 이를 `ours` val 표의 역CDF로 다시 `ours` 척도에 사상한 뒤 [0,1]로 clip한다. 표는 `ours > 0`(무음이 아닌) val 행으로 적합한다(`train_backbone_heads.py:529-538`). 런타임이 무음이 아닌 모든 파일을 융합하는 것과 모집단이 일치한다. 가중치를 가진 멤버가 빠지면 `KeyError`가 난다(`:19-21`).
- FILE: `decision_heads.file_score` (`RT7/decision_heads.py:88-98`). 후보 `max`, `noisy_or`, `max_nopresence`(`:70-75`)의 val CDF를 가중평균한다. 가중치는 max 0, noisy_or 9, max_nopresence 1이다. S7은 val 격자로 골랐고(`file_ensemble.py:37-52`), s7_new는 `--file-weights 0,9,1`로 고정했으며 s7_new 자체 val 격자의 최선은 1/1/8이었다(REPORTED `run_identity.json`). FILE 표는 **융합 후** Fv, 원시 Fm, presence 헤드 Pv/Pm으로 적합한다(`train_backbone_heads.py:539-551`). 런타임 입력과 일치한다.
- presence: `presence_head.json`(로지스틱, train split만으로 적합, `data_v4/presence.py:95-97`)은 S7과 s7_new에서 바이트까지 같고, 학습에 쓴 `H.PRESENCE`와도 같다(VERIFIED).

### 1.8 패키지 추론에서 submission.csv까지
- `run()` (`RT7:220-253`): manifest의 candidate 상수를 확인하고 manifest에 적힌 78개 멤버의 sha256을 모두 검증한다(`:223-228`). TF32와 cudnn.benchmark를 끄고(`:236-238`) sample_submission 순서대로 파일을 **하나씩** `Engine.predict`에 넣는다(`:244`). 이어 `validate_predictions`로 열, ID 순서, 유한성, [0,1] 범위를 확인하고(`graph_scoring.py:48-78`) CSV를 쓴다. 예외를 잡는 코드는 없다. 파일 하나가 실패하면 전체가 중단된다(F10).
- 파일당 출력: MUSIC은 seed 3개 평균이고 무음 검사가 없다(`RT7:153-156,171`). VOICE는 무음이면 0이고, 아니면 `fuse(ours, ad, wavlm)`다(`:169-181`). PRESENT는 presence 헤드에서 온다(`:168`). FILE은 `file_score`다(`:182`).

---

## 2. 과거 우려 5가지의 현재 상태

### 2.1 반복 held-out 선택 → **여전히 유효 (F1)**
| 세트 | 크기 | 쓰인 곳 | 결론 |
|---|---|---|---|
| old val (은행 val) | 4,931 (FILE 4,927) | 모든 헤드의 조기종료, 음성 CDF 표, FILE CDF 표와 가중치 격자(S7 0/9/1 선택), `validation_metrics.json`, `ads.py`·`s8_eval.py`의 val 단계 선택, FILE 가중치 결정(S8 1/1/1) | in-sample. 개발용 |
| E1 (bank test, DB 고정 4,822) | 4,822 | S6 연구 held-out(`ads.py`). `rt_s6enc`에서 S7 채택: manifest에 “fails the per-seed lock rule, RT held-out +0.0104 vs S6”(VERIFIED, S7 manifest). `s8_eval heldout`, `new_eval`(s7_new vs s7_ref +0.0064), `select_newval` 거부 조건. **S7과 s7_new 패키지의 학습 데이터**(음성 5,139, 음악 2,855행) | 개발용. 패키지 평가 불가 |
| E2 (newtest) | 12,426 | `new_eval`·`virtual_configs`(구성 7개 이상), `leak_check`, `e2_subsets`, `select_newval` 거부 조건. **s7_new 패키지의 학습 데이터**(음성 8,186, 음악 4,663행) | 개발용. 패키지 평가 불가 |
| newval | 3,841 | W_newB 조기종료, round-16 mix 선택(`select_newval.py:1-3,39-41`). **s7_new 패키지의 학습 데이터**(음성 2,445, 음악 1,584행) | 개발용 |
| LB (public = private) | 1,200 | S3~S8 비교(S7 최고, S8은 낮음) | 적응적. S7↔S8은 교란됨(F4) |

### 2.2 단일 클래스·전부 양성 슬라이스 → **평가기 일부는 올바르고 보고 코드에는 결함 (F7)**
`graph_scoring.official_metrics`는 단일 클래스를 `invalid`로 표시한다(`:155-157`). `s8_eval.fpr_guard`는 realV+fakeM에 대해 val TPR95 임계값에서의 FPR을 쓰고(`research/top15/work5/s8/s8_eval.py:143-157`), `new_eval.per_source`는 단일 클래스일 때 FPR/TPR로 바꾼다(`new_eval.py:74-92`). 둘 다 올바르다. 반면 `fit_ft_fusion.Boot`와 `ads_of`에는 결함이 있다(F7).

### 2.3 서로 다른 척도의 비교 → **FILE 층에 남아 있음 (F4)**
VOICE 융합 자체는 CDF를 쓰므로 척도에 불변이다. 그런데 결과를 `ours` 척도로 되돌린 뒤 FILE 후보(max, noisy_or)에서 원시 Fm과 비교하기 때문에 FILE은 척도에 의존한다. FILE 표와 가중치를 융합 후 값으로 적합했으므로 **학습과 런타임 사이에는 일관된다**(1.7). 문제는 규칙이 척도에 따라 달라진다는 점이다.

### 2.4 빠진 멤버와 제로필 → **런타임에는 없음. 과거 평가기에만 있음 (F3)**
런타임은 모든 파일에서 모든 멤버를 계산하고(`RT7:175-181`, 주석에서도 rule 2-4를 언급) 누락 시 예외를 던진다(`component_fusion.py:19-21`). try/except나 기본값 대체는 없다. 선택적 JSON이 빠졌을 때의 조용한 강등만 잠재 위험으로 남는다(F10). RT 평가기인 `rt_s6enc.py:66`, `s8_eval.py:51-53`(누락 assert), `new_eval.py:57-58`(엄격한 인덱싱)은 제로필을 하지 않는다. `research/top15/work2/backbones/ads.py:67,73`에는 제로필이 남아 있다.

### 2.5 SSLAM과 WavLM이 실제 출력에 들어가는가 → **들어간다 (VERIFIED)**
- 설정: `music_sslam_layers.json` = [5], `voice_wavlm_layer.json` = [8]. `component_fusion.json`의 voice 가중치는 {ours 1, ad 0.5, wavlm 1.0}이고 wavlm 표가 1,001점이다. 두 zip 모두 SSLAM/WavLM 자산과 `wavlm_4x.pt`를 담고 있고, 모두 manifest 해시로 결속돼 있다(VERIFIED).
- 코드 경로: `eat_family`에 SSLAM이 들어가 음악 특징에 이어 붙는다(`RT7:102-104,150-152`). `self.wavlm`은 fusion 가중치가 있을 때만 로드되는데(`:107-118`) 가중치가 1.0이므로 항상 로드되고, 무음이 아닌 모든 파일에서 `members['wavlm']`이 계산된다(`:178-180`).
- 수치(10개 parity 파일, CPU, `_parity_ablation.py`):
  - 음악 헤드 첫 층의 제곱노름 중 SSLAM 열 비중은 S7 21.9%, s7_new 22.3%다. SSLAM 블록을 학습 평균으로 바꾸면 MUSIC은 최대 0.304(S7)/0.379(s7_new), FILE은 최대 0.077/0.030 변한다.
  - WavLM 멤버를 빼고 같은 표로 융합하면 VOICE는 최대 0.706(S7)/0.333(s7_new), FILE은 최대 0.107/0.065 변한다. 예를 들어 S7에서 `ctrsvdd_…2454`는 VOICE 0.2791이 0.9855로 바뀐다.
  - 이것은 민감도 측정이지 재학습 절제(ablation)가 아니다.

---

## 3. 추가 점검 항목

| 항목 | 결과 | 근거 |
|---|---|---|
| unknown 라벨 마스크 | 현 데이터에서는 안전(VERIFIED 1.2). 잠재 위험은 `head_labels`의 route 라벨과 `_label`(`graph_scoring.py:81-88`)이 `*_mask`를 보지 않는다는 점(F10) | CSV 전수 검사 |
| 손실 축약 | 가중합/배치 크기, 창 손실은 알려진 창 수로 정규화. 이상 없음 | `full_bank_train.py:101-110` |
| 기울기와 동결 | 헤드만 학습, 인코더는 캐시. 런타임 인코더는 모두 `eval().requires_grad_(False)`와 `inference_mode`. DF-Arena는 `load_df_arena_model`에서 `.eval()` | `RT7:40,68-70,96,113`, `script.py:331` |
| 체크포인트와 아키텍처 호환 | 12개 헤드 모두 shape가 런타임 빌더와 맞고 `strict=True`로 로드, 유한성·lock sha 일치(두 패키지 VERIFIED). `fake_label_index` 0, DF-Arena `spoof` 인덱스와 일치 검사 | `RT7:43-60,119-121` |
| 캐시 특징 ↔ 런타임 | 1.4 표 참조. 차이는 특징 fp16 저장과 AD 가중치 정밀도(F8). parity 재실행 PASS | F5 |
| NaN/Inf | 오디오 비유한이면 예외(`script.py:148`), 무음이면 0 stem, 정규화에 eps. 최종 출력은 `validate_predictions`가 비유한과 범위 이탈에 예외. 조용한 대체값은 없음 | — |
| 비결정성 | 학습은 seed와 `epoch_order` RNG로 고정되지만 CPU/GPU 모두 비트 재현은 안 됨. 그래서 재사용할 때 바이트 복사(`train_backbone_heads.py:380-384`). 런타임은 TF32와 benchmark가 꺼져 있지만 `use_deterministic_algorithms`는 없어 1e-6 수준 흔들림 가능(HYPOTHESIS, 영향 무시 가능) | — |
| 파일 간 의존 | 없음. 파일 단위 루프이고 배치는 한 파일 안에서만, 테스트 통계 없음. CPU에서 역순 2파일 부분집합의 최대 차이 0.0(REPORTED `verify.log`, 두 패키지). 코드 정독으로도 확인 | `RT7:244`, `verify_datav4_submission.py:72-76,87` |

---

## 4. S7 ↔ s7_new 런타임 차이 (전수)

- zip 멤버 이름 79개가 같다. manifest 해시가 다른 멤버는 18개다(VERIFIED). 목록: `model/runtime/full_bank_submission.py`; 헤드 12개(`music_4x`, `voice_4x`, `voice_ad_4x`, `wavlm_4x`); `candidate_lock.json`, `component_fusion.json`, `file_ensemble.json`, `run_identity.json`, `validation_metrics.json`. 여기에 manifest 자체가 더해진다(pkg_chain의 “19 changed members”와 일치). **인코더 가중치, HTDemucs, PANNs, ffmpeg, 다른 런타임 모듈, presence 헤드, 층 JSON은 모두 바이트까지 같다.**
- `full_bank_submission.py` diff(`diff RT7 RTN`)와 행동상 의미:
  1. `CANDIDATE` 상수(`RTN:21`)는 manifest 대조용일 뿐 예측과는 무관하다.
  2. `load_ssl`, `load_encodec` 추가(`RTN:43-60`). 호출되지 않는다.
  3. `eat_family` 구성을 리스트 내포에서 루프와 `self.eat_models` dict로 바꿨다(`RTN:120-124`). 로드 순서, 블록, fp16 플래그가 같으므로 의미는 동일하다.
  4. S8/S9 선택 멤버(`RTN:139-169`, `204-212`, `241-244`, `283-314`)는 `voice_ssl_members.json`과 `music_extra.json`이 있을 때만 켜진다. s7_new zip에는 둘 다 없으므로(VERIFIED 멤버 목록) 빈 리스트가 되어 **실행되지 않는다.** 참고로 이 분기에서는 추가 멤버 특징에 `.half().float()`를 적용하는데(`RTN:212,243`), S7 멤버에는 적용하지 않는다(F8).
- 따라서 s7_new의 행동 차이는 전부 **데이터**에서 온다. 헤드 가중치 12개가 바뀌었고(voice/voice_ad는 `final_s7_newL`에서 재사용, music/wavlm은 재학습), 음성 CDF 표 3개와 FILE CDF 표 3개를 old val에서 다시 적합했으며, FILE 가중치 값(0/9/1)은 S7과 같다(선택 방식만 격자에서 고정으로 바뀜).

---

## 5. 발견 사항

### F1. 패키지의 held-out 추정치 부재와 반복 선택 — CONFIRMED (평가 타당성 결함)
- **근거**: `data_v4/train_backbone_heads.py:349`(bank test를 train으로), `:369-372`(모든 새 행을 train으로, `NEW_VAL=1`일 때만 newval을 val로). 체크포인트 `train_ids` 교집합(VERIFIED): s7_new `voice_41.pt` 58,275행 = bank train 25,987 + bank test 5,139 + new train 16,518 + **newval 2,445 + newtest 8,186**. `music_41.pt` 27,300행 = 9,363 + 2,855 + 8,835 + **1,584 + 4,663**. S7도 bank test 5,139/2,855행을 학습했다. old val과의 교집합은 두 패키지 모두 0이다. DB 구성은 `jobs_nd.txt`의 `s7_new_*`이 `NEW_TRAIN=1 NEW_VAL=0`이고 `add_new_rows`(`:81-86`)는 pool 'train' 23,955행만 추가한다. S7 manifest `heads`의 “exploratory (fails the per-seed lock rule, RT held-out +0.0104 vs S6)”, s6 `choice_lock.json` 규칙 C(“every seed's component EER improves”).
- **영향 경로**: 슬롯 1 결정(s7_new)과 S7 채택 근거.
- **기대 동작**: 최종 패키지(또는 그와 같은 레시피와 행 풀)를 평가할 수 있고 선택에 쓰지 않은 세트가 하나는 있어야 한다. 아니면 모든 수치를 개발용으로 명시해야 한다.
- **관찰 동작**: old val, E1, E2, newval, LB가 모두 선택이나 학습에 쓰였다. manifest의 “DB protocol vs S7 recipe: E1 +0.0064, E2 +0.0285”는 newval/newtest 16,267행을 학습하지 않은 DB 모델의 수치이고, 패키지는 그 행들(AT-ADD 평가 생성기 포함)까지 학습한 **다른 모델**이다. S7은 사전 등록 규칙 C(시드별 개선)에 실패했지만 E1 RT 결과(+0.0104)를 근거로 제출됐고, 이후 LB 결과로 현행 최선이 됐다.
- **예상 영향**: 로컬 이득은 선택 낙관이 섞인 개발 증거다. 패키지 수준 일반화의 크기는 UNKNOWN이다. P1 재적합 자체는 흔한 관행이고 결함이 아니다. 결함은 해석에 있다.
- **제안 수정**: (a) 모든 보고에 “dev/in-sample”을 표기한다. (b) 추정치가 꼭 필요하면 P1에서 뺄 봉인 부분집합(예: E2_unlinked의 lineage 그룹 무작위 일부)을 미리 정하고 한 번만 비교한다. 데이터가 줄어드는 비용과 저울질해야 한다. (c) LB는 적응적이라는 점을 명시한다.
- **검증 방법**: 봉인 세트 ID와 모든 패키지 체크포인트의 `train_ids` 교집합이 0인지 확인한다(이번에 쓴 교집합 계산과 같은 방식). 봉인 세트로는 사전 등록한 비교 하나만 실행한다.

### F2. 음악 헤드가 음악 없는 파일을 채점해 FILE에 기여 — PLAUSIBLE risk (수치 VERIFIED)
- **근거**: 음악 헤드 학습 행은 music 라벨이 있는 행뿐이고(`full_bank_train.py:37-40`, `train_backbone_heads.py:183-185,214-218`), route=music인 행은 모두 MUSIC_PRESENT=1이다(VERIFIED). 체크포인트 `train_ids` 12,218 = 음악 라벨 행 전체. 런타임은 모든 파일의 backing을 채점하고(`RT7:141-156`) FILE 후보 `max_nopresence = max(Fv, Fm)`(`decision_heads.py:73`)와 `noisy_or`(Pm·Fm)에 넣는다. `_scale_check.py`는 S7 레시피 DB 예측(`research/top15/work4/runs/*.json`, frozen rows)을 재구성했고, 기록된 E1 ADS 0.90921과 정확히 일치한다.
- **영향 경로**: 음성만 있는 파일(음성 합성 탐지 대상)의 FILE_FAKE_PROB.
- **기대 동작**: 음악이 없는 파일에서 Fm은 정보가 없거나 FILE에 영향을 주지 않아야 한다.
- **관찰 동작**: E1의 음악 없는 파일 2,673개(진짜 1,729)에서 Fm q50/q90은 진짜 0.004/0.469, 가짜 0.001/0.035다. 진짜 파일의 36.1%에서 Fm > Fv라서 `max_nopresence`를 음악 헤드가 결정한다(가짜는 0.0%). val에서도 56.8% 대 0.8%다. parity 파일 중 LibriSpeech 진짜 음성의 MUSIC은 S7 0.038, s7_new 0.494로, 재학습마다 크게 흔들린다. 음성 쪽 대조군(음성 없는 파일에서 Fv=0 오라클)은 FILE EER이 그대로다(0.0741).
- **예상 영향**: 오라클(라벨로 게이트하는 진단)로 Fm=0을 주면 E1 FILE EER이 0/9/1에서 0.0741→0.0716, val 적합 가중치(1/1/1)에서 0.0753→0.0700이다. 총점으로 대략 −0.001~−0.0025에 해당하는 **상한**이다. 실제로 쓸 수 있는 게이트는 이보다 작을 것이다.
- **제안 수정**: 새 데이터 없이 할 수 있는 순서로 적는다. (a) `max_nopresence`의 Fm을 Pm으로 게이트하거나(`max(Fv, Pm·Fm)`) 후보에서 뺀다. (b) FILE 가중치를 사전에 고정하고, 조기종료에 쓰지 않은 세트로 적합한다. (c) 장기적으로는 음악 헤드에 음악 없는 backing을 보는 보조 목표(예: MUSIC_PRESENT)를 더한다.
- **검증 방법**: 변경 규칙을 사전 등록한 뒤 E1+E2 RT에서 FILE EER과 ADS를 부트스트랩한다(개발 증거). 음악 없는 진짜 파일의 FILE 점수 분포(q90)와 FPR@val-TPR95가 줄었는지 본다. 최종 판단은 LB가 하게 된다(적응성 명시).

### F3. `ads.py`의 제로필 평가(AC 의미론) — CONFIRMED defect (평가 코드, 과거 결정에 영향)
- **근거**: `research/top15/work2/backbones/ads.py:67`(“a missing member score is 0”), `:73`(`comp[m].get(i, 0.)`). run 파일에는 라벨이 있는 행만 들어 있다(`run.py:303-305`). S6 연구 E1 4,822행 중 음악 멤버 점수가 없어 0이 된 행이 2,673개(FILE 가짜 944), 음성 멤버 점수가 전부 0인 행이 981개(가짜 286)다(VERIFIED). `fit_ft_fusion.py:143-144`의 AC와 RT 경로에도 `.get(…, 0.)`가 있고, RT 커버리지 검사는 `ft`에만 있다(`:130-132`).
- **영향 경로**: S6 잠금 선택(SSLAM 교체, WavLM w1 추가)과 그 held-out 수치(joint +0.00945).
- **기대 동작**: 런타임처럼 모든 행을 모든 멤버로 채점해야 한다.
- **관찰 동작**: 라벨이 없다는 이유로 점수를 0으로 넣는다. 이는 F2의 오라클 게이트와 같은 효과(Fm=0 on no-music)라서 FILE을 낙관적으로 만든다.
- **예상 영향**: S6 시기 FILE EER은 RT 대비 약 0.0025~0.0054 낙관적이었을 것으로 본다(ESTIMATE: F2의 오라클 수치를 옮긴 것이고, 음성 멤버 제로필은 F2 대조군에서 효과가 없었다). 방향은 이후 `rt_s6enc`(RT, +0.0104)가 확인했다. 현행 `new_eval`/`s8_eval`은 RT와 엄격한 인덱싱을 쓴다.
- **제안 수정**: `ads.py`의 `component()`는 누락 시 예외를 던지도록 바꾸거나 “AC, runtime 아님”으로 표시한다. `fit_ft_fusion` RT 경로는 누락 assert를 모든 멤버로 넓힌다.
- **검증 방법**: `s8_eval.py:51-53`처럼 val/test의 모든 행이 모든 멤버의 every-row 점수를 갖는지 assert하고, 기존 보고서 수치(S6 RT 0.90921)를 그대로 재현하는지 본다.

### F4. FILE 규칙의 척도 의존과 가중치 결합 — CONFIRMED (설계 성질), 영향 PLAUSIBLE
- **근거**: `component_fusion.py:25-28`(`ours` 척도로 역사상), `decision_heads.py:71-73`(Pv·Fv와 Pm·Fm의 max, noisy_or, max(Fv, Fm)). `ours` 표의 val 중앙값은 0.861(S7)/0.845(s7_new)로 원시 Fm 척도와 다르다(VERIFIED, 표 요약). `_scale_check.py`에서 Fv를 자기 val CDF로 바꾸면(단조변환이므로 VOICE EER 변화 0) FILE EER이 달라진다. 가중치를 0/9/1로 고정하면 val +0.01219, E1 −0.00044이고, 재적합하면 val +0.00166, E1 −0.00018이다. S8의 `file_ensemble`은 동률 규칙으로 1/1/1(격자 이득 0.0008 < 0.002), S7은 0/9/1이다(VERIFIED `run_identity.json`). DB 평가(`rt_s6enc`/`s8_eval`, val 3,680행)가 S7 레시피 E1 0.90921을 낼 때의 FILE 가중치는 1/1/1이었고(재현으로 VERIFIED), 패키지는 0/9/1이다.
- **영향 경로**: FILE_FAKE_PROB, 그리고 구성 간 비교(S7↔S8 LB, S7 레시피 E1).
- **기대 동작**: 성분 결합은 공통으로 보정된 척도에서 해야 한다. 적어도 비교할 때는 FILE 규칙을 같게 맞춰야 한다.
- **관찰 동작**: FILE 가중치와 표가 `ours` 헤드의 보정 상태에 묶여 있다. s7_new는 `ours` 헤드가 달라졌는데도 S7의 0/9/1을 그대로 쓴다(자체 격자 최선은 1/1/8, 차이 0.00015). S7↔S8 LB 차이(−0.0034)는 음악 특징과 FILE 가중치가 **동시에** 바뀐 결과다.
- **예상 영향**: E1에서는 작다(≤0.0005 FILE EER). 은닉 세트에서의 크기는 UNKNOWN이다. “mix-SSLAM이 LB를 낮췄다” 같은 결론은 교란돼 있다.
- **제안 수정**: FILE 후보 전에 Fv와 Fm을 각자의 보정 척도(예: 별도 보정 split의 CDF나 Platt)로 옮긴다. 또는 비교 실험에서 FILE 규칙을 고정한다. LB 비교는 한 번에 요소 하나만 바꾼다.
- **검증 방법**: 성분별 단조변환을 가했을 때 FILE EER 변화가 ≈0이면 척도 불변이다. E1과 E2 RT에서 가중치를 사전 고정한 채 비교한다.

### F5. s7_new parity 영수증 공백 — CONFIRMED gap (이번 재실행으로 해소), CUDA는 UNKNOWN
- **근거**: `submissions/DataV4_s7_new/runtime_parity.txt`는 15:01에 생성됐고 출력 형식에 `file` 키가 없다. `research/top15/work5/w/w_runtime_parity.py`의 FILE 비교와 영수증 sha 결속(`:19-22,46-52`)은 commit `a5b552a`(16:13)에서 추가됐다(VERIFIED `git diff`). S7은 packet에도 submissions에도 `runtime_parity.txt`가 없다. `DataV4_s7_new.chain.log`는 “staged for CUDA check”에서 끝난다.
- **영향 경로**: 학습 특징 경로와 패키지 런타임의 일치 여부.
- **기대 동작**: FILE까지 포함한 parity를 해당 패키지 영수증에 결속해 남겨야 한다.
- **관찰 동작**: 엄격판 로직으로 재실행했다(`_parity_ablation.py`, 10파일, CPU, verification.json sha = receipt sha 확인). s7_new는 WORST VOICE 0.00018, MUSIC 0.00089, **FILE 0.00011**로 PASS. S7은 0.00078, 0.00227, 0.00019로 PASS(허용치 0.02/0.01/0.01).
- **예상 영향**: 낮음. CUDA 코드 경로는 S7(LB에서 L4 실행 성공)과 같고, 자산도 모두 같다(4절).
- **제안 수정**: 현행 스크립트로 s7_new 영수증을 다시 만들어 보관한다. 시간이 되면 `verify_s8_vessl.py`로 CUDA 검증을 돌린다.
- **검증 방법**: 영수증에 `file` 키와 `n 10`이 있고 sha가 일치하는지 본다. CUDA와 CPU 예측의 최대 차이를 기록한다.

### F6. 새 데이터에 내용 해시가 없음, 추출 실패는 조용히 제외 — CONFIRMED gap (데이터 수용)
- **근거**: `atadd_rows.py:56`, `open_rows.py:42`가 `'audio_sha256': '', 'source_sha256': ''`를 넣는다. `data/new_rows.csv` 40,222행 전부 빈 값이다(VERIFIED). `dump_features.py:153-163`은 예외가 난 행을 목록에만 남기고 건너뛴다. `train_backbone_heads.py:61-73`은 npz가 없는 행을 버린다(8행).
- **영향 경로**: 새 행 수용, 누수 검사(`leak_check.py`는 group_id와 OpenBMAT 트랙 메타데이터만 본다), E2의 독립성.
- **기대 동작**: 은행 행처럼 내용 해시를 기록하고 추출 전에 대조해야 한다(`full_bank_features.py:170`). split 경계를 넘는 정확 중복이 0임을 보여야 한다.
- **관찰 동작**: 내용 기준 중복·동일성 검사를 할 수 없다. 메타데이터 키 기준으로는 old val ∩ 학습 행 = 0, new train ∩ newtest group_id = 4(atadd_speech 10행, leak_check가 제외한 10행과 일치), remix 기부자는 train 풀에만 있다(MUSDB 곡 키, GTSinger ID로 VERIFIED).
- **예상 영향**: 메타데이터가 틀리거나 같은 오디오가 다른 이름으로 들어 있으면 E2가 오염될 수 있다(HYPOTHESIS). 규모는 UNKNOWN.
- **제안 수정**: 새 행의 파일 sha256과 디코드 PCM 해시를 계산해 CSV에 넣고, split 간·은행 간 정확 중복을 검사한다. 실패 행 목록(`_failed_*.json`)의 유형을 확인한다.
- **검증 방법**: 해시 중복 교집합이 0인지 보고, 실패 행 8개의 코덱이나 길이 분포가 은닉 테스트 조건(MP3/WAV/FLAC, 4~60초)과 겹치지 않는지 확인한다.

### F7. 단일 클래스 슬라이스와 출처 교란 — CONFIRMED (평가 성질과 보고 코드 결함)
- **근거**: realV+fakeM 행 수는 bank train 5, **old val 5**, E1 상위집합 107, new train 156, **newval 0, newtest 0**이다. 이 슬라이스에서는 FILE이 전부 1, VOICE가 전부 0, MUSIC이 전부 1이라 세 성분 EER이 모두 정의되지 않는다(VERIFIED). E2 데이터셋 9개 중 7개는 FILE이 단일 클래스다(gtsinger_ko 880·gtsinger_ko_speech 383·musdb 111·sdrr 146은 전부 진짜, haim_b3 209·b4 209·b6 327은 전부 가짜). 두 클래스가 모두 있는 것은 atadd 10,000과 openbmat 161뿐이다. 출처 교란도 심하다. E1 VOICE에서 두 클래스를 모두 가진 source에 속한 진짜 행은 5%뿐이고, old val MUSIC에서는 가짜 7%, 진짜 29%다(VERIFIED). 코드: `fit_ft_fusion.py:104-111`의 `Boot.ads`는 sklearn `roc_curve`를 써서 단일 클래스 재표본에서 NaN을 내고, `p_gt_0 = mean(d > 0)`은 NaN을 음수처럼 센다. `ads_of`(`:67-71`)는 `round(None)`으로 중단된다. `e2_subsets.txt`에는 “nan (+nan, p 0.00)”이 찍혀 있다.
- **영향 경로**: E2 세부 슬라이스 보고, `select_newval`의 거부 조건(E2_open_unlinked는 FILE 가짜가 거의 전부 HAIM, 진짜가 GTSinger/MUSDB/SDRR), realV+fakeM 같은 은닉 테스트 혼합 유형의 평가.
- **기대 동작**: 단일 클래스 슬라이스는 “정의되지 않음”으로 표시하고 고정 임계값에서의 FPR/TPR만 보고해야 한다. 풀링 EER은 출처 교란을 명시해야 한다.
- **관찰 동작**: `s8_eval`과 `new_eval.per_source`는 올바르다. 부트스트랩 p 값은 0.00으로 잘못 찍힌다. old val에 realV+fakeM이 5행뿐이라 조기종료와 CDF 표는 이 유형을 사실상 보지 못한다.
- **예상 영향**: 보고서를 읽는 사람이 p=0.00을 “확실한 악화”로 오해할 수 있다. 출처가 갈린 슬라이스의 EER은 출처 식별 능력을 잰다.
- **제안 수정**: `Boot.delta`에서 NaN 재표본을 제외하고 개수를 보고하거나 “undefined”로 표시한다. 단일 클래스 E2 데이터셋은 FPR/TPR@val-TPR95만 보고한다. 거부 조건은 두 클래스가 모두 있는 소스(atadd, openbmat) 안에서 평가한다.
- **검증 방법**: 단일 클래스 부분집합에서 `Boot.delta`가 “undefined”를 반환하는 단위 검사, 그리고 거부 조건 슬라이스별 (source, label) 교차표를 확인한다.

### F8. 특징과 가중치 정밀도 불일치 — PLAUSIBLE risk (측정상 작음)
- **근거**: 학습 캐시의 fp16 저장 위치는 `extract_backbones.py:71,79`, `extract_new_backbones.py:109`, `extract_s6.py:53,65`, `dump_features.py:53,56,58,65`다. 런타임 S7 멤버 특징은 fp32로 쓰인다(`RT7:148,152,176,179-180`). AD 가중치는 패키지 쪽이 `_workspace/models/xls-r-1b-anti-deepfake-fp16`(F16 815 텐서, `pkg_chain.sh:13`)이고, 로컬 원본은 F32(VERIFIED, safetensors 헤더)다. 학습 추출에 어느 쪽을 썼는지는 인자(`extract_backbones.py:48`)라서 UNKNOWN이다. `w_runtime_parity.py:3`은 오프라인 쪽을 “fp32 AntiDeepfake”로 적고 있다.
- **영향 경로**: 모든 멤버의 헤드 입력.
- **기대 동작**: 학습 입력과 런타임 입력의 표현 정밀도가 같아야 한다.
- **관찰 동작**: parity 오차는 VOICE ≤ 7.8e-4, MUSIC ≤ 2.27e-3, FILE ≤ 1.9e-4(10파일, CPU)다.
- **예상 영향**: 낮음. EER 임계값 근처의 극소수 순위 뒤바뀜 정도(HYPOTHESIS).
- **제안 수정**: 선택 사항이다. S8 분기처럼 S7 멤버 특징에도 `.half().float()`를 적용한다. 비용은 거의 없다.
- **검증 방법**: 적용 전후 parity WORST를 비교(작아지는지)하고, E1 RT 점수가 바뀌지 않았는지 확인한다.

### F9. s7_new의 조기종료 세트가 새 데이터 도메인을 대표하지 못함 — PLAUSIBLE risk
- **근거**: `run_identity.json`에 `"new_val": false`. val은 은행 old val 4,931행뿐이다(`train_backbone_heads.py:370-372,376-378`). s7_new music seed 41은 epoch 6에서 선택됐다(S7은 15). 음악 학습 행의 약 55%(15,082/27,300)가 새 도메인이다.
- **영향 경로**: 모든 s7_new 헤드의 epoch 선택.
- **기대 동작**: 학습 분포와 비슷한 세트로 조기종료해야 한다.
- **관찰 동작**: 은행 분포만으로 epoch를 고른다. W_newB가 이 점을 겨냥한 변형이다(REPORTED MANIFEST).
- **예상 영향**: UNKNOWN. 과소·과다 학습 모두 가능하다.
- **제안 수정**: DB 실행의 epoch별 newval EER 곡선을 보고, 선택 epoch와 newval 최적 epoch의 차이를 확인한다.
- **검증 방법**: 곡선 비교. 이미 학습된 DB 모델의 `history`에 기록이 있으면 추가 GPU가 필요 없다.

### F10. 런타임 잠재 위험 — PLAUSIBLE latent risk (현 패키지에는 낮음)
- **근거**: `read()`가 None을 돌려주면 legacy presence, max 규칙, 융합 없음, MERT [12], EAT/SSLAM 없음으로 **조용히** 강등한다(`RT7:76-82,102-104`). try/except 없이 파일마다 순차 실행한다(`:244`). `ad_heads`/`wavlm_heads`에는 유한성 검사가 없다(`:86-90,114-118` vs `:52`). `graph_scoring._label`은 `*_mask`를 무시하고(`:81-88`), `head_labels`는 route 라벨의 마스크를 무시한다(`full_bank_train.py:37`).
- **영향 경로**: 앞으로 만들 패키지와 CSV.
- **기대 동작**: 필수 구성요소가 빠지면 즉시 실패해야 하고, 라벨 사용은 마스크를 따라야 한다.
- **관찰 동작**: 현 zip은 모든 JSON이 있고 manifest 해시로 결속돼 있어 강등이 발동할 수 없다(VERIFIED). 두 패키지의 `ad`/`wavlm` 헤드는 유한하다(VERIFIED). 현 CSV에서는 마스크 0 셀의 값이 비어 있다(VERIFIED). 다만 `component_fusion.json`만 빠지면 FILE 표(융합 후 Fv로 적합)가 융합 전 Fv에 조용히 적용된다. 파일 하나라도 디코드나 OOM으로 실패하면 제출 전체가 사라진다(S7은 은닉 테스트에서 통과했고 s7_new도 같은 경로).
- **예상 영향**: 현재 낮음.
- **제안 수정**: manifest `choice.component_fusion`이 있으면 `component_fusion.json`을 필수로 요구한다. `ad`/`wavlm` 헤드에도 `isfinite` 검사를 넣는다. `_label`과 `head_labels`는 `*_mask`를 확인한다.
- **검증 방법**: 임시 사본에서 JSON 하나를 지우면 예외가 나는지 보는 단위 검사와, 마스크 0에 값 '1'을 넣은 합성 행을 `_label`이 무시하는지 보는 단위 검사.

### F11. 기록(provenance) 불일치와 in-sample 지표 표기 — CONFIRMED (기록 결함, 낮음)
- **근거**: `train_backbone_heads.py:578`은 이번 실행의 플래그만 기록한다. S7과 s7_new 모두 `label_completion: false`인데 voice/voice_ad는 L 실행(`final_s6L`, `final_s7_newL`)에서 재사용했다(`voice_heads_reused_from`). `:574`의 `train_with_backbone`은 `a.cache/ID.npz`만 세므로 s7_new도 37,108이지만, 실제 `train_ids`는 58,275(음성)/27,300(음악)이다(VERIFIED). `validation_metrics.json`(S7 FILE 0.02983, s7_new 0.02736)은 조기종료, CDF, FILE 가중치를 모두 적합한 old val에서 계산했다(`:554`).
- **영향 경로**: 사람의 해석과 향후 재현.
- **기대 동작**: 재사용한 헤드의 라벨 규칙과 실제 행 수를 기록하고, in-sample임을 표시해야 한다.
- **관찰 동작**: 위와 같이 불일치한다.
- **예상 영향**: 낮음. 다만 validation 수치가 개선 증거로 인용되면 오해를 부른다.
- **제안 수정**: `run_identity`에 재사용 원천의 `label_completion`과 `len(train_ids)`를 기록하고, `validation_metrics`에 `"in_sample": true`를 붙인다.
- **검증 방법**: `run_identity`의 행 수 필드가 체크포인트 `len(train_ids)`와 같은지 확인한다.

### F12. `parity.py`의 presence 비교 대상 불일치 — CONFIRMED (QA 스크립트 결함, 낮음)
- **근거**: `research/top15/work5/newdata/parity.py:17`은 dump의 `presence`와 cache_v4의 `presence`를 비교한다. dump 쪽은 presence **헤드 출력**이고(`dump_features.py:67,77`), cache_v4 쪽은 **legacy PANNs max**다(`full_bank_features.py:205,212`). 400개 val 행에서 두 양의 차이는 최대 0.915다(VERIFIED). 반면 `new_eval`이 쓰는 `presence_val.json`은 헤드 출력과 차이 0이다(VERIFIED). 평가 자체는 일관된다.
- **영향 경로**: 새 행 특징 parity 보고의 presence 줄. 출력 파일은 저장소에 없어 사용 여부는 UNKNOWN이다.
- **기대 동작**: 같은 양끼리 비교해야 한다.
- **관찰 동작**: 서로 다른 양을 비교한다.
- **예상 영향**: 낮음. presence 불일치 경보가 오탐이 되거나 무시된다.
- **제안 수정**: cache 쪽을 `dh.presence(heads, presence_features, legacy)`로 바꾼다.
- **검증 방법**: 겹치는 행에서 max|diff|가 ≈1e-6 이하인지 확인한다.

### F13. WavLM 점수 포화 — HYPOTHESIS (낮음)
- **근거**: `component_fusion.json` wavlm 표에서 1−1e−6 이상인 비율은 S7 18.6%, s7_new 29.1%이고, 1e−4 이하도 약 14%다(VERIFIED). DB every-row 예측에서 포화된 비율은 가짜 음성 47.2%(val)/41.2%(E1), 진짜 0%다(VERIFIED).
- **영향 경로**: 최상위 구간에서 CDF 해상도가 줄어든다(동률 블록 전체가 같은 CDF 하한을 받는다).
- **기대 동작**: 순위 정보가 보존돼야 한다.
- **관찰 동작**: 포화 블록에 진짜가 없어 EER 영향은 측정되지 않았다.
- **예상 영향**: 은닉 세트에서 진짜가 포화될 때만 문제가 된다.
- **제안 수정**: 멤버 점수를 softmax 확률 대신 창 최대 log-odds로 기록한다(단조변환이므로 비용이 작다).
- **검증 방법**: 동률 수와 E1/E2 RT의 VOICE·FILE EER을 비교한다.

---

## 6. 이번 감사에서 실행한 검증 (요약 출력)

1. zip 대조: `unzip -Z1`로 멤버 79개를 얻었고, 패킷 `MEMBERS.txt`와 일치했다. 패킷 runtime 파일 40개와 zip 멤버의 sha가 일치했다. manifest 78항목 중 소형 멤버 해시가 일치했다. 로컬 `_workspace/final_s7_20260928`, `final_s7_new_20260928`의 trained 멤버 21개가 zip과 일치했다. S7↔s7_new 해시 차이는 18개다.
2. 체크포인트 12×2의 메타데이터, shape, 유한성, lock sha를 확인했고 `train_ids`와 평가 세트의 교집합을 계산했다(F1 수치).
3. 라벨과 마스크 전수 검사: route 라벨 미지 0, 마스크 0인 값 0, route 성분 PRESENT≠1 0, FILE 라벨 모순 0. 라벨 완성 1,364행.
4. `_parity_ablation.py`(엄격판 parity 재실행과 SSLAM/WavLM 민감도):
   - S7: `PARITY WORST {'voice': 0.00078, 'music': 0.00227, 'file': 0.00019} n 10`, SSLAM 비중 0.2189, WavLM 제거 시 max|dVOICE| 0.7064, max|dFILE| 0.1073, SSLAM을 평균으로 바꾸면 max|dMUSIC| 0.3038, max|dFILE| 0.0772.
   - s7_new: `{'voice': 0.00018, 'music': 0.00089, 'file': 0.00011} n 10`, 0.2232, 0.3331/0.0645, 0.3789/0.0301.
5. `_scale_check.py`(S7 레시피 DB 재구성, val 3,680 / E1 4,822):
   - 재현: E1 ADS 0.90921(FILE 0.07531, VOICE 0.06116, MUSIC 0.13635), val 적합 FILE 가중치 1/1/1.
   - Fv를 val CDF로 바꾼 경우(0/9/1 고정): dFILE val +0.01219, E1 −0.00044, dVOICE 0. 재적합: +0.00166, −0.00018.
   - 음악 없는 파일: val 2,394개(진짜 741) Fm q90 진짜 0.38, 가짜 0.119, Fm>Fv 비율 0.568/0.008. E1 2,673개(진짜 1,729) q90 0.469/0.035, 0.361/0.000.
   - 오라클 Fm=0: 0/9/1에서 val 0.03268→0.03185, E1 0.0741→0.07159. 1/1/1에서 val 0.0337→0.03102, E1 0.07531→0.06995. 음성 대조(Fv=0, 음성 없는 파일): 변화 없음.
6. 표 포화, 슬라이스 교차표, 출처 교란, lineage 키 교집합, 새 행 해시 공백(F6, F7, F13 수치).
7. 쓰기는 이 폴더에만 했다: `code_audit.md`, `_parity_ablation.py`, `_scale_check.py`, `_members_*.txt`, `_manifest_*.json`.

---

## 7. 커버리지

| 범위 | 수준 | 파일 |
|---|---|---|
| 패키지 런타임 | 전부 | `RT7`·`RTN`의 `full_bank_submission.py`(RTN은 diff 전수), `decision_heads.py`, `component_fusion.py`, `voice_heads.py`, `mert12_submission.py`, `antideepfake_runtime.py`, `localization_models.py`, `assets/baseline/script.py`, `script.py`, `MEMBERS.txt`, `model/trained/*.json`(두 패키지), 체크포인트 메타데이터 |
| 패키지 런타임 | 일부 | `eight_heads.py`(`FrozenHead`만), `graph_scoring.py`(검증과 지표), S8 `trained/*.json`(가중치와 extra만) |
| 패키지 런타임 | 안 봄 | `eat_core/*`, `v6_runtime.py`·`v6_core.py`·`v6_data.py`, `ast_multitask.py`, `graph_scoring`의 나머지, DF-Arena/MERT 모델 코드, S8 런타임 본문 |
| 학습·보정 | 전부 | `data_v4/train_backbone_heads.py`, `full_bank_train.py`, `full_bank_batches.py`, `data_v4/weighting.py`, `data_v4/label_rules.py`, `data_v4/file_ensemble.py`, `data_v4/split_rules.json`, `full_bank_evaluate.py`, `full_bank_features.py`, `research/top15/work2/backbones/run.py`, `research/top15/work4/voice_data.py` |
| 특징 추출 | 전부 또는 일부 | `data_v4/extract_backbones.py`(전부), `research/top15/work2/s6/extract_s6.py`(전부), `research/top15/work2/backbones/extract_new_backbones.py`(EAT 부분), `research/top15/work5/newdata/dump_features.py`(전부), `data_v4/dump_stems.py`(grep), `data_v4/presence.py`(적합 split만) |
| 평가·선택 | 전부 | `research/top15/work2/backbones/ads.py`, `research/top15/work3/integration/fit_ft_fusion.py`, `research/top15/work5/s8/s8_eval.py`, `research/top15/work5/newdata/{new_eval,virtual_configs,select_newval,score_split,leak_check,parity}.py`, `research/top15/work5/w/w_runtime_parity.py`, 결과 txt/json 4종, `rt_s6enc_report.json`, s6 `choice_lock.json` |
| 평가·선택 | 일부 | `research/top15/work4/rt_s6enc.py`(grep), `jobs_nd.txt`(앞부분), S6 `ads_report`(요약) |
| 패키징·QA | 전부 | `package_datav4_submission.py`, `verify_datav4_submission.py`, `pkg_chain.sh`, receipts(S7, s7_new) |
| 안 봄 | — | `atadd_rows.py`/`open_rows.py`/`openbmat_extra.py`/`assign_splits.py`/`music_newval.py`(행 생성과 split 규칙. 해시 공백 줄만 grep), `extract_s8.py`/`extract_s9.py`, W_new/W_newB/s7_new_ob 패키지 본문, `rows_r2.txt`/`frozen_rows.json` 구성 근거, history·ledger·debate 문서(독립성 유지), E3·AI-Hub 관련 전부(지시에 따라 제외), 이 폴더에 있는 다른 에이전트 산출물(읽지 않음) |

---

## 8. 남은 불확실성

- CUDA(L4)에서의 s7_new 수치 parity는 NOT RUN이다(CPU parity만 있음). 코드와 자산이 S7과 같으므로 위험은 낮다고 보지만 증거는 아니다.
- AD 학습 특징을 fp32 가중치로 추출했는지 fp16 가중치로 추출했는지는 UNKNOWN이다(parity로 영향 크기만 측정).
- F2와 F4의 수치는 S7 레시피 DB 모델(E1/val)에서 얻었다. s7_new DB 예측은 쓰지 않았다(E3가 섞인 위치를 피하려고 `research/top15/work4/runs`만 사용). s7_new에서는 parity 파일 사례(LibriSpeech Fm 0.494)만 있다.
- 오라클 수치는 라벨로 게이트한 **상한**이고, 실제 적용할 규칙의 이득은 사전 등록 평가가 필요하다.
- 은닉 테스트 분포(전화 채널, 한국어, 혼합 유형)에 대한 대표성은 이 감사로 판단할 수 없다.
