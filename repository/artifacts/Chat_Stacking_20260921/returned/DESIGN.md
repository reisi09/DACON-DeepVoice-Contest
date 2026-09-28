# DACON 236749 — File-only stacking 설계와 코드 감사

## 1. 기준과 변경 경계

기준 입력 SHA-256: `7adfb7b426c45b0efbb23de581ca1a5a073079d2110f4f4f84cdcf0fca2fc75a`. INPUT_MANIFEST의 133개 파일 해시/크기를 실제 검사했다. 현재 기준은 `existing9504_MERT330M_WindowMIL_DFfc5_701515`이며 제출 snapshot의 `current/submission_source/model/runtime/full_bank_submission.py`가 우선한다. `current/training_code/dacon9504a100/`에서 실제 학습·캐시·평가 구현을 확인했다. `current/training_source/full_bank_submission.py`는 제출본과 해시가 다르므로 덮어쓰지 않았다. `reference/legacy/`를 import하거나 현재 head/schema에 연결하지 않았다.

첫 실험은 FILE_FAKE_PROB만 바꾼다. VOICE_FAKE_PROB, MUSIC_FAKE_PROB, VOICE_PRESENT_PROB, MUSIC_PRESENT_PROB는 현재 Engine의 출력 그대로다. PANN, HTDemucs, DF-Arena, MERT와 기존 여섯 head를 새로 학습하지 않는다. 개선은 가설이며, 이 반환물에는 새로운 독립 데이터에 학습한 메타 모델이 없다.

## 2. 실제 현재 추론 순서

1. baseline `load_audio`: librosa mono, 16 kHz, float32. 빈 파일/비유한 입력을 거부한다.
2. 원본 peak를 구한다. `preserve_gain=False`이고 peak>0.6이면 0.6/peak를 곱한다. 추론 기본값은 False. 기존 legacy_crop/constructed 학습의 gain-preserving 경로를 임의로 통일하지 않는다.
3. 정규화 음원을 FLOAT WAV로 임시 저장하고 HTDemucs로 분리한다. pinned torchaudio loader, 모델 채널/샘플레이트 변환을 사용한다. Demucs 입력 평균/표준편차 정규화, `shifts=0`, `split=True`, `overlap=.25`, 통계 복원을 거친다. vocals가 voice, 나머지 source 합이 backing이며 채널 평균과 16 kHz 재표본화를 수행한다. 두 stem을 raw 길이에 맞춰 자르고 부족하면 오른쪽에 0을 붙인다.
4. backing의 음악 windows를 만들고 16→24 kHz soxr_hq, processor padding, MERT hidden_states[12]의 시간평균을 계산한다. 파일당 window×1024 float32 특징이다.
5. seed 41/42/43별 음악 head는 **각 window를 checkpoint mean/scale로 정규화 → 1024→128 GELU→1 MLP → window logit의 log-mean-exp → sigmoid**다. 이 파일 확률들을 seed 평균한다. window probability 평균이나 seed logit 평균으로 바꾸지 않는다.
6. 음성/PANN 격자의 각 start에서 voice segment를 DF-Arena에 넣는다. fc5 pre-forward hook으로 window×1280 embedding을 얻는다. 같은 start의 **raw** segment를 PANN에 넣어 선택된 음성/음악 class 그룹의 clip 확률을 계산한다. 파일 Presence는 window와 해당 클래스에 걸친 maximum이다.
7. 음성 seed별 linear(1280→2), softmax의 fake index(**0**)를 취하고 window maximum을 구한 뒤 seed 평균한다. 현재 checkpoint들에서 fake index가 일치하는지 확인한다.
8. voice stem RMS<1e-5이면 최종 Fv=0으로 고정한다. 현재 Engine은 이 gate 이전에도 DF forward를 실행한다. Fm에는 같은 침묵 gate가 없다.
9. `File=max(Pv*Fv,Pm*Fm)`을 반환한다. 선택적 stacking 활성화 시 이 마지막 File만 메타 모델 확률로 교체한다.

수치 정의는 `_snapshot_math.py`에 원문 AST를 보존해 옮기고 무관한 transformers import만 분리했다. `configs/snapshot_symbols.json`에 원본 경로·해시·줄 번호가 있다. 원본 scorer는 `_snapshot_scoring.py`에 byte-identical하게 복사했다. AST/해시 검사를 실제 실행했다.

## 3. 시간축 및 padding

| 항목 | 음악/MERT | 음성/DF 및 PANN |
|---|---|---|
| 원본 시간 단위 | 16 kHz samples | 16 kHz samples |
| window | 64,600 samples = 4.0375 s | 동일한 길이 |
| nominal hop | 32,300 = 2.01875 s | 64,600 = 4.0375 s |
| 마지막 window | 마지막 원본 sample에서 끝나도록 exact tail start 추가 | 동일한 원칙이나 격자가 다름 |
| 짧은 파일 | 오른쪽 0 padding | 짧은 신호 전체를 반복 후 절단 |
| encoder 내부 SR | 24 kHz | DF 16 kHz, PANN 32 kHz |

예: N=100,000이면 음악 starts=[0,32300,35400], 음성/PANN starts=[0,35400]. 기존 cache `starts`는 음악 축이지 DF/PANN 축이 아니다. `valid_samples=min(64600,N-start)`이며 짧은 파일의 반복/zero-padding 가상 구간을 실제 지속 길이에 포함하지 않는다. MERT processor와 PANN 내부 경계 효과 자체를 이 valid 길이로 제거했다고 주장하지 않는다.

`aligned_summary`는 양쪽 start/end 전체의 합집합으로 원본 시간을 작은 구간으로 나누고, 각 구간을 덮는 window 확률을 branch별 평균한 뒤 두 값을 곱한다. 겹치는 음악 window가 시간을 중복 가중하지 않게 한다. mean은 구간 길이 가중, max는 구간 maximum, coverage는 product>=0.5인 원본 시간 비율, longest_fraction은 그 조건을 만족하는 최장 연속 길이/N이다. 0.5는 사전 고정한 descriptor 기준이며 OOD로 조정하지 않는다. 양쪽 중 하나라도 실제 시간을 덮지 못하면 실패한다. PANN 출력은 **clip-level 527 classes**이지 정밀 framewise localization이 아니다.

## 4. 피처 가용성

- F0(4): Pv, Pm, Fv, Fm. 기존 max와 noisy-OR `1-(1-Pv*Fv)*(1-Pm*Fm)`를 비학습 기준으로 비교한다.
- F1(32): F0 + 각 branch seed 3개 확률 + seed variance/std/range + 현재 cache에 저장된 원래 DF-Arena fc5 파일 점수 + 각 branch window 확률의 mean/max/std/q25/q50/q90/top3mean + original DF−Fv. voice window는 softmax fake, music window는 sigmoid(logit), 각 window에서 먼저 seed 평균한다. 이것은 **진단 요약**이며 원래 파일 집계 공식을 대신하지 않는다.
- F2(48): F1 + PANN clip Presence mean/std/q90 + 527-class 점수 총합으로 정규화한 entropy의 mean/std + branch별 정렬 product mean/max/coverage/longest_fraction. 이 entropy는 multilabel 점수의 요약일 뿐 categorical ground truth나 확률 보정의 증거가 아니다.
- F2R(52): F2 + raw와 stem fake 확률 차이 2개 + voice/raw, music/raw 에너지 비율 2개. 분리 실패/신뢰도라는 해석은 가설이다.
- F1S(38): F1 + log1p(초 길이), 원래 peak, gain, 정규화 RMS, silence flag, 음악 padding 비율. shortcut 위험군으로 별도 추가/제거 실험한다.

기존 diagnostic CSV 2,852행에는 F0, seed 점수, original DF scalar가 있지만 **window 점수/PANN class 시계열/raw embedding이 없다**. 10개 NPZ에서는 현재 head를 CPU로 실행해 F1 전체를 재구성할 수 있다. original DF per-window는 저장되지 않아 복구했다고 하지 않는다. `window_truth`는 존재해도 읽어서 모델 피처로 사용하지 않는다. F2는 old cache만으로 복원할 수 없다.

새 GPU 경로의 F1/F2 telemetry는 기존 encoder 호출 결과를 기록하므로 추가 encoder 호출이 없다. 이전 캐시에 없던 PANN 정보를 얻으려면 원본 음원이 필요하다. 이번 extractor는 새 음원에 현재 Engine을 그대로 실행한다. F2R는 raw DF N_voice회와 raw MERT ceil(N_music/batch)회가 추가된다. F0/F1/F2 수치형 메타 피처는 작지만 MERT/DF/PANN/분리기의 원래 배포 비용은 사라지지 않는다. GPU 속도·VRAM·L4 적합성은 미측정이다.

정답/mask/window_truth/route/source/generator/ID/경로/split/노출/정답 기반 SNR·fake ratio는 allowlist에 없다. 길이·음량 등은 F1S 및 에너지군 F2R로 분리했다. 수천 차원 embedding, PCA, feature selection, calibration, imputation은 기본안에 없다. 후속 PCA/보정은 매 fitting fold 안에서만 학습해야 한다.

## 5. 주 경로: 현재 전문가 고정 + 새 meta-dev

전문가 학습/선택/과거 개발에 쓰이지 않은 원본으로 meta-dev 및 locked OOD를 만든다. 원본 lineage 검증 후 split하고 그 안에서만 변형한다. 합성·배깅을 많이 한 parent의 영향은 연결 component별 inverse-count weight로 줄인다. 이 경우 최적화 대상이 원래 파일 빈도 분포에서 component 균등 분포로 바뀌므로 class prior와 확률 보정이 달라질 수 있다. class_weight는 추가하지 않으며, 평가 Brier/logloss는 자연 파일 빈도에 대한 비가중값이다. `experiment_file_equal.json`은 가중치 민감도 비교용이며 같은 split에서 별도로 실행한다. LR StandardScaler의 평균/분산은 fitting rows에서 비가중으로 학습한다.

outer 5-fold grouped CV, inner 3-fold grouped CV. 기본 그룹은 두 부모·speaker·track·hash의 추이적 연결 component다. 이는 **source-disjoint CV라는 뜻이 아니다**. source grouping은 `group_mode=source`로 별도 실행할 수 있으며, cross-source mixtures가 모두 연결되어 fold를 만들 수 없다면 실패시킨다. 임의 그룹 분리로 수량을 맞추지 않는다. 최종 locked OOD는 dev와 source 및 fake generator family를 분리하고 alias 표를 사전에 검토한다.

각 outer fold의 train에서만 두 개 이내 grid 후보를 inner logloss로 선택한다. 선택된 파라미터로 outer-train known File 라벨만 fit하고 outer-eval의 모든 행을 예측한다. XGBoost early stopping용 내부 holdout도 **현재 fit partition 안**에서만 만든다. best_iteration을 정한 다음 현재 fit partition 전체로 고정 round 재학습한다. outer-eval/locked OOD는 이 과정에 없다. 빈 fold·단일 클래스 fold는 이유와 함께 NOT_EVALUATED이며 일부 행만 남겨 시스템별 다른 모집단을 비교하지 않는다.

LR: C=0.1/1, L2, max_iter=2000. ExtraTrees: bootstrap=True, 128 trees, depth 3 또는 5, min_leaf 12 또는 16, log_loss split criterion, OOB 점수 사용 안 함. XGBoost: depth 2/3, 최대 200 rounds, lr=.05, min_child_weight=10, lambda=10, alpha=.1, subsample=.8, CPU hist, early stopping 15. 깊은 트리·대규모 탐색·자동 다른 모델 대체는 없다. F0/F1 6개 시스템의 outer folds와 입력 ID는 동일하다.

outer 점수로 6개 중 하나를 고르면 winner selection optimism이 생긴다. 이 결과 자체를 선택된 전체 시스템의 최종 불편 추정치라 하지 않는다. 최종 fit은 dev 내부 CV로 파라미터를 결정하고, 모델·feature order·버전·개발 provenance·OOD manifest/cache index digest를 lock한다. OOD 파일은 lock 명령에서 읽지 않는다. 별도 `ood` 명령은 고정 모델을 로드하고 한 번만 평가하며 어떤 fit도 하지 않는다. receipt 삭제나 코드 수정까지 막는 보안 경계는 아니므로 실제 운영에서는 담당자/접근권한 분리도 필요하다.

## 6. 기존 9,504개를 재사용하는 대안 — 이번 기본 entrypoint에서는 금지

학습한 head의 in-sample 확률을 OOF라고 부를 수 없다. 현재 val은 checkpoint 선택에 사용됐고 test도 과거 개발에 노출됐다. 전역 base-OOF 테이블을 한 번 만든 뒤 그 행을 meta-CV로 나누기만 하면, meta 평가 그룹이 다른 행의 base 모델 학습/선택에 들어간 경로가 남을 수 있다.

올바른 전체-stack 평가에서는 outer 평가 그룹 H를 먼저 봉인한다. 나머지 T에서 meta 모델을 선택할 때 inner 평가 그룹 I를 다시 봉인한다. T\I 안에서 base-head cross-fitting을 새로 수행해 meta-train 피처를 만든다. 이때 각 base fit의 음악 mean/scale, head 최적화, early stopping/checkpoint 선택에서 그 base validation 그룹과 I, H가 모두 배제되어야 한다. I용 base prediction은 I/H를 전혀 보지 않은 base heads로 생성한다. 이 구조로 meta 파라미터를 선택한 다음 T 안에서 다시 cross-fitted meta-train 피처를 만들고, T 내부에서만 base 선택/학습한 heads로 H를 예측해 최종 outer 평가를 수행한다. 마지막 전체 개발집합 fit도 별도로 다시 만든다. 모든 단계의 train/selection/evaluation group ID를 저장한다.

재사용 가능한 것은 고정·파일 독립 전처리 및 고정 encoder로 만든 **정규화 전 MERT windows, DF fc5 입력 embeddings, 고정 PANN 출력**이다. supervised하게 전체 데이터로 fit한 scaler, 이전 head 확률, 외부 평가로 고른 checkpoint는 그대로 재사용할 수 없다. encoder 자체를 바꾸거나 그 선택이 평가셋에 의존하면 이 캐시 재사용 근거도 깨진다. 현재 NPZ의 embeddings는 head scaler 이전 값이므로 조건부 재사용 후보다. window_truth/identity 속 라벨은 모델 피처가 아니다.

이 대안은 encoder를 다시 돌리지 않아도 각 nested split마다 음악/음성×3seed base heads를 다시 fit해야 한다. checkpoint 선택용 추가 내부 분할과 최종 refit도 필요하다. 기존 6개 head를 그대로 쓰는 주 경로와 계산량이 다르며, 실제 재학습 시간은 측정하지 않았다. 이번 반환물은 이 대안을 자동 실행하지 않고 문서로 명확히 분리한다.

## 7. 평가와 통계 해석

현재 scorer의 tie-group 단위 이산 ROC에서 |FPR−FNR| 최소 지점을 선택하고 EER=(FPR+FNR)/2로 계산한다. 보간 EER나 미분 가능한 학습 loss로 바꾸지 않는다. 학습/선택은 logloss/BCE이고 EER은 평가에 쓴다. mask=0/unknown은 None으로 바꾸어 원래 scorer에 전달한다. 성분 Fake는 실제 Presence=1이며 라벨이 알려진 행만 평가한다.

모든 모델에 다섯 출력의 지표, 분모, unknown 개수, coverage, invalid 사유와 Brier/logloss를 보고한다. 관련 라벨이 불완전하거나 클래스가 빠진 지표가 있으면 공식 총점은 null이다. 네 출력이 고정되고 완전한 라벨로 공식 점수가 계산 가능할 때만 ΔScore=−0.45×ΔFileEER다. EER 개선은 calibration 개선과 동일하지 않다.

paired bootstrap은 connected-parent component를 복원추출하고 동일한 행/중복 가중을 baseline과 candidate에 적용한다. File 클래스가 사라진 draw는 제외하고 valid/skipped 수를 보고한다. 유효 draw가 max(30,반복의 절반) 미만이면 CI를 보고하지 않는다. CI는 고정 예측에 조건부이며 모델 선택/재학습 불확실성까지 포괄하지 않는다. category/composition/source/generator/codec/duration/SNR 조건별 결과를 별도 보고하되, 해당 메타데이터는 피처로 입력하지 않는다.

## 8. 저장·배포와 검증 범위

LR/ExtraTrees는 skops의 표준 dump/load, 추가 trusted type 없이 로드한다. XGBoost는 표준 JSON save_model/load_model이다. pickle/joblib fallback은 없다. manifest digest를 bundle 바깥에서 pin하고 파일 hash, 입력 순서, 라이브러리 버전을 검사한다. skops는 이 환경에서 설치되지 않아 해당 두 모델의 직렬화는 NOT_RUN이다. LR/ExtraTrees 학습·예측·CV는 실행했다. XGBoost round-trip 최대 차이는 0이다.

`patches/full_bank_submission.py`는 완전한 수정본이며 unified diff도 함께 제공한다. 기본 실행은 여전히 max다. 환경변수 두 개로 명시적 production lock을 제공할 때만 stacking을 켠다. `infer_record`는 한 파일의 고정 피처만 사용한다. 기본 네 출력은 그대로 복사한다. 새 오디오/GPU/L4/공식 제출/독립 OOD는 모두 미실행이며, stub encoder 테스트는 wiring 확인에 한정된다.
