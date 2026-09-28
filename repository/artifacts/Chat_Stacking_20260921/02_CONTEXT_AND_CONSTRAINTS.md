# 현재 기준과 변경 범위

작성일: 2026-09-21. 현재 checkout HEAD: `67b2bc61f6ce8a32cfdcf5d17837872f7c5b109c`. 작업 디렉터리에는 다른 작업의 미커밋 파일이 있으므로 HEAD만으로 모든 입력 파일을 식별하지 않는다. `INPUT_MANIFEST.json`의 파일별 해시가 실제 첨부본의 기준이다.

## 기준 모델

현재 대상은 `existing9504_MERT330M_WindowMIL_DFfc5_701515`다. `B_full1696` noisy-OR나 과거 G0~G4를 현재 모델로 오인하지 않는다. `current/submission_source/model/runtime/full_bank_submission.py`가 최종 추론 기준이고, `current/training_source/full_bank_submission.py`에는 이전 candidate 문자열이 남아 있을 수 있다.

- 실제 head 최적화에는 train 6,652행을 사용했다. 전체 selected 9,504행은 특징 추출 대상이다.
- val/test는 각 1,426행. val 997행, test 1,001행은 과거 개발 노출이 기록돼 있다. val은 checkpoint 선택에도 사용됐다.
- 고정 encoder: PANNs, HTDemucs base, DF-Arena, MERT330M layer 12.
- 학습된 head: 음악 WindowMIL MLP와 음성 Linear(1280,2), 각각 seed 41/42/43.
- File: `max(vp*vf, mp*mf)`. 음악은 window logits의 log-mean-exp 뒤 sigmoid, 음성은 window softmax Fake의 max, 이후 각각 seed 평균이다.
- 일반 신규 입력은 peak 0.6 제한. 과거 constructed/legacy의 gain 보존 검사는 별도 경로다. 개발 중 임의로 음량 정책을 바꾸지 않는다.
- 현재 패키지의 CPU parity와 과거 로컬 지표는 확인 기록이다. 새 스태킹의 성능·GPU 실행·L4 충족을 뜻하지 않는다.

## 캐시에서 얻을 수 있는 것

실제 NPZ에는 `mert_windows`, `starts`, `valid_samples`, `voice_embeddings`, `voice_original`, `voice_silent`, `presence`, `samples`, `source_peak`, `normalization_gain`, `normalized_rms`, `identity`, `window_truth`가 있다. `window_truth`는 정답이므로 입력 피처로 쓰면 안 된다. 존재하는 값과 shape/dtype은 `fixtures/cache_schema.json`에도 기록한다.

`presence`는 파일 수준의 두 최댓값이다. 기존 캐시에는 PANN 전체 `[T,527]`, 구간별 Presence, raw 경로 Fake 점수, 두 stem RMS가 없다. 존재하지 않는 시간 정보를 파일 수준 Presence로 복제해 ‘구간 예측’이라고 부르면 안 된다. `starts`는 MERT 시간축이며 음성 embedding의 시간축을 자동으로 뜻하지 않는다. 실제 source의 `get_segment_starts`와 `window_layout`을 각각 확인한다.

첨부한 작은 head와 캐시로 per-seed 및 per-window 점수를 CPU에서 재계산할 수 있다. 그 재계산은 parity 확인이며 새 성능 검증이 아니다. 전체 cache와 큰 encoder 가중치, raw 오디오는 첨부하지 않았다.

## 실험 계약

1차 범위는 File head의 스태킹이다. 네 성분 출력까지 동시에 재학습하면 효과를 분리하기 어렵다. F0(기본 네 확률) → F1(기존 출력/캐시로 얻는 확장 피처) → F2(추가 추출 피처)의 비교를 구분한다. LR·ExtraTrees·XGBoost를 같은 split에서 비교한다.

독립 원본과 생성기의 확보를 우선한다. 초기 3,200개는 목표이며 확보 사실이 아니다. 모든 코드 작성은 새 데이터가 미확보여도 진행할 수 있지만, 모델 채택 결정은 실제 독립 평가 뒤에 한다. 기존 데이터의 path/ID/source/generator/label/mask는 메타데이터이지 모델 입력이 아니다.

PANN top-k/LME, Presence floor, noisy-OR는 참고 ablation이다. 최종 수식 탐색을 대규모로 늘리기보다 피처 및 학습형 결합의 효과를 분리한다. 학습 가중치/샘플링은 실제 발생 확률과 다를 수 있으므로 ‘0.8=현실에서 80%’ 같은 해석은 별도 calibration 근거 없이는 하지 않는다.

## 환경 경계

이 패키지의 대회 규정과 라이선스 기록은 보관본이다. 실제 실행 전 공식 원문을 새로 확인해야 한다. 모델 UI의 제공 이름·사용량·일반 채팅의 파일/코드 실행 도구는 확인하지 않았다. 가능한 도구만 사용하고 제약을 명시한다.

원격 인스턴스 생성/재시작, 로그인·토큰 요청, 브라우저 조작, 실제 대회 제출은 이번 작업이 아니다. 기존 VESSL을 재사용하고 저장소 확인·업로드를 GPU 시작 전에 완료해야 한다는 운영 제약은 이후 실행 문서에 보존한다. 특정 workspace가 현재 실행 가능하다고 가정하지 않는다.
