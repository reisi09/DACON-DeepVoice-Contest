# 반환 코드 검토 — 2026-09-21

판정: 스태킹 실험 코드는 구현돼 있다. 전처리 계약 누락을 수정한 뒤 신규 데이터 실험으로 넘어가야 한다. 성능 향상이나 제출 적합성은 아직 확인되지 않았다. 반환 원본은 수정하지 않았다.

## 수정할 사항

### 1. 학습·추론의 전처리 일치를 보장하지 못함

`src/dacon_stacking/workflow.py:34`의 `load_records`는 코드·체크포인트·numeric mode·feature schema만 비교한다. `preprocessing`은 캐시 자체 해시에 포함되지만 manifest와의 일치, 캐시 간 정책 일치, 배포 정책 일치는 검사하지 않는다. `lock_candidate`도 같은 네 항목만 저장한다.

실제 추출은 `gpu.py:116`에서 행별 `preserve_gain`을 허용하지만, 배포의 `predict_engine_file`은 항상 `preserve_gain=False`를 사용한다. 따라서 preserve_gain=True로 만든 피처에 학습한 모델이 다른 전처리로 배포될 수 있다. 현재 기본값 False만 사용하는 경로에서 오류가 발생했다는 뜻은 아니다.

재현: 정상 해시로 저장한 preserve_gain=False/True 캐시를 함께 읽어도 통과했다. manifest를 False로 바꾸고 True 캐시를 그대로 읽어도 통과했다. `reproduce_preprocessing_gap.py`와 `preprocessing_gap.log` 참조. 이 재현은 캐시 로더의 누락을 검증하며, 전체 production lock이나 실제 GPU 추론을 실행한 것은 아니다.

수정: 파일마다 달라지는 source_peak·normalization_gain 측정값과 공통 전처리 정책을 구분한다. 정책을 cache → dev → lock → OOD → deployment 전체에서 검증한다. 현재 배포를 유지한다면 production 경로에서는 preserve_gain=True를 거부하는 것이 가장 간단하다. 진단용 예외 경로는 production 후보와 분리한다.

### 2. XGBoost가 없으면 smoke의 저장 검증 보고서가 사라짐

`src/dacon_stacking/demo.py:42`의 `fit_final_dev`가 모델별 `try` 밖에 있다. 로컬에서 XGBoost가 없는 상태로 실행하자 CLI 전체가 NOT_RUN으로 종료했고, 이미 계산한 LR/ExtraTrees 상태도 `serialization_results.json`에 저장되지 않았다. `synthetic_cv.json`은 남아 있다.

수정: fit부터 save/load까지 같은 모델별 NotRunError 처리 안에 넣고 각 후보 상태를 저장한다. 모델 대체나 pickle 우회는 필요 없다.

## 직접 재검증한 범위

- 반환 SHA-256 목록 74개, 입력 manifest 133개가 모두 일치했다.
- 실제 head 6개와 캐시 10개를 CPU에서 재생했다. 과거 참조 예측과 최대 절대 차이는 0.0012563467025756836으로 제공된 0.002 허용치 이내였다. 동일 GPU 실행 결과라는 의미는 아니다.
- 반환 테스트 중 실행 가능한 12개가 통과했다. 부모 그룹 누수·노출 데이터 거부, mask 처리, 학습 fold 내부 scaler, 피처 순서·해시 검증, 시간 구간 정렬, stub Engine의 기존 출력 보존을 확인했다. stub 검사는 실제 오디오/GPU parity 증거가 아니다.
- synthetic 72행에서 F0/F1 × LR/ExtraTrees의 네 CV 경로가 OK였다. 합성 데이터 결과이므로 탐지 성능 근거로 사용하지 않는다.
- 로컬에 xgboost와 skops가 없어 XGBoost 검증 두 항목은 제외했고, 모델 저장·재로드는 재검증하지 못했다. 반환 ZIP의 XGBoost round-trip 성공 기록은 작성자 측 결과로만 취급한다.

신규 meta-dev·locked OOD 음원 수집, 실제 GPU 추출, L4 시간·메모리, 최종 제출 패키지, 일반화 성능 향상은 이번 검토 범위에서 실행하지 않았다.

## 다음 작업

같은 ChatGPT 대화에 `FIX_PROMPT.md`와 검토 ZIP을 보내 위 두 부분만 수정한다. 수정본을 재검토한 뒤 미노출 데이터로 비교 실험한다. 기존 val/test나 synthetic 점수로 최종 후보를 승인하지 않는다.
