# 검증이 방지하는 결함

이 테스트들은 성능 개선 검증이 아니다. 실제 캐시 재생과 synthetic wiring을 구분한다.

| 테스트 함수 | 방지하는 결함 | 자료 |
|---|---|---|
| test_real_heads_cache_baseline_and_seed_replay | logit/probability 또는 seed/window 집계 혼동 | 실제 6 heads + 10 NPZ |
| test_different_timebases_and_no_invalid_padding_weight | 음악/음성 index-wise 정렬, padding 시간 과대계산 | 결정적 작은 시간 fixture |
| test_forbidden_and_missing_features_are_rejected | 라벨 feature 유입, silent reorder, NaN 0 대치 | 작은 feature table |
| test_parent_bridges_and_training_exposure_are_refused | mixed 두 부모의 간접 split 누수, 선택 노출, registry 충돌 | 명시적 lineage fixture |
| test_masked_unknown_is_not_real_and_total_stays_null | mask=0의 numeric 0을 real 정답으로 취급 | unknown label fixture |
| test_fold_preprocessing_and_estimator_use_training_only | 평가 데이터의 평균/분산이 scaler에 유입 | LR fit + 극단적 held-out 값 |
| test_xgb_json_reload_order_and_single_file_independence | 잘못된 모델 hash/order, 저장 후 변화, cohort 의존 | synthetic XGBoost |
| test_ood_requires_immutable_production_lock | 후보 lock 없이 OOD 열기 | lock gate |
| test_manifest_requires_parent_scope_and_consistent_component_truth | 부모 일부만 검토, File/성분 라벨 모순 | lineage/label fixture |
| test_paired_bootstrap_skips_one_class_draws_without_imputing | 한 클래스 bootstrap을 가짜 EER=0으로 처리 | 불균형 6개 그룹 |
| test_numerical_symbols_remain_verbatim_current_not_legacy | 다른 snapshot 수치 코드 혼입 | 원문 AST/byte hash |
| test_extended_cache_binds_provenance_and_rejects_tampering | cache identity/file 변경을 무시 | 실제 cache-derived 피처 |
| test_opt_in_telemetry_leaves_baseline_and_four_outputs_unchanged | telemetry가 baseline 수식을 변경 | 실제 heads + synthetic stub encoders, GPU parity 아님 |
| test_six_matched_cv_systems_keep_same_outer_rows_and_four_outputs | 모델마다 split/모집단이 달라짐, outer가 내부 tuning에 유입 | 72 synthetic rows / 36 groups |

실행: `PYTHONPATH=src STACKING_INPUT_ROOT=/path/to/extracted/input OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 pytest -q`.
v1의 기존 테스트 14개는 그대로 유지했다. v2 최종 실행 결과는 VERIFICATION.json과 logs/v2/에 기록한다. XGBoost 없는 최소 환경에서 전체 suite가 통과한다고 주장하지 않는다. skops save/load는 별도 smoke의 status로 남으며 이 세션에서는 LR/ET NOT_RUN이다. 원래 upstream 프로젝트의 전체 대형 GPU suite를 실행한 것은 아니다.


## v2 추가 회귀 검증

| 위치 | 먼저 정의한 방지 결함 | 실제 검사 범위 |
|---|---|---|
| `test_preprocessing_contract.py`의 cache/manifest 검사 | False/True 혼합, manifest False/cache True, 전처리 정책 누락·변경을 허용함 | 합성 수치의 실제 NPZ·JSON sidecar·index를 만들어 다시 읽음. 해시를 맞게 갱신해 단순 tamper 검사가 아니라 정책 검사인지 확인 |
| 같은 파일의 peak/gain·extraction 검사 | 모든 파일의 측정값을 같게 강제하거나 잘못된 행을 GPU 초기화 뒤에 검출함 | 서로 다른 peak/gain 허용, 모든 행 사전거부, 실제 adapter/캐시 쓰기·읽기. native encoder만 CPU test double |
| 같은 파일의 workflow·deployment 검사 | dev에서 확인한 정책이 lock/OOD/배포에서 사라짐 | 실제 grouped fit·XGBoost JSON·lock/OOD 경로와 production bundle/cache inference. 외부 자산 identity/engine은 명시적 test double이므로 GPU 검증 아님 |
| `test_smoke_optional.py` | XGBoost 미설치로 fit에서 중단해 다른 후보의 결과와 최종 보고서를 잃음 | import 차단으로 미설치 재현. LR/ET는 실제 sklearn fit. fit/save/load별 NotRunError와 보고서 보존 검사 |
| 같은 파일의 skops 미설치 검사 | skops 미설치인데 저장 성공을 기록하거나 pickle로 우회함 | metadata/import 미설치 재현, LR/ET 저장 NOT_RUN과 원인 확인 |

`reproduce_review_findings.py`는 사용자 설명 기반의 작은 독립 재현기다. 첨부되지 않은 원본 reproducer를 실행한 것으로 표기하지 않는다. 이전 코드에서 결함이 재현되는 로그와 v2에서 거부/보고서 보존을 확인하는 로그를 따로 보관한다.
