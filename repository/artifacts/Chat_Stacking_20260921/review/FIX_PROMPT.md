이전 DACON_Stacking_Return.zip을 Codex에서 검사했다. 첨부 검토 자료를 확인하고 아래 두 부분만 수정한 DACON_Stacking_Return_v2.zip을 반환해라. 기존 입력 ZIP과 반환 코드가 이 대화에 있다. 설계를 다시 만들거나 후보 모델을 추가하지 마라.

1. 전처리 계약 누락을 수정해라.
   - workflow.load_records는 preprocessing을 비교하지 않는다. preserve_gain=False/True 캐시 혼합과 manifest=False/cache=True 불일치가 모두 통과했다. 첨부 reproduce_preprocessing_gap.py로 재현할 수 있다.
   - extract_audio는 행별 preserve_gain을 허용하지만 predict_engine_file은 False로 고정돼 있다. 학습 피처와 배포 피처의 전처리 계약을 맞춰라.
   - 파일별 source_peak/normalization_gain 측정값 자체를 모든 행에서 같도록 강제하지 마라. 공통 정책과 파일별 측정값을 구분해라.
   - 최소 수정안을 우선해라. 현재 배포 정책을 유지한다면 production 경로에서 preserve_gain=True를 거부하고, 진단용 예외는 production 후보에 들어가지 못하게 해라. cache/manifest/dev/lock/OOD/deployment의 실제 호출 경로에 적용해라.
   - 허용된 동일 정책은 통과하고 다른 정책은 거부되는지 검사해라. 서로 다른 음원에서 측정된 peak/gain 차이는 허용돼야 한다. CPU 검사와 실제 GPU parity를 구분해라.

2. optional dependency가 없는 smoke 경로를 수정해라.
   - demo.run_smoke의 fit_final_dev가 모델별 try 밖이라 XGBoost 미설치 시 serialization_results.json 저장 전에 종료된다.
   - 후보별 fit/save/load에서 NotRunError를 기록하고 나머지 후보 결과도 저장해라. 임의 모델 대체나 pickle 우회는 금지한다.
   - XGBoost 미설치를 재현하는 작은 검사로 다른 후보의 상태와 최종 보고서가 보존되는지 확인해라. skops가 없으면 해당 저장 검사는 NOT_RUN으로 기록해라.

추가 확인: LR/ExtraTrees의 skops 저장·재로드가 이 환경에서 가능하면 실제 실행해라. 불가능하면 NOT_RUN과 원인을 유지해라. 최종 결과에 적용 파일, 실행 명령, 검증 로그, 미실행 범위를 포함하고 OUTPUT_SHA256.json을 갱신해라. 실제 신규 음원·GPU·L4·공식 점수를 검사하지 않았다면 성공한 것으로 쓰지 마라. GPU 시작, VESSL 작업, 공식 제출은 하지 마라.
