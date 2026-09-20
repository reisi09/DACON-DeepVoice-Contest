# 외부 원문 확인 기록 — 2026-09-21

이 목록은 데이터 수집/권리 최종 승인이나 GPU 호환성 보장이 아니다. 코드와 수치는 입력 snapshot 및 이 세션 로그가 우선한다.

| 목적 | 공식 원문 | 이번 확인 |
|---|---|---|
| 대회 규칙 | https://dacon.io/competitions/official/236749/overview/rules | 공개 자원/최소 비영리 허용, 출처 명시, 평가 데이터 학습 금지, 파일별 독립 예측 확인 |
| sklearn persistence | https://scikit-learn.org/stable/model_persistence.html | skops/비-pickle 경로와 버전 호환 경고 확인; rolling 최신 문서이며 실행 sklearn은 1.8.0 |
| skops persistence | https://skops.readthedocs.io/en/stable/persistence.html | 신뢰할 출처 및 type 검토 계약 확인 |
| skops 고정 패키지 | https://pypi.org/project/skops/0.13.0/ | PyPI 릴리스/해시 확인; 다운로드 및 pip 설치 실패, 실행 호환성 미검증 |
| XGBoost 3.1 IO | https://xgboost.readthedocs.io/en/release_3.1.0/tutorials/saving_model.html | JSON/UBJ 지원 포맷과 save_model 사용 확인; 3.1.3 JSON round-trip CPU 실행 |
| 데이터 후보 라이선스 | DATA_CANDIDATES.csv의 공식 URL들 | ASVspoof LICENSE 원문, OpenSLR 표기, HAIM/SONICS card 확인; 오디오는 다운로드하지 않음 |

GitHub repository/API 및 VESSL API/인스턴스에는 접속하지 않았다. 검색 결과에 표시되는 링크를 remote repository 접근/실행으로 사용하지 않았다. 공식 페이지의 현재 내용이 포함된 과거 로컬 revision의 권리·내용을 자동으로 대체하지 않는다.
