# 입력 파일 읽는 순서

| 순서 | 경로 | 용도 |
|---|---|---|
| 1 | `02_CONTEXT_AND_CONSTRAINTS.md` | 현재 모델, 작업 범위, 누수 위험 |
| 2 | `current/PROJECT.md`, `current/DATASETS.md` | 현재 실험과 데이터 설명 |
| 3 | `current/submission_source/model/runtime/full_bank_submission.py` | 현재 최종 추론 및 head 로딩 |
| 4 | 같은 폴더의 `mert12_submission.py`, `localization_models.py`, `eight_heads.py` | 시간축, WindowMIL, MLP 정의 |
| 5 | `current/submission_source/model/assets/baseline/script.py` | PANN, DF-Arena, 분리기 연결 및 기본 시간축 |
| 6 | `current/training_code/dacon9504a100/features.py`, `training.py`, `evaluation.py` | 캐시 생성, train 정규화·학습, val 선택 |
| 7 | `current/training_source/` | 학습 당시 공통 모듈 snapshot |
| 8 | `data/selected_list.csv`, `runtime_manifest.csv`, `catalog_list.csv` | 실제 선택·노출·예비 후보 목록 |
| 9 | `diagnostic/val_predictions.csv`, `test_predictions.csv` | 기존 실제 예측. 진단·wiring 확인 전용 |
| 10 | `fixtures/heads/`, `fixtures/cache/`, `fixtures/cache_schema.json` | 작은 실제 checkpoint 6개, NPZ 10개, schema |
| 11 | `evidence/` | 후보 lock·val/test 지표·parity 기록 |
| 12 | `reference/legacy/`, `reference/EXPERIMENTS_ALL_2026-09-17.md` | 기존 logistic 결합·OOF 검증과 과거 결과 참고 |
| 13 | `reference/COMPETITION_CONTRACT.md` | 보관된 규정과 원문 URL |

`current/training_code/` 아래 package가 참조하는 공통 모듈은 `current/training_source/`에 있다. 현재 제출 모듈은 `current/submission_source/model/runtime/`에 있다. 서로 같은 이름의 모듈을 무작정 한 폴더에 합치지 않는다. 실제 통합 시 원래 프로젝트 상대 경로로 매핑한 변경 목록을 반환한다.

캐시 파일명은 원래 ID를 안전한 이름과 짧은 해시로 바꾼 값이다. `fixtures/index.json`의 ID 매핑을 사용한다. 문자열 치환으로 ID를 추측하지 않는다. index에는 실제 parity 기록의 기준 예측도 들어 있으며, 이는 head/cache 재계산의 비교값으로만 쓴다.

현재 데이터의 최종 split은 `current_split` 또는 runtime manifest의 실제 `split`이 기준이다. `previous_split`, `prior_split`, `assigned_group_split`으로 덮어쓰지 않는다. `*_mask=0`은 해당 라벨을 모른다는 뜻이다.

`reference/legacy/`는 학습형 결합과 누수 방지 구현의 참고본이다. 전체 과거 런타임을 포함한 것이 아니므로 단독 실행을 보장하지 않는다. 필요한 로직을 확인해서 현재 schema와 연결해야 한다. `reference` 문서의 과거 상대 링크는 이 축소 패키지에 없는 파일을 가리킬 수 있다.

`INPUT_MANIFEST.json`은 각 파일의 바이트 수·SHA-256·원래 위치를 기록한다. ZIP 내부에는 중첩 ZIP, 대형 encoder 가중치, raw 오디오가 없다. `fixtures/heads/*.pt`는 큰 encoder가 아니라 작은 head다.
