# DACON DeepVoice 실험 코드

지금까지 진행한 딥보이스 탐지 실험의 모델·학습·추론 코드, 노트북, 설정과 결과 기록을 모았습니다. 원래 프로젝트의 상대 경로는 `repository/` 아래에 유지했습니다.

- [현재 계획과 남은 작업](plan.md)
- [실험별 코드 안내](EXPERIMENT_INDEX.md)
- [전체 실험 기록 — 2026-09-17 기준](docs/EXPERIMENTS_ALL_2026-09-17.md)
- [최신 Existing9504 구조](docs/PROJECT.md) · [상세 코드 위치](docs/CODE_MAP.md)
- [실제 소스 파일과 SHA-256](SOURCE_MANIFEST.json)

## 최신 완료 실험: Existing9504

[H100 학습 노트북](repository/results/existing9504_h100_20260920/DACON_Existing9504_H100.ipynb), [학습 제어 코드](repository/results/existing9504_h100_20260920/code/dacon9504a100/), [학습 당시 공통 소스](repository/results/existing9504_h100_20260920/training_snapshot/source/), [최종 제출 추론 코드](repository/submissions/Existing9504_MIL/source/)를 구분해 보존했습니다. 같은 이름의 과거 루트 파일보다 이 고정본이 해당 완료 실험의 기준입니다.

9,504개 데이터의 분할은 train 6,652 / val 1,426 / test 1,426입니다. 고정 MERT330 특징으로 WindowMIL을 학습하고, DF-Arena 음성 분류기 및 PANNs·HTDemucs 경로와 결합합니다. 기존 test 노출 이력이 있으므로 독립 평가로 해석하면 안 됩니다.

## 실행과 결과 해석

학습 코드가 존재한다는 사실만으로 모든 설정의 실행 완료나 성능을 뜻하지 않습니다. 실험별 완료·중단·미실행 상태와 공식 점수/로컬 지표 구분은 결과 기록을 따릅니다. 9월 17일 기록은 당시의 결과이며 최신 리더보드 조회 결과가 아닙니다.

작업 디렉터리는 `repository/`입니다. 실행에는 별도의 오디오, 사전학습·학습 가중치, GPU 환경과 경로 설정이 필요합니다. 환경 파일은 최신 실험의 `environment/`와 `code/dacon9504a100/`에 있습니다. 제출용 requirements는 대회 사전 설치 환경을 전제로 하므로 빈 환경용 전체 의존성 목록이 아닙니다. 이번 업로드에서는 학습을 재실행하지 않았습니다.

원본 오디오·모델 가중치·특징 캐시·가상환경·인증정보는 이 실험 소스 묶음에 포함하지 않았습니다. 노트북 출력과 실행 번호를 제거했으며, 데이터 분할 CSV와 출처·검증 메타데이터는 보존했습니다. 전체 후보 중복 목록 `combined_catalog.csv`와 작업 지시 원문은 제외했습니다. 기존 소스 안내의 전체 후보 목록 설명은 원본 로컬 패키지 기준입니다.

앞서 추가한 외부 모델·데이터 목록과 AASIST 외부 배포본은 [참고자료](references/external_resources/)로 이동했습니다. 이 자료는 실제로 수행한 실험 목록과 별개입니다.
