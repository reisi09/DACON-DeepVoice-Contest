# DACON DeepVoice 실험 코드

지금까지 진행한 딥보이스 탐지 실험의 모델·학습·추론 코드, 노트북, 설정과 결과 기록을 모았습니다. 원래 프로젝트의 상대 경로는 `repository/` 아래에 유지했습니다.

## 대회 개요

**[딥보이스 범죄 대응을 위한 AI 탐지 모델 경진대회](https://dacon.io/competitions/official/236749/overview/description)** (DACON 236749)

| 항목 | 내용 |
|---|---|
| 주제 | 신종 보이스피싱 범죄 대응을 위한 AI 딥보이스 탐지 모델 개발 |
| 주최 / 주관 / 운영 | 행정안전부·한국지능정보사회진흥원 / 국립과학수사연구원 / 데이콘 |
| 상금 | 4,200만 원 |
| 일정 | 참가 2026-08-18 ~ 09-29, 대회 시작 08-26, 리더보드 제출 마감 09-29 10:00, 대회 종료 09-30 10:00. 2차 평가 자료 제출 ~10-05, 최종 발표 10-16 |
| 과제 | 오디오 1,200개 각각에 대해 AI 생성 여부(파일·음성 성분·음악 성분)와 음성·음악 성분의 존재 확률을 예측 |
| 평가 | **Score = 0.9 × ADS + 0.1 × CPS** (아래 식 참고) |
| 순위 | 테스트 100%가 Public이고, 종료 시점의 Public이 Private(1차 평가)이다. 1차 상위 15팀이 2차 평가(코드·보고서·데이터 검증) 대상이다 |
| 제출 방식 | 코드 제출(`submit.zip` ≤ 10 GB, 압축 해제 후 ≤ 32 GB). 오프라인 L4 22.4 GiB, 6 vCPU, 28 GB RAM에서 추론 60분 이내, 패키지 설치 10분 이내 |
| 주요 규칙 | 최소 비영리 이용이 허용된 공개 자원만 사용하고 출처를 명시한다. 평가 데이터로 학습·튜닝·pseudo-labeling을 하지 않으며, 파일마다 독립적으로 예측한다 |

평가식:
- ADS = 0.5·(1 − File EER) + 0.2·(1 − Voice EER) + 0.3·(1 − Music EER)
- CPS = 0.5·Voice Presence AUC + 0.5·Music Presence AUC

Voice EER은 음성이 있는 샘플에서만, Music EER은 음악이 있는 샘플에서만 계산한다.

## 최종 결과 (대회 종료 2026-09-29 10:00 KST)

| 항목 | 값 |
|---|---|
| 최종 제출 모델 | W_new (`repository/submissions/DataV4_W_new/submit.json`, zip sha256 `58e7e3bd460cde93b6cb65aae02da6985b9e2d5c17c5f9f88622f58df04d1c68`) |
| 리더보드 | total **0.86652**, ADS **0.85195**, CPS **0.99767** |
| 순위 | **29위** |
| 이전 최고 | S7 ADS 0.8374 → W_new가 +0.0146 |

- **모델 구조**: S7 구조를 그대로 쓰고, 모든 head를 W 가중치로 다시 학습했다. 학습 데이터는 bank v4 P1 행에 새 데이터 40,214행을 더했다.
- **고정 인코더**: HTDemucs, DF-Arena-1B, AntiDeepfake XLS-R-1B, WavLM-Large, MERT-330M, EAT-large, SSLAM, PANNs.
- **학습한 head**:
  - 음성: DF-Arena·AntiDeepfake(라벨 보완 L)·WavLM h8 멤버를 1/0.5/1로 융합
  - 음악: MERT 6·12층, EAT 16, SSLAM 5 특징 위의 WindowMIL
  - FILE: 후보 규칙을 0/9/1로 가중
- **자세한 기록**:
  - 구성과 검증: [협업 문서](repository/docs/collab/README.md), [제출 계획과 LB 결과](repository/docs/collab/SUBMISSION_PLAN_20260929.md)
  - 마지막 날의 기각 실험 X1–X3: [x2](repository/research/top15/work5/x2_20260928/SUMMARY.md), [x3](repository/research/top15/work5/x3_20260929/SUMMARY.md)
- 음원, 특징, 가중치, 제출 zip은 이 저장소에 없다. 대회가 끝난 뒤 작업 환경의 데이터(AI-Hub 포함)는 삭제했다.

- [협업 문서: 모델·데이터·검증 요약 — 2026-09-28](repository/docs/collab/README.md)
- [Top-15 research plan across all branches — 2026-09-27](docs/TOP15_RESEARCH_PLAN_2026-09-27.md)
- [Research results across all branches — 2026-09-27](docs/RESEARCH_RESULTS_2026-09-27.md)
- [현재 계획과 남은 작업](plan.md)
- [E08 성능 하락 진단·기존 DataV4와 CPU 비교](docs/E08_REGRESSION_2026-09-24.md)
- [E08 제출 ZIP 생성·실제 음원 CPU 추론 검증](docs/E08_SUBMISSION_2026-09-24.md)
- [완료 캐시 5,952개 보존·E08/기준선 학습 완료](docs/A100_PARTIAL_TRAINING_2026-09-24.md)
- [새 A100 workspace·E00/E04 특징 추출 시작](docs/A100_RUN_2026-09-24.md)
- [A100 지원 변경·실행 묶음·음원 이전 검증 완료](docs/A100_PREPARATION_2026-09-24.md)
- [GPU 학습 사전준비·E00/E04·CPU 검증 완료](docs/GPU_PREPARATION_2026-09-24.md)
- [대회 규칙 재검토·MLAAD 146개 추가·roles-v6](docs/MLAAD_DECISION_2026-09-24.md)
- [CPU 자동 검토·신규 데이터 1,331개 학습 편입 — 2026-09-24](docs/CPU_ADMISSION_2026-09-24.md)
- [추가 수집 후 CPU 검수·roles-v4·캐시 확인 — 2026-09-24](docs/CPU_FOLLOWUP_2026-09-24.md)
- [추가 원본 2,800개 수집·검증 기록 — 2026-09-24](docs/ADDITIONAL_DATA_COLLECTION_2026-09-24.md)
- [CPU 준비 결과·VESSL 업로드·남은 조건 — 2026-09-24](docs/CPU_PREPARATION_2026-09-24.md)
- [LB 개선 workflow](repository/docs/superpowers/plans/2026-09-24-dacon-lb-improvement-workflow.md)
- [기존 계획 기록](plan.md)
- [실험별 코드 안내](EXPERIMENT_INDEX.md)
- [전체 실험 기록 — 2026-09-17 기준](docs/EXPERIMENTS_ALL_2026-09-17.md)
- [최신 Existing9504 구조](docs/PROJECT.md) · [상세 코드 위치](docs/CODE_MAP.md)
- [초기 소스 묶음과 당시 SHA-256](SOURCE_MANIFEST.json)
- [Data v4 수집·생성·split 설계와 코드 — 2026-09-23](repository/data_v4/DESIGN.md)

## 최신 완료 실험: Existing9504

[H100 학습 노트북](repository/results/existing9504_h100_20260920/DACON_Existing9504_H100.ipynb), [학습 제어 코드](repository/results/existing9504_h100_20260920/code/dacon9504a100/), [학습 당시 공통 소스](repository/results/existing9504_h100_20260920/training_snapshot/source/), [최종 제출 추론 코드](repository/submissions/Existing9504_MIL/source/)를 구분해 보존했습니다. 같은 이름의 과거 루트 파일보다 이 고정본이 해당 완료 실험의 기준입니다.

9,504개 데이터의 분할은 train 6,652 / val 1,426 / test 1,426입니다. 고정 MERT330 특징으로 WindowMIL을 학습하고, DF-Arena 음성 분류기 및 PANNs·HTDemucs 경로와 결합합니다. 기존 test 노출 이력이 있으므로 독립 평가로 해석하면 안 됩니다.

## 실행과 결과 해석

학습 코드가 존재한다는 사실만으로 모든 설정의 실행 완료나 성능을 뜻하지 않습니다. 실험별 완료·중단·미실행 상태와 공식 점수/로컬 지표 구분은 결과 기록을 따릅니다. 9월 17일 기록은 당시의 결과이며 최신 리더보드 조회 결과가 아닙니다.

작업 디렉터리는 `repository/`입니다. 실행에는 별도의 오디오, 사전학습·학습 가중치, GPU 환경과 경로 설정이 필요합니다. 환경 파일은 최신 실험의 `environment/`와 `code/dacon9504a100/`에 있습니다. 제출용 requirements는 대회 사전 설치 환경을 전제로 하므로 빈 환경용 전체 의존성 목록이 아닙니다. 이번 업로드에서는 학습을 재실행하지 않았습니다.

원본 오디오·모델 가중치·특징 캐시·가상환경·인증정보는 이 실험 소스 묶음에 포함하지 않았습니다. 노트북 출력과 실행 번호를 제거했으며, 데이터 분할 CSV와 출처·검증 메타데이터는 보존했습니다. 전체 후보 중복 목록 `combined_catalog.csv`와 작업 지시 원문은 제외했습니다. 기존 소스 안내의 전체 후보 목록 설명은 원본 로컬 패키지 기준입니다.

앞서 추가한 외부 모델·데이터 목록과 AASIST 외부 배포본은 [참고자료](references/external_resources/)로 이동했습니다. 이 자료는 실제로 수행한 실험 목록과 별개입니다.
