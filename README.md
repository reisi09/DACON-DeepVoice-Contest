# DACON DeepVoice Contest · External Resources

딥보이스·AI 음악 탐지에 사용할 외부 모델과 데이터 후보를 정리한 저장소입니다. GitHub 스타가 많은 오디오 프로젝트를 우선 조사하고, 직접 위조 탐지·음성 표현·음악 표현·전처리 용도를 구분했습니다.

**기준일: 2026-09-21 KST.** 스타는 GitHub API의 실제 조회값이며 성능 점수가 아닙니다. 목록은 선별한 후보의 순위로, 전체 GitHub를 망라하지 않습니다.

| 자료 | 내용 |
|---|---|
| [모델 목록](MODEL_CATALOG.md) / [CSV](MODELS.csv) | 12개 후보, 스타순, 역할·가중치 위치·라이선스 |
| [데이터 목록](DATA_CATALOG.md) / [CSV](DATASETS.csv) | 8개 데이터 원천, 공식 다운로드·라벨·노출 조건 |
| [다운로드 안내](DOWNLOADS.md) | 실제 포함 파일과 별도 다운로드 범위 |
| [출처 스냅샷](SOURCES.json) | 조회 시각, GitHub 메타데이터, HF revision |
| [AASIST 공식 코드·가중치](AASIST_official_with_weights.zip) | 고정 commit의 소스와 AASIST/AASIST-L 가중치 2개, MIT LICENSE |
| [파일 무결성](INVENTORY.json) | 업로드 파일 SHA-256 및 AASIST 가중치별 해시 |

## 현재 상태

외부 자원 선정과 AASIST 원본 파일 보관을 완료했습니다. 대형 가중치와 원본 음원은 공식 배포 링크로 관리합니다. 새 후보 모델의 학습·추론이나 성능 비교는 실행하지 않았습니다. 이 자료는 DACON 제출 ZIP이 아닙니다.

기존 프로젝트는 PANNs·HTDemucs·DF-Arena·MERT 특징으로 다섯 확률을 계산합니다. 현재 검토한 스태킹 구현은 File 확률을 별도 분류기로 결합하는 구조입니다. 이 목록의 모델을 추가하려면 새 특징 추출과 미노출 데이터 검증을 거쳐야 하며 기존 스태킹에 자동으로 연결되지 않습니다.

## 선정 원칙

공식 저장소·배포처를 우선하고 스타 수와 과제 적합성을 함께 기록합니다. GitHub가 공식 배포처가 아닌 데이터는 N/A로 남깁니다. 코드·가중치·데이터의 라이선스를 구분하며, 기존 모델이 본 데이터는 새 독립 평가셋으로 간주하지 않습니다.
