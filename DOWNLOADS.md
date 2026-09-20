# 가중치와 데이터 받기

## 포함된 실제 모델 파일

`AASIST_official_with_weights.zip`은 [clovaai/aasist의 고정 commit](https://github.com/clovaai/aasist/tree/a04c9863f63d44471dde8a6abcb3b082b07cd1d1)에서 받은 공식 소스 ZIP이다. AASIST와 AASIST-L의 가중치·config·모델 코드·LICENSE를 함께 보존했다. 실행 환경과 사용법은 ZIP 안의 원본 README를 따른다. 사용자 모델로 재학습하거나 DACON 제출 형식으로 변환하지 않았다.

## 큰 모델

MODELS.csv의 weights_url로 받는다. HF 후보는 SOURCES.json의 sha로 revision을 고정할 수 있다. 예시는 실행 안내이며 이번 업로드에서 실행하지 않았다.

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="openai/whisper-small",
    revision="973afd24965f72e36ca33b3055d56a652f456b4d",
    local_dir="external_weights/whisper-small",
)
```

모델마다 전처리기·설정·가중치를 함께 보존한다. MERT, PANNs, AST, Whisper의 sample rate와 입력 형식은 동일하지 않다. 공통 환경에 모든 상류 requirements를 한꺼번에 설치하지 않고 후보별 환경에서 원본 예제를 확인한다.

## 원본 음원

DATASETS.csv의 source_url에서 필요한 split·언어·생성기만 받는다. 원본 음원과 대형 가중치는 이번 Git 저장소에 복제하지 않았다. 각 배포처의 조건과 원본 해시를 기록하고 저장소에는 manifest를 유지한다. SONICS의 배포 fake 음원과 real_songs.csv의 외부 링크는 구분한다.

## 재현 범위

이 카탈로그는 후보 선정과 파일 출처 관리용이다. 새 모델 추론·학습, 신규 데이터 다운로드·디코딩·라벨 검증, DACON 제출 환경 검증을 완료했다는 뜻이 아니다. 대회 규정은 [공식 페이지](https://dacon.io/competitions/official/236749/overview/description)에서 실행 시점에 확인한다.
