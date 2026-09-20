# DACON 236749 대회 규정 확인 기록

확인일: **2026-09-13**. 공식 페이지와 공개 토크의 운영진 답변을 확인한 기록이다. 계정의 현재 점수·순위·남은 제출 횟수는 확인하지 않았다.

## 데이터와 예측 대상

학습 데이터는 제공하지 않는다. 공개 `data/test/`의 WAV 3개는 입력 형식 확인용 더미다. **더미와 `sample_submission.csv`를 정답이 있는 검증셋으로 사용하면 안 된다.** 실제 평가에서는 서버가 같은 경로의 데이터를 비공개 오디오 1,200개로 교체한다. 출력은 실제 평가 ID·행 수를 따르는 UTF-8 CSV이며, ID와 아래 5개 예측값으로 구성한다. [공식 데이터](https://dacon.io/competitions/official/236749/data)

| 출력 컬럼 | 의미 |
|---|---|
| `FILE_FAKE_PROB` | 파일 전체가 FAKE일 확률 |
| `VOICE_FAKE_PROB` | 음성 성분이 FAKE일 확률 |
| `MUSIC_FAKE_PROB` | 음악 성분이 FAKE일 확률 |
| `VOICE_PRESENT_PROB` | 음성 존재 확률 |
| `MUSIC_PRESENT_PROB` | 음악 존재 확률 |

각 값은 0–1이다. 평가 오디오는 4–60초, 16 kHz이며 모노·스테레오, MP3/WAV/FLAC 등과 전화채널 샘플이 포함된다. 보컬은 음성, 보컬+반주는 혼합이다. 어느 성분이든 AI 생성이면 파일 전체가 FAKE다. 성분을 새로 생성하지 않는 잡음 제거·음량 조정 등은 REAL로 본다. [공식 개요](https://dacon.io/competitions/official/236749/overview/description)

## 평가 산식

```text
ADS = 0.5*(1-File EER) + 0.2*(1-Voice EER) + 0.3*(1-Music EER)
CPS = 0.5*Voice Presence ROC-AUC + 0.5*Music Presence ROC-AUC
Score = 0.9*ADS + 0.1*CPS
```

공식 EER 계산 코드는 다음과 같다. FAKE가 양성 클래스 1이다.

```python
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1, drop_intermediate=False)
fnr = 1 - tpr
idx = np.argmin(np.abs(fpr - fnr))
EER = (fpr[idx] + fnr[idx]) / 2
```

총점에서 파일·음성·음악의 `1-EER` 가중치는 각각 0.45, 0.18, 0.27이며 두 존재 AUC는 각각 0.05다. Public은 평가 데이터 100%, Private은 종료 시점 Public이다. [공식 평가](https://dacon.io/competitions/official/236749/overview/evaluation)

- Voice/Music EER은 **정답 PRESENT=1인 파일만** 평가한다. 예측 존재확률로 마스크를 만들지 않는다. 파일 어느 구간에든 성분이 있으면 PRESENT=1이다. [운영진 답변](https://dacon.io/competitions/official/236749/talkboard/417333)
- EER은 교차점을 보간하지 않는다. 최소 차이가 같으면 `np.argmin`이 첫 번째를 선택한다. `drop_intermediate=False`를 유지한다. [NumPy](https://numpy.org/doc/stable/reference/generated/numpy.argmin.html), [scikit-learn 1.8](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.roc_curve.html)
- 빈 성분 집합·단일 클래스 검증셋에 대한 대회 예외 처리 규칙은 공개되지 않았다. 자체 검증에서 임의 점수로 대체하지 않고 평가 불가로 구분한다.

## 제출 환경과 금지 사항

| 항목 | 조건 |
|---|---|
| ZIP 최상위 | `model/`, `script.py`, `requirements.txt` |
| 입력·출력 | `data/test/` → `output/submission.csv` |
| 실행 | 패키지 설치 후 `python script.py` |
| 하드웨어 | L4 22.4 GiB, 6 vCPU, RAM 28 GB |
| 환경 | Ubuntu 22.04.5, Python 3.11.15, CUDA 12.8 |
| 제한 | 설치 10분, 추론 60분, ZIP 10 GB, 해제 후 32 GB |
| 오류 차감 | 설치 오류는 미차감, 코드 실행 후 오류는 차감 |

기본 설치 버전 중 주요 항목은 `torch/torchaudio==2.7.1+cu128`, `numpy==1.26.4`, `pandas==2.0.3`, `scikit-learn==1.8.0`, `transformers==4.57.6`, `librosa==0.10.2.post1`, `soundfile==0.12.1`, `demucs==4.0.1`, `panns-inference==0.1.1`이다. 기본 패키지는 requirements에 중복해서 넣지 않도록 권장한다. [공식 평가](https://dacon.io/competitions/official/236749/overview/evaluation)

패키지 설치 이후 인터넷은 차단된다. 추론에 필요한 가중치·설정·토크나이저를 모두 ZIP에 포함하고 로컬 경로로 읽어야 한다. 추론 중 외부 API 호출·모델 다운로드는 불가능하다. [공식 평가와 제출 환경](https://dacon.io/competitions/official/236749/overview/evaluation)

하루 최대 **3회**, 사용 언어는 Python이다. 비공개 평가 데이터로 추가 학습·튜닝·Pseudo-Labeling을 할 수 없다. 파일마다 독립적으로 예측해야 하며 다른 파일의 정보·예측값·통계로 보정할 수 없다. 한 파일 내부의 세그먼트 추론·집계는 허용된다. [공식 규칙](https://dacon.io/competitions/official/236749/overview/rules)

규정 안에서 후처리 변경·진단 제출은 허용된다. 점수가 나온 사실만으로 규정 준수가 입증되지는 않으며, 대회 중 불시 수동 확인과 종료 후 검증을 진행한다. [후처리 답변](https://dacon.io/competitions/official/236749/talkboard/417353?category=&dtype=&fType=&page=&ptype=), [독립 예측 답변](https://dacon.io/competitions/official/236749/talkboard/417344?category=&dtype=&fType=&page=&ptype=)

## 외부 자원과 2차 평가 자료

공개 접근이 가능하고 최소 비영리 사용이 허용된 데이터·모델·API를 활용할 수 있으나 개별 이용조건을 준수하고 출처를 밝혀야 한다. 2차 평가에는 재현 가능한 학습 코드, **모델 개발 보고서와 학습데이터 구성 보고서 각 HWP 1개**, 실제 학습 데이터와 팀원 정보를 제출한다. [공식 규칙](https://dacon.io/competitions/official/236749/overview/rules)

| 구분 | 운영진이 확인한 제출·권리 조건 |
|---|---|
| 외부 학습 데이터 | 실제 사용 파일이 필요하다. URL·스크립트·checksum으로 대체할 수 없다. 제3자 제공 자체를 금지하는 데이터는 사용 불가다. [09-01 답변](https://dacon.io/competitions/official/236749/talkboard/417280) |
| 사전학습 모델 | 참가자가 fine-tuning한 데이터만 제출하며 upstream pretraining corpus는 제외된다. [09-01 답변](https://dacon.io/competitions/official/236749/talkboard/417280) |
| 증강 데이터 | 중간 파일은 별도 저장하지 않아도 된다. 원본·코드·설정·seed 등으로 동일 결과를 재현해야 한다. [09-01 답변](https://dacon.io/competitions/official/236749/talkboard/417280) |
| 생성 학습 오디오 | 실제 생성 파일과 생성 스크립트·모델·버전/revision·seed·prompt·처리 매개변수를 함께 제출한다. [08-27 답변](https://dacon.io/competitions/official/236749/talkboard/417198) |
| 대용량 2차 자료 | 용량 제한 없이 Google Drive 등으로 전달할 수 있다. 추론 ZIP의 10 GB 제한과 별개다. [09-07 답변](https://dacon.io/competitions/official/236749/talkboard/417336) |
| 외부 자원 권리 | 운영진 검증으로 기존 권리가 바뀌지 않는다. 라이선스가 대회 사용과 검증 제출을 허용해야 한다. [09-03 답변](https://dacon.io/competitions/official/236749/talkboard/417310) |

CC BY-NC와 CC BY-NC-SA도 공개 접근성·최소 비영리 사용 허용·라이선스 준수를 조건으로 허용된다. 데이터별 원저작물 조건까지 확인해야 한다. [CC BY-NC](https://dacon.io/competitions/official/236749/talkboard/417193), [CC BY-NC-SA](https://dacon.io/competitions/official/236749/talkboard/417212)

2차 평가 대상은 검증을 통과한 상위 15팀이다. 공식 평가 페이지의 내장 이미지 표를 직접 확인한 배점은 다음과 같다.

| 항목 | 배점 |
|---|---:|
| 모델 성능 | 30 |
| 학습데이터 구성 | 20 |
| 모델 개발 | 25 |
| 결과 해석 및 일반화 | 15 |
| 실무 활용성 | 10 |

성능 환산은 `30*(대상팀 점수/대상팀 최고점수)^N`이며 `N`은 비공개 1–5다. [공식 평가](https://dacon.io/competitions/official/236749/overview/evaluation)

## 일정과 문서 불일치

| 항목 | 공식 본문 표기 |
|---|---|
| 팀 병합 마감 | 2026-09-23 23:59 |
| **리더보드 제출 마감** | **2026-09-29 10:00** |
| 대회 종료 | 2026-09-30 10:00 |
| 2차 자료 제출 | 2026-09-30 12:00–2026-10-05 10:00 |

제출 마감은 대회 종료보다 하루 빠르다. 해당 일정 본문에 시간대는 별도로 명시되어 있지 않다. [공식 일정](https://dacon.io/competitions/official/236749/overview/schedule)

| 불일치·미확인 항목 | 근거와 적용 기준 |
|---|---|
| `open/` 대 `data/` | 평가 페이지의 한 유의 문장은 `open/`을 쓰지만 데이터 설명·제출 구조·공식 가이드는 `data/test/`에 일치한다. 구현 기준은 `data/test/`로 둔다. |
| 종료 연도 | 일정의 기간 문장에는 `2025-09-30`이 있으나 개별 종료·제출 마감 항목은 `2026`이다. 위의 명시적 마감 항목을 적용한다. |
| 공지에서의 정정 | 공개 토크 1개 고정 글과 일반 글 9개를 확인했으며 두 불일치를 해소한 공지는 없었다. |
| 현재 계정 상태 | 점수·순위·남은 일일 제출 횟수는 이 문서의 확인 범위 밖이다. |

FAQ상 리더보드는 최고 점수를 반영하고 동점이면 먼저 기록한 팀이 우선한다. 제출 횟수가 차감된 실패 건은 팀 병합 기준에도 포함된다. [공식 FAQ와 09-10 답변](https://dacon.io/competitions/official/236749/talkboard/417136)
