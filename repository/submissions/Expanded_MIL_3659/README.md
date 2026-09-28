# 확장 데이터 MIL 제출 파일

제출할 파일은 `submit.zip` 하나다. 실제 데이콘 제출은 하지 않았다.

- 용량: **6,267,015,648 bytes (6.267 GB / 5.837 GiB)**. 해제 후 6,267,007,026 bytes.
- 모델: 확장 데이터 학습 3,659행에서 학습한 MERT layer12 + MIL, seed 41·42·43의 확률 평균.
- 선택: Dev에서 선정된 MIL. 체크포인트는 각각 96·72·72 update. 추가 전체 데이터 재학습 없이 선정 당시 가중치와 정규화 값을 그대로 사용한다.
- 음악 구간 집계: 64,600 samples, hop 32,300, 정확한 마지막 구간 포함, log-mean-exp(temperature 1).
- 음성: DF-Arena + HTDemucs. 존재 확률: PANNs. 파일 확률: `max(voice_present*voice_fake, music_present*music_fake)`.
- 학습 결과의 로컬 Music EER: Dev 0.0788888889, Test 0.0707142857. 공식 제출 점수가 아니다.

ZIP 최상위에는 `script.py`, `requirements.txt`, `model/`만 있다. 오디오나 특징 캐시는 포함하지 않았다. 세 체크포인트와 27개 사전학습 자산은 학습 기록의 SHA-256과 일치한다. 모든 ZIP 구성 파일의 해시와 CRC를 읽어 검증했다. 전체 해시는 `submit.zip.sha256`, 구성 파일별 해시는 `submit.json`에 있다.

검증은 Mac CPU에서 진행했다. 압축을 실제로 해제하고 네트워크 연결을 차단한 상태에서 인간 음악 8초, 생성 음악 8초, 무음 4초를 추론했다. 실제 `script.py`가 5개 확률 컬럼을 가진 CSV를 생성했다. 저장된 가중치로 직접 계산한 MIL 음악 확률과 비교하고, 구간 전체를 복제해도 집계가 변하지 않는지 확인했다. 앞의 음악 2개만 역순으로 다시 추론해 파일 순서·부분집합 변경에 따른 차이도 확인했다. 수치 결과는 `verification.json`, 실행 시간·메모리는 `cpu_runtime_receipt.json`에 기록했다. 이 소규모 실행은 탐지 성능 평가가 아니다.

**L4에서 1,200개 추론을 60분 안에 마치는지, CUDA bfloat16 경로가 원래 H100 예측을 재현하는지는 이번 작업에서 검증하지 않았다.** CPU 검증 환경의 일부 라이브러리 버전은 공식 서버와 다르다. GPU 워크스페이스를 새로 시작하지 않았다. 공식 ZIP 구조·용량·환경 기준은 [데이콘 평가 페이지](https://dacon.io/competitions/official/236749/overview/evaluation)를 2026-09-18에 확인했다.

재생성 명령(저장소 루트에서 실행):

```sh
.venv/bin/python package_expanded_submission.py \
  --assets _workspace/redesign_full_20260917/assets \
  --results results/expanded_h100_20260918/downloaded/unpacked/output \
  --output submissions/Expanded_MIL_3659/submit.zip
```

실제 오디오 실행 이후 수치 검증:

```sh
.venv/bin/python verify_expanded_submission.py \
  --qa _workspace/expanded_submission_qa \
  --training results/expanded_h100_20260918/downloaded/unpacked/output/train \
  --output submissions/Expanded_MIL_3659/verification.json
```
