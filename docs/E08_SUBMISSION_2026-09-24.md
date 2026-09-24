# E08 제출 파일 — 2026-09-24

부분 캐시 5,952개로 학습한 `E08_partial5952_seed41_voice_top30` 결과를 제출 ZIP으로 만들었습니다. 공식 대회 제출은 하지 않았습니다.

## 파일과 구성

- 로컬 파일: `/Users/choiwuseck/Documents/ChatGPT/데이콘/submissions/E08_Partial5952_Top30/submit.zip`
- 크기: **6,345,996,113 bytes**, 압축 해제 후 **6,345,987,235 bytes**.
- SHA-256: `fc23d7800ad1ce8b132336fd83f6163b482706772cd6b6f1f701bcd66411ae06`
- 최상위 구성: `model/`, `script.py`, `requirements.txt`. 총 48개 파일.
- 학습 가중치: `music_41.pt`, `voice_41.pt`. 음성은 상위 30% 집계, 존재 확률은 기존 PANNs max, FILE 결합은 max입니다.
- MERT330 layer12·DF-Arena·HTDemucs·PANNs와 전처리 파일은 학습 당시 해시와 대조했습니다. 학습 가중치 외에 데이터·특징 캐시·중간 체크포인트·미리 계산한 제출 CSV는 포함하지 않았습니다.

기존 `DataV4_DurationBalanced/submit.zip`의 고정 모델 파일을 재사용하고 학습 가중치와 추론 코드를 교체했습니다. 기존 ZIP의 보정기·결합기 파일을 제거해 E08 평가 설정과 맞췄습니다. 원본 가중치는 보존했습니다. 전체 파일 해시와 ZIP CRC를 확인했고, 검증용으로 추출한 파일도 해시를 대조했습니다. 파일별 해시와 구성은 [패키지 영수증](evidence/lb_e08_package_20260924/package_receipt.json)에 있습니다.

## 실제 추론 검증

최종 ZIP에서 추출한 `script.py --device cpu`를 원본 음원 3개(WAV·MP3)에 실행했습니다. 추가 GPU는 사용하지 않았습니다. 두 파일 실행은 51.19초, 별도 긴 음원 실행은 31.82초에 종료 코드 0으로 끝났습니다. CSV 열·ID 순서·유한한 0~1 확률을 확인했습니다. 저장된 A100 예측과의 최대 절대 차이는 **0.000491679**입니다.

긴 음원은 음성 window 5개 중 2개를 집계합니다. 같은 가중치의 CPU 캐시 기준 top-30% 예측은 `0.1137860492`, max로 바꾸면 `0.2156186253`입니다. 최종 ZIP의 원본 음원 추론 결과는 `0.1137824729`로 top-30% 기준과의 차이가 `0.0000035763`이었습니다. 실행 전에 정한 절대 오차 허용값 `0.001` 안에 들어왔으며, max로 잘못 처리되는 문제도 구분했습니다.

[검증 결과](evidence/lb_e08_package_20260924/inference_qa.json)와 [재검증 코드](evidence/lb_e08_package_20260924/verify_inference.py)를 보존했습니다. 패키징 회귀 검증에서는 과거 보정기 유입, 잘못된 encoder 재사용, 설정과 체크포인트 불일치, 기존 출력 덮어쓰기를 검사했습니다.

이 검증은 소수 파일의 실행·수치 일치 확인입니다. 독립 성능 평가나 공식 ADS 점수를 뜻하지 않습니다. **L4 전체 테스트의 실행 시간·메모리와 새 환경의 의존성 설치는 검증하지 않았습니다.** ZIP 크기·구성은 [공식 평가 안내](https://dacon.io/competitions/official/236749/overview/evaluation)에 맞췄습니다. 학습 데이터 범위와 성능 해석의 한계는 [학습 기록](A100_PARTIAL_TRAINING_2026-09-24.md)을 따릅니다.

## 재생성

`repository/`에서 아래 명령을 실행합니다. 입력 파일은 별도로 보존한 로컬 산출물이며 Git에는 포함하지 않았습니다. 기존 출력은 덮어쓰지 않으므로 재생성할 때는 새 출력 경로를 사용합니다.

```sh
PROJECT=/Users/choiwuseck/Documents/ChatGPT/데이콘
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
  "$PROJECT/.venv/bin/python" package_datav4_submission.py \
  --candidate-lock "$PROJECT/_workspace/lb_a100_run_20260924_v1/final_readback/unpacked/runs/partial_training/E08_topk/candidate_lock.json" \
  --base-archive "$PROJECT/submissions/DataV4_DurationBalanced/submit.zip" \
  --legacy-decision-layer \
  --candidate E08_partial5952_seed41_voice_top30 \
  --output "$PROJECT/submissions/E08_Partial5952_Top30/submit.zip"
```
