# C08 제출 패키지

제출 파일: `C08_submit.zip`

- C08 seed 41, epoch 12 체크포인트. 88,257개 파라미터.
- MERT-v1-95M → 구간당 평균 벡터 1개 → 64채널 Conv1D 3층·GELU·residual → max pooling → Linear → sigmoid.
- baseline의 PANNs 존재 확률과 DF-Arena 음성 판정을 유지한다. 음악 판정만 C08으로 교체하며 파일 확률은 `max(음성 존재 확률 × 음성 가짜 확률, 음악 존재 확률 × 음악 가짜 확률)`로 계산한다.
- 기존 체크포인트를 패키징했다. 전체 데이터 재학습과 앙상블은 하지 않았다.

## 확인한 범위

- ZIP 최상위 구조: `script.py`, `requirements.txt`, `model/`.
- ZIP CRC, 파일별 SHA-256, C08 체크포인트와 학습 당시 MERT 가중치의 일치를 확인했다.
- 실제 평가 캐시 190개에서 제출 코드의 다섯 확률이 H100의 C08 seed 41 결과와 절대오차 1e-6 이내로 일치했다.
- 압축을 새로 푼 패키지에서 모델을 실제로 불러와 CPU 추론을 수행했다. 공식 더미 오디오 1개는 22.46초, 4초짜리 실제 Echoes 오디오 1개는 17.49초에 CSV를 생성했다. 이 시간에는 패키지 해시 확인과 모델 로딩이 포함된다.
- 실제 오디오 1개의 음악 확률은 H100 기록과 약 5.36e-7, 다섯 확률 중 최대 차이는 약 4.62e-5였다. 자세한 값은 `cpu_real_audio.json`에 있다.
- L4에서 전체 평가 데이터를 실행한 시간과 리더보드 점수는 확인하지 않았다. 대회 업로드는 수행하지 않았다.

## 환경

평가 서버에 기본 설치된 라이브러리를 사용하며, requirements에는 torch 2.7.1 / torchaudio 2.7.1 / torchvision 0.22.1의 CUDA 12.8 조합을 지정했다. torchvision과 torch의 버전 불일치를 방지하기 위한 설정이다.

공식 환경·제출 구조: https://dacon.io/competitions/official/236749/overview/evaluation

## 점수 해석

이 패키지는 seed 41 단일 모델이다. 앞서 안내한 C08 음악 EER 0.2954는 세 시드의 평균이며 이 ZIP의 단일 모델 점수가 아니다. seed 41의 기존 평가 기록은 음악 EER 0.313889, 파일 EER 0.316026이다. 공식 리더보드 점수로 해석하지 않는다.
