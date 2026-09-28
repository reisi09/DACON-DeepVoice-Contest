# AF-Whisper 637M A100 비교 결과

## 결론

이번 frozen encoder 비교에서는 기존 MERT330을 유지하는 편이 낫다. AF-Whisper는 Tune Music EER를 낮췄지만, 이전부터 사용한 평가 데이터의 Music EER와 File EER는 모두 높아졌다. 이번 모델 교체로 평가 성능이 개선되지는 않았다.

## 성능

seeds 41·42·43의 평균 ± 표준편차, 단위 %. EER는 낮을수록 좋다.

| 구성 | Tune Music EER | 평가 Music EER | 평가 File EER |
|---|---:|---:|---:|
| MERT330 + MLP | 25.56 ± 1.34 | 22.04 ± 0.92 | 26.67 ± 1.16 |
| MERT330 + MLP + LS 0.05 | 24.18 ± 2.40 | 21.76 ± 0.26 | 25.73 ± 0.76 |
| AF-Whisper637 + MLP | 23.23 ± 2.08 | 25.74 ± 0.65 | 27.74 ± 0.94 |
| AF-Whisper637 + MLP + LS 0.05 | 23.23 ± 2.08 | 26.20 ± 0.65 | 27.86 ± 0.76 |

LS 0.05끼리 비교하면 AF-Whisper의 평가 Music EER는 4.44%p, File EER는 2.14%p 높다. Tune EER 평균·표준편차·BCE 순서로 평가 전에 고정한 선택 결과는 AF637_mlp_ls5였다. AF 두 설정의 Tune EER는 같았고 BCE로 LS 0.05를 선택했다.

## 실험 조건

- Frozen audio encoder: 실제 추론 파라미터 315,428,992 → 636,968,960, 약 2.02배. encoder는 학습하지 않았다.
- 입력 1024 → 1280차원. 기존 compact residual Conv1d trunk + MLP32 + Softmax를 유지했다. head 파라미터는 107,234 → 124,130개다.
- 모델: nvidia/audio-flamingo-3/sound_tower, revision ee26c58423988d7d7cda7b85dd3ce5d97ee8753d. LLM은 로드하지 않았다.
- AdamW, 고정 learning rate 1e-4, dropout 0.1, weight decay 0.01, batch 64. 이 비교에는 학습률 스케줄러를 사용하지 않았다.
- LS 0/0.05 × seeds 41/42/43의 6개 신규 학습. 동일 조건의 기존 MERT 체크포인트 6개는 hash와 예측 일치를 확인한 뒤 재사용했다.
- train 1008 / tune 248 / 기존 평가 190개. source group 분리 유지. 대조군과 맞춰 실험의 NSynth mass는 0이다.
- Early stopping: unsmoothed balanced validation CE, patience 20, min_delta 0.0001, 최대 300은 안전 상한. AF 모델은 두 LS 조건 모두 seed 순서대로 11/12/14 epoch를 선택했고 31/32/34 epoch에 종료했다.
- 특징 추출 1,446개 완료: 5,234.27초(약 87.24분). 6개 신규 학습·Tune 예측 합계 약 86.80초. A100-SXM4-80GB, torch 2.9.1+cu130.

## 검증과 해석 범위

- 실제 pretrained A100 실행에서 full masked encoder와 패딩 계산을 줄인 구현의 최대 절대 오차 1.6093e-6. batch/단일 실행 오차 5.0068e-6. 허용 범위 내다.
- 결과 ZIP SHA256·크기·CRC 검증, 특징 ZIP 및 1,446개 member SHA256 검증 완료. 입력·분할·체크포인트 출처와 early stopping 기록을 확인했다.
- 저장된 예측에서 지표를 다시 계산해 보고서와 일치하는지 확인했다. 원격 평가에서 체크포인트 재로드 예측 허용 오차는 1e-6이다.
- 평가 데이터는 이전 실험에서도 사용했다. 새로운 독립 테스트 결과가 아니며, 3개 seed만으로 일반적인 우열을 단정하지 않는다.
- 모델 교체로 사전학습 데이터·encoder 구조·오디오 frontend·입력 차원도 달라졌다. 파라미터 수만의 효과를 분리한 실험이 아니다.
- 공식 총점은 산출하지 못했다. voice_presence_auc에 양쪽 label이 모두 없고 일부 voice label이 미확인이다. 위 결과를 리더보드 점수로 환산하지 않는다.
- 기존 M330_MLP_full1206_submit.zip은 변경하거나 공식 업로드하지 않았다. AF 모델은 제출 파일로 패키징하지 않았다.
- NVIDIA 모델 가중치는 연구 목적의 비상업 라이선스다. 상업적 사용 가능 모델이라고 해석하지 않는다.

## 산출물

- verification.json: 12개 모델의 지표, seed별 epoch, 수치 검증 결과.
- archive_verification.json: 결과 및 특징 ZIP 해시와 member 검증 기록.
- unpacked/C08_AFWhisper_A100_Executed.ipynb: A100 실행 노트북.
- unpacked/results/: 모델 체크포인트, 학습 기록, 예측, 선택 기록.
- AFWhisper_features.zip: 재사용 가능한 특징 캐시.
- resource_stop.json: GPU 종료와 과금 상태.
