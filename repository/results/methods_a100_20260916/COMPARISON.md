# MERT330 분류기 비교 결과

기존 MLP 제출 ZIP을 먼저 완성한 뒤, A100에서 16개 설정을 seed 41·42·43으로 각각 학습했다. 총 48회 모두 얼리 스토핑으로 종료됐고, 저장된 예측으로 지표를 다시 계산했다.

## 제출 파일

- [기존 MLP 전체 데이터 제출 ZIP](/Users/choiwuseck/Documents/ChatGPT/데이콘/submissions/M330_MLP_full1206/M330_MLP_full1206_submit.zip): 6.27GB. 공식 업로드는 하지 않았다.
- 승인된 음악 1,206개를 5-fold로 나눴다. 각 파일은 학습에 4번, 검증에 1번 쓰이며 NSynth 90개도 양수 가중치로 반영했다.
- 선택 에포크: 36·43·40·42·28. 실제 종료 에포크: 56·63·60·62·48.
- 다섯 음악 확률을 평균 낸 뒤 기존 file fusion을 적용한다. MERT 특징은 한 번만 추출한다.
- 이 전체 데이터 앙상블에는 독립적인 최종 평가 점수가 없다. 아래 비교 실험 결과를 이 ZIP의 점수로 해석하면 안 된다.

## 동일 조건

- Frozen MERT-v1-330M, 입력 1024차원, 약 4초 구간당 시간 벡터 1개.
- 공통 trunk: LayerNorm → 1×1 projection 64채널 → GELU → residual Conv1d 3층. 활성화 함수는 GELU를 유지했다.
- Dropout0.1, AdamW lr1e-4, weight decay0.01, batch64. 기존 feature crop 확률0.7·feature mask 확률0.3을 유지했다.
- 기존 분할: train1008 / tune248 / 과거 test190. source group이 겹치지 않는지 검사했다.
- 비교 실험의 NSynth mass는 기존 대조군과 같은 0이다. 전체 데이터 제출 모델의 NSynth mass0.05와 구별해야 한다.
- 모든 실행: unsmoothed balanced validation cross entropy 기준, patience20, min_delta0.0001, 최대300에포크는 안전 상한. 최저 검증 손실 가중치를 복원했다.
- 실제 종료 범위는 24~55에포크였다. Plateau는 ReduceLROnPlateau(factor0.5, patience4, min_lr1e-6)이며 5개 연속 정체 에포크 뒤 학습률을 낮춘다.
- MLP와 cosine은 classifier × LS(0/0.05) × Plateau(없음/있음)의 2×2×2 비교다. 추가 4개 구조는 기본 설정과 LS+Plateau 묶음만 비교했으므로 두 효과를 각각 분리할 수 없다.

## 비교한 후단 구조

- MLP: max pooling → Linear64→32 → GELU → dropout → Linear32→2.
- Cosine: max pooling → 특징·클래스 벡터 L2 정규화 → cosine logits × 10.
- 다중 prototype: 클래스당 대표 벡터3개, cosine logits × 10을 log-mean-exp로 합산.
- Attention: 학습한 구간별 가중치로 pooling한 뒤 같은 MLP 적용.
- Conv1d: pooling 전에 1×1 Conv64→2를 적용하고 유효 구간의 mean 또는 log-mean-exp로 합산.
- 모두 학습 시 2-class logits를 사용하고, 추론 시 softmax의 fake 확률을 읽는다.

## 전체 결과

EER는 낮을수록 좋다. 평균±표준편차는 seed 3개 기준이며, 아래 표는 tune 평균 EER 순이다. 과거 test는 이전 실험에서도 사용했으므로 새 독립 평가셋이 아니다.

| 설정 | Tune 음악 EER | 과거 test 음악 EER | 과거 test 파일 EER |
|---|---:|---:|---:|
| Cosine + LS0.05 | 0.23140 ± 0.05525 | 0.22407 ± 0.02179 | 0.27735 ± 0.00937 |
| Attention + MLP | 0.23233 ± 0.01164 | 0.21296 ± 0.00917 | 0.26795 ± 0.01308 |
| Attention + MLP + LS0.05 + Plateau | 0.23233 ± 0.01164 | 0.21296 ± 0.00917 | 0.26261 ± 0.00755 |
| MLP + LS0.05 | 0.24182 ± 0.02401 | 0.21759 ± 0.00262 | 0.25726 ± 0.00755 |
| MLP | 0.25558 ± 0.01342 | 0.22037 ± 0.00917 | 0.26667 ± 0.01155 |
| MLP + Plateau | 0.25558 ± 0.01342 | 0.22037 ± 0.00917 | 0.26667 ± 0.01155 |
| MLP + LS0.05 + Plateau | 0.25558 ± 0.01702 | 0.22222 ± 0.00818 | 0.25726 ± 0.00755 |
| Conv1d + mean | 0.27121 ± 0.01342 | 0.21759 ± 0.02734 | 0.28397 ± 0.00000 |
| Conv1d + mean + LS0.05 + Plateau | 0.27883 ± 0.00811 | 0.21944 ± 0.02722 | 0.28932 ± 0.00755 |
| Cosine | 0.27976 ± 0.02984 | 0.23981 ± 0.01982 | 0.28141 ± 0.01155 |
| Cosine + Plateau | 0.27976 ± 0.02984 | 0.24444 ± 0.01571 | 0.27735 ± 0.01330 |
| Conv1d + log-mean-exp + LS0.05 + Plateau | 0.28069 ± 0.01342 | 0.22685 ± 0.01982 | 0.28397 ± 0.00000 |
| Conv1d + log-mean-exp | 0.28404 ± 0.01851 | 0.22870 ± 0.02179 | 0.28397 ± 0.00000 |
| Cosine + LS0.05 + Plateau | 0.28590 ± 0.01412 | 0.23519 ± 0.01367 | 0.27735 ± 0.00683 |
| 다중 prototype | 0.29967 ± 0.02683 | 0.26019 ± 0.07551 | 0.30534 ± 0.03298 |
| 다중 prototype + LS0.05 + Plateau | 0.31436 ± 0.01412 | 0.28889 ± 0.08016 | 0.31325 ± 0.02853 |

### 해석

- 사전에 정한 tune 기준 1위는 **Cosine + LS0.05, Plateau 없음**이다. 다만 과거 test 음악 EER는 0.22407로 기존 MLP의 0.22037보다 높았다.
- Attention + MLP는 과거 test 음악 EER가 0.21296으로 가장 낮았다. 기존 MLP 대비 0.00741 낮지만, seed 3개와 재사용한 평가셋만으로 일반적인 우위를 확정할 수는 없다.
- MLP + LS0.05는 과거 test 음악 EER0.21759, 파일 EER0.25726이었다. 파일 EER 평균은 이번 학습 설정 중 가장 낮았다.
- 기존 MLP에 Plateau만 추가한 경우 선택 가중치와 성능이 같았다. 스케줄러가 실제로 학습률을 낮춰도 더 좋은 검증 가중치를 얻는다는 보장은 없다.
- 다중 prototype은 이번 실험에서 성능과 seed 안정성이 좋지 않았다.
- 공식 baseline을 같은 과거 test에 적용한 음악 EER는 0.55278, 파일 EER는 0.43205였다. 이 로컬 수치는 사용자에게 제공된 공식 리더보드 점수와 다른 지표다.

## 추론 시 temperature·확률 smoothing

기존 MLP의 저장된 확률만 변환했다. test에 맞춰 temperature를 학습하거나 선택하지 않았다. 모두 음악 EER는 같았으며, 음악 확률과 고정 voice 확률을 결합하는 방식 때문에 파일 EER는 달라졌다.

| 변환 | 음악 EER 평균 | 파일 EER 평균 | 음악 balanced BCE 평균 |
|---|---:|---:|---:|
| 변환 없음 | 0.22037 | 0.26667 | 0.52023 |
| T0.5 | 0.22037 | 0.27201 | 0.63615 |
| T2 | 0.22037 | 0.26261 | 0.54659 |
| prob_smoothing0.05 | 0.22037 | 0.26261 | 0.51846 |

확률 smoothing은 p′=0.95p+0.025다. 학습 시 정답 분포를 바꾸는 label smoothing과는 별개다.

## 검증과 남은 한계

- 48개 checkpoint의 SHA-256, 실제 로드와 파라미터 수, train/tune ID, 최저 검증 손실 선택, 얼리 스토핑 중단 시점, 실제 학습률 이력을 검사했다.
- A100에서는 각 checkpoint를 다시 읽어 예측 일치를 확인했다. 로컬 지표 재계산은 허용 오차1e-12로 확인했다. AUC 합산에서 약4e-16의 부동소수점 차이가 있었다.
- 제출 ZIP의 CPU 오프라인 실제 오디오 추론을 확인했다. A100 패키지 음악 확률과 학습 캐시 기준의 차이는 약1.91e-7이었다.
- CPU와 A100 환경의 음악 확률 차이 약0.00694는 별도로 남아 있다. 기기와 torch/librosa/numpy 버전이 달라 정확한 원인은 아직 분리하지 않았다.
- 공식 전체 점수, L4 전체 추론 시간은 검증하지 않았다. 제한된 라벨과 과거 test로 리더보드 성능을 보장할 수 없다.
- GPU는 stopped/standby, 시간당 과금0을 확인했다. 5분 모니터는 일시중지했다.

## 원본 자료

- [검증 및 seed별 전체 수치](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/methods_a100_20260916/verification.json)
- [실행된 노트북](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/methods_a100_20260916/unpacked/C08_Methods_A100_Executed.ipynb)
- [실험 결과 ZIP](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/methods_a100_20260916/C08_Methods_A100_results.zip)
- [제출 ZIP 검증 기록](/Users/choiwuseck/Documents/ChatGPT/데이콘/submissions/M330_MLP_full1206/package_verification.json)
- [GPU 종료 기록](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/methods_a100_20260916/resource_stop.json)
