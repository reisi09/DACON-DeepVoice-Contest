실험과 결과 다운로드를 마쳤고, H100은 2026-09-17 10:31:52 KST에 stopped/standby로 정지됐습니다. 15분 점검도 PAUSED로 바꿨습니다. 워크스페이스 생성부터 정지 확인까지 1시간 34분 42초가 걸렸습니다.

결과 ZIP은 `downloaded/results.zip`입니다. 40,853,841바이트이며 전체 SHA-256은 `ae0c589a3772ec37a3f54ebd721a6ab929204b1297b2e5f1c6ecd0e147bc0c06`입니다. 내부 파일 89개를 해시로 검증했고, 체크포인트 36개(본 학습 6개와 OOF 30개), 예측, 실행 노트북, 비교 결과가 포함돼 있습니다. 원본 ZIP과 압축을 푼 결과는 그대로 보존했습니다.

이번 실험은 기존 자료를 재사용한 진단용 A/B 비교입니다. 학습 1,469개, 검증 129개를 사용했고 sealed 98개는 열지 않았습니다. 분할 간 계보 중복은 없지만, 기존에 재사용한 자료라 새로운 독립 평가셋은 아닙니다. 검증셋은 실제 음악 7개, NSynth 20개, 생성 음악 102개로 구성됩니다.

사전학습 인코더는 고정하고 저장한 특징으로 음악 분류기를 학습했습니다. A는 stem 단독, B는 raw+stem residual입니다. 시드는 41·42·43이며, 각 시드의 확률을 평균한 앙상블을 비교했습니다. AdamW(lr=1e-4, weight_decay=0.01), batch_size=64, 최대 300 epoch, patience=20을 사용했습니다. 체크포인트는 클래스·연결 그룹 가중 검증 손실(0.27×bag CE + 0.10×window BCE)로 선택했습니다. 이 선택에 사용한 검증셋에서 아래 지표를 측정했습니다.

EER는 낮을수록 좋습니다. 아래 값은 3개 시드의 확률 앙상블 결과입니다.

| 방식 | Music EER | File EER |
|---|---:|---:|
| A: stem + max | 8.12% | 10.95% |
| B: raw+stem + max | 8.12% | 10.95% |
| B + noisy-OR | 8.12% | 8.12% |
| B + OOF 관계 모델 | 8.12% | 10.46% |

max 결합에서는 A와 B의 앙상블 EER가 같습니다. 이 결과만으로 raw+stem의 이득이 없다고 일반화할 수는 없습니다. noisy-OR의 File EER는 max보다 2.83%p 낮지만, 개선량의 그룹 bootstrap 95% 신뢰구간이 0을 포함하므로 개선을 확정할 수 없습니다. OOF 관계 모델도 noisy-OR보다 좋은 결과를 내지 못했습니다.

실험 종료 후 저장된 예측에 같은 max/noisy-OR 규칙을 적용해 A/B를 다시 비교했습니다. GPU 사용이나 추가 학습은 없었습니다. A + noisy-OR의 File EER는 10.95%로 그대로였고, B + noisy-OR만 8.12%였습니다. 이 검증셋에서 관찰된 개선은 원음 분기와 결합 방식의 조합에서 나타났습니다. 따라서 독립 검증에서 확인할 후보는 B + noisy-OR입니다. 이 사후 비교도 2,000회 그룹 bootstrap의 가중 지표 차이 구간이 [-0.043559, 0.037099]로 0을 포함하므로 확정적인 개선 근거는 아닙니다. 계산값과 원본 예측 해시는 `posthoc_raw_file_interaction.json`에 저장했습니다.

| 방식 | 시드별 Music EER (41 / 42 / 43) | 평균 ± 표준편차 |
|---|---|---|
| A | 7.63% / 10.46% / 14.76% | 10.95% ± 2.93%p |
| B | 10.95% / 10.46% / 10.95% | 10.78% ± 0.23%p |

표준편차는 ddof=0이며, 평균 EER와 확률 앙상블 EER는 서로 다른 값입니다. B의 시드 편차가 작았다는 관찰은 이 3개 시드에 한정합니다. 본 학습에서 선택된 epoch는 A 23/15/14, B 17/17/8입니다. File 관계 모델은 학습 자료의 5-fold grouped OOF 예측으로 학습했습니다.

| 비교 | 계산 가능한 가중 지표 차이 | 그룹 bootstrap 95% 구간 |
|---|---:|---:|
| A → B | 0.000000 | [-0.050560, 0.029771] |
| B max → noisy-OR | +0.012745 | [-0.003798, 0.028066] |
| B max → OOF 관계 | +0.002206 | [-0.024021, 0.054495] |

이 차이는 같은 검증 자료에서 바뀐 출력의 계산 가능한 기여분입니다. 공식 총점이나 리더보드 점수가 아닙니다. 2,000회 재표집 구간이 모두 0을 포함하고 필수 하위집단도 없어, 모든 비교는 `diagnostic_only_not_accepted`로 남았습니다.

검증 자료에 충분한 음성 진위 라벨이 없고, 존재 여부 AUC를 계산할 양쪽 클래스도 갖춰지지 않아 공식 총점은 null입니다. C/D 혼합 학습은 검증된 가짜 반주가 없어 실행하지 않았습니다. 음성 적응 학습과 raw=stem 용량 통제도 실행하지 않았으므로 해당 효과는 확인하지 못했습니다. L4 제출 환경 검증과 DACON 제출은 진행하지 않았습니다.

ZIP의 `preflight.json`은 학습 전 상태이며 `actual_gpu_training_verified=false`를 유지합니다. `experiments/summary.json`에도 CPU fixture용 일반 제한 문구가 남아 있습니다. 이를 학습 완료 판정으로 사용하지 않았습니다. 실제 H100 실행은 CUDA 검사, 특징 추출 기록, optimizer update와 체크포인트, 실행 노트북, 종료 코드 0으로 확인했습니다. 검증된 원본 결과 파일은 수정하지 않았습니다.

<!-- HUMANIZE-SUMMARY
Final Korean editing pass: retained all measurements and uncertainty, qualified the earlier max-only conclusion, and labeled the additional matched postprocessing calculation as post hoc.
No wording-only changes after drafting; S1/S2 findings 0->0; checks6/6; GradeB.
-->
