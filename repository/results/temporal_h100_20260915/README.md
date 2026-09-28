# H100 Conv1D 하이퍼파라미터 실험

실험, 결과 다운로드·검증, H100 Pause를 완료했다. 워크스페이스 `temporal-conv1d-h100` (`wsp-yiuufw51tke8`)은 stopped이며 billing burnRatePerHour=0이다. 기존 A100도 정지 상태다. 15분 모니터링을 해제했다.

보고용 노트북은 `Temporal_H100_Report.ipynb`, 원본 실행 기록은 `unpacked/Temporal_H100_Executed.ipynb`이다. 5개 코드 셀이 모두 정상 실행됐다. 서버 생성 그래프와 결과 표를 확인했으며, 보고용 사본에는 실제 PNG를 첨부했다. 원본 실행 기록은 수정하지 않았다.

## 결과

같은 평가 190개 중 음악 EER 대상 150개를 비교했다. 낮을수록 좋다.

| 모델 | Music EER | File EER |
|---|---:|---:|
| 공식 베이스라인 | 0.552778 | 0.432051 |
| 같은 분할 Linear | **0.366667** | **0.351923** |
| T7 seed41 | 0.547222 | 0.380128 |
| T7 seed42 | 0.600000 | 0.396154 |
| T7 seed43 | 0.552778 | 0.400000 |

Conv1D의 seed 평균 Music EER는 0.566667, 모집단 표준편차는 0.023679다. 설정을 고를 때는 튜닝 자료만 썼으며 평가 점수로 seed를 선택하지 않았다. 이 실험에서는 Conv1D 교체를 채택할 근거가 없다. 공개 train/dev/sealed를 재사용했고 기존 dev은 이미 과거에 확인한 자료이므로 새 외부 검증이 아니다. 리더보드 개선을 증명하지 않는다. Voice presence 정답 클래스가 부족해 종합 점수는 계산하지 않았다.

## 학습 설정과 GPU 사용

MERT 본체는 동결하고 기존 4.0375초 구간의 768차원 특징을 재사용했다. 학습 대상은 414,017개 파라미터의 6층 residual Conv1D와 gated attention/max pooling이다. 학습 1,008개, 튜닝 248개, 평가 190개를 원본·혼합 부모 그룹별로 분리했고 frozen `locked_split.json`을 사용했다. 기존 C3 품질 마스크는 학습 자료에만 적용했다.

LR 3e-4/1e-4 × Dropout .2/.4 × weight decay .01/.05의 8개 설정을 비교했다. AdamW, batch16, clipping1.0, 최대300epoch, patience20, min_delta1e-4다. 튜닝 EER→balanced BCE 순으로 T7(lr1e-4,dropout.4,wd.05)을 선택했다. best epoch는 seed41/42/43에서 1/2/1이고, 21/22/21epoch에서 종료했다. 검증 손실의 상승은 추가 epoch보다 일반화 문제를 먼저 검토해야 한다는 근거다.

노트북 실행 시작 07:01:14 UTC, 정상 종료 07:03:19 UTC로 약2분5초였다. 학습 중 GPU 사용률을 시계열로 수집하지 않아 평균 사용률 수치는 없다. 종료 후 유휴 상태는 확인했다. 작은 분류기와 batch16은 H100을 충분히 활용하기 어려운 구성이라는 해석이며, 병목을 프로파일링한 결과는 아니다.

## 이전, 비용, 보존

CPU 작업 `job-fml78fjtmo81`에서 원본 음원의 SHA256을 확인하고 1,446개 캐시를 내보냈다. 82,829,419바이트의 특징 ZIP을 object storage로 이전했다. H100에서 캐시 해시·원래 추출 식별자를 검증했다. 테스트는 내보내기/불러오기 값 보존과 변조된 음원·캐시 거부를 확인했다.

H100은 NVIDIA H100 80GB HBM3, torch2.9.1+cu130, Python3.13.9이며 `unpacked/requirements-head.txt`에 추가 의존성을 고정했다. 기존 A100 자료의 분할이 라이브러리 버전에 따라 바뀌지 않도록 split JSON을 그대로 사용했다.

작업 전 잔액12.793524→Pause 확인 시12.224411, 감소0.569113크레딧이다. CPU 이전, 환경 준비, 노트북 실행, 다운로드 재시도 및 정지 전 대기를 포함한 계정 잔액 차이다. 학습만의 비용은 분리 측정하지 않았다. Object-storage CLI 다운로드가 1회 실패했으며 Jupyter 직접 다운로드로 복구했다.

결과 ZIP은 16,738,675바이트, SHA256 `016db533f56b40641e45943a4342557eb4a2ac934c0335dfaf521a39b5c4afc6`이다. 크기·SHA256·CRC·추출 경로를 검증한 뒤 Pause했다. 기존 object volume의 `temporal/h100-run-20260915`에도 보존했다. 새 유료 cluster storage는 만들지 않았다. H100의 임시 /root는 Pause 시 삭제되므로 재개하더라도 이 결과와 원본 캐시는 object storage/로컬 자료를 사용해야 한다.

학습 데이터나 성능을 변경하는 추가 실험, 리더보드 제출, 크레딧 구매는 하지 않았다.
