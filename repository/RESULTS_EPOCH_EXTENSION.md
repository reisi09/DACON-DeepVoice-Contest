# AST epoch 연장 실험 결과

2026-09-13 Colab A100에서 기존 6 epoch 모델을 이어 학습했다. 최대 30 epoch로 설정했지만 개발 proxy가 5 epoch 연속 개선되지 않아 22 epoch에서 종료했고, 처음 최고 점수를 기록한 17 epoch checkpoint를 복원했다. 개발 성능은 올랐으나 기존 평가 274개에서의 개선은 작았다. 새 DACON 제출은 하지 않았다.

| 부분 라벨 proxy | 기존 6 epoch | 선택된 17 epoch | 변화 |
| --- | ---: | ---: | ---: |
| 개발 122개, AST 단독 | 0.865720 | 0.903801 | +0.038081 |
| 기존 평가 274개, AST 단독 | 0.774420 | 0.775809 | +0.001389 |
| 기존 평가 274개, 고정 결합 모델 | 0.792205 | 0.792205 | 0 |
| 기존 생성기 평가 176개, AST 단독 | 0.726733 | 0.726733 | 0 |
| 기존 생성기 평가 176개, 고정 결합 모델 | 0.756339 | 0.756339 | 0 |
| 기존 source 평가 98개, AST 단독 | 0.894835 | 0.894835 | 0 |

위 수치는 리더보드 점수나 정확도가 아니다. 미상 라벨을 제외하고 계산 가능한 지표의 가중치를 정규화한 proxy다. 구간별로 계산 가능한 지표가 달라 서로 다른 구간의 점수 크기를 직접 비교하지 않는다. 274개는 이전에 관찰한 평가 자료를 다시 쓴 후속 비교이며, 새로운 독립 검증이 아니다.

## 실험 조건

- `AST_LAST_BLOCK`의 마지막 encoder block·최종 LayerNorm·다섯 head, 7,093,253개 파라미터를 계속 학습했다. `AST_FROZEN`은 이번에 재학습하지 않았다.
- 기존 6 epoch의 모델·AdamW·난수 상태를 복원했다. head LR `1e-4`, backbone LR `1e-5`, batch 8, accumulation 2, weight decay 0.01, gradient clip 1.0을 유지했다.
- 학습 633개, 개발 선택 122개, 기존 생성기 평가 176개, 기존 source 평가 98개를 유지했다. 전체 1,029개를 학습에 넣은 실험이 아니다.
- checkpoint와 두 manifest 해시를 확인했다. 추가 학습 전 개발 예측은 기존 CSV와 최대 차이 `0.0`이었다.
- 변경 가능한 설정은 `epochs`와 `patience`로 제한했다. 모델 구조·학습률·crop/gain 설정은 그대로 두고 epoch 증가 효과를 비교했다.
- epoch 선택은 개발 자료만 사용했다. 개발 선택 결과와 checkpoint hash를 저장한 뒤 기존 평가 274개를 계산했다.

## 결과 해석

학습 loss는 6 epoch `0.092527`, 17 epoch `0.017512`, 22 epoch `0.007588`이었다. 개발 proxy는 17 epoch부터 22 epoch까지 `0.903801`로 같았다. 추가 학습이 훈련 loss를 낮췄지만 이 구간에서는 개발 지표를 더 높이지 못했다.

개발 파일 EER은 `0.155711 → 0.106412`, 음악 EER은 `0.147186 → 0.088312`로 개선됐다. 음성 EER은 `0.135945`로 같았다. 기존 평가 274개에서는 파일 EER만 `0.233355 → 0.230269`로 조금 낮아졌고, 음성 EER `0.113300`과 음악 EER `0.371024`는 같았다. 고정 결합은 G0의 파일·음성 출력과 AST의 음악·존재 출력을 사용하므로 종합 proxy가 그대로였다.

동일한 EER이 나온 것을 학습 실패나 동일한 예측으로 해석하면 안 된다. 파일·음성·음악 진위 확률은 각각 274개 모두 `1e-6` 이상 바뀌었고, 최대 변화는 각각 `0.430013`, `0.416148`, `0.423827`이었다. 선택된 모델 가중치 해시도 기존과 달라졌다. 추가 16 epoch 모두 head와 backbone의 gradient 및 가중치 갱신이 기록됐다.

이번 결과만으로 과적합의 원인을 확정할 수는 없다. 다만 epoch 증가로 얻은 개발 개선이 기존 생성기 평가의 개선으로 이어지지는 않았다. 같은 조건에서 epoch만 더 늘릴 근거는 약하다. 기존 선택 규칙에서도 음성 EER·대응 AUC 악화 조건을 통과하지 못해 G0가 유지됐다. 새 AST ZIP은 비교용 후보로 저장했다.

![학습 loss와 개발 proxy](results/epoch_extension_20260913_013624/learning_curve.png)

## 실행과 저장 검증

실행 환경은 `NVIDIA A100-SXM4-40GB`, Python `3.13.15`, PyTorch `2.11.0+cu128`, Transformers `4.57.6`이었다. 추가 16 epoch의 학습·개발 평가 시간 합은 `123.476`초다. 이 합은 checkpoint 저장과 설치 시간을 포함하지 않는다. 첫 셀부터 런타임 해제 요청까지는 `345.715`초, 약 5분 46초였다.

새 ZIP을 별도 디렉터리에 풀고 네트워크를 차단한 새 프로세스에서 실제 제출 스크립트를 실행했다. 형식 확인용 3개와 60초 stress audio의 ID 순서·모든 확률이 기준 예측과 일치했다. 최대 차이는 `0.0`이었다. 이 검증은 배포 경로의 수치 재현성 확인이며 모델 품질은 위 평가로 측정했다. L4 전체 데이터 실행 시간은 아직 측정하지 않았다.

- ZIP: `344,836,171 B`, SHA-256 `5aa3c61dc017a85cdf795de5b730da85d461d51ec3c4b9f731dfe54ef43bc712`.
- 선택된 가중치 SHA-256: `b51e8645730626c2b5c26647e4fb5d02b5d117a542ac2d71c77ee9cde282a107`.
- Drive 복사 뒤 ZIP hash가 원본과 같은지 확인했다. 모델, optimizer checkpoint, epoch별 결과, 예측 CSV, 노트북 출력, 그래프를 저장했다.
- Colab 화면에서 활성 세션 `0개`, 사용률 `0/시간`을 확인했다. 표시 잔액은 `54.49 → 54.36`이었다. 잔액 반영 지연 여부는 확인하지 않았으므로 최종 과금량으로 단정하지 않는다.
- 로컬에서는 이어 학습한 Adam 갱신이 중단 없는 갱신과 일치하는지, 학습률 변경을 거부하는지, 조기 종료 후 최적 가중치를 복원하는지 확인했다. 기존 gradient·resampling·패키지 검사와 함께 통과했다.
- 저장된 예측 CSV로 기존 평가 274개의 지표를 로컬에서 다시 계산해 Colab 결과와 일치함을 확인했다. 실행 노트북에는 7개 코드 셀의 실행 기록이 있고 오류 출력은 없다.

[실행 Colab](https://colab.research.google.com/drive/18AFcp2u_enex4BMraCO_iZu_la7oURjv) · [전체 결과 Drive](https://drive.google.com/drive/folders/1FffrAQ7aOW3vJZfNA1f9_xT2V-FkWG22) · [AST 후보 ZIP](https://drive.google.com/file/d/12NeVok-WmvRmbDXE6L6gg0VvlNjJ0ery/view)

원자료는 `results/epoch_extension_20260913_013624/`에 저장했다. `training_report.json`, `comparison.json`, `measurement_audit.json`, `executed.ipynb`, `packaged_verification.json`, `runtime_released.png`에서 수치와 실행 상태를 확인할 수 있다.
