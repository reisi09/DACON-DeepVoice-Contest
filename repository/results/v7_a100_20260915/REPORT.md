# A100 데이터 전처리 비교 결과

실험과 결과 다운로드를 완료하고 A100 workspace `jolly-routine`을 Pause했다. `state=stopped`, `burnRatePerHour=0` 확인. 15분 확인 자동화도 중지했다. 대회 제출은 하지 않았다.

## 결과

동일한 개발 자료 190개(210개에서 동일 PCM 중복 제거)로 평가했다. 파일 EER는 190개, 음악 EER는 음악이 있는 150개에서 계산한다. EER는 낮을수록 좋다.

| 조건 | 파일 EER | 음악 EER |
|---|---:|---:|
| V6_saved | 33.59% | 46.67% |
| C0_class_balance | 33.59% | 46.67% |
| C1_nsynth_5pct | 31.60% | 46.67% |
| C2_no_nsynth | 35.19% | 41.39% |
| C3_equal_windows | 31.60% | 31.39% |
| Official_baseline | 43.21% | 55.28% |
| C4_four_seconds_seed41 | 39.62% | 34.72% |
| C5_four_seconds_no_nsynth_seed41 | 31.60% | 35.28% |
| C4_four_seconds_seed42 | 38.40% | 38.61% |
| C5_four_seconds_no_nsynth_seed42 | 31.60% | 35.28% |
| C4_four_seconds_seed43 | 36.79% | 35.28% |
| C5_four_seconds_no_nsynth_seed43 | 31.60% | 31.94% |

## 판단

- C3는 FMA 45%, NSynth 5%, 생성 음악 50%의 손실 가중치를 적용한다. 생성 음악은 생성기별로 균등 가중치를 주고, 정규화 통계에 파일당 같은 비중을 반영하며 epoch마다 파일당 구간 하나를 뽑는다. 기존 4.0375초 MERT 캐시를 사용한다.
- C3의 음악 EER는 기존 V6보다 15.28%p 낮고, 파일 EER는 1.99%p 낮다. 혼합 음악 EER도 52.50%에서 37.50%로 낮아졌다. 음악 판별의 개선 폭보다 파일 판별의 개선 폭이 작으므로 결합기와 다른 성분의 병목이 남는다.
- NSynth 비중만 낮춘 C1은 음악 EER가 그대로였다. NSynth를 완전히 제외한 C2는 음악 EER가 낮아졌지만 파일 EER는 높아졌다. 단순 삭제만으로 모든 지표가 개선되지는 않았다.
- 정확히 4초인 C4/C5가 C3보다 좋지는 않았다. C4/C5는 HTDemucs 재실행, crop 위치·pool과 길이가 함께 달라지므로 길이 0.0375초 차이만의 인과 효과로 해석할 수 없다.
- C4/C5는 seed 41·42·43을 모두 기록했다. C3는 seed 41 한 번이므로 가장 유망한 후보로 판단하되 안정적인 우승 조건으로 확정하지 않는다.
- 기존 개발 자료를 재사용한 탐색 비교다. 리더보드 점수 개선이나 7위 이내 진입을 입증하지 않는다. 음성 존재 라벨의 두 클래스가 모두 확보되지 않아 종합 점수는 계산하지 않았다.
- 공식 베이스라인 행은 제공된 DF-Arena 음성·음악 경로를 다시 실행하고, 같은 입력·자산의 기존 PANNs 존재 확률 캐시를 재사용한 결과다.

## 검증과 비용

- 기존 입력·특징 캐시 1,806개 검증. 새 4초 특징은 1,492개 파일에 대해 계산했다.
- 실행 노트북의 코드 셀 7개가 모두 완료됐고 오류 출력이 없다. 이는 인코더 전체 미세조정이 아니라 동결 MERT의 분류층·결합기 비교다.
- 저장된 예측으로 12개 조건·참조 결과를 다시 계산해 파일·음악 등 모든 가용 지표가 일치했다.
- 기존 가중치 재현의 최대 확률 차이는 1.4597686859296743e-7, C0 재학습 재현은 6.256836212470418e-7이었다. 기존 가중치의 평가 지표는 모두 동일했다.
- 다운로드 ZIP은 36,588,277 bytes이며 SHA256 `62a277693c4b5cbbcd57573a08419b86765b0f4e60f3b25d4df0335bdd3c9cae` 및 ZIP CRC를 검증했다. 그래프도 시각적으로 확인했다.
- 시작 잔액 15.392910 → 종료 잔액 13.977432, 차감 **1.415478크레딧**. 종료 시 시간당 과금 0. 약 1시간 안에 종료했다. CLI의 totalRuntimeSeconds 필드는 0으로 반환돼 정확한 GPU 실행시간 근거로 쓰지 않았다.

## 다음 실험의 기준

C3를 기준 후보로 두고 여러 seed에서 재현되는지 먼저 확인한다. 이후 원본 그룹을 분리한 추가 평가에서 개선이 유지되는지 확인하고 모델 미세조정을 비교한다. 현재 가중치는 저장돼 있지만 C3용 제출 ZIP 제작이나 DACON 제출은 아직 하지 않았다.

## 파일

- `V7_A100_Executed.ipynb`: 실행 결과와 비교 그래프가 포함된 노트북.
- `results/comparison.csv`, `results/comparison.json`: 전체 조건 비교.
- `results/C3_equal_windows/heads.npz`: C3 음악 분류층·결합기 및 정규화 파라미터.
- `results/*/predictions.json`, `report.json`, `losses.json`: 각 조건의 근거.
- `results/crop_audit.json`, `results/four_second_cache/`: 4초 구간 위치·무음 비율과 재계산 특징.
- `download_manifest.json`, `download_verification.json`, `cloud_final_state.json`, `billing_final.json`: 다운로드와 정지·과금 확인.
