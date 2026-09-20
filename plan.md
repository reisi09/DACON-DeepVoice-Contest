# 현재 계획과 남은 작업

기준일: 2026-09-21. 목표는 현재 모델의 고정 수식 `max(Pv*Fv, Pm*Fm)`과, 예측 과정에서 얻는 다양한 피처를 사용하는 학습형 스태킹을 동일한 독립 데이터에서 비교하는 것입니다. 개선 여부는 아직 확인하지 않았습니다.

## 현재 기준

현재 모델은 `existing9504_MERT330M_WindowMIL_DFfc5_701515`입니다. PANNs·HTDemucs base·DF-Arena·MERT330M layer 12를 고정하고, 음악 WindowMIL과 음성 Linear head를 seed 41/42/43으로 학습했습니다. 첫 비교에서는 `FILE_FAKE_PROB`만 변경하고 나머지 네 확률 출력은 유지합니다. 과거 B_full1696이나 G0–G4는 현재 기준 모델이 아닙니다.

9,504개 중 head 학습은 train 6,652행이며 val/test는 각 1,426행입니다. val 997행과 test 1,001행에 과거 개발 노출 기록이 있고 val은 checkpoint 선택에도 사용했습니다. 이 데이터를 새 독립 평가로 재사용하지 않습니다.

## 완료한 작업

- [x] Existing9504 학습·제출 코드와 이전 실험 소스 확보. [코드 안내](EXPERIMENT_INDEX.md)
- [x] 일반 ChatGPT에서 Stacking v2 구현 산출물 작성. [구현](experimental/stacking_v2/src/), [설계](experimental/stacking_v2/DESIGN.md)
- [x] 로컬 Codex에서 v2 반환 파일 110개의 해시와 전처리 정책·누수 방지·피처 순서·직렬화 경로 검토.
- [x] 기존 검토 환경에서 CPU suite 35 passed, 합성 F0/F1 × LR/ExtraTrees/XGBoost 6개 경로, 저장·재로드 예측 일치 확인. 이 숫자는 당시 검토 기록이며 이번 업로드에서 재실행한 결과가 아닙니다.
- [x] v1의 전처리 정책 누락과 XGBoost 미설치 시 보고서 중단 문제에 대한 수정 확인. [검토 근거](experimental/stacking_v2/codex_review/REVIEW.md)

## 1. 신규 데이터와 분할 확보 — 미완료

- [ ] 데이터 후보의 공식 배포처·라이선스·대회 사용 규정을 실행 전에 확인한다.
- [ ] 현재 전문가의 학습·선택에 사용하지 않은 meta-dev와 locked OOD를 확보한다. 초기 목표는 약 3,200개: Voice-only 800 / Music-only 800 / Mixed 1,200 / Stress 400. 이는 확보 수량이 아니다.
- [ ] dev 약 2,560 / locked OOD 약 640을 출발점으로 하되, 독립 원본·화자·곡·생성기가 부족하면 숫자를 억지로 맞추지 않는다. 파일 수와 독립 parent 수를 별도로 보고한다.
- [ ] 원본을 먼저 분할한 다음 각 split 안에서 혼합·partial·codec 변형을 만든다. Mixed는 두 부모를 모두 추적하고 연결된 부모 그룹을 분리하지 않는다.
- [ ] 기존 9,504개 및 과거 개발 데이터와 hash·parent·speaker·track·generator family·provenance를 대조한다. upstream 사전학습 노출의 불확실성은 unknown으로 남긴다.

완료 기준: 라벨·출처·부모·생성기·split·노출 상태가 기록된 manifest와 누수 검사 결과. [데이터 계획](experimental/stacking_v2/DATA_PLAN.md)

## 2. 실제 오디오 추출과 parity — 미완료

- [ ] 현재 encoder/head를 고정하고 신규 오디오의 기본 확률과 캐시를 추출한다. v1 production bundle/lock은 재사용하지 않고 v2 전처리 계약으로 다시 만든다.
- [ ] 일반 신규 입력의 peak 0.6 제한을 유지한다. 정책·feature order·코드/오디오/checkpoint hash와 timebase를 캐시에 기록한다.
- [ ] 같은 오디오에서 기존 제출 추론과 새 adapter 결과를 비교한다. 합성 fixture나 일부 test double을 실제 GPU parity로 대신하지 않는다.
- [ ] 음악·음성·PANN 시간축을 각각 확인한다. 기존 파일 수준 Presence 두 값으로 구간 Presence를 만들어내지 않는다.

완료 기준: 실제 음원/GPU parity 보고서와 재사용 가능한 캐시. 오차 허용치는 실행 전에 기존 계약을 확인해 고정한다.

## 3. F0/F1 비교와 후보 선택 — 미완료

| 비교 | 피처·모델 |
|---|---|
| 기준 수식 | 현재 max 및 참고 noisy-OR |
| F0 | Pv, Pm, Fv, Fm 네 기본 확률 |
| F1 | 실제 캐시에서 얻는 seed 분산·범위, window mean/max/std/quantile/top-k, 원래 DF-Arena 점수 등 |
| 메타 모델 | L2 Logistic Regression, ExtraTrees, 얕고 정규화한 XGBoost |

- [ ] 같은 meta-dev 행과 parent 그룹 CV에서 F0/F1 × 세 모델을 비교한다. 탐색 조합은 소수로 사전 고정한다.
- [ ] scaling·imputation·feature selection·early stopping·calibration은 fold 안에서만 학습한다. OOB 점수를 source-disjoint 검증으로 대체하지 않는다.
- [ ] label·mask·window_truth·ID·경로·source·generator·split·노출 여부는 입력에서 제외한다. 길이·음량·codec은 shortcut 위험군으로 따로 제거 비교한다.
- [ ] File EER을 주 지표로 비교하고 다섯 출력의 지표·분모·누락 라벨, Brier/logloss, 조건별 결과와 parent 그룹 paired bootstrap을 기록한다. 완전한 라벨이 없으면 공식 총점은 null이다.
- [ ] dev에서 후보·피처·전처리·threshold/보정 설정을 고정한 다음 locked OOD를 한 번 평가한다. OOD 결과로 재튜닝하면 이후 독립 평가용으로 쓰지 않는다.

기존 학습 데이터를 활용하는 대안은 base-head 학습·정규화·checkpoint 선택까지 평가 그룹을 제외하는 nested/cross-fitting이 필요합니다. 기존 head의 in-sample 점수를 OOF라고 부르지 않습니다.

## 4. F2 추가 피처 — 후속 비교

- [ ] F0/F1 결과와 추출 비용을 확인한 뒤 PANN 클래스·시간 요약, 정렬된 Presence×Fake, coverage, raw/stem 차이, 에너지 비율을 추가한다.
- [ ] 필요한 추가 encoder 호출·저장 용량·추론 시간을 기록한다. 없는 피처를 0으로 채워 측정값처럼 취급하지 않는다.

현재 캐시에는 PANN 전체 시간 출력, 구간 Presence, raw Fake 점수, 두 stem RMS가 없습니다. 수천 차원 전체 임베딩이나 새 encoder 도입은 첫 비교 범위에 넣지 않습니다. [피처 명세](experimental/stacking_v2/FEATURE_SPEC.csv)

## 5. 채택·배포 검증 — 미완료

- [ ] 독립 평가에서 기준 대비 개선과 조건별 실패 양상을 확인한 뒤 채택 여부를 결정한다. 개선되지 않으면 현재 수식을 유지한다.
- [ ] 실제 L4에서 시간·VRAM과 원본 대비 추론 일치를 측정한다. H100 실행이나 CPU 검증으로 L4 충족을 주장하지 않는다.
- [ ] 최종 모델 저장·재로드, ID+5개 확률의 6열 CSV 계약, 기존 네 출력 보존, 최종 제출 ZIP의 오프라인 실행을 검증한다.
- [ ] 공식 제출과 점수 확인은 별도 실행 단계로 남긴다. 현재 새 스태킹의 공식 점수는 없다.

## 실행 분담과 자원

일반 ChatGPT에서는 설계·코드·CPU 검증 산출물을 먼저 만들고, Codex는 반환 파일과 결과를 검사합니다. 실제 데이터 확보·GPU 추출·독립 성능 측정은 아직 남아 있습니다. [실행 안내](experimental/stacking_v2/RUNBOOK.md)의 fixture·오디오·가중치 등 별도 입력과 경로를 먼저 확인해야 합니다.

GPU 작업은 기존 VESSL workspace를 재사용합니다. `wsp-9683432r72m7`는 현재 지침에 따라 중지 상태를 유지하며, 이번 GitHub 업로드에서는 시작하지 않습니다. 이후 실행 시 저장소 확인과 새 설정 업로드를 먼저 끝내고, 자원·시간 제한을 정한 뒤 시작합니다. 실행 중에는 5분 간격으로 확인하고 결과 다운로드·검증 후 compute를 중지합니다. 새 인스턴스를 만들거나 유료 실행 범위를 늘리는 작업은 이 계획 파일 작성에 포함되지 않습니다.
