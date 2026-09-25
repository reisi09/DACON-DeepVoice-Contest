# 기존 점수 캐시를 이용한 CPU 진단

2026-09-25. 실제 저장된 점수로 네 가지 진단을 실행했다. GPU·클라우드·네트워크·추가 설치를 사용하지 않았으며, 새 PANNs 모델이나 제출 파일은 만들지 않았다. `ponytail`·`programming` 지침을 적용하고 한국어 본문은 `humanize-korean`으로 마지막에 다듬었다.

**이 결과는 과거 DataV4 후보의 개발 진단이다. 현재 PANNs v1, 사용자 최고점 제출물, 독립 평가의 결과로 해석할 수 없다.**

## 입력과 재현 범위

입력은 `results/data_v4_20260923/data_v4_results_20260923T1639Z`의 과거 점수와 당시 `file_ensemble.json`이다. 진단 전에 다음 항목을 확인했다.

- 과거 val 4,927개와 test 6,444개, 총 11,371개를 `orig_id`로 `audit_v3`와 연결했다. 원본 manifest·bank·현재 audit의 음원 SHA와 알려진 라벨이 모두 일치했다. 해당 음원의 현재 검증 기록도 있다. 이번 작업에서 음원을 다시 디코딩하거나 모델 추론을 반복하지는 않았다.
- 당시 CDF와 저장된 네 가지 component 확률로 FILE을 다시 계산했다. 최대 절대 오차는 val `1.734723475976807e-18`, test `5.551115123125783e-17`이었다.
- 현재 보존 패키지와 과거 후보는 presence head의 바이트만 같았다. fake checkpoint 목록과 CDF는 다르다. 현재 패키지의 CDF를 과거 점수에 적용하면 FILE 확률의 최대 차이가 val `0.056572735253964`, test `0.055985467513716025`였다. 이번 실험에는 당시 CDF만 사용했다.
- 과거 `candidate_lock.json`이 고정한 파일은 4,931행의 `validation_predictions.csv`다. 이 파일과 겹치는 val decision 4,927행의 두 fake 확률은 정확히 같다. test decision은 그 잠금 파일과 ID가 겹치지 않는다. test decision에도 별도 SHA를 기록했지만, 이를 checkpoint부터 test 출력까지의 암호학적 증명으로 주장하지 않는다.

과거 test 6,444개는 현재 전부 `development` 역할이다. 과거 val 4,927개는 현재 `calibration` 2,021개, `selection` 2,384개, `development` 522개로 나뉜다. 당시 CDF는 과거 val 전체로 적합했으므로 현재 역할 이름만으로 미노출 평가가 되지는 않는다.

## A. 기존 지표와 자료 구성

아래 지표는 알려진 라벨에만 공식 component 계산식을 적용했다. 일부 정답이 없어 전체 점수는 두 패널 모두 `null`이다. ADS·CPS 또는 리더보드 점수로 바꿔 부르지 않는다.

| 과거 패널 | FILE EER | Voice fake EER | Music fake EER | Voice presence AUC | Music presence AUC |
|---|---:|---:|---:|---:|---:|
| val 4,927 | 0.073421 | 0.059922 | 0.124699 | 0.998799 | 0.996951 |
| test 6,444, 현재 development | 0.113911 | 0.122801 | 0.196189 | 0.997946 | 0.998654 |

분모는 다음과 같다. `알려짐/관련 정답 미확인`이며, 성분이 없다고 확인된 파일은 해당 fake EER 대상에서 제외한다.

| 패널 | FILE | Voice fake | Music fake | Voice presence | Music presence |
|---|---:|---:|---:|---:|---:|
| val | 4,927 / 0 | 3,785 / 437 | 1,748 / 0 | 4,802 / 125 | 4,927 / 0 |
| test | 6,444 / 0 | 5,139 / 261 | 2,849 / 0 | 6,268 / 176 | 6,444 / 0 |

**두 과거 패널 모두 음성과 음악이 둘 다 없다고 확인된 `(0,0)` 파일이 0개다.** 따라서 이 점수만으로 잡음·무음에서 presence가 개선됐는지 판단할 수 없다. 아래 test 민감도 패널은 음성만 3,595개, 음악만 1,044개, 둘 다 존재 1,629개다.

source별로 양쪽 music fake 정답이 존재하는 구간에서는 `constructed_mix` EER `0.225947`(518개), `constructed_song` `0.248418`(400개), `haim` `0.188345`(786개)였다. 표본 구성과 생성 방식이 달라 source 간 차이를 원인으로 단정하지 않는다. 나머지 source에는 한 종류의 정답만 있어 EER을 정의할 수 없는 경우도 있다.

## B. 정답 presence로 치환한 민감도

FILE 정답과 두 presence 정답이 모두 있는 test 6,268개만 고정했다. presence 미확인 176개를 제외했으며, component는 2,572개다. fake 확률과 당시 FILE CDF는 바꾸지 않았다.

| 치환 | FILE EER | 기준 대비 EER 변화 | component bootstrap 95% 구간 |
|---|---:|---:|---:|
| 없음 | 0.10289136 | 0 | — |
| Voice presence만 정답 | 0.10289136 | 0 | [-0.00066210, +0.00081953] |
| Music presence만 정답 | 0.09923562 | -0.00365574 | [-0.00635765, +0.00012922] |
| 두 presence 모두 정답 | 0.09923562 | -0.00365574 | [-0.00637977, +0.00021478] |

같은 component를 함께 뽑는 paired bootstrap 1,000회, seed `20260925`를 사용했다. 구간은 모두 0을 포함한다. 정답을 넣는 개입은 실현 가능한 성능 상한이나 회복 예측이 아니다. 고정 fake·CDF에서 presence 입력에 얼마나 민감한지 확인한 결과다.

## C. 적합하지 않은 temperature 변화

같은 6,268개에 미리 정한 presence temperature `0.5 / 1 / 2`를 두 target에 함께 적용했다. 어떤 temperature도 결과를 보고 채택하지 않았다.

| Temperature | FILE EER | 기준 대비 EER 변화 | 95% 구간 |
|---|---:|---:|---:|
| 0.5 | 0.10242605 | -0.00046531 | [-0.00128287, +0.00045632] |
| 1 | 0.10289136 | 0 | [0, 0] |
| 2 | 0.10242605 | -0.00046531 | [-0.00168406, +0.00121759] |

Presence 순위와 AUC는 유지됐고, 서로 다른 확률의 수도 변하지 않았다. 이 패널의 AUC는 voice `0.99794622`, music `0.99856735`다. 확률의 크기만 바꿔도 FILE 결합 결과는 달라질 수 있으므로 presence AUC만으로 FILE 개선을 판정할 수 없다.

## D. 역할을 구분한 탐색적 보정

사용자의 추가 실험 지시에 따라 C까지의 결과와 별도로 조건을 기록한 뒤 수행했다. 현재 `calibration` 2,021개에서 target별 temperature와 Platt 보정을 각각 적합하고 `selection` 2,384개에 한 번씩 적용했다. 두 역할의 component 수는 1,193개와 1,268개이며, component·lineage key 중복은 0개였다. 앞서 설명한 과거 CDF 노출은 남아 있으므로 탐색적 개발 진단으로만 취급한다.

Temperature는 `T ∈ [0.25, 4]`에서 calibration binary log-loss를 최소화했다. Platt는 logit 하나를 입력으로 하는 `LogisticRegression(C=1, penalty=l2, solver=lbfgs, max_iter=1000, random_state=41)`이다. fit 결과를 저장한 뒤 selection 지표를 계산했으며, 선택한 보정법이나 배포 변경은 없다. 적합에 쓴 알려진 voice 정답은 1,967개, music 정답은 2,021개다.

| 보정 | Selection FILE EER | EER 변화 | 95% 구간 | Voice Brier | Music Brier |
|---|---:|---:|---:|---:|---:|
| 원본 | 0.06667462 | 0 | — | 0.00485216 | 0.01669037 |
| 적합한 temperature | 0.06667462 | 0 | [-0.00155742, +0.00169560] | 0.00474595 | 0.01696711 |
| Platt | 0.07091731 | +0.00424268 | [-0.00153369, +0.00713226] | 0.00387950 | 0.01957598 |

적합한 temperature는 voice `0.722897715601194`, music `0.8554714819682245`다. Selection presence AUC는 세 조건 모두 voice `0.99882332`, music `0.99587812`로 같았다. Voice Brier가 좋아져도 music Brier나 FILE이 함께 좋아지지는 않았다. 두 보정법의 FILE 변화 구간은 모두 0을 포함한다.

## 후속 실험에 반영할 내용

- PANNs 구조 비교는 실제로 수행할 가치가 남아 있다. 다만 이 자료만으로 큰 회복을 예상하지 않으며, `(0,0)` 잡음과 낮은 SNR·짧은 사건·가창의 검증 자료를 별도로 확보한다.
- 추가 데이터는 총량보다 빠진 조건을 채우는 데 우선 쓴다. 이미 준비한 잡음 자료가 현재 평가 역할에도 충분히 들어가는지 확인하고, 혼합 비율·시간 점유율을 고정한 비교를 추가한다.
- Voice/Music fake EER은 이 실험에서 고정한 fake 출력의 지표다. 이를 개선하려면 fake 분류 경로의 데이터·방법론은 PANNs presence와 별도 실험으로 비교해야 한다. 특히 혼합·가창 구간은 다음 비교 대상으로 삼을 근거가 있다.
- 보정은 후보로 유지하되 presence AUC 또는 calibration loss 하나만 보고 채택하지 않는다. 알려진 노출을 기록하고, 고정된 비교 패널의 FILE·presence calibration·source별 변화도 함께 확인한다.

## 재현 자료

개인 작업 산출물은 프로젝트 상대 경로 `_workspace/panns_cnn_v1/cpu_score_diagnostics/`에 보관했다. 원시 음원·행별 점수·개인 경로는 이 문서에 포함하지 않았다. `execution.sb`로 네트워크를 차단하고 thread 상한 4, CPU 실행 상한 1,200초, 18 GiB 미만 조건으로 실행했다. A–C는 `7.243087500042748`초, 최대 RSS `0.9306640625` GiB, D는 `1.8758300420013256`초, `0.793701171875` GiB였다. 이는 계산 실행 시간이며 자료 확인·문서 작성 시간은 제외한다.

| 파일 | SHA-256 |
|---|---|
| `predeclared.json` | `7cd19ec6ed0b10a6616455da690e01eb077a26ab1020342ed9ed6a8fc95614e7` |
| `run.py` | `b755ffdcdf5fa2ac53135db9dc371285f27cd6260b058ec2e0290aadc2936871` |
| `results.json` | `d9fac085398f499151b9ef7f11019ba5d93f8487a3dade912572027b7d821d83` |
| `calibration_predeclared.json` | `b5268b02748e142788074ff8acd26970f55e0d2a5672bd9457763f55e4ce3785` |
| `calibration.py` | `c2b0d93072e4e746e8bb1343775489587f27a982f4590ee2e6e198496db16ea1` |
| `calibration_fit.json` | `ccd566846293300949095c0296a3d3409089d0842a4289593be3dc9a4bec9413` |
| `calibration_results.json` | `1458fba1ededcdfec2d031a6c7ecd3411e69b037f2272d167051cad89fc052ff` |
| 과거 `candidate_lock.json` | `02364a6a3d03061d11c3511f295e19f220b95fa162014f321431090496030165` |
| 과거 `file_ensemble.json` | `1df117a29a90c00fa0a47479160cd37ee1fcc983fd9539295d99251e257df1aa` |
| 과거 `val_predictions_decision.csv` | `e4e24b787c2500607cd203cf8790821ad3454ce93635524024e73c53623f5544` |
| 과거 `test_predictions_decision.csv` | `b767d4ab16b9308b09912a0386bf62e4dc8ac7cb4e62974626ccb071f9d5a76c` |
| `audit_v3/presence_rows.jsonl` | `0e54b38495fb4d9acdfc311789ea3879936eee3abdf3e4c89e6280c0ef2ea691` |

`results.json`에 전체 source별 분모·지표·입력 해시를, `calibration_fit.json`에 적합 계수·역할 경계 검증을 남겼다. 검증 근거는 실제 저장 점수의 재계산과 수치 비교이며, 테스트 통과 개수나 다른 모델의 동의가 아니다.
