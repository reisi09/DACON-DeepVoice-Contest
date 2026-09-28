# DACON 236749 V4 결과 분석: 1차 비교와 matched 재검증

## 분석 범위

이 문서는 Colab에서 남긴 실행 기록을 로컬 CPU로 다시 읽어 분석한 결과다. 새 GPU 학습은 하지 않았으며 모델 가중치와 음원도 로드하지 않았다. sealed 데이터에 접근하거나 대회에 제출하지도 않았다.

분석 대상은 source commit `a01a50e0c774f181c20238109127c5b81cc95dc1`에서 실행한 G0와 P1–P4의 1차 비교, source commit `d2397ad8cc96667e5bc2e4132d86405afe093f4e`에서 따로 실행한 matched P1/P3 재검증이다. 기존 P1/P3에는 두 파이프라인의 초기 RNG 상태가 같았다는 기록이 없어 인과 비교에서 제외했다. 새 재검증은 공통 초기 가중치·RNG·seed와 같은 window exposure를 기록했다.

## 결론

P1–P4는 모두 학습을 마쳤지만, 사전에 고정한 품질 게이트를 전부 통과한 후보는 없었다. 최종 선택은 `G0`였고 이유는 `no_candidate_passed`다. `.omo/evidence/v4-retry-finished.json`의 `sealed_opened`도 `false`다.

개발 데이터에는 음성 관련 unknown 라벨이 남아 있다. voice-presence의 알려진 라벨에는 한 클래스만 있다. 이 때문에 5개 헤드 종합 점수는 `null`이며, 이 보고서는 그 값을 0이나 다른 수치로 바꾸지 않는다.

## 1차 비교

| 파이프라인 | 선택 에포크 | 최소 FILE EER 에포크 | FILE EER | MUSIC EER | seen D | OOD D | rank | 통과 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| P1 | 5 | 3 | 0.342857 | 0.341429 | 0.109500 | -0.045000 | -0.045000 | 아니오 |
| P2 | 4 | 9 | 0.289286 | 0.370714 | 0.177000 | -0.045000 | -0.045000 | 아니오 |
| P3 | 5 | 4 | 0.353571 | 0.358571 | 0.045000 | -0.117000 | -0.117000 | 아니오 |
| P4 | 2 | 5 | 0.367857 | 0.446429 | 0.109500 | -0.045000 | -0.045000 | 아니오 |

`선택 에포크`는 `min(seen D, OOD D)`를 최대화한 에포크다. `최소 FILE EER 에포크`는 전체 known-label FILE EER만 가장 낮았던 에포크다. 네 파이프라인 모두 두 에포크가 다르다. 전체 FILE EER만 보고 모델을 고르면 seen/OOD 평가 목표와 다른 결론에 이를 수 있다.

P1, P2, P4의 선택 에포크 OOD D는 -0.045였고, P3는 -0.117이었다. 모든 후보가 요구치 0.005보다 낮았다. P1은 OOD gain·FILE EER·MUSIC AUC·bootstrap win·bootstrap p10 게이트를 통과하지 못했다. P2는 여기에 resource 게이트까지 실패했다. P3는 OOD 관련 게이트 외에 seen bootstrap win과 p10도 통과하지 못했다. P4는 OOD 관련 게이트와 seen bootstrap p10을 통과하지 못했다.

## matched P1/P3 재검증

재검증 계약은 seed `20260913`, 각 arm 10에포크, `one_epoch_warmup_then_30_epoch_cosine` scheduler를 고정했다. P1과 P3의 initial weights SHA-256은 `0784cf50ef834751c93f813f981ca249799c0ebac12b93950ec1c4e2a1b1be10`, seed RNG SHA-256은 `373d3bd3579f4e77cc224197612499148e34968d0f3a69ded66a682559b1beda`로 서로 같다. 두 arm은 모두 10에포크, 350업데이트, 11,160 window exposure를 완료했다. 다만 계약에 적힌 대로 CUDA bitwise 동일성은 주장하지 않는다.

| 파이프라인 | 완료 에포크 | 전체 업데이트 | 선택 에포크 | 선택 checkpoint 업데이트 | 최소 FILE EER 에포크 | FILE EER | MUSIC EER | seen D | OOD D | rank | 실행 시간(초) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P1 | 10 | 350 | 6 | 210 | 2 | 0.371429 | 0.387857 | 0.067500 | -0.045000 | -0.045000 | 127.962 |
| P3 | 10 | 350 | 8 | 280 | 10 | 0.335714 | 0.341429 | 0.049500 | -0.045000 | -0.045000 | 383.178 |

전체 실행의 350업데이트와 선택 checkpoint의 210·280업데이트는 구분해야 한다. 선택 에포크도 파이프라인별로 다르므로 증강 효과는 같은 에포크끼리 비교했다. 에포크 10에서 P3-P1은 FILE EER 0.000000, MUSIC EER +0.012143, seen D +0.037500, OOD D 0.000000, rank 0.000000이다. P3은 seen D가 높았지만 MUSIC EER도 높았고 OOD D와 rank는 같았다. 단일 seed에서 얻은 이 결과만으로 P3의 전반적 우위를 주장할 수는 없다.

이 재검증은 학습 설계의 한 부분을 확인한 분석 자료다. 기존 `G0` 선택을 자동으로 바꾸지 않았고, 새 refit·sealed 평가·대회 제출도 하지 않았다. unknown 음성 라벨을 임의로 채울 수 없으므로 공식 5-head 종합 점수는 여전히 `null`이다.

## 데이터 범위와 refit

탐색 학습은 train의 music-eligible 1,116행을 사용했다. 후보가 게이트를 통과했다면 train+dev의 music-eligible 1,286행으로 full refit을 실행하는 계획이었다. 후보가 전부 탈락했으므로 1,286행 refit은 실행하지 않았다.

## G0 raw/normalized 확인

`v4-retry-G0_predictions.raw.csv`와 `v4-retry-G0_predictions.csv`는 파싱한 210행의 ID, 순서, 확률 값이 같다. 파일 해시가 다른 이유는 개행 표현이 다르기 때문이다. `v4-retry-g0-parity.json`은 2개 ID에서 FILE 공식을 반올림된 head로 다시 계산할 때 발생하는 5e-11 미만의 잔차를 기록한다. 이는 확률 값이 다른 두 G0 CSV를 의미하지 않는다.

## 산출물과 재현

`_workspace/v4_results_report/v4_results_analysis.ipynb`는 `.venv`의 `nbformat`/`nbclient`로 처음부터 끝까지 실행했다. 노트북은 입력 JSON/CSV의 SHA-256, 1차 파이프라인 비교표, 에포크별 FILE/MUSIC EER, seen/OOD D, matched P1/P3 재검증 표와 그래프를 포함한다. 해시와 계산 결과는 `.omo/evidence/v4-results-summary.json`에도 저장했다. 실행 결과가 노트북에 포함되어 있으므로 Drive에서는 재실행하지 않고도 표와 그래프를 볼 수 있다. 원래 Colab 학습 노트북의 capture는 `current_cell_may_be_incomplete=true`이며, 이 노트북은 그 학습 노트북을 완료된 것으로 바꾸어 표현하지 않는 별도의 분석 재실행이다. 재실행할 때는 `V4_analysis_replay_bundle.zip`을 현재 로컬 workspace 루트에 풀고 CPU로 실행한다.

이 보고서는 1차 비교와 새 matched P1/P3 재검증의 현재 결과를 기록한다. 새 재검증은 통제된 한 seed 비교를 완료했지만, P3의 전반적 우위나 공식 점수 개선을 보여 주지는 않았다.

<!-- HUMANIZE-SUMMARY
원본 4061자 / 윤문본 4066자 / 변경률 0.41%
카테고리별 탐지: A2 불필요 접속어 1→0, B1 피동형 0→0, D1 관용구 1→0, I2 문장 리듬 1→0
자체검증: 6/6 통과
등급: B - 기존 윤문본에 새로 추가된 matched 재검증 문단만 보수적으로 다듬었다.
주요 변경:
- "다른 값이다" → "구분해야 한다"
- "이 한 seed 결과로" → "단일 seed에서 얻은 이 결과만으로"
- "바꾸지 않고" → "바꾸지 않았고"
-->
