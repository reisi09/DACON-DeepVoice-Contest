# DACON 236749 V4 1차 결과 분석

## 분석 범위

이 문서는 Colab에서 남긴 실행 영수증을 로컬 CPU로 다시 읽어 분석한 결과다. 새 GPU 학습은 하지 않았으며 모델 가중치와 음원도 로드하지 않았다. sealed 데이터에 접근하거나 대회에 제출하지도 않았다.

분석 대상은 source commit `a01a50e0c774f181c20238109127c5b81cc95dc1`에서 실행한 G0와 P1–P4의 1차 비교다. 기존 P1/P3에는 두 파이프라인의 초기 RNG 상태가 같았다는 영수증이 없다. 기존 P3 결과로는 증강의 인과 효과를 주장할 수 없다. 15에포크 matched P1/P3 재검증은 따로 진행 중이며, 아직 이 문서에 결과를 포함하지 않았다.

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

## 데이터 범위와 refit

탐색 학습은 train의 music-eligible 1,116행을 사용했다. 후보가 게이트를 통과했다면 train+dev의 music-eligible 1,286행으로 full refit을 실행하는 계획이었다. 후보가 전부 탈락했으므로 1,286행 refit은 실행하지 않았다.

## G0 raw/normalized 확인

`v4-retry-G0_predictions.raw.csv`와 `v4-retry-G0_predictions.csv`는 파싱한 210행의 ID, 순서, 확률 값이 같다. 파일 해시가 다른 이유는 개행 표현이 다르기 때문이다. `v4-retry-g0-parity.json`은 2개 ID에서 FILE 공식을 반올림된 head로 다시 계산할 때 발생하는 5e-11 미만의 잔차를 기록한다. 이는 확률 값이 다른 두 G0 CSV를 의미하지 않는다.

## 산출물과 재현

`_workspace/v4_results_report/v4_results_analysis.ipynb`는 `.venv`의 `nbformat`/`nbclient`로 처음부터 끝까지 실행했다. 노트북은 입력 JSON/CSV의 SHA-256, 파이프라인 비교표, 에포크별 FILE/MUSIC EER 그래프, seen/OOD D 그래프를 포함한다. 해시와 계산 결과는 `.omo/evidence/v4-results-summary.json`에도 저장했다.

이 보고서는 1차 비교의 현재 상태를 기록한다. matched P1/P3 재검증 영수증이 완료되기 전에는 증강의 인과 효과를 최종 판단할 수 없다.

<!-- HUMANIZE-SUMMARY
원본 2436자 / 윤문본 2412자 / 변경률 2.85%
카테고리별 탐지: A2 불필요 접속어 3→0, B1 피동형 2→0, D1 관용구 1→0, I2 문장 리듬 2→1
자체검증: 6/6 통과
등급: B - 수치·고유명사·판단 범위를 유지하면서 번역투와 기계적 접속을 제거했다.
주요 변경:
- "Colab에서 생성된" → "Colab에서 남긴"
- "따라서 기존 P3의 결과로" → "기존 P3 결과로는"
- "학습을 완료했지만" → "학습을 마쳤지만"
- "실제로 네 파이프라인" → "네 파이프라인"
-->
