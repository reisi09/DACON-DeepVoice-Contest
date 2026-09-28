# 평가 산식·후보 선택·비교 해석 감사

검토일은 2026-09-18이다. 운영 코드는 수정하지 않았고 GPU를 실행하지 않았다. 저장 예측을 CPU에서 다시 계산했다. 통계·OOF·sampler 상세 근거는 `evaluation_statistics.md`와 `evaluation_statistics_*` 산출물에 연결한다.

## 판단

현재 `graph_scoring.py`의 EER 산식과 성분 평가 마스크에서 오류를 발견하지 못했다. 실예측을 넣어 계산한 `graph_scoring`·`v2_scoring`·`eight_metrics`·sklearn 비교 356개에서 최대 절대 차이는 `2.220446049250313e-16`이었다. 다만 산식 일치와 모델의 일반화는 별개다. 저장 보고서에는 E5 test의 독립 성분 수를 전체 split 합계로 표시한 오류가 남아 있고, 기존 eight sampler의 실제 노출 누락과 bootstrap 유효 재표집 비율 검사 부재도 확인됐다. 후자의 두 건은 통계 상세 절을 참조한다.

최신 공식 제출 수치는 주 감사자가 현재 제출 화면에서 확인한 MERT12 `0.7479394074`, ADS `0.7211111111`, CPS `0.9893940741`이다. M330은 `0.7396465503`, ADS `0.7118968254`, CPS는 같다. 총점 차이는 `0.0082928571`, ADS 차이는 `0.0092142857`이다. 표시 자리수의 반올림 오차 안에서 `0.9 × ADS 차이`가 총점 차이와 일치한다. ADS 하나만으로 File/Voice/Music EER 각각의 변화량은 역산할 수 없다. 이 감사 하위 작업은 공식 화면을 별도로 재조회하지 않았으며 주 감사자의 현재 증거를 인용한다.

## 남아 있는 실제 오류

### EVAL-01 · E5 test 성분 수가 1개인데 요약 JSON은 4개로 표시한다

- 위치: `results/eight_h100_20260917/FINAL_REPORT.metrics.json:162-165`, 키 `experiments.E5.test_counts.components`.
- 원시 예측 `downloaded/unpacked/output/predictions_E5.csv`는 test 144개(REAL 72 / FAKE 72), 고유 `component_id` 1개다. train/calibration/dev/test는 각각 1개이며 모두 합쳐 4개다.
- `corrected_summary/final_aggregation.corrected.json:538`의 `component_count:4`는 `population:648`(545행)과 함께 전체 E5 모집단을 가리킨다. 해당 정정본은 test 성분 수를 별도로 고치지 않았다. 따라서 이미 고친 178→256 전체 성분 수 오류와 다른 잔여 문제다.
- 영향: EER·AUC·recall·FPR 값은 재현됐으나 test 독립 단위 수를 4배로 오인할 수 있다. 본문 `FINAL_REPORT.md:85`의 “전체 provenance 성분은 4개”는 범위를 명시했으므로 그 문장 자체는 틀리지 않다.
- 최소 수정: 최종 metrics JSON의 E5 test components를 1로 고치고 전체 4와 분리한다. 원시 예측과 GPU 보고서는 그대로 보존한다.
- 근거: `evaluation_recompute.json → eight.E5.split_components`, `evidence/evaluation_statistics_probe_populations.json`.

### EVAL-02 · fractional label을 정수로 잘라 정상 이진 라벨로 받아들인다

- 위치: `eight_metrics.py:17-23`.
- 원인: 원래 값을 검사하기 전에 `np.asarray(labels, dtype=np.int64)`로 변환한다.
- 재현: `binary_summary([0.5,1.5],[0.1,0.9])`가 오류 없이 `EER=0`, `AUC=1`, REAL 1 / FAKE 1을 반환한다.
- 영향 경계: 이번 저장 예측의 `label`은 0/1이며 이 결함이 기존 보고 EER를 바꿨다는 증거는 없다. 향후 잘못된 라벨을 조용히 점수화할 수 있는 입력 검증 결함이다. 원시 라벨의 이진성·유한성을 확인한 뒤 정수로 변환해야 한다.
- 근거: `evaluation_recompute.json → contracts.fractional_label_coercion_reproduced`.

## 통과한 평가 계약

`graph_scoring.py:91-110`은 높은 확률을 FAKE(1)로 처리하고 동점 score를 묶어 full ROC를 만든 뒤 `|FPR−FNR|`가 최소인 점의 `(FPR+FNR)/2`를 쓴다. 선형 보간 EER가 아니다. `v2_scoring.py:64-89`와 `eight_metrics.py:27-41`도 같은 이산 규칙이다. 저장 E1/E3/E4/E6/E8 예측을 직접 대입했을 때 EER와 AUC가 sklearn의 `drop_intermediate=False` 결과와 부동소수점 반올림 범위에서 같았다.

`graph_scoring.py:135-147`은 `VOICE_PRESENT`/`MUSIC_PRESENT`의 정답 값으로 성분 EER 대상 여부를 정한다. 실제 six test 예측의 존재 확률을 전부 반전해도 Voice/Music EER와 평가 대상 수가 그대로였다. 예측 존재 확률로 낮은 확신의 성분을 평가에서 빼는 동작은 없다.

`graph_scoring.py:153-165`는 unknown relevant label과 single-class를 구분하며 하나라도 있으면 공식 총점을 내지 않는다. six 기본 test 29개에서 File EER n=29, Voice EER n=8, Music EER n=21이다. voice 관련 미확정은 21개이고 known `VOICE_PRESENT` 8개도 한 class만 있어 Voice Presence AUC는 `null`이다. 여섯 모델의 dev/test 12개 보고서가 모두 `score:null`을 유지했다. 완전 라벨·양 class가 있는 별도 계약 사례에서는 score 1을 계산했고, 관련 성분 라벨 한 개를 비우자 총점이 다시 `null`이 됐다.

공식 가중식은 `0.45*(1−File EER)+0.18*(1−Voice EER)+0.27*(1−Music EER)+0.05*Voice Presence AUC+0.05*Music Presence AUC`다. `v2_scoring.py:150-158`의 proxy는 계산 가능한 지표만 남긴 뒤 그 가중치 합으로 나눈다. 이는 의도된 개발용 proxy이며 공식 총점으로 읽으면 안 된다. `redesign`의 변경 출력 가중 개선량은 이 proxy와도 다르므로 서로 다른 목적함수의 후보 순위를 직접 비교하면 안 된다.

## 저장 예측에서 재현한 후보 비교

아래 EER는 로컬 기록 데이터의 값이다. MERT12 full1322의 공식 score와 같은 지표가 아니다.

| 비교 | 원본·독립 성분 | seed EER 평균 | 확률 평균 후 EER |
|---|---:|---:|---:|
| E6 MERT L6 | test 127 / 24 | 0.12290969899665549 | 0.12290969899665549 |
| E6 MERT L12 | test 127 / 24 | 0.08995261984392416 | 0.09155518394648826 |
| E6 MERT L24 | test 127 / 24 | 0.17349498327759194 | 0.17349498327759194 |
| E3 CE, Opus+telephone | test 127 / 24 | 0.23292920847268675 | 0.21927257525083615 |
| E3 CE+consistency, Opus+telephone | test 127 / 24 | 0.22408026755852842 | 0.22408026755852845 |

E6 L12 우세는 두 집계 모두에서 나타난다. E3은 seed 평균으로 `−0.00884894091415833` 개선되지만 실제 확률 ensemble은 `+0.0048076923076923` 악화된다. seed 평균을 제출 ensemble 성능으로 바꾸어 쓰면 방향까지 틀린다. `eight_ablation.py:274-281`은 평균 dev EER로 layer를 고른다. 최종 제출이 probability ensemble이면 dev에서도 그 결과를 직접 비교하는 선택 규칙이 더 맞으며, 이미 확인한 test 결과로 규칙을 소급 변경하면 안 된다.

E6 Music test는 REAL 23 / FAKE 104다. EER은 FPR/FNR를 반반 반영하므로 단순한 fake 비율 104/127이 점수의 가중치가 되지는 않는다. 다만 각 class 안에서는 행이 많거나 파생본이 많은 source가 더 크게 작용한다. source 균형 학습을 했다는 이유로 평가도 source 균등 추정량이 되는 것은 아니다. 연결 성분 bootstrap 역시 성분을 재표집할 뿐 표본에 포함된 각 성분의 행 수를 동일하게 만들지는 않는다.

## six_h100 선택은 이름 순서와 학습량의 영향을 받는다

`results/six_h100_20260917/downloaded/unpacked/output/experiment_4.json`과 `experiment_5.json`의 dev 지표는 각각 실제 예측과 완전히 같았다. 두 후보는 File EER `0.07107843137254904`, Music EER `0.13942307692307693`으로 동점이고 B seed41은 각각 `0.3137254901960784`, `0.3798076923076923`이다. 코드의 `.45*File 개선+.27*Music 개선`은 둘 다 `0.17409502262443438`이다.

`six_h100_experiments.py:211-218`의 `max((gain,name))` 때문에 `mixture_seed41`이 선택된다. 실제 `selection_locked.json.selected`도 같다. 테스트를 보고 고른 흔적은 아니지만 혼합 자료가 더 좋다는 근거도 아니다. 후속 실험에서는 동점일 때 기존 모델이나 더 작은 추가 데이터 구성을 유지하는 규칙을 미리 정해야 한다.

동일 17 epoch·batch 64라도 최적화 횟수는 달랐다. B seed41은 1,696행·459 updates, 새 음악 후보는 1,758행·476 updates, 혼합 후보는 1,814행·493 updates다. 근거는 `results/redesign_full_20260917/downloaded/unpacked/output/models/B_seed41/training.json`, six `experiment_4.json:88-99`, `experiment_5.json:88-99`다. `redesign_fit.py:200-221`은 모든 행을 epoch마다 돌기 때문에 행 추가가 곧 updates 증가로 이어진다. 따라서 “같은 B architecture/seed/epochs”라는 설명은 맞지만 데이터 구성만의 순수 효과를 분리한 비교는 아니다.

B full-data와 six 선택 ensemble은 seed 41/42/43에 각각 17/17/8 epoch를 쓴다(`redesign_full_refit_run.py:65-69`, `six_h100_experiments.py:225-228`). 세 모델을 평균한 결과는 실제 ensemble 결과로 보고할 수 있지만, 그 모델 간 분산을 seed만 바꾼 변동성으로 해석하면 학습 예산 차이가 섞인다.

이 선택의 최종 확인셋에서 B seed41 File EER는 `0.10049019607843139`, 새 음악은 `0.1715686274509804`, 혼합은 `0.07107843137254904`였다. dev에서 큰 개선이 그대로 유지되지 않았다는 진단까지는 가능하다. 이미 열어 본 이 test를 다음 후보의 새 독립 검증셋으로 부를 수는 없다.

## threshold·짧은 삽입 패널의 해석

E1/E3/E4/E6의 `evaluate(...,threshold=None)`은 해당 test의 EER 점에서 threshold를 구한다(`eight_metrics.py:61-69`, `eight_fixed_io.py:123-134`). 그래서 EER/AUC는 유효한 순위 지표지만 함께 저장한 test FPR/recall은 test 정답으로 맞춘 operating point다. 사전에 고정한 배포 threshold의 성능으로 보고하면 안 된다.

E5와 E8은 calibration REAL에서 threshold를 고정한다. E5의 6개 threshold와 test의 모든 요약 항목이 재현됐다. calibration 108 REAL에서 목표 FPR 5%를 만족시키는 empirical cutoff이며, 새 모집단에서 FPR 5%를 보장하지는 않는다. 실제 E5 test Anti-max는 recall `0.20833333333333334`, FPR `0.1111111111111111`; DF-top30은 recall `0.09722222222222222`, FPR `0`이다. test는 REAL/FAKE 72개씩이지만 독립 성분은 1개다.

`eight_probes.py:203-215`는 calibration threshold 후 dev recall, dev FPR, 고정 pool 순서로 선택을 잠근다. `:222-225`의 삽입 길이·위치별 패널은 FAKE만 있거나 REAL만 있는 경우 EER/AUC가 `null`인 것이 정상이다. 같은 배경·화자·길이·위치에서 파생한 행들을 독립 관측으로 세거나 `null` EER를 0으로 바꾸면 안 된다.

## 공식 MERT12와 이전 제출을 비교할 때 남는 혼동

MERT12 full1322는 이전 train/calibration/dev/test를 합친 1,322개 음악에 대해 seed 41/42/43을 각각 500 updates로 다시 학습했다. receipt의 `all_former_splits_are_training:true`(2행), `eligible_rows:1322`(4402행), `independent_validation:false`(4433행)가 이를 명시한다. 과거 E6 test EER를 이 재학습 모델의 독립 성능으로 옮겨 적을 수 없다.

M330 대비 공식 score 개선은 확인된 결과지만 L12 층 하나의 인과 효과로 확정할 수는 없다. full1322 재학습 자료, head·sampler와 노출, seed ensemble, 전처리 구현을 함께 구분해야 한다. 특히 `mert12_submission.py:109-120,155`는 Demucs 입력을 `torchaudio.load+demucs.audio.convert_audio`로 고정한다. 이전 FFmpeg 경로와 stem이 달라지면 음성 가중치가 같아도 음성 예측은 바뀔 수 있다. 같은 자료·같은 sampler 노출·같은 updates·같은 decoder로 L12/L24를 비교한 실험만 layer의 효과를 좁혀 설명할 수 있다.

## 범위와 재실행

직접 읽은 중심 파일은 `graph_scoring.py`, `v2_scoring.py`, `eight_metrics.py`, `eight_fixed_io.py`, `eight_ablation.py`, `eight_heads.py`, `eight_probes.py`, `eight_probe_stats.py`, `eight_e2_music_file_analysis.py`, `six_h100_experiments.py`, `redesign_fit.py`, `mert12_full_refit.py`, `mert12_submission.py`다. 통계 보조 감사는 `redesign_evaluate.py`, `redesign_evaluation_common.py`, `redesign_oof.py`, `redesign_oof_validation.py`를 추가 검토했다.

실제 재계산은 E1 음악·음성, E3 음악·음성의 네 codec, E4, E6 MERT·Anti, E8 dev/test, E5 calibration threshold·test 전체, six의 B seed41/새 음악/혼합/B ensemble/M330/선택 ensemble의 dev/test 기본 조건을 다룬다. `evaluation_recompute.py`는 이 범위를 다시 실행하며 입력 예측 SHA-256을 `evaluation_recompute.json`에 기록한다.

```sh
.venv/bin/python results/precision_audit_20260918/evaluation_recompute.py
```

이번 계산은 모델 재학습·소리의 라벨 청취·공식 숨김 test 평가·L4 전체 실행을 대신하지 않는다. 표본 수 300을 보편적인 적합 기준으로 두지 않는다. 필요한 독립 표본 수는 구분하려는 개선량, class·source별 구성, 연결 구조와 목표 불확실성에 따라 정해야 한다. 현재 문제는 256이 300보다 작다는 숫자 비교보다 일부 핵심 voice/partial test의 독립 성분이 실제로 1개라는 점이다.

<!-- humanize-korean final pass: meaning, numbers, paths, uncertainty and technical terms preserved. -->
