# 학습·최적화·용량 감사

확인 범위는 최신 MERT12 full1322, 실제 실행한 eight E1/E3/E6, redesign B full1696, temporal/fullFT의 학습 경로다. GPU를 켜거나 가중치를 다시 학습하지 않았다. 저장된 manifest·feature·checkpoint를 CPU에서 재생했고, 아래 학습 데이터 지표를 독립 검증 성능으로 해석하지 않았다.

## T1. 최신 coverage prefix는 최소 1회 노출만 고쳤고, 실제 학습 편향은 남아 있다 — 확인됨, 우선순위 높음

`eight_heads.py:49-60`은 같은 전역 `ordinal`로 source, generator, parent의 나머지를 계산한다. source가 두 개이고 한 source의 parent 수가 짝수이면, 그 source에 도달하는 ordinal의 홀짝이 고정되어 parent 절반에 도달하지 못한다. seed는 마지막 member 선택과 배치 안 순서에만 영향을 주므로 이 누락은 seed를 바꿔도 사라지지 않는다.

실제 E6 음악 학습 831개에 500×32 schedule을 재생했다. 789개만 노출됐고 REAL 42개가 빠졌다. FMA 84개 중 42개만 학습했고 FMA/FMA 55개는 모두 학습했다. 이 재생의 ID 순서 SHA-256은 저장된 `E6_mert_l12_seed41.pt`의 `draw_sha256`과 같았다. parent-uniform 대조군은 831개 모두 노출됐다. 따라서 당시 E1 balanced 대조는 source 재가중뿐 아니라 실제 학습 원본의 범위도 달랐다.

최신 `mert12_full_refit.py:138-163`의 prefix는 총 1,322개 최소 노출을 보장한다. 그러나 기존 schedule에서 빠지던 **109개 모두 REAL**이고, 세 seed 모두 이 109개는 정확히 한 번씩만 등장한다. 마지막 등장 시점은 zero-based update 13이며, 나머지 update 14–499에는 등장하지 않는다. REAL 218개 중 절반을 처음 14회 업데이트에만 사용한 셈이다. source별 나머지 REAL은 약 61–88회씩 등장한다. 이 계산은 세 seed의 실제 `full_refit_receipt.json.checkpoints[].after_counts_by_id`와 정확히 일치했다.

저장된 최종 세 head의 확률을 평균해 1,322개 학습 feature에 적용한 결과:

| 학습 부분집합 | 행 수 | 0.5 기준 오분류 | MUSIC_FAKE 확률 중앙값 | 평균 BCE |
|---|---:|---:|---:|---:|
| prefix에서만 1회 사용한 REAL | 109 | 42 | 0.1900049299 | 1.6916062916 |
| 계속 사용한 REAL | 109 | 0 | 0.0007048869 | 0.0009352582 |
| FAKE | 1,104 | 0 | 0.9999449849 | 0.0008384482 |

학습 데이터의 오분류 42개가 모두 prefix-only REAL에서 발생했다. 이는 현재 학습 구성에서 관찰한 뚜렷한 fit 격차다. 0.5는 공식 EER threshold가 아니며, 이 표가 sampler만으로 공식 점수 하락을 설명하거나 잘못된 라벨 가능성을 배제하지는 않는다. 그래도 단순히 MLP 용량이 작아서 전체 데이터를 못 외운다는 설명보다, 반복 노출의 차이를 먼저 확인할 근거가 강하다.

- 근거: `results/precision_audit_20260918/training_sampling.json`, `training_feature_replay.json`.
- 범위: 최신 제출 head에도 남은 현상이다. “109개가 한 번도 학습되지 않았다”는 이전 문제는 고쳐졌고, “109개가 충분히 반복 학습됐다”는 주장은 성립하지 않는다.
- 결정적 다음 확인: 전역 ordinal 대신 source/generator별 parent cursor를 따로 진행하는 schedule에서 모든 원본의 epoch별 노출량을 먼저 검사한다. 이후 원래 split을 유지한 동일 500-update 비교로 일반화 영향을 확인해야 한다. 이번 감사에서 수정·재학습·재제출하지 않았다.

## T2. E3 consistency는 831개 원본 중 앞의 250개에만 적용됐다 — 확인됨, 실험 해석 제한

`eight_ablation.py:63-72`는 manifest 삽입 순서대로 clean/processed pair를 만든다. `eight_heads.py:157-163`은 매 update마다 `pairs[step % len(pairs)]` 하나만 consistency에 사용한다. 실제 음악 train은 원본 831개, base/telephone/mp3_32k 총 2,493행이며 pair는 1,662개다. 500 update 동안 앞의 500 pair, 즉 250개 원본만 사용했다. 이 pair 순서는 seed마다 같고 pair update는 FAKE 412 / REAL 88이다.

CE schedule 자체는 2,493개 모두 노출됐고 CE-only/CE+consistency의 draw digest도 같은 구조다. 문제는 consistency가 전체 원본에 고르게 적용된 것으로 읽으면 안 된다는 데 있다. consistency gradient는 실제로 연결되어 있지만 적용 대상과 label 비중이 CE의 1:1 노출과 다르다. 이 E3 결과만으로 “codec consistency가 효과 없다”는 일반 결론을 낼 수 없다.

- 근거: `training_sampling.json.historical.E3_music.consistency_pairs`.
- 다음 확인: 전체 pair를 한 번 이상 순회하거나 seed별 pair permutation을 사용한 동일한 추가 forward 예산 비교. 여러 pair를 배치로 처리하면 loss scale도 함께 고정해야 한다.

## T3. 3-window mean feature 학습과 긴 파일 추론의 분포 차이 — 구조는 확인, 성능 영향은 가설

최신 학습 NPZ 1,322개를 모두 읽었다. 모든 원본이 정확히 3개 window를 갖는다. `eight_features.py:99-109`는 MERT hidden_states[12]의 frame mean을 만들고, `mert12_full_refit.py:74-87`은 유효 window feature를 평균한다. 최종 경로도 `mert12_submission.py:87-99,137-146`에서 전체 half-hop window의 평균 feature를 만든 뒤 head와 sigmoid를 적용한다. 따라서 학습과 추론에서 평균·비선형 연산 순서가 뒤집히는 구현 오류는 없다.

다만 학습은 3-window 짧은 clip이고 추론은 파일 전체의 가변 window 수다. 긴 파일의 짧은 가짜 구간은 feature mean에서 비중이 작아진다. 평균만 받은 head는 window 순서와 구간별 변동을 복원할 수 없다. partial-fake·긴 음악에서 이것이 실제 성능 병목인지는 해당 파일의 구간 truth가 있어야 판정할 수 있다.

실제 저장 feature와 head로 `head(mean(feature))`와 `mean(sigmoid(head(feature_window)))`를 비교했다. 세 seed 확률 평균의 절대 차이 최대값은 **0.3897524774**, 0.01 초과는 **148/1,322개**다. 둘을 같은 aggregation으로 취급하거나 후처리만으로 교환해서는 안 된다. window 순서만 바꾸거나 동일 window 집합을 반복하면 mean은 보존되므로, 그 불변성을 통과해도 국소 탐지 능력이 입증되지는 않는다.

`stem_mask`를 사용하지 않는 설정도 확인했다. 그러나 이번 1,322개에는 mask=0인 window가 **0개**여서, 현재 학습 데이터에 audibility mask를 추가하는 것만으로 feature가 달라지지는 않는다. 짧은 파일의 zero-padding frame 평균과 실제 무음·긴 파일의 희석은 별도 가설로 남긴다.

- 근거: `training_feature_replay.json.window_count_histogram`, `inaudible_windows`, `mean_feature_vs_mean_window_probability_abs_diff`.
- 다음 확인: 길이와 가짜 구간 비율을 통제한 labeled partial set에서 mean-feature head와 window-score pooling을 동일 backbone·train split으로 비교한다. 새 pooling으로 바꿀 때는 별도 학습/선정 실험이 필요하다.

## T4. redesign의 global class/group weight가 minibatch 안에서 재정규화된다 — 계산 확인, 영향은 제한적으로 해석

`redesign_fit.py:92-109`는 전체 train의 class/group weight 합을 1로 만든 뒤, 각 minibatch에서 다시 `weights[eligible].sum()`으로 나눈다. 이렇게 나눈 weighted ratio는 uniformly shuffled minibatch에서 전체 weighted objective의 unbiased estimate가 아니다. 극단적으로 batch=1이면 양수 weight 크기는 완전히 상쇄된다. 같은 프로젝트의 temporal 경로는 `music_temporal.py:321-331`에서 global weight에 `N/B`를 곱하므로 이 문제를 피한다.

B full1696의 실제 manifest·seed·epoch·batch64 순서를 재생했다. 라벨 eligible 1,216개(REAL 190 / FAKE 1,026)이고 receipt상 모두 audible하다. loss에 들어가는 정규화 계수의 누적 REAL 비중은 seed41 **0.4804508470**, seed42 **0.4804735859**, seed43 **0.4773421091**이었다. 원래 global weight 비중은 약 0.5다. 개별 행의 누적 계수/의도 weight 비율은 seed43에서 **0.8000–1.7191** 범위였다.

이는 고정된 loss값을 가정했을 때 각 행에 부여되는 계수의 재구성이다. 실제 gradient는 가중치 변화, dropout, CE와 event loss에 따라 달라지므로 이를 최종 gradient 기여율이라고 부르지 않는다. 약 2 percentage points의 class weight drift가 공식 점수에 미친 영향은 측정하지 않았다. 모든 group을 균등하게 최적화했다는 엄밀한 설명에는 맞지 않는다.

- 근거: `training_redesign_weights.json.epochs`.
- 다음 확인: 작은 알려진 loss/weight population에서 minibatch 추정의 기대값이 global loss와 일치하는지 수치로 확인하고, 필요할 때 global denominator를 유지한 실제 split 비교를 한다.

## 용량·학습 연결·선정에 관해 확인한 비문제와 한계

| 항목 | 확인 결과 | 해석 경계 |
|---|---|---|
| 최신 train 대상 | core/base/MUSIC_PRESENT=1/known MUSIC_FAKE만 1,322개; 218 REAL / 1,104 FAKE | 7,101개 전체 파생 row 학습이 아니다. |
| latest trainable parameters | `FrozenHead(1024)` 131,329개. 저장 state는 MLP뿐이다. | MERT330 backbone fine-tuning이 아니다. |
| MLP gradient·update | 세 receipt 모두 네 parameter tensor의 첫 gradient L1>0, update L2 약 6.62–6.75 | 연결·변화의 증거이며 일반화의 증거가 아니다. |
| optimizer | AdamW lr=1e-3, weight_decay=1e-2, clip=1, batch32, 500 updates; scheduler·early stop·dropout 없음 | 500 update가 최적인지, convergence가 충분한지 판단할 학습 곡선은 남기지 않았다. |
| latest normalizer | 각 checkpoint의 mean/std를 1,322개 원본으로 다시 계산했을 때 최대 차이 모두 0 | full-data refit의 train population이다. test-time 파일 간 normalization은 아니다. |
| feature binding | 1,322개 NPZ의 ID/audio hash/NPZ hash가 training manifest와 모두 일치; extractor code hash도 하나로 동일 | 실제 cache 혼합이나 라벨/ID 어긋남을 발견하지 않았다. |
| E6 normalization | `eight_ablation.py:58-62`에서 train만 fit, dev/test에는 그 통계 적용 | latest full-data normalization과 E6 validation normalization을 혼동하면 안 된다. |
| E6 selection | `eight_ablation.py:274-281`에서 seed별 dev EER의 평균으로 layer 선택 후 lock | 최종 probability ensemble의 EER와 seed EER 평균은 다른 통계다. |
| latest selection | 이전 layer12 recipe를 고정하고 all-former-splits로 재학습, 마지막 500-update state 저장 | 독립 validation이 없으므로 full1322의 품질을 local train 값으로 비교할 수 없다. |
| redesign head 크기 | A control 107,234개, B residual 115,785개; B는 8,551개 추가(약 7.97%) | “더 큰 residual”의 효과와 raw 정보의 효과를 분리하려면 capacity control이 필요하다. |
| B encoder | `MusicResidual`의 encoder는 cached MERT vector 위에 둔 학습 가능한 TemporalMusic head다. | MERT 자체가 학습됐다는 뜻이 아니다. |
| B 무라벨 speech | 480 speech row의 music weight가 0이므로 music negative로 강제 학습하지 않는다. | 1,696개 전부가 supervised music gradient에 기여했다는 뜻은 아니다. |
| redesign unknown event | `_targets`는 명시된 music/accompaniment interval에 완전히 들어가는 window만 known으로 지정; BCE는 valid/audible와 함께 mask | 모르는 target을 0으로 채워 학습하는 문제는 확인하지 않았다. 경계 window는 supervision에서 빠진다. |

Latest normalizer 검증은 모든 1,322개를 대상으로 했다. normalization을 fitting한 행과 gradient에 반복 사용한 행의 비중은 다르지만 이것 자체가 라벨 leakage는 아니다. 모델 용량을 늘리거나 줄이는 제안보다 T1의 실제 노출 schedule을 먼저 바로잡아 비교하는 편이 원인 분리에 유리하다.

추가로 `redesign_models.py:77-81`는 동일 shared encoder를 stem과 raw에 각각 호출한다. raw=stem capacity control이라도 train mode의 dropout은 두 호출에서 독립적으로 적용되어 hidden 차이가 항상 0은 아니다. 이 점은 capacity-only 해석에 주의가 필요한 설계 특성이다. 해당 효과 크기를 별도로 측정하지 않았으므로 실제 실패로 분류하지 않는다.

## Temporal/fullFT 별도 확인

상세 근거와 실제 optimizer update 표는 `sections/training_temporal.md`에 있다. 핵심은 다음과 같다.

- 실제 MERT/EAT fullFT는 encoder gradient와 optimizer update가 연결되어 있다. frozen-feature head 학습과 구분해야 한다.
- fullft8 microbatch1×accum8과 microbatch4×accum2는 effective batch8을 유지했다. 마지막 짧은 accumulation group의 loss도 실제 표본 수로 보정했다.
- temporal batch16/batch128 비교는 selected checkpoint 기준 126/32 updates, completed run 기준 1,386/192 updates다. equal epoch가 equal update는 아니므로 batch 하나만의 효과를 격리한 실험이 아니다.
- S0 MERT fullFT의 train loss는 감소하지만 선택 epoch의 tune BCE는 0.5993429717→1.4516976697로 증가했다. EER는 0.3030119019→0.2304671687로 개선되어 EER 우선 선정 규칙대로 선택됐다. 곡선은 ranking/calibration의 차이를 보여주며, 특정 LR·용량 원인까지 증명하지는 않는다.
- separator adaptation은 pseudo-stem L1을 학습한 별도 단계이고, classifier 학습 때 separator는 다시 frozen이다. end-to-end joint separation/classification 학습으로 표현하면 안 된다.

## 조사한 코드·실행 원본·재현 명령

실행 코드는 원래 source.zip과 다를 수 있어 확인했다. eight source.zip의 `eight_ablation.py`, `eight_features.py`는 현재 root와 다르지만, 최종 attempt6의 `_workspace/eight_h100_20260917/runtime_overrides_full/`과 root는 SHA-256이 같다. `results/eight_h100_20260917/runtime_sources/full_overlay_then_main.attempt6.sh:44-56`의 선언된 runtime hash도 같다. `eight_heads.py`, `eight_fixed_io.py`는 source.zip과 root가 같다. MERT12 upload/source.zip의 `mert12_full_refit.py`, `eight_heads.py`와 redesign full source.zip의 `redesign_fit.py`, `redesign_models.py`도 root와 같았다. `training_runtime_hashes.json`에 final override hash를 남겼다.

직접 읽은 주요 코드: `mert12_full_refit.py`, `mert12_submission.py`, `eight_heads.py`, `eight_fixed_io.py`, `eight_ablation.py`, `eight_features.py`, `eight_feature_identity.py`, `eight_lora.py`, `eight_lora_runtime.py`, `eight_lora_train.py`, `antideepfake_runtime.py`, `redesign_fit.py`, `redesign_models.py`, `redesign_train.py`, `redesign_full_refit_run.py`, `redesign_checkpoint.py`, `redesign_features_cache.py`, `music_temporal.py`; temporal/fullFT의 추가 원본은 별도 section에 기록했다.

직접 읽은 핵심 데이터: `_workspace/eight_h100_20260917/dataset/manifest.csv`, `_workspace/mert12_a100_20260918/upload/features.zip`의 1,322개 NPZ, `results/mert12_a100_20260918/downloaded/unpacked/output/mert12_full/`의 training manifest·full-refit receipt·3개 head, `results/eight_h100_20260917/downloaded/unpacked/output/checkpoints/`의 E6/E1/E3 saved draw digest, `results/redesign_full_20260917/downloaded/unpacked/output/full_training.csv`와 full-refit receipt.

저장된 sampler와 최신 head의 핵심 현상은 다음 명령으로 다시 확인한다. 학습·모델 다운로드·GPU 실행은 하지 않는다.

```sh
PYTHONPATH=. .venv/bin/python results/precision_audit_20260918/training_reproduce.py
```

이 재현은 saved draw counts 일치, prefix-only REAL 109개의 마지막 update 13·노출 1회, saved ensemble의 학습 오분류 42개가 모두 그 집합에 속하는지를 실제 artifact에서 확인한다. 원래 인라인 감사에서는 추가로 모든 cache hash·normalizer·audibility·pooling 비교와 redesign의 실제 epoch order를 계산해 JSON으로 보존했다.

<!-- HUMANIZE-SUMMARY: Korean prose final pass applied conservatively; technical identifiers, numerical claims, uncertainty, and evidence boundaries preserved. -->

## 표현 차원과 layer12 강건성 확인

`spectrum_stem` cache의 **448차원**과 E4 보고서의 **384차원**은 서로 모순되지 않는다. `eight_spectral.py:11-12,28-46`은 1–7 kHz의 local-trend residual 384차원과 coarse log-power band 64차원을 이 순서대로 붙인다. `eight_feature_schema.py:81-82`는 전체 448차원을 검증하고, 실제 E4는 `eight_ablation.py:236,248,251` 및 `selection_E4_locked.json.feature`의 `spectrum_stem[:384]`처럼 앞의 residual만 사용한다. 따라서 E4에서 뒤의 64차원 band를 사용했다는 설명은 틀리고, “384차원 residual”이라는 FINAL_REPORT 설명은 맞다. spectral과 MERT는 둘 다 linear head지만 input width가 달라 trainable head parameter는 385 대 1,025개다. 같은 architecture 형식과 같은 parameter 수는 구분해야 한다.

반면 **layer12의 codec·swap 강건성은 이번 eight cache로 확인되지 않았다.** `eight_feature_schema.py:13-22`에서 base music만 L6/L12/L24를 모두 추출하고, processed music와 swap은 L24만 추출한다. `eight_ablation.py:183-192`의 E3 codec 실험은 L24이고, postrun E2 music/file 분석도 `eight_e2_music_file_analysis.py:100,176` 및 `postrun_e2/normalizer_receipt.json.normalizers.music.key`에서 L24다. E6 layer12 선택은 base music의 dev EER를 대상으로 했다. 따라서 E3/E2의 강건성 결과를 최신 L12 head가 직접 통과한 결과로 옮기면 안 된다. 같은 split의 processed/swap audio에서 L12 feature를 별도로 추출하고 선정에 쓰지 않은 자료로 확인하는 것이 다음 검증이다.
