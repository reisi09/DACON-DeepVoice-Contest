# 데이터·라벨·분할 정밀 감사

감사일은 2026-09-18이다. 최신 eight H100, MERT12 full1322, six H100, voice/music pilot과 historical v4를 조사했다. 이번 감사에서는 모델을 학습하거나 클라우드를 시작하지 않았다. CSV를 직접 다시 집계하고 실제 오디오 파일의 SHA-256·sample rate·channel·duration을 확인했으며, E6 checkpoint에 저장된 sampling hash를 재현했다. 숫자는 아래 증거 JSON과 원본 manifest에서 확인할 수 있다.

- `/Users/choiwuseck/Documents/ChatGPT/데이콘/results/precision_audit_20260918/data_manifest_measurements.json`
- `/Users/choiwuseck/Documents/ChatGPT/데이콘/results/precision_audit_20260918/data_sampler_measurements.json`
- `/Users/choiwuseck/Documents/ChatGPT/데이콘/results/precision_audit_20260918/data_sampler_checkpoint_replay.json`
- `/Users/choiwuseck/Documents/ChatGPT/데이콘/results/precision_audit_20260918/data_audio_integrity.json`

여기서 P1은 학습 비교나 일반화 판단을 바꿀 수 있는 문제, P2는 특정 집계·해석·추적 범위의 문제다. CONFIRMED는 이번 측정 또는 직접 읽은 코드·실행 산출물로 확인한 사실이다. HYPOTHESIS는 아직 모델의 실제 동작이나 정답 오류를 확인하지 못한 가능성이다. 데이터셋의 한계와 프로그램 결함은 구분했다.

## 핵심 판정

최신 데이터의 기록된 부모·artist·speaker·source hash가 split을 넘는 사례는 없었다. unknown 라벨을 real로 바꾸는 문제도 최신 selector에서 발견하지 못했다. 그러나 E6의 실제 학습에서는 음악 train 831개 중 FMA 42개가 한 번도 선택되지 않았다. 또한 최신 세 데이터셋 모두 `VOICE_PRESENT=0`인 확정 라벨이 없어 Voice presence AUC를 계산할 수 없다. 음악 출처와 정답은 완전히 결합돼 있으며, 수천 개의 변환 파일이 늘어나도 test의 독립 화자는 eight 2명, 두 pilot 각각 1명이다.

MERT12 full1322는 1,322개 전부를 최소 한 번 노출하도록 수정했지만 source sampler의 순회 결함은 남아 있다. REAL 109개는 update 0–13에 각각 한 번만 쓰였고 이후 486회 update에는 등장하지 않았다. 이 모델은 former train·calibration·dev·test를 모두 사용했으므로, 같은 자료에서 새 모델의 독립 검증 성능을 얻을 수는 없다.

## 데이터 단위와 실제 규모

| 코호트 | manifest 행 | base 원본 뷰 | base 음악 | base 음성 | base의 연결 그룹 | 모든 파생 뷰를 연결한 그룹 |
|---|---:|---:|---:|---:|---:|---:|
| eight | 7,101 | 1,834 | 1,322 | 512 | 306 | 256 |
| six | 839 | 136 | 104 | 32 | 108 | 35 |
| voice/music | 568 | 134 | 102 | 32 | 106 | 93 |

연결 그룹은 `group_id`의 원자와 파생 부모를 연결해 계산한 provenance 단위다. 실제 통계적 effective sample size를 추정한 값은 아니다. 알려지지 않은 동일 곡·artist alias·유사 음원 관계가 있으면 독립성은 더 낮아질 수 있다. 반대로 사용하지 않는 합성 뷰까지 연결하면 특정 core-only 추정에 필요한 그룹보다 보수적으로 합쳐진다. 분석 대상과 bootstrap 단위를 함께 밝혀야 한다.

eight의 base 음악 1,322개는 real 218개, fake 1,104개이며 base 음성 512개는 real 256개, fake 256개다. 음악만 떼어 본 기록된 연결 그룹은 296개, 화자 그룹은 10개다. 그 296개를 특정 목표 숫자와 비교하는 것만으로 충분성이나 부족함을 판정할 수 없다.

| split | eight base 음악 / 음악 그룹 | eight 음성 / 화자 | six base 음악 / 화자 | voice/music base 음악 / 화자 |
|---|---:|---:|---:|---:|
| train | 831 / 187 | 212 / 3 | 62 / 2 | 52 / 1 |
| calibration | 200 / 42 | 204 / 3 | 해당 없음 | 21 / 1 |
| dev | 164 / 35 | 48 / 2 | 21 / 1 | 16 / 1 |
| test | 127 / 32 | 48 / 2 | 21 / 1 | 13 / 1 |

eight 전체 그래프에서 test 음악이 속한 그룹은 24개다. E2 혼합이 여러 음악 부모와 화자를 연결하므로 core만으로 계산한 32개와 다르다. 같은 이유로 six test 전체는 6개, voice/music test 전체는 11개 연결 그룹이다. `connected_components=306`인 source receipt와 실제 분석의 256개는 단위가 달라서 생긴 차이다.

근거: `/Users/choiwuseck/Documents/ChatGPT/데이콘/_workspace/eight_h100_20260917/dataset/manifest.csv:1`, `/Users/choiwuseck/Documents/ChatGPT/데이콘/results/six_h100_20260917/probes.csv:1`, `/Users/choiwuseck/Documents/ChatGPT/데이콘/results/voice_music_h100_20260917/probes.csv:1`; 이번 `data_manifest_measurements.json`의 각 코호트 `splits`.

## D01. E6의 실제 sampler가 real FMA 42개를 영구적으로 건너뛴다

**P1 · CONFIRMED measured/checkpoint · source sampler 순회 결함은 full1322에도 남아 있음; 최소 1회 노출만 수정됨.**

현재 `source_generator_parent` sampler는 source 선택과 parent 선택에 같은 전역 `ordinal`을 사용한다. source를 `ordinal % len(sources)`로 선택한 뒤 parent도 `ordinal % len(parents)`로 선택하므로, 특정 source에 도달하는 ordinal의 나머지가 일부 parent에만 대응한다. 오래 학습하거나 seed를 바꾸어도 그 부분집합이 바뀌지 않는 경우가 생긴다.

실제 E6 MERT layer12의 seed 41·42·43 checkpoint에서 `steps`, `batch_size`, `seed`, `mode`를 읽고 순서를 다시 만들었다. 세 seed 모두 저장된 `draw_sha256`과 정확히 일치했다. 500 step × batch 32에서 831개 중 789개만 노출됐으며, 빠진 42개는 모두 FMA다. 같은 자료에 parent sampler를 적용하면 831개가 전부 노출된다. 따라서 source balance 실험은 가중치만 바꾸는 비교가 아니었다. 일부 real 자료를 제거하는 효과까지 섞였다. 이것이 성능 차이의 원인인지는 별도 비교가 필요하다.

근거: `/Users/choiwuseck/Documents/ChatGPT/데이콘/eight_heads.py:49`, `:59`, `:178`; `/Users/choiwuseck/Documents/ChatGPT/데이콘/eight_ablation.py:77`; `/Users/choiwuseck/Documents/ChatGPT/데이콘/results/eight_h100_20260917/downloaded/unpacked/output/checkpoints/E6_mert_l12_seed41.pt`의 `draw_sha256`; 이번 `data_sampler_checkpoint_replay.json`의 `exact_replay=true`, `unique_exposed=789`.

full1322에서 최소 노출만 고친 뒤 남은 순회 결함과 학습 fit 격차는 D06 및 `training_sampling.json`·`training_feature_replay.json`에 기록했다. 앞선 E6의 42개 누락과 최신 refit의 prefix-only REAL 109개는 서로 다른 학습 모집단의 측정이다.

가장 작은 다음 검증은 source별 독립 순번 또는 독립 RNG로 parent를 선택한 schedule에서 모든 eligible ID의 노출 횟수와 원래 class/source 목표 비중을 확인한 뒤, 동일한 frozen feature·step·seed로 head 비교만 다시 하는 것이다. 새 encoder나 ViT를 도입하기 전에 해결할 수 있는 비교 조건 문제다.

## D02. Voice presence AUC를 계산할 음성 부재 라벨이 없다

**P1 · CONFIRMED measured · 데이터의 평가 범위 제한.**

| 코호트 | `VOICE_PRESENT=1` | `VOICE_PRESENT=0` | unknown |
|---|---:|---:|---:|
| eight 전체 | 2,644 | 0 | 4,457 |
| six 전체 | 192 | 0 | 647 |
| voice/music 전체 | 160 | 0 | 408 |
| v4 dev | 120 | 0 | 90 |
| v4 sealed | 120 | 0 | 90 |

이 표는 예측값과 무관한 라벨 적격성 측정이다. 어떤 모델도 이 데이터에서 Voice presence AUC를 식별할 수 없다. 최신 자료의 Music EER 또는 File EER가 좋아져도 공식 전체 목적의 개선을 확인했다고 할 수 없는 이유 중 하나다. v4의 실제 `official_metrics` 경로에서도 두 평가 split의 `voice_presence_auc`가 invalid이고 `score=None`임을 확인했다. 상수 예측을 넣은 이 실행은 라벨 적격성 확인용이며 성능 검증이 아니다.

음악에 보컬이 없다고 추측해서 0을 채우면 이 결함을 해결한 것이 아니다. 현재 빈칸 보존은 옳다. 필요한 것은 검증된 voice-absent 자료와 verified singing·speech·혼합 자료를 평가 목적에 맞게 구분하는 일이다. 먼저 실제 사용할 최소 평가 패널의 voice presence를 독립 청취로 확정하고, 라벨러 간 불일치를 남기는 검증이 결정적이다. 파일 수 목표보다 양·음성 클래스의 구성과 화자·곡의 독립성이 먼저다.

근거: 각 원본 manifest의 `VOICE_PRESENT` 열; `/Users/choiwuseck/Documents/ChatGPT/데이콘/prepare_fresh_fma.py:108`, `/Users/choiwuseck/Documents/ChatGPT/데이콘/prepare_fresh_sonics.py:46`; `/Users/choiwuseck/Documents/ChatGPT/데이콘/graph_scoring.py:113`, `:135`, `:155`.

## D03. 음악의 source만 알아도 정답을 모두 맞힌다

**P1 · CONFIRMED measured confounding / HYPOTHESIS model shortcut.**

eight·six·voice/music의 base 음악 각각에서 source별 majority label 규칙의 일치율은 100%였다. six는 FMA 64개가 전부 real, suno 20개와 udio 20개가 전부 fake다. voice/music은 FMA 64개 real, suno 19개·udio 19개 fake다. eight에서도 `FMA`, `FMA/FMA`는 real이고 Echoes의 source 세부 종류와 suno·udio는 fake다. v4 순수 음악도 train에서 FMA 70개·NSynth acoustic 100개가 real, Echoes 946개가 fake다.

생성기의 출력이 fake인 것은 라벨 정의상 자연스럽다. 문제는 수집 코퍼스·원본 codec·처리 경로·길이·배경 특성이 source와 함께 바뀌어도 이 실험만으로 분리하기 어렵다는 점이다. 모델이 그 특징을 실제 사용한다고 확인한 것은 아니다. source/parent sampling만 균등하게 해도 class 내 노출 불균형은 줄지만 corpus와 label의 결합 자체는 남는다.

가장 작은 다음 검증은 현재 score를 source·원본 codec·artist·generator별로 나누고, matched real reference와 생성본에 같은 후처리를 적용한 비교에서 성능 차이가 유지되는지 확인하는 것이다. 그 다음 코퍼스와 생성기가 모두 바뀐 양 클래스 패널을 한 번만 평가하면 된다. 단순히 기존 FMA와 SONICS를 같은 비율로 더 모으는 작업은 이 식별 문제를 해결하지 못한다.

근거: 이번 `data_manifest_measurements.json`의 `base_music_source_labels`, `base_music_source_majority_rule_accuracy`; `/Users/choiwuseck/Documents/ChatGPT/데이콘/prepare_fresh_fma.py:108`; `/Users/choiwuseck/Documents/ChatGPT/데이콘/prepare_fresh_sonics.py:97`; `/Users/choiwuseck/Documents/ChatGPT/데이콘/eight_data.py:71`.

## D04. 화자와 파생 뷰 수를 구분해야 한다

**P1 · CONFIRMED measured · 일반화 범위 제한.**

eight test의 base 음성은 48개지만 화자는 2명이다. six와 voice/music test는 각각 8개 음성, 1명 화자다. eight의 partial voice 648개는 9개 화자 그룹에서 길이·위치·삽입 길이를 교차해 만든 데이터다. 새 화자 648명이 아니다. codec 변환과 배경 교체도 같은 부모의 반복 측정이다.

이 자료는 동일 화자에서 압축·전화 대역·짧은 fake event를 바꿨을 때의 원인 분리에 유용하다. 새 화자 집단이나 singing 전체에 대한 좁은 confidence interval을 뒷받침하지는 않는다. 최신 E7은 두 test 화자 때문에 population CI를 주장하지 않는다고 명시하므로, 이 제한을 숨기는 구현 결함으로 분류하지 않았다.

다음 검증은 seed 평균만 늘리기보다 화자별 paired effect가 서로 일치하는지 보고, 다음 추가 화자에서 결론이 뒤집히는지 확인하는 것이다. 필요한 추가 표본 수는 허용할 효과 차이와 화자 간 분산, 목표 CI 폭에 맞춰 정해야 한다. 현재 파일에 있는 300-group 조건을 보편적인 통계 기준으로 사용할 근거는 없다.

근거: `/Users/choiwuseck/Documents/ChatGPT/데이콘/eight_audio.py:126`, `:143`, `:148`; `/Users/choiwuseck/Documents/ChatGPT/데이콘/prepare_voice_music_h100.py:58`; `/Users/choiwuseck/Documents/ChatGPT/데이콘/eight_lora_train.py:239`, `:258`; 이번 `data_manifest_measurements.json`의 `splits.*.speech_speakers`.

## D05. provenance split 통과는 unseen-generator 통과를 뜻하지 않는다

**P1 · CONFIRMED measured · 보고 범위 제한.**

최신 eight split은 reference·artist·speaker를 전이적으로 묶지만 generator 자체를 그룹으로 강제하지 않는다. test에 있는 16개 비어 있지 않은 generator/provider 값은 모두 train에도 있다. 이 값에는 real의 `original`도 포함된다. 따라서 기본 test는 알려진 generator에서 새 부모·화자·곡을 평가하는 자료다. 완전히 새로운 generator에 대한 검증으로 해석하면 안 된다.

E1에는 generator를 제외해 다시 학습하는 별도의 sensitivity 경로가 있다. 그 경로의 존재와 기본 split의 범위는 구분해야 한다. generator가 train에서 빠져도 dev에 양 클래스가 충분히 남는지, 제외한 generator의 후처리 버전이나 같은 생성 서비스의 alias가 다른 이름으로 남지 않는지를 함께 확인해야 한다.

근거: `/Users/choiwuseck/Documents/ChatGPT/데이콘/eight_data.py:52`, `:71`, `:94`; `/Users/choiwuseck/Documents/ChatGPT/데이콘/eight_ablation.py:133`; 이번 `data_manifest_measurements.json`의 `eight.test_generators_also_train`.

가장 작은 다음 검증은 주장하려는 generator 한 개를 정하고, 그 generator와 alias·variant 전체를 학습/선택에서 제외한 ID 목록을 고정한 뒤 같은 head 조건으로 평가하는 것이다. 기본 split의 이름을 바꾸는 것으로 대체할 수 없다.

## D06. full1322는 모든 ID를 학습했지만 독립 holdout은 남기지 않았다

**P1 · CONFIRMED measured · 의도된 refit의 평가 한계.**

MERT12 training manifest는 최신 eight의 eligible core/base 음악 1,322개와 ID·MUSIC_FAKE·audio hash가 정확히 일치했다. 업로드된 `full_data_manifest.csv`의 row dictionary도 해당 1,322개를 원래 순서대로 필터한 결과와 동일했다. 전체 7,101행 CSV와 SHA-256이 다른 것은 행을 필터했기 때문이며 integrity 오류가 아니다.

full1322에 포함된 former split은 train 831, calibration 200, dev 164, test 127개다. seed 41·42·43 각각 16,000번의 draw에서 1,322개를 모두 노출했고, ID당 노출 횟수는 최소 1회·최대 88회다. deterministic coverage prefix가 고친 것은 원래 schedule에서 누락되던 109개의 최소 1회 노출이다. source sampler의 순회 결함은 그대로여서 세 seed 모두 이 109개 REAL은 zero-based update 0–13에 각각 한 번만 등장하고 update 14–499의 486회에는 등장하지 않았다. 이 재현은 실제 receipt의 ID별 노출 횟수와 일치한다.

저장된 ensemble을 학습 feature에 다시 적용했을 때 threshold 0.5의 오분류 42개는 모두 이 109개 REAL에 속했다. 반복 노출 차이와 함께 관찰된 학습 fit 격차이며, sampler가 공식 점수 하락의 원인임을 입증한 결과는 아니다. 0.5는 공식 EER threshold가 아니고 독립 품질도 여전히 미확인이다. 근거와 재현 범위는 `/Users/choiwuseck/Documents/ChatGPT/데이콘/results/precision_audit_20260918/training_sampling.json`의 `latest.*`, `/Users/choiwuseck/Documents/ChatGPT/데이콘/results/precision_audit_20260918/training_feature_replay.json`의 `in_sample_diagnostics`, `/Users/choiwuseck/Documents/ChatGPT/데이콘/results/precision_audit_20260918/sections/training.md`의 T1을 함께 참조한다.

이 모델의 같은 former test에 대한 성능은 training-data 성능이다. eight의 전체 코호트도 prior project work에 이미 노출됐다는 data receipt가 있다. 새 split을 만들었다고 이전 선택 과정의 노출이 사라지지는 않는다. `historical=false`인 238개 source record는 canonical cohort가 아니라 fresh acquisition에서 왔다는 코드 구분이지, 프로젝트 전체에서 전혀 보지 않은 자료라는 보증이 아니다.

근거: `/Users/choiwuseck/Documents/ChatGPT/데이콘/mert12_full_refit.py:138`, `:229`, `:238`; `/Users/choiwuseck/Documents/ChatGPT/데이콘/results/mert12_a100_20260918/downloaded/unpacked/output/mert12_full/full_refit_receipt.json`의 `all_former_splits_are_training`, `checkpoints`; `/Users/choiwuseck/Documents/ChatGPT/데이콘/results/eight_h100_20260917/data_receipt.json`의 `historical_exposure`; `/Users/choiwuseck/Documents/ChatGPT/데이콘/eight_data.py:59`, `:82`.

다음 검증은 기존 자료를 재사용한 진단과 새 자료에서의 확인을 나누는 것이다. encoder·layer·pooling·threshold를 먼저 고정하고 이전 수집·평가·선택에 사용하지 않은 부모 단위 패널에서 평가해야 한다. 전 모델의 사전학습 자료와 겹치지 않았다는 사실까지 현재 증거로 확인한 것은 아니다.

## D07. full-data는 전체 곡의 모든 구간을 학습했다는 뜻이 아니다

**P1 · CONFIRMED code/artifact; 실제 목표 분포 영향은 HYPOTHESIS.**

eight의 `core()`는 각 입력의 중앙 최대 8초를 한 번 자르고 peak만 제한한다. MERT12 full1322는 이 core/base 음악 뷰만 선택하며, 1,322개 모두 feature manifest에 valid window 3개가 기록돼 있다. 원본 파일의 앞·뒤 전 구간이나 자연 발생 partial fake 구간을 모두 노출한 refit이 아니다.

eight E5는 speech를 이어붙인 real bed에 0.25·0.5·1·2초짜리 speech event를 넣은 자료다. 길이는 12·30·60초이고 event 위치는 시작·중간·끝이다. 이것은 짧은 사건에 대한 pooling 민감도를 점검하기에 적합하지만, 편집된 실제 노래·가창 합성·부분 악기 교체·오래 이어지는 음악의 모든 유형을 대표하지는 않는다. six에는 별도로 3초 partial-music construction이 있으나 실제 edited-song 라벨 검증을 대체하지 못한다.

가장 작은 다음 검증은 같은 원본에서 중앙 외의 고정 구간을 별도 뷰로 추출해 prediction 변화와 오류 위치를 보고, 검증된 partial-music 사례에서 full-file 집계가 사건을 놓치는지 측정하는 것이다. 이 결과를 본 뒤 random crop 학습, MIL, temporal head 또는 새 ViT 계열 모델의 필요성을 판단할 수 있다.

근거: `/Users/choiwuseck/Documents/ChatGPT/데이콘/eight_audio.py:50`, `:143`; `/Users/choiwuseck/Documents/ChatGPT/데이콘/mert12_full_refit.py:48`, `:87`; `/Users/choiwuseck/Documents/ChatGPT/데이콘/prepare_six_h100.py:175`; 이번 `data_manifest_measurements.json`의 `mert12.windows`.

## D08. v4 splice에는 의도하지 않은 중복이 있고 six에는 의도된 같은 뷰가 있다

**P2 · CONFIRMED actual bytes · v4의 historical 결함; 최신 eight에는 재현되지 않음.**

v4 dev와 sealed는 각각 210행이지만 고유 PCM은 190개다. 각 split의 splice 40행 중 20쌍이 ID만 다르고 WAV byte가 정확히 같다. 10개 music group을 20회 순환하면서 같은 부모를 다시 선택하고 고정 crop key를 재사용한 것이 원인이다. splice 내부 지표는 동일 표본이 균등하게 두 번 들어가므로 그대로일 수 있지만, 전체 집계에서는 splice가 고유 파일 기준보다 크게 가중된다. 이를 새로운 source 증거로 셀 수 없다.

근거: `/Users/choiwuseck/Documents/ChatGPT/데이콘/prepare_v4_data.py:290`; `/Users/choiwuseck/Documents/ChatGPT/데이콘/v4_data_augmentation.py:34`; 중복 예시는 `/Users/choiwuseck/Documents/ChatGPT/데이콘/v4_validation/dev.csv:2`와 `:39`, `/Users/choiwuseck/Documents/ChatGPT/데이콘/v4_validation/sealed.csv:5`와 `:77`이다.

six의 839행은 735개 고유 audio hash를 가진다. 104쌍은 모두 `15s`와 `base`가 같은 구간을 의도적으로 공유한 경우다. `/Users/choiwuseck/Documents/ChatGPT/데이콘/prepare_six_h100.py:138`와 `:143`이 이를 만든다. 길이 실험과 codec 기준 뷰의 짝이므로 각각의 비교 안에서 쓰는 것은 오류가 아니다. 다만 둘을 독립 표본처럼 합쳐 confidence interval을 줄이거나 데이터 크기를 주장하면 중복 가중된다.

최신 eight 7,101행과 voice/music 568행에는 실제 hash 중복이 없었다. v4를 재사용할 때는 중복을 제외한 전체 집계와 기존 집계를 비교하는 것이 가장 작은 결정적 검증이다. six는 실험별 variant 필터와 parent 단위 집계가 중복 가중을 막는지 확인하면 된다.

## D09. exact hash 검증을 넘어선 near-duplicate·alias 검증은 확보되지 않았다

**P2 · CONFIRMED coverage limit / HYPOTHESIS latent overlap.**

최신 세 코호트의 실제 오디오 8,508개를 다시 읽어 manifest SHA-256·16 kHz·mono·duration 일치를 확인했다. 불일치는 0건이었다. 모든 split 쌍에서 기록된 atomic group·parent ID·audio hash 교집합도 0이었다. 이 범위에서는 누출을 발견하지 못했다.

이 검증은 시간 이동·부분 겹침·재인코딩·EQ·artist alias·같은 노래의 다른 upload·prompt 재사용까지 배제하지 않는다. FMA importer는 artist ID를 사용하고 SONICS는 song ID만 사용한다. SONICS의 artist·prompt·musical-content 관계는 현재 데이터에 없다. 동일 byte가 없다는 사실을 음향적 독립성으로 확대하면 안 된다. 실제 near-duplicate가 있다는 증거도 이번에는 얻지 않았다.

가장 작은 다음 검증은 train과 평가 사이의 의심 pair만 retrieval로 좁힌 뒤, 시간 정렬된 음향 fingerprint와 청취로 확인하는 것이다. embedding cosine만 높다는 이유로 같은 label의 서로 다른 곡을 제거해서는 안 된다. 확정 pair만 provenance atom으로 추가하고 split을 다시 계산해야 한다. 사전학습 overlap은 접근 가능한 model card와 training inventory 범위 밖이면 unknown으로 남긴다.

근거: 이번 `data_audio_integrity.json`, `data_manifest_measurements.json`의 `cross_split_*`; `/Users/choiwuseck/Documents/ChatGPT/데이콘/prepare_fresh_fma.py:129`, `:130`, `:131`; `/Users/choiwuseck/Documents/ChatGPT/데이콘/prepare_fresh_sonics.py:48`; `/Users/choiwuseck/Documents/ChatGPT/데이콘/eight_data.py:79`.

## 통과한 라벨·분할·결합 검사

최신 eight의 모든 manifest ID는 유일하다. byte hash를 다시 계산한 결과도 7,101개 모두 유일했다. source·파생 parent의 기록된 lineage를 split별로 다시 비교했을 때 충돌은 없었다. `eight_contract`는 모든 parent의 provenance atom이 child에 포함되는지 검사하고, 동일 atom이 두 split에 들어가면 중단한다. root grouping도 reference·artist·speaker를 union-find로 전이적으로 연결한다. 직접 부모만 비교해서 간접 공유 부모를 놓치는 단순 분할은 아니다.

라벨은 `0`, `1`, 빈 문자열로 구분된다. 최신 music selector는 `kind`가 mixed인지 music인지에 의존하지 않고 `MUSIC_PRESENT=1`, known `MUSIC_FAKE`를 기준으로 선택한다. 따라서 FMA/SONICS의 `kind=mixed` 때문에 음악 학습에서 빠지는 과거 유형의 selector 문제는 현재 경로에서 확인되지 않았다. 음성 selector는 known `VOICE_FAKE`만 사용한다.

FMA·SONICS 음악의 `VOICE_FAKE`와 `VOICE_PRESENT`는 알 수 없으면 빈칸으로 유지된다. real speech를 음악과 섞어도 음악 속 미확인 보컬 때문에 전체 voice authenticity를 real로 확정하지 않는다. fake speech가 들어간 혼합만 `VOICE_FAKE=1`로 정한다. eight E2의 noise 배경은 예외적으로 음성 기여자를 정확히 알 수 있어 real voice label을 유지한다. 이 보수적 mask 규칙에서 의미가 뒤집힌 사례는 찾지 못했다.

MERT12 feature loader는 ID·audio hash·고정 모델 asset hash를 확인한 뒤 valid window mean을 만든다. 이번에는 저장된 training manifest를 source manifest와 직접 결합해 ID·label·audio hash 일치를 확인했다. feature NPZ 전체 수치와 추론 parity는 다른 감사 범위이며 이 결과만으로 품질을 주장하지 않는다.

근거: `/Users/choiwuseck/Documents/ChatGPT/데이콘/eight_contract.py:124`, `:128`; `/Users/choiwuseck/Documents/ChatGPT/데이콘/eight_data.py:94`, `:214`; `/Users/choiwuseck/Documents/ChatGPT/데이콘/eight_ablation.py:24`; `/Users/choiwuseck/Documents/ChatGPT/데이콘/eight_lora_runtime.py:92`; `/Users/choiwuseck/Documents/ChatGPT/데이콘/mert12_full_refit.py:76`.

v4도 manifest/provenance 1,936행의 일치, split 간 atomic parent·artist·speaker·content ID·기록된 원본/PCM hash 중복 없음, 파생 240행의 부모 split·group 합집합 일치를 확인했다. v4 전체 평가 그래프는 splice가 모든 artist를 연결해 하나의 component가 되지만, 실제 V4 bootstrap은 music_seen/music_ood 각각의 10개 artist component만 사용한다. 전체 그래프가 1개라는 이유만으로 해당 bootstrap 구현을 오류라고 판단하지 않았다.

## 확인하지 못한 것과 다음 결정

source annotation이 실제 clip의 의미와 일치하는지는 독립 청취로 전수 검증하지 않았다. 학습 데이터의 fake generator label이 틀렸다는 확정 사례, 음악 속 singing의 구간별 진위, 숨은 artist alias, 사전학습과 평가 음원의 overlap, 실제 비공개 평가의 장르·길이·언어·편집 분포도 현재 자료로 확정하지 못했다. v4는 train 영어/평가 독일어와 화자 변경이 함께 일어나므로 언어 변화와 speaker 변화의 효과를 분리할 수 없다.

우선순위는 이미 확보한 feature로 D01의 sampler 차이를 제거하는 것, D02의 빠진 평가 라벨을 검증하는 것, D03/D05의 source와 generator 변화에 대한 별도 확인 패널을 고정하는 것이다. ViT·새 오디오 encoder·가창 특화 모델을 비교하더라도 이 세 조건과 부모 단위 평가를 공통으로 사용해야 구조 차이의 효과를 판단할 수 있다. 데이터를 무조건 늘리거나 특정 파일 수를 채우는 대신, 현재 결론을 뒤집을 수 있는 검증 단위를 추가하는 편이 결정 가치가 높다.

<!-- HUMANIZE-SUMMARY: Korean final editing pass applied conservatively; numerical values, file paths, identifiers, evidence status, and uncertainty boundaries preserved. No unsupported causality or performance claim added. -->
