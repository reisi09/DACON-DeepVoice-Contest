[VERIFIED] **이 패킷에서는 독립 최종평가로 인정할 세트를 확인하지 못했습니다.** DB 실험의 E1·E2 점수와 P1 재학습 패키지의 성능을 구분해야 합니다. P1 코드는 기존 test를 학습에 합치며, `s7_new`는 신규 `newtest`도 학습에 사용합니다. (`code/data_v4/train_backbone_heads.py:343-372`)

[VERIFIED] 아래의 VERIFIED는 소스·CSV·해시를 직접 확인했다는 뜻입니다. 기존 실험 수치는 REPORTED로 표시했습니다. [NOT RUN] 모델 추론·재학습·네트워크·GPU 실행은 하지 않았습니다.

인용 경로 약어:

- `N/` = `code/research/top15/work5/newdata/`
- `E/` = `results/research/top15/work5/newdata/`
- `S7/` = `runtime_pkg/DataV4_S7_SSLAMWavLM/model/`
- `NEW/` = `runtime_pkg/DataV4_s7_new/model/`

## 1. 누수·적응적 과적합: 영향이 큰 순서

1. **[VERIFIED] 최종 패키지에 대한 평가 독립성이 소실됩니다.** DB 실험은 신규 train만 학습하지만, P1은 기존 test 전체와 신규 행 전체를 학습합니다. `NEW_VAL=1`일 때만 신규 newval을 조기 종료용으로 남깁니다. 따라서 저장된 DB 개선치는 **레시피 비교 근거**이며, 배포 패키지 자체의 독립 성능 추정치가 아닙니다. (`code/data_v4/train_backbone_heads.py:76-86,349-372`; `NEW/trained/run_identity.json:2-5,42-45`)

2. **[VERIFIED] E1·E2·newval을 이용한 반복 선택이 확인됩니다.** 여러 가중치·데이터·혼합 후보를 E1/E2에서 비교했고, 이후 newval 및 E1/E2 veto로 다시 선택했습니다. [REPORTED] 이전 MERT 층과 AD 가중치 선택에도 test 결과가 영향을 주었다고 기록돼 있습니다. [HYPOTHESIS] 이 과정의 선택 편향은 고정 모델을 재표집하는 현재 bootstrap CI에 반영되지 않습니다. `p_gt_0`는 향후 LB 개선 확률로 해석할 수 없습니다. (`E/virtual_configs_eval.txt:2-18`; `N/select_newval.py:1-3,39-47`; `history/research/top15/experiment-ledger.csv:27`; `code/research/top15/work3/integration/fit_ft_fusion.py:85-111`)

3. **[VERIFIED] old val이 여러 적합 단계에 중복 사용됩니다.** 체크포인트를 val EER/BCE로 고른 뒤 같은 val에서 멤버 CDF와 FILE 가중치를 적합하고 성능까지 계산합니다. [HYPOTHESIS] val 성능에는 선택 낙관성이 포함됩니다. 다만 저장된 CDF를 파일별로 적용하는 것은 평가 파일 사이의 통계를 이용하는 누수와는 구별됩니다. (`code/data_v4/train_backbone_heads.py:239-258,526-554`; `code/data_v4/file_ensemble.py:37-52`; `code/decision_heads.py:78-98`)

4. **[VERIFIED] 신규 데이터에서 실제 그룹 교차가 있습니다.** CSV 전체 검사에서 AT-ADD의 4개 그룹이 train/newtest에 걸쳐 있고, 해당 newtest 행은 10개입니다. split 코드는 eval 풀을 그룹 충돌 확인 없이 newtest로 보냅니다. [REPORTED] OpenBMAT 음악 원본 교차 65행을 더해 총 75행이라는 결과가 있습니다. [UNKNOWN] OpenBMAT 원본 메타데이터가 없어 65행은 독립 재검산하지 못했습니다. (`N/assign_splits.py:24-31`; `data/new_rows.csv:5646,26520`; `E/leak_check.txt:1`; `N/leak_check.py:16-29`)

5. **[UNKNOWN] 데이터셋 간 원본·파생본 누수와 사전학습 중복은 배제할 수 없습니다.** 신규 40,222행의 두 SHA 필드는 모두 비어 있습니다. OpenBMAT 그룹은 첫 번째 음악 트랙만 반영하고, remix 그룹은 보컬 donor만 반영합니다. [VERIFIED] 반면 현재 bank의 기록된 ID·group·component·두 SHA는 split 사이에 교차하지 않았고, 신규 remix 636행의 명시된 donor는 모두 train이었습니다. 이는 기록된 계보에 대한 확인이며 실제 음원의 근접 중복 부재까지 증명하지 않습니다. (`N/atadd_rows.py:52-56`; `N/open_rows.py:42,122-129,191-192`; `N/assign_splits.py:32-38`; `data/bank/splits.report.json:1813-1814`; `data/new_data_summary.csv:9`)

## 2. 평가 타당성

| 세트 | 가능한 해석과 한계 |
|---|---|
| old val | **[VERIFIED]** 조기 종료·CDF·fusion 적합 진단용입니다. 독립 일반화 평가가 아닙니다. 패키지 지표에도 미상 라벨로 인한 부분 평가와 `score: null`이 명시돼 있습니다. (`S7/trained/validation_metrics.json:2-36`; `code/data_v4/train_backbone_heads.py:526-554`) |
| E1, 4,822행 | **[VERIFIED]** 동일 DB 조건의 후보 비교에 사용됩니다. 반복 노출됐으며 P1에는 학습 데이터입니다. **[UNKNOWN]** 고정 ID 목록이 없어 bank test 6,450행에서 빠진 행의 대표성을 재검증하지 못했습니다. (`MANIFEST.md:52`; `code/research/top15/work2/backbones/run.py:46-47,261-263`; `code/data_v4/train_backbone_heads.py:349`) |
| E2, 12,426행 | **[VERIFIED]** AT-ADD는 10,000행, **80.48%**입니다. MANIFEST의 80.9%는 전체 E2 비율과 다릅니다. **[REPORTED]** 연결 행 제거 후 개선도 AT-ADD에서 훨씬 큽니다. 특정 신규 도메인 전이의 근거이지 hidden evaluator 전체의 대표 표본은 아닙니다. (`data/new_data_summary.csv:2-13`; `E/leak_check.txt:3-5`) |
| newval, 3,841행 | **[VERIFIED]** W_newB의 조기 종료에 사용됩니다. s7_new/W_new 비교에서는 선택용으로 사용됐으므로 최종평가 지위를 얻지 못합니다. **[UNKNOWN]** 보고된 선택 표본 2,281행은 전체 3,841행에서 연결 39행을 뺀 수와도 다릅니다. 나머지 제외 내역이 필요합니다. (`N/jobs_nd.txt:18-21`; `E/select_newval.txt:1-2`; `N/new_eval.py:42-48`) |
| E3, 한국어 REAL 257행 | **[REPORTED]** 한국어 REAL의 오탐 진단용입니다. **[VERIFIED]** REAL만으로 EER·ADS나 FAKE 탐지율을 평가할 수 없습니다. E1에 합쳐 계산한 점수도 E3 단독 탐지력의 근거가 아닙니다. (`MANIFEST.md:55`; `E/new_eval_e3.json:303-329`; `S7/runtime/graph_scoring.py:155-157`) |

- **[VERIFIED] 단일 클래스 판정은 성분별로 해야 합니다.** `realV+fakeM`만 모으면 FILE/MUSIC은 모두 양성, VOICE는 모두 음성이므로 각 EER가 정의되지 않습니다. OpenBMAT처럼 FILE/MUSIC에는 양쪽 클래스가 있어도 VOICE 라벨이 없으면 전체 ADS는 계산할 수 없습니다. 고정된 별도 calibration 임계값의 FPR 또는 TPR를 보고해야 합니다. (`history/research/top15/work/lane1/official_text.txt:137-142`; `N/new_eval.py:74-91`; `E/e2_subsets.txt:3-7`)

- **[VERIFIED] CDF fusion은 보정된 정답 확률이 아닙니다.** 멤버 CDF 평균을 해당 후보의 DF head 척도로 역변환합니다. 후보마다 CDF가 달라 raw score 평균이나 동일 숫자 임계값의 차이를 성능 차이로 해석할 수 없습니다. 같은 행의 EER/AUC, 또는 동일 절차로 별도 적합한 임계값의 FPR/TPR가 적절합니다. 포화·clipping으로 생기는 동점도 확인해야 합니다. (`S7/runtime/component_fusion.py:14-28`; `N/new_eval.py:41,53-63,79`)

- **[VERIFIED] Public과 Private은 별개의 최종 holdout이 아닙니다.** 패킷의 공식 문구는 Public이 test 100%, Private이 종료 시 Public이라고 명시합니다. **[REPORTED]** S7/S8 LB 수치는 사용자 전사 기록이며 이번 검토에서 서버와 ZIP의 대응을 확인하지 않았습니다. (`history/research/top15/work/lane1/official_text.txt:144-147`; `MANIFEST.md:29-38`)

## 3. 구현·통합 결과

### 확인된 결함

**A. 과거 AC 평가의 누락 점수 0 대입**

- **[VERIFIED] 근거 → 경로 → 기대 → 관찰:** 라벨별 예측만 읽는 `backbones/ads.py`가 누락 멤버를 `.get(ID, 0.)`으로 채웁니다. FILE 계산에는 실제 런타임처럼 모든 파일의 성분 점수가 필요하지만, 라벨 유무에 따라 점수가 달라집니다. (`code/research/top15/work2/backbones/ads.py:48-74`)
- **[REPORTED] 영향:** 과거 감사에는 FILE EER가 약 0.004 낙관적이었다고 기록돼 있습니다. 이 크기를 현재 후보 전체에 적용할 수는 없습니다. (`history/research/top15/work4/AUDIT.md:12`)
- **[HYPOTHESIS] 수정 → 확인:** every-row 예측과 필수 멤버 검증으로 통일하고, 같은 실제 파일·체크포인트에서 평가 경로와 패키지의 다섯 출력을 비교해야 합니다. **[VERIFIED]** 현재 `new_eval.py`는 직접 ID 조회를 하고, runtime `fuse()`는 가중치가 있는 멤버 누락 시 예외를 냅니다. (`N/new_eval.py:57-63`; `S7/runtime/component_fusion.py:19-25`)

**B. 단일 클래스 bootstrap의 잘못된 `p_gt_0`**

- **[VERIFIED] 근거 → 경로 → 기대 → 관찰:** bootstrap의 `eer()`에는 두 클래스 검사 없이 ROC 계산이 들어갑니다. `safe()`는 예외만 잡아 NaN을 걸러내지 못합니다. 실제 E3 결과에는 `delta: NaN`, `ci95: NaN`, `p_gt_0: 0.0`이 함께 저장돼 있습니다. (`code/research/top15/work3/integration/fit_ft_fusion.py:60-64,100-111`; `N/new_eval.py:95-100`; `E/new_eval_e3.json:116-122`)
- **[HYPOTHESIS] 영향 → 수정 → 확인:** 개선 확률 0처럼 오독될 수 있습니다. 성분별 클래스 충족 여부와 유한성을 검사해 불가 지표를 사유가 있는 `null`로 반환해야 합니다. E3와 all-positive 부분집합에서 확률·CI가 생성되지 않고, 유효한 양·음성 세트의 기존 계산은 유지되는지 확인하면 됩니다.

**C. 불완전한 계보가 split과 가중치 상한에 함께 사용됨**

- **[VERIFIED] 근거 → 경로 → 기대 → 관찰:** OpenBMAT는 첫 음악 트랙, remix는 보컬 donor 기반 그룹을 만들지만 W의 상한은 이 `group_id`를 완전한 계보처럼 사용합니다. 따라서 공유 반주·donor 파생본 전체의 2% 상한을 보장하지 않습니다. (`N/open_rows.py:122-129,191-192`; `code/data_v4/weighting.py:21-22,131-135`)
- **[HYPOTHESIS] 영향 → 수정 → 확인:** 원본 하나의 영향력이 과소 집계될 수 있습니다. 모든 donor·원곡·파생본의 연결요소를 만든 뒤 split과 상한에 같이 사용해야 합니다. 연결요소의 split 교차와 **실제 최종 학습 가중치 합**을 검사해야 합니다.

### 통합 확인과 남은 위험

- **[VERIFIED] SSLAM·WavLM은 S7 실행 경로에 연결돼 있습니다.** SSLAM block 5는 MERT 6+12, EAT block 16 뒤에 붙어 music head로 들어갑니다. WavLM hidden 8은 voice head를 거쳐 `ours/ad/wavlm = 1/0.5/1` fusion에 들어갑니다. “S7 런타임 미구현”이라는 이전 문서는 현재 소스에 적용되지 않습니다. **[NOT RUN]** 가중치·음원이 없어 실제 기여도 ablation은 하지 못했습니다. (`S7/runtime/full_bank_submission.py:97-118,143-156,172-182`; `S7/trained/component_fusion.json:1`)

- **[VERIFIED] 학습·출력의 핵심 연결은 일치합니다.** 학습은 알려진 성분 라벨, weighted BCE, 음악 구간 보조 손실을 사용하고, voice는 window max, music은 log-mean-exp로 집계합니다. 최종 출력은 ID 순서와 유한한 `[0,1]` 값을 검사합니다. CSV 전체에서 `mask=0`인데 값이 남아 있는 셀은 없었습니다. **[UNKNOWN]** 실제 optimizer 실행·checkpoint 내부 train ID는 검증하지 못했습니다. (`code/full_bank_train.py:30-41,77-129`; `S7/runtime/full_bank_submission.py:153-183,244-245`; `S7/runtime/graph_scoring.py:48-77`)

- **[VERIFIED] fp16 stem 보정과 feature 저장 정밀도는 구별해야 합니다.** SSLAM/WavLM 입력 stem은 fp16 왕복을 하지만, 학습 cache에 fp16으로 저장한 embedding을 runtime 기본 멤버가 모두 같은 방식으로 반올림하지는 않습니다. **[REPORTED]** s7_new의 10파일 parity 최대차는 voice `0.00018`, music `0.00089`입니다. **[UNKNOWN]** 이 로그에는 현재 검사 코드가 요구하는 FILE 오차와 표본 수가 출력되지 않아 강화된 검사의 통과 증거로 볼 수 없습니다. 수정 여부는 실제 목표 환경에서 FILE·성분 점수와 순위 변동을 측정해 결정해야 합니다. (`N/dump_features.py:53-65`; `S7/runtime/full_bank_submission.py:148-156,179-180`; `receipts/DataV4_s7_new/runtime_parity.txt:11`; `code/research/top15/work5/w/w_runtime_parity.py:48-52`)

- **[VERIFIED] 채널·구간 처리는 명시적입니다.** 입력은 16 kHz mono로 내려가고 peak 0.6 제한 후 분리됩니다. MERT는 24 kHz, PANNs는 32 kHz로 재표본화합니다. MERT/WavLM은 64,600-sample 반창 간격과 tail, DF는 한 창 간격과 tail을 사용합니다. 짧은 입력은 경로별로 zero-padding·반복·가변 길이가 다릅니다. **[HYPOTHESIS]** stereo 위상 상쇄, 짧거나 끝부분에만 있는 성분, 저음량 음성의 위험은 남습니다. 확인은 해당 실제 음원에서 cache/runtime 출력 및 FPR/FNR를 측정해야 합니다. (`S7/assets/baseline/script.py:144-175,206-217`; `S7/runtime/mert12_submission.py:47-64`; `S7/runtime/full_bank_submission.py:131-159`; `S7/runtime/antideepfake_runtime.py:93-106`)

- **[VERIFIED] 현재 S7에는 시간 초과에 따른 멤버 생략 분기가 없습니다.** 무음 voice는 명시적으로 0이 됩니다. 설정 파일 부재 시 legacy fallback은 있으나, 정상 패키지 진입점은 먼저 manifest 해시를 검사합니다. **[HYPOTHESIS]** `Engine` 직접 호출에서는 필수 설정을 명시적으로 요구하는 편이 안전합니다. 필수 멤버·설정 누락은 실패하고, 무음 분기만 의도대로 유지되는지 확인해야 합니다. (`S7/runtime/full_bank_submission.py:75-83,169-181,220-228`; `code/decision_heads.py:63-66,88-98`)

- **[VERIFIED] s7_new의 runtime 확장 기능은 현재 설정에서 비활성입니다.** S7보다 추가 SSL/music encoder 경로가 있지만 `voice_ssl=[]`, `music_extra=[]`입니다. 주요 변경은 학습 데이터·head·CDF와 FILE 가중치 고정입니다. S8은 mix-SSLAM뿐 아니라 FILE 가중치도 S7의 `0/9/1`에서 `1/1/1`로 달라져 순수 단일 요인 비교가 아닙니다. (`NEW/runtime/full_bank_submission.py:139-169`; `NEW/trained/run_identity.json:35-45,65-70`; `runtime_pkg/DataV4_S8_MixSSLAM/model/trained/file_ensemble.json:1`)

## 4. 개선 선택지

### 신규 수집 없이

- **[HYPOTHESIS] 먼저 평가 결함과 계보를 수정하고, head 학습·조기 종료·CDF/fusion 적합을 그룹 기준으로 분리한 비교를 우선해야 합니다.** 현재 자료를 다시 나누는 것은 개발 검증을 개선하지만 이미 노출된 자료를 독립 최종평가로 되돌리지는 못합니다. (`code/data_v4/train_backbone_heads.py:239-258,526-554`; `N/select_newval.py:39-47`)

- **[REPORTED] W 재가중치만의 채택 근거는 약합니다.** 신규 데이터를 쓰지 않은 W_ref는 S7_ref 대비 E1 `−0.00284`, E2 `−0.00865`입니다. val 개선만으로 채택하면 안 됩니다. W의 고정 음성 strata 비중은 검증 대상 가정입니다. (`E/new_eval_newtest.json:233-248`; `code/data_v4/weighting.py:21,96-108`)

- **[REPORTED] voice=s7_new/music=W_new 혼합은 자체 채택 기준을 충족하지 못했습니다.** newval 이득은 `+0.0013`, CI `[-0.0020,+0.0057]`로, 요구한 `≥+0.002`와 CI 하한 `>0`을 모두 통과하지 못했습니다. E2 결과를 보고 기준을 완화하면 추가 선택 편향이 생깁니다. (`N/select_newval.py:1-3`; `E/select_newval.txt:2`)

- **[HYPOTHESIS] 가능한 작은 ablation은 WavLM에도 검증된 L 라벨을 적용하거나, 멤버 하나를 제거하고 CDF/FILE을 다시 적합하는 것입니다.** WavLM은 원래 라벨을 사용합니다. **[REPORTED]** SSLAM·WavLM 추가는 기존 DB에서 각각 양의 개선을 보여, 제거 자체를 성능 개선으로 가정할 근거는 없습니다. pooling·mask 변경도 재학습 없이 추론에만 적용하면 안 됩니다. (`receipts/DataV4_S7_SSLAMWavLM/submit.json:9`; `results/research/top15/work4/rt_s6enc_report.json:630-652`; `history/research/top15/experiment-ledger.csv:30`)

### 데이터 추가 시

- **[REPORTED] 기존 신규 데이터 추가는 유망하지만 개선이 균일하지 않습니다.** s7_new의 DB 이득은 E1 `+0.00641`, E2 `+0.02851`입니다. 그러나 AT-ADD singing에서는 VOICE EER가 `0.0800→0.0611`로 좋아져도 FILE EER는 `0.1440→0.1509`로 나빠집니다. 성분 개선을 FILE 개선으로 대신 보고하면 안 됩니다. (`E/new_eval_newtest.json:39-47,568-574,647-662`)

- **[HYPOTHESIS] 추가 수집의 우선 후보는 실제 방송 혼합과 성분 진위가 다른 자연스러운 혼합입니다.** s7_new도 OpenBMAT FILE/MUSIC EER가 각각 `0.3357`이며, 신규 remix는 train에만 있습니다. 보컬·반주 진위, 동시/순차 등장, SNR을 함께 기록하고 방송·화자·모든 음악 원본을 분리해야 합니다. 저SNR이 현재 실패 원인이라는 주장은 아직 가설입니다. (`E/new_eval_newtest.json:630-637`; `data/new_data_summary.csv:9-10`; `history/research/top15/work/lane1/official_text.txt:24-25`)

- **[ESTIMATE] 패킷의 잔액 $18.75만 사용하면 H100 약 6.29시간이며, s7_new ZIP 여유는 약 254.7 MB입니다.** 다운로드·QA·다른 비용은 별도입니다. **[NOT RUN]** 새 backbone이나 대규모 재학습의 완료 시간, L4 메모리·60분 제한 충족은 확인하지 않았습니다. 현재 자료는 CPU 10파일 검증입니다. (`MANIFEST.md:60-63`; `receipts/DataV4_s7_new/submit.json:4`; `receipts/DataV4_s7_new/verify.log:3-10`)

## 5. 독립 최종평가에 필요한 증거

**[UNKNOWN] 현재 패킷에는 아래 조건을 모두 입증한 세트가 없습니다.** E1/E2는 반복 선택에 노출됐고 P1에서 소비되며, newval은 선택·조기 종료용, E3는 REAL만 있는 진단 세트입니다. (`MANIFEST.md:50-55`; `code/data_v4/train_backbone_heads.py:349-372`)

**[HYPOTHESIS] 최종평가로 인정하려면** 모델·전처리·threshold·CDF·fusion을 먼저 해시로 고정한 뒤, 개발 중 보지 않은 데이터에서 한 번 평가해야 합니다. 필요한 증거는 다음과 같습니다.

- 원음 및 모든 파생본의 해시·계보, 화자·가수·원곡·remix donor의 학습/적합 세트 교차 부재.
- FILE/VOICE/MUSIC 각각의 유효 양·음성 표본과 충분한 독립 그룹, 성분 존재·진위 라벨 및 미상 비율.
- 공식 유형·길이·채널·형식에 대응하는 구성, 사전 고정된 비교·실패 기준.
- **그 최종 패키지**의 파일별 예측, 공식 계산법의 지표, 그룹별 불확실성, 실제 L4 실행 기록.

근거가 되는 현재 공백: `N/open_rows.py:122-129,191-192`; `S7/runtime/graph_scoring.py:113-165`; `history/research/top15/work/lane1/official_text.txt:16-44`; `receipts/DataV4_s7_new/verify.log:10`.

## 6. 검토 범위

- **[VERIFIED] 전체 소스 흐름 검토:** `code/`의 제공된 Python·split 설정·신규 데이터 jobs/패키징 shell, S7의 inference·decision·fusion·voice head·WindowMIL·전처리 경로. S7/new/S8 Python 파일은 전체 바이트 비교로 차이를 추적했습니다. (`code/full_bank_train.py:1-314`; `code/data_v4/train_backbone_heads.py:1-594`; `S7/runtime/full_bank_submission.py:1-253`)

- **[VERIFIED] 전체 행·구조 검사:** 네 CSV, 신규 요약 CSV, 주요 결과 JSON/TXT. 신규 CSV와 요약은 **40,222행**으로 일치하지만 패키지 기록은 **40,214행**입니다. **[UNKNOWN]** 8행 차이의 정확한 제외 ID·사유는 패킷에 없습니다. 코드에는 feature 없는 행을 제외하는 경로가 있습니다. (`data/new_data_summary.csv:2-13`; `NEW/trained/run_identity.json:42-45`; `code/data_v4/train_backbone_heads.py:61-72`)

- **[VERIFIED] 파일 정체성·산술 확인:** S7/new/S8의 제공된 파일은 각각 40개가 해당 receipt 해시와 일치했습니다. S7/new의 저장된 QA 출력에서 FILE 공식을 재계산한 최대차는 각각 `2.22e-16`, `1.11e-16`입니다. 이는 소스 정체성과 후처리 산술 확인이며 원음 추론 재현은 아닙니다. (`receipts/DataV4_S7_SSLAMWavLM/submit.json:35-112`; `receipts/DataV4_s7_new/submit.json:35-112`; `code/decision_heads.py:70-98`)

- **[VERIFIED] 부분 검토:** history의 공식 규정 발췌·실험/claim ledger·이전 감사, 기타 receipt 메타데이터, bank 분석 JSON. **[NOT RUN] 미검증:** vendored MERT/EAT/DF 내부 전체 의미 검토, 실제 모델 가중치·음원·feature cache, ZIP 전체, 설치 및 L4 실행. 패킷에는 `full_bank_batches.py`, `full_bank_evaluate.py`, `s8_eval.py`, 고정 평가 ID 목록, FakeMusicCaps 보컬 caption 목록도 없어 완전한 재현 감사는 불가능합니다. (`code/full_bank_train.py:21-26`; `N/new_eval.py:19-23`; `code/research/top15/work2/backbones/run.py:47,140`; `code/data_v4/label_rules.py:13`)