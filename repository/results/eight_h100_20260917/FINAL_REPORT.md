# H100 8개 실험 최종 분석 보고서

## 1. 결론과 권고

이번 실행에서 후속 검증으로 가져갈 가장 유력한 후보는 **E6의 MERT 12층 표현**이다. 음악 test 127개(REAL 23, FAKE 104, 연결 성분 24)에서 3개 seed 평균 EER가 **8.995%**로, MERT 24층의 **17.349%**보다 **8.354%p 낮았다**. seed별 EER도 12층이 `9.636%, 8.675%, 8.675%`, 24층이 모두 `17.349%`여서 방향이 일관됐다. 고정된 test 예측을 연결 성분 단위로 1,000회 재표집한 사후 분석에서는 `MERT12 − MERT24` EER 차이의 95% 구간이 seed 41에서 `[-19.009%p, 0.000%p]`, seed 42에서 `[-19.835%p, -0.641%p]`, seed 43에서 `[-20.241%p, -1.085%p]`였다. 두 seed에서는 0을 제외했고 한 seed에서는 상한이 0에 닿았다. 따라서 **MERT12는 후속 검증의 우선 후보**로 삼을 근거가 있지만, 이 구간은 학습 불확실성 전체가 아니라 고정된 세 head와 현재 split에 조건부인 근거다.

나머지 변경은 채택할 근거가 부족했다. E1의 균형 sampler는 기본 parent-uniform sampler보다 나아지지 않았고, E3의 consistency loss는 seed 평균에서만 소폭 개선됐으며 ensemble EER는 오히려 소폭 나빠졌다. E4의 spectral branch 단독 성능은 MERT보다 낮았고, fusion은 seed에 따라 방향이 섞였다. E5의 짧은 가짜 구간 검출은 recall이 낮았고, E7은 모든 방식이 dev/test EER 0이라는 ceiling에 걸려 LoRA의 추가 가치를 판단할 수 없었다. E8의 REAL-only 모델도 test EER 30.602%로 경쟁력이 없었다.

여기에 제시한 결과는 **공식 DACON 점수, 제출 결과, L4 환경 검증, 새로운 완전 독립 holdout 성능**이 아니다. Echoes 계열과 FMA/SONICS를 포함한 기록 데이터에서 얻은 로컬 진단이며, 특히 FMA REAL과 SONICS FAKE 비교에는 source와 label이 얽혀 있다. 일부 자료는 과거 실험에 노출된 이력이 있고, 전체 provenance 연결 성분도 256개라 사전에 둔 독립 source 300개 이상 조건을 충족하지 못했다. 음성 label은 별도의 사람 검수를 거치지 않았고 singing 유효성도 검증하지 않았다. 음성 test는 화자 2명뿐이어서 EER 0도 모집단 일반화 근거로 쓸 수 없다.

## 2. 데이터, 평가 규칙, 감사 상태

전체 manifest는 7,101행이며 core 6,293행, swap 160행, partial 648행이다. split은 train 3,321, calibration 1,880, dev 1,024, test 876행이다. 모든 `group_id`와 `parent_ids`를 합친 뒤 전이적 연결 관계를 계산하면 전체 provenance 연결 성분은 **256개**다. 음악 base test의 주된 평가 단위는 127개 파일(REAL 23, FAKE 104, 연결 성분 24)이고, 음성 base test는 48개 파일(REAL 24, FAKE 24)이지만 연결 성분은 1개이며 실제 화자는 2명이다.

EER는 저장된 파일별 score에 공식 ROC 계산 규칙을 적용한 값이다. 학습 head가 있는 E1·E3·E4·E6은 seed 41·42·43을 따로 평가했다. 이 보고서에서 **seed 평균 EER**는 세 모델의 EER를 산술평균한 값이고, **ensemble EER**는 세 score를 먼저 평균한 뒤 다시 계산한 값이다. 두 값은 같은 통계량이 아니므로 따로 적었다. 신뢰구간은 파일이나 window가 아니라 provenance 연결 성분 전체를 재표집했으며, 연결 성분이 5개 미만인 slice에는 모집단 신뢰구간을 제시하지 않았다.

독립 QA에서는 저장된 prediction의 주 평가 EER 77건을 같은 ROC 규칙으로 다시 계산했고 보고값과의 최대 차이는 0이었다. 7,314개 archive member의 hash가 모두 일치했고, 실행 notebook에는 오류가 없었으며, 7,101개 feature 배열은 모두 유한값이었다. E7 checkpoint 9개도 모두 유한값이었고, LoRA CE/CE+KL checkpoint에서는 각 96개 LoRA-B tensor가 실제로 0이 아닌 값으로 바뀐 것을 확인했다. 이 검증의 판정은 **수치 및 실행 산출물 통과, 봉인된 집계 보고 필드 2건은 차단**이었다. 두 필드의 수정 내용은 아래 정정본에 반영됐으며, 원 archive는 그대로 보존했다.

정본 메타데이터는 [`corrected_summary/final_aggregation.corrected.json`](corrected_summary/final_aggregation.corrected.json)이다. 봉인된 GPU archive 안의 `output/final_aggregation.json`은 전체 연결 성분을 178개로 세고 E8에 500 updates를 기입했다. 정정본은 이를 각각 256개와 `null`로 바로잡고 실험별 모집단 분모를 복원했다. [`corrected_summary/executed_notebook_errata.md`](corrected_summary/executed_notebook_errata.md)에 정정 범위가 기록되어 있다. 원 archive의 score, prediction, checkpoint, fitted model은 바꾸지 않았고 GPU를 다시 실행하지 않았다.

## 3. 실험별 결과

### E1 — sampler 균형화

두 arm은 같은 MLP head, seed, 500 updates, batch당 REAL 16개와 FAKE 16개를 사용했다. 차이는 parent-uniform 추출과 label 내부 source×generator×parent 균형 추출뿐이다.

음악 dev에서 parent arm의 3-seed 평균 EER는 **23.631%**, balanced arm은 **24.110%**였다. 음악 test에서는 parent의 seed 평균이 **17.189%**, balanced가 **17.349%**로 balanced가 **0.160%p 나빴다**. ensemble EER는 양쪽 모두 **17.349%**였다. seed별 paired CI는 모두 0을 가로질러 따라서 결론은 **불확실**하며, 균형 sampler를 채택할 근거는 없다. 음성에서는 두 arm 모두 dev/test EER 0이었지만 test가 하나의 연결 성분에 묶여 있어 개선 증거로 보지 않았다.

생성기 하나를 학습에서 완전히 제외한 민감도 분석도 결과의 방향은 일정하지 않았다. 아래 값은 각 생성기를 따로 제외했을 때의 3-seed 평균 EER이며, 생성기 사이를 합친 하나의 aggregate가 아니다.

| 제외 생성기 | 평가 수 | 성분 | parent EER | balanced EER | balanced-parent |
|---|---:|---:|---:|---:|---:|
| acestep | 37 | 17 | 32.667% | 41.500% | +8.833%p |
| audioldm | 46 | 17 | 41.968% | 44.889% | +2.921%p |
| brev | 31 | 17 | 49.000% | 51.667% | +2.667%p |
| diffrhythm | 47 | 17 | 27.636% | 27.636% | 0.000%p |
| elevenlabs | 30 | 17 | 40.000% | 22.000% | -18.000%p |
| mubert | 29 | 17 | 26.500% | 34.667% | +8.167%p |
| musicgen | 46 | 17 | 36.254% | 25.365% | -10.889%p |
| producer | 30 | 17 | 19.333% | 22.000% | +2.667%p |
| songgen | 46 | 17 | 28.286% | 39.048% | +10.762%p |
| stableaudio | 26 | 17 | 12.667% | 60.667% | +48.000%p |
| suno | 38 | 24 | 23.538% | 30.718% | +7.179%p |
| udio | 33 | 20 | 49.000% | 37.417% | -11.583%p |

ElevenLabs, MusicGen, Udio에서는 balanced가 나았지만 Stable Audio, SongGen, ACE-Step 등에서는 나빴다. 이 결과는 단일 sampler 변경이 생성기 일반화를 안정적으로 높인다는 주장을 뒷받침하지 않는다. 음성 생성기 holdout 가운데 Cartesia, Chatterbox, Higgs, Kani는 두 arm 모두 EER 0이었지만 모두 하나의 연결 성분이었고, 나머지는 평가 수가 부족했다.

### E2 — backing truth 변화에 대한 Music/File/Voice 반응

원 E2는 2,644개 파일의 같은 window에서 raw와 recovered-vocals score를 비교했다. 알려진 VOICE label은 2,584개(REAL 1,262, FAKE 1,322), 미확정은 60개였고 provenance 성분은 4개였다. test에서 recovered-vocals 사용 시 raw 대비 EER가 여섯 조합 모두 낮아졌다. DF mean은 `26.125%→24.438%`, DF max는 `23.641%→22.750%`, DF top30은 `26.125%→24.438%`, Anti mean은 `25.000%→23.313%`, Anti max는 `24.532%→22.750%`, Anti top30은 `24.438%→22.750%`였다. 그러나 dev에서는 6개 중 4개만 개선됐고 REAL과 FAKE score가 함께 움직였으므로, vocal 분리가 backing만 순수하게 제거했다고 볼 수 없다.

GPU 실행 뒤 미리 고정한 CPU 보충 분석에서는 E1 checkpoint와 train-only normalizer를 그대로 사용해 fake SONICS backing과 두 real FMA backing 평균의 score 차이를 계산했다. 별도 선택 없이 parent와 balanced arm을 모두 보고했다.

| split/arm | Music 반응 평균 | 양의 pair | File truth-changing 평균 | 양의 pair |
|---|---:|---:|---:|---:|
| dev parent | +0.184111 | 8/8 | +0.029755 | 2/4 |
| dev balanced | +0.135048 | 8/8 | +0.026527 | 2/4 |
| test parent | +0.439034 | 8/8 | +0.380183 | 4/4 |
| test balanced | +0.363382 | 8/8 | +0.381545 | 4/4 |

test threshold 진단에서 parent Music은 n=24(REAL 16, FAKE 8), AUC 1.000, EER 0, FPR 0, recall 75.0%였고 balanced Music은 AUC 0.96875, EER 12.5%, FPR 0, recall 62.5%였다. File은 n=12(REAL 8, FAKE 4)였으며 두 arm 모두 AUC 0.75, EER 50.0%, recall 50.0%였다. parent File FPR은 25.0%, balanced는 0이었다.

다만 test의 real-backing control도 크게 움직였다. parent에서 Music은 -0.398171, File은 -0.807171이었고 balanced에서는 각각 -0.383070, -0.789576이었다. 각 slice가 하나의 연결 성분에 불과하며 FMA REAL과 SONICS FAKE가 source까지 달라진다. 따라서 이 결과는 **모델이 구성된 probe에서 backing 변화에 반응한다**는 진단까지만 뒷받침하고, Music/File authenticity를 인과적으로 포착했다는 근거는 아니다. 보충 결과와 hash 영수증은 [`postrun_e2/`](postrun_e2/)에 따로 보관했다.

### E3 — codec consistency loss

CE arm과 `CE + 0.1×paired consistency` arm은 똑같은 500 updates와 파일/variant 일정을 사용했다. 학습에는 base, telephone, mp3_32k를 넣었고, 주 평가는 보지 않은 `opus_telephone`에 두었다.

음악 Opus dev의 seed 평균 EER는 CE **34.839%**, consistency **33.333%**로 1.506%p 개선됐다. test 127개에서는 CE seed 평균이 **23.293%**, consistency가 **22.408%**로 0.885%p 낮았다. 반면 ensemble EER는 CE **21.927%**, consistency **22.408%**로 consistency가 0.481%p 나빴다. codec별 test seed 평균은 base `20.401%→17.670%`, telephone `24.659%→23.453%`, mp3_32k `21.767%→19.036%`, Opus `23.293%→22.408%`였다. 점 추정치는 대체로 좋아졌지만 seed별 paired CI가 모두 0을 가로질렀고 ensemble 결과도 반대여서 **미세한 개선 신호, 결론은 불확실**로 판단했다. 음성은 모든 arm이 EER 0이었으나 연결 성분 1개라 판단 근거로 삼기 어렵다.

### E4 — spectral forensic branch와 fusion

MERT와 spectral branch는 같은 stem view와 선형 head를 사용했다. spectral은 1–7 kHz 384차원 local-trend residual이며, dev에서 고른 fusion weight는 seed 41/42/43에 각각 `0.50/0.75/0.50`이었다.

| arm | dev 3-seed 평균 EER | test 3-seed 평균 EER | test ensemble EER |
|---|---:|---:|---:|
| MERT | 27.002% | 24.659% | 26.024% |
| spectral | 30.561% | 27.390% | 26.024% |
| fusion | 23.990% | 24.338% | 21.927% |

fusion의 test seed 평균은 MERT보다 0.321%p 낮았지만, seed 41에서는 더 나빴고 seed 42·43에서만 나았다. 모든 test 차이의 CI가 0을 가로질렀다. spectral 단독은 MERT보다 나빴다. 결론은 **fusion에 약한 신호가 있으나 혼재되어 있으며, spectral 단독 채택 근거는 없음**이다.

### E5 — 짧은 가짜 구간 탐지

calibration은 REAL/FAKE 각 108개, dev와 test는 각각 72/72개였고 전체 provenance 성분은 4개였다. dev recall을 우선하고 FPR을 함께 본 사전 규칙으로 Anti는 max pooling, DF는 top30 pooling을 선택했다.

test에서 Anti max는 EER **43.056%**, AUC 0.59481, recall **20.833%**, FPR 11.111%였다. DF top30은 EER **43.750%**, AUC 0.64323, recall **9.722%**, FPR 0이었다. 선택되지 않은 pooling을 test에서 다시 고르지 않았다. 짧은 구간을 실제로 잡는 recall이 너무 낮으므로 **현재 설정은 실용 후보가 아니다**.

### E6 — encoder layer 선택

음악 MERT는 dev에서 12층을 선택했다. dev 3-seed 평균 EER는 6층 21.218%, 12층 16.153%, 24층 24.110%였다. test 결과는 다음과 같다.

| MERT layer | seed 41 | seed 42 | seed 43 | seed 평균 | ensemble |
|---|---:|---:|---:|---:|---:|
| L6 | 12.291% | 12.291% | 12.291% | 12.291% | 12.291% |
| L12 | 9.636% | 8.675% | 8.675% | **8.995%** | **9.156%** |
| L24 | 17.349% | 17.349% | 17.349% | 17.349% | 17.349% |

L12는 L24보다 seed 평균 기준 8.354%p 낮고 L6보다 3.296%p 낮았다. 별도 사후 연결 성분 bootstrap에서도 L12-L24 차이는 세 seed 모두 음수였고, seed 42·43의 95% 구간은 0을 제외했다. 현재 여덟 실험 가운데 가장 행동 가능한 결과다. 다음 실험에서는 L12를 고정 후보로 두되, source-disjoint split과 더 많은 독립 성분으로 다시 확인해야 한다.

음성 Anti는 dev에서 L24와 L48이 모두 EER 0이어서 고정 tie-break로 L24를 선택했다. test에서는 L12/L24/L48과 세 seed가 모두 EER 0이었다. 그러나 test 화자는 2명이고 연결 성분은 1개라 층 우열이나 일반화를 말할 수 없다.

### E7 — frozen/head-only/LoRA 비교

음성 데이터는 train 318/318, calibration 204/204, dev 48/48, test 48/48 REAL/FAKE였고, 화자는 train 3명, calibration 3명, dev 2명, test 2명이었다. frozen, head-only CE, LoRA CE, LoRA CE+KL 모두 dev와 test에서 EER 0, AUC 1.0이었다. 동률 규칙으로 head-only CE가 선택됐다.

calibration에서 고정한 threshold를 test에 적용하면 recall은 모두 100%였고, FPR은 frozen 2.083%, head-only 각 seed 4.167%, LoRA CE와 LoRA CE+KL은 각 seed 0이었다. 그러나 선택 목적함수인 dev EER가 전 arm에서 이미 0이며 test도 화자 2명뿐이다. 따라서 LoRA가 더 낫다고 결론 내릴 수 없고, **현재 증거만으로 추가 LoRA 학습 비용을 정당화할 수 없다**. 먼저 더 어려운 source/speaker-disjoint 평가셋이 필요하다.

### E8 — REAL-only 이상치 탐지

E8은 반복 최적화가 없는 REAL-only diagonal Gaussian 거리 모델이다. train REAL 139개(127개 성분)만으로 정규화와 분포를 적합했고, calibration REAL 31개(19개 성분)만으로 FPR threshold를 고정했다. FAKE label은 fit이나 threshold 선택에 사용하지 않았다. 따라서 `fixed_updates`는 500이 아니라 **`null`**이다.

dev 164개(REAL 25, FAKE 139, 성분 27)에서 EER **40.144%**, AUC 0.58763, FPR 4.0%, recall 19.424%였다. test 127개에서 EER **30.602%**, AUC 0.67977, FPR 13.043%, recall 21.154%였다. optional fusion은 실행하지 않았다. REAL-only 접근은 label leakage 없이 작동했지만 현재 표현과 거리식으로는 **성능이 낮아 채택하기 어렵다**.

## 4. 종합 판정

| 실험 | 핵심 질문 | 관측 결과 | 판정 |
|---|---|---|---|
| E1 | 균형 sampler가 일반화를 높이는가 | test +0.160%p 악화, holdout 방향 혼재 | 불확실/미채택 |
| E2 | backing truth 변화에 반응하는가 | 구성 probe에서 반응, control도 크게 이동 | 진단 신호만 있음 |
| E3 | consistency가 codec 강건성을 높이는가 | seed 평균 -0.885%p, ensemble +0.481%p | 미세 신호/불확실 |
| E4 | spectral/fusion이 MERT를 보완하는가 | spectral 악화, fusion seed 평균 -0.321%p | 혼재/불확실 |
| E5 | 짧은 가짜 구간을 잡는가 | 선택 arm recall 20.833%, 9.722% | 나쁨/미채택 |
| E6 | 어느 encoder layer가 좋은가 | MERT12 8.995% vs L24 17.349% | **우선 후속 후보** |
| E7 | LoRA가 head-only보다 나은가 | 전 arm dev/test EER 0 ceiling | 판단 불가 |
| E8 | REAL-only로 탐지 가능한가 | test EER 30.602%, recall 21.154% | 나쁨/미채택 |

후속 작업의 우선순위는 다음과 같이 잡을 수 있다.

1. **MERT12를 음악 head의 기본 후보로 고정**하고 새로운 source-disjoint 평가에서 재검증한다.
2. E3 consistency는 MERT12 위에서 다시 비교하되, 현재처럼 같은 파일 일정과 fixed updates를 유지하고 ensemble과 seed 평균을 모두 보고한다.
3. E2는 실제 backing source를 교차한 설계와 더 많은 독립 성분으로 다시 만든다. 현재 FMA/SONICS 비교를 authenticity 성능으로 쓰지 않는다.
4. E4 fusion은 spectral 입력을 그대로 추가 학습하기 전에 독립 dev에서 효과가 반복되는지 확인한다.
5. E5, E8, E7 추가 LoRA 학습은 현 단계에서 우선순위를 낮춘다.

## 5. 실행 및 산출물 무결성

성공 실행 archive는 292,178,089 bytes, 7,314 members이며 SHA-256은 `681d41de10397d31a371daaa276a4a90d19f04789b85e01fdd6f936399dd23d2`다. 다운로드 뒤 모든 member hash를 검증했다. VESSL workspace `wsp-9683432r72m7`는 Unix `1789658398.128507`에 standby 상태가 확인됐고, bootstrap은 SHA-256 `d027558a7774b4581c6ed1485ed9e94f8bec5b0b8fc88687aa5366e520b07a94`로 복구됐다.

성공 결과와 setup 실패 기록은 분리했다. attempt 1~5의 시작·환경·native admission 실패 기록은 각 attempt 디렉터리에 따로 보존했으며, 이 실패 산출물을 최종 실험 결과에 합치지 않았다. attempt 6의 검증된 archive만 본 분석의 GPU 실행 근거로 사용했다.

분석에 사용한 주요 근거는 다음 위치에 있다.

- 정정 집계: [`corrected_summary/final_aggregation.corrected.json`](corrected_summary/final_aggregation.corrected.json)
- GPU 원시 보고서: [`downloaded/unpacked/output/`](downloaded/unpacked/output/)
- E7 보고서: [`downloaded/unpacked/output/e7/report.json`](downloaded/unpacked/output/e7/report.json)
- E2 Music/File CPU 보충 분석: [`postrun_e2/experiment_2_music_file_postrun.json`](postrun_e2/experiment_2_music_file_postrun.json)
- E6 사후 bootstrap: [`postrun_e2/E6_mert12_vs_l24_test_bootstrap.json`](postrun_e2/E6_mert12_vs_l24_test_bootstrap.json)
- archive 검증: [`download_verification.json`](download_verification.json)
- resource 정지 확인: [`resource_stop.json`](resource_stop.json)

이 보고서의 수치는 재현 가능한 로컬 EER와 진단 지표다. 공식 제출 점수나 대회 순위로 해석해서는 안 되며, 현재 자료만으로 L4 호환성이나 공개 환경의 강건성을 주장할 수도 없다.

<!-- HUMANIZE-SUMMARY
원본 글자수: 12033 / 윤문본 글자수: 12059 / 변경률: 0.47%
탐지: A1 번역투 3→0, D1 상투적 연결 2→0, G1 기계적 문장 시작 3→0, I1 리듬 반복 2→0
자체검증: 의미·수치·고유명사·장르·격식·기술 식별자 6/6 통과
등급: B — 수치와 감사 경계를 보존하는 보수적 리포트 윤문
주요 변경: “다음 단계로 가져갈”→“후속 검증으로 가져갈”; “그 밖의 변경”→“나머지 변경”; “채택 순서”→“후속 작업의 우선순위”
-->
