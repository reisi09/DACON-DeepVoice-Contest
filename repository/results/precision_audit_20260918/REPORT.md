# DACON 236749 전 과정 정밀 감사

2026-09-18 · 현재 제출 92294 / MERT12 full1322 기준

**실제로 놓친 문제가 있다. 가장 먼저 해결할 것은 학습 sampler다.** 최신 모델은 실제 음악 218개 중 109개를 초반에 한 번씩만 학습한 뒤 남은 486회 업데이트에서 제외했다. 저장된 최종 모델을 학습 특징에 다시 적용했을 때 0.5 기준 오류 42개가 모두 이 109개에서 나왔다. 새 모델을 검토할 근거도 있다. 현재 음악 판별기는 8초 전체 특징의 평균만 학습했고, 긴 파일 속 짧은 위조 구간을 직접 검증하지 않았다.

다만 이 두 현상만으로 공식 점수 손실 전체를 설명할 수는 없다. 검증 데이터의 출처와 정답이 강하게 결합돼 있고, 음성·부분 위조 평가의 독립 그룹이 적으며, 모든 성분의 정답을 갖춘 로컬 평가도 없다. **학습 결함 수정, 검증 데이터 재구성, 시간 구간을 다루는 새 구조 비교를 함께 추진하는 것이 타당하다.** 무작위 분할을 다시 하거나 ViT로 이름만 바꾸는 것으로는 이 문제들이 해결되지 않는다.

## 1. 실제 데이콘 결과

인증된 [제출 페이지](https://dacon.io/competitions/official/236749/mysubmission)와 [리더보드](https://dacon.io/competitions/official/236749/leaderboard)를 이번 감사에서 직접 확인했다. 아래는 로컬 proxy가 아닌 공식 결과다.

| 제출 | 식별 | 공식 총점 | ADS | CPS | 실행 시간 |
|---:|---|---:|---:|---:|---|
| 92294 | MERT12 full1322 | **0.7479394074** | 0.7211111111 | 0.9893940741 | 19분 12초 |
| 90735 | M330 MLP full1206 | 0.7396465503 | 0.7118968254 | 0.9893940741 | 19분 |
| 90692 | C08 | 0.7314822646 | 0.7028253968 | 0.9893940741 | 17분 22초 |
| 88418 / 88809 | 공식 baseline | 0.6909110688 | 0.6577539683 | 0.9893249735 | 21분 59초 / 21분 16초 |
| 90217 | C3 | 0.6843536931 | 0.6504603175 | 0.9893940741 | 21분 24초 |
| 91610 | submit.zip, 모델 연결 미확보 | 0.6837822646 | 0.6498253968 | 0.9893940741 | 21분 10초 |
| 89524 | V5 B1 fixed | 0.6776485873 | 0.6451031746 | 0.9705573016 | 2분 55초 |
| 90004 | V6 fixed | 0.6727822646 | 0.6376031746 | 0.9893940741 | 21분 19초 |
| 88630 | AST epoch17 | 0.6453405503 | 0.6212619048 | 0.8620483598 | 1분 21초 |
| 91238 | submit.zip, 과거 사용자 식별 S2 EAT | 0.6378679788 | 0.5988095238 | 0.9893940741 | 14분 33초 |
| 88673 | DACON_Sumbit_1.zip, 기존 기록상 AST full1029 | 0.6174819841 | 0.5881111111 | 0.8818198413 | 1분 22초 |

관측 당시 우리 팀은 **167위**였다. 1위 표시 점수는 0.88836, 15위는 0.85469다. 순위와 상위 점수는 시점에 따라 바뀌며 리더보드 값은 반올림 표시다. MERT12는 M330보다 **0.0082928571**, baseline보다 **0.0570283386** 높다. 개선은 있었지만 상위권과의 차이를 해소한 정도는 아니다.

공식 점수는 다음과 같다.

```text
Score = 0.9 × ADS + 0.1 × CPS
      = 0.45 × (1 − File EER)
      + 0.18 × (1 − Voice EER)
      + 0.27 × (1 − Music EER)
      + 0.05 × Voice Presence AUC
      + 0.05 × Music Presence AUC
```

현재 CPS가 0.9893940741이므로 **ADS가 그대로라는 조건에서 CPS만 1로 높여 얻을 수 있는 총점은 최대 약 0.0010605926**이다. 개선의 큰 부분은 진위 판별에서 찾아야 한다. 다만 존재 확률은 File 결합에도 쓰이므로 CPS가 높다는 이유로 presence gate의 오류를 무시해서는 안 된다. ADS만 공개되어 File·Voice·Music 중 어느 EER가 얼마나 나쁜지는 역산할 수 없다. [공식 평가 기준](https://dacon.io/competitions/official/236749/overview/evaluation)

원본 관측: [제출 목록](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/precision_audit_20260918/live_submissions_20260918.txt), [순위 기록](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/precision_audit_20260918/live_leaderboard_20260918.json). 로컬의 과거 `official_submission.json`은 아직 grading 상태이므로 현재 결과를 대표하지 않는다. 91610을 순서와 파일명만 보고 B full1696이라고 단정하지 않았다.

## 2. 현재 모델은 실제로 무엇을 하는가

```text
원음 ── PANNs ─────────────── Voice/Music Presence의 파일별 최대값
  └── HTDemucs ─ vocal ─ DF-Arena ─ Voice Fake의 window 최대값
               └ backing ─ MERT330 layer12
                            └ frame 평균 → window 평균 → MLP 3개 확률 평균
                                                      └ Music Fake

File Fake = max(Voice Presence × Voice Fake,
                Music Presence × Music Fake)
```

최신 학습 대상은 core/base 음악 **1,322개(REAL 218 / FAKE 1,104)**다. 각 seed는 1024→128→1 MLP의 **131,329개 파라미터**만 batch32로 500회 업데이트했다. MERT330·HTDemucs·DF-Arena·PANNs는 이번 재학습에서 고정돼 있다. layer12를 읽지만 추론 때는 24-layer MERT 전체가 실행되므로 이를 12-layer 모델이라고 부르면 안 된다.

V6의 시간별 learned fusion, B의 raw residual·noisy-OR, 과거 fullFT가 최신 제출에서 실행되는 것은 아니다. 파일이 ZIP 안에 있다는 사실과 실제 호출된다는 사실을 구분해 추적했다. [추론 상세](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/precision_audit_20260918/sections/inference.md)

## 3. 우선순위별 발견 사항

아래의 우선순위는 후속 조사의 순서다. 공식 점수 손실 기여도를 측정한 순위는 아니다.

| 우선 | 발견 | 판정 | 적용 범위 |
|---|---|---|---|
| P0 | sampler가 일부 원본을 구조적으로 제외하며, 최신 prefix는 초반 1회 노출만 보장 | 실제 schedule·receipt·checkpoint로 확인 | 최신 MERT12 및 eight의 관련 실험 |
| P0 | 완전한 성분 라벨·음성 부재 negative 부족 | manifest 집계로 확인 | 최신 로컬 평가 전반 |
| P0 | 출처와 정답 결합, 음성/partial test의 독립 그룹 1개 | 원시 lineage와 예측으로 확인 | 모델 선택·일반화 해석 |
| P1 | 8초·3-window 평균 학습과 4–60초 추론 사이 검증 공백 | cache 전수 검사, 점수 영향 미측정 | 최신 Music |
| P1 | Voice와 Presence의 서로 다른 시간대 최대값 결합 가능 | 실제 함수와 수치 반례 | 최신 File, B |
| P1 | 실제 음성에 실제 음악만 더해도 Voice Fake가 크게 오르는 사례 | 저장된 paired 예측에서 관측 | six probe, 좁은 표본 |
| P1 | E3 consistency가 831개 중 250개 원본에만 적용 | 실제 pair 순서 재현 | codec 방법의 해석 |
| P1 | E5 test 그룹 수 1을 4로 보고, bootstrap의 무효 표집 제한 누락 | 원시 예측 및 재현 사례 | 보고·불확실성 판단 |
| P1 | 같은 epoch라도 update 수가 달라 순수 효과 분리 불가 | 실제 실행 기록 | temporal batch, six 추가 데이터 |
| P2 | fractional label을 정수로 잘라 수용 | 입력 경계 결함 재현, 현재 라벨 영향 없음 | eight_metrics |
| P2 | B minibatch weight 재정규화가 global 목표와 달라짐 | 실제 schedule의 계수 재계산 | B 학습, 최신 MERT12 아님 |
| P2 | 무음·역상 stereo·padding 처리의 품질 확인 부족 | 실제 출력 또는 입력 반례 | 최신 경계 입력 |
| 기록 | 최신 공식 상태 누락, 이전 코드와 archived code 혼동 위험 | hash·시점 대조 | 실험 관리 |

### 3.1 sampler 결함과 최신 학습 결과

[`eight_heads.py:49`](/Users/choiwuseck/Documents/ChatGPT/데이콘/eight_heads.py:49)는 source와 parent를 같은 전역 ordinal의 나머지로 고른다. source를 방문하는 간격과 parent 수가 공약수를 가지면 일부 parent에 영구히 도달하지 못한다. seed를 바꿔도 이 순회 구조는 유지된다.

과거 E6 음악 831개 중 실제로 학습한 것은 789개였다. FMA REAL 84개 중 42개가 빠졌다. 재생한 ID 순서 해시는 저장 checkpoint의 `draw_sha256`과 일치했다. 음성도 source-balanced 경로에서 212개 중 129개만 노출됐고 REAL 53개·FAKE 30개가 빠졌다. parent-uniform 대조군은 각 모집단 전체를 사용했다. 당시 비교는 source 가중치뿐 아니라 실제로 사용한 원본 집합까지 달랐다.

최신 [`coverage_draws`](/Users/choiwuseck/Documents/ChatGPT/데이콘/mert12_full_refit.py:138)는 앞부분에 모든 원본을 한 번씩 넣는다. 그래서 1,322개 전체를 사용했다는 기록은 맞다. 그러나 기존 순회가 건너뛰는 REAL 109개는 세 seed 모두 **딱 한 번**, zero-based update 13 이내에만 등장한다. 이후 update 14–499에서는 다시 빠진다. 다른 REAL 원본은 약 61–88회씩 등장한다.

| 최신 저장 ensemble의 학습 feature 재생 | 원본 수 | 0.5 기준 오류 | 평균 BCE |
|---|---:|---:|---:|
| 초반 1회만 학습한 REAL | 109 | **42** | 1.6916062916 |
| 계속 반복 학습한 REAL | 109 | 0 | 0.0009352582 |
| FAKE | 1,104 | 0 | 0.0008384482 |

**최소 1회 등장 여부만 확인한 기존 검사로는 이 문제를 잡지 못했다.** source/generator별 parent 순회를 따로 진행하고, 각 구간의 반복 노출과 클래스 비율을 검사해야 한다. 이후 같은 split·500 updates·decoder·seed로 비교해야 수정의 성능 효과를 말할 수 있다. 위 표는 학습 데이터의 fit 격차이며 독립 검증이나 공식 EER가 아니다. [학습 상세와 재현 코드](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/precision_audit_20260918/sections/training.md)

### 3.2 데이터 재분배만으로 해결되지 않는 출처와 라벨 문제

최신 eight의 7,101행은 독립적인 음악·화자 7,101개가 아니다. 기반 원본 기록은 1,834개이며 파생본을 모두 연결하면 **256개 연결 성분**이다. 전체 화자는 10명이고 core speech test에는 2명뿐이다. 코드가 명시한 parent·artist·speaker·원본 참조를 묶어 split 간 겹침을 다시 검사했으며 누출은 발견하지 못했다. 그러나 이 자료 상당수는 이전 실험에서 이미 사용됐다. 분할이 겹치지 않는 것과 새로운 확인용 자료인 것은 별개다.

base 음악의 `source` 문자열만으로 MUSIC_FAKE를 100% 구분할 수 있다. REAL은 FMA 계열, FAKE는 Echoes·Suno·Udio 계열처럼 수집 출처와 정답이 결합돼 있다. **모델이 실제로 출처만 보고 답했다는 증거는 아니지만, 이를 배제할 수 없는 데이터 구성이다.** 같은 자료를 8:1:1이나 7:2:1로 다시 나눠도 출처-정답 결합은 남는다. FMA와 FMA/FMA 같은 이름도 실제 provider·수집 batch·원본을 분리한 메타데이터로 정규화해야 한다.

라벨 문제는 더 직접적이다. eight에서 알려진 `VOICE_PRESENT`는 2,644개이며 모두 1이다. 나머지 4,457개는 unknown이고 **확정된 0은 하나도 없다.** six의 알려진 192개, voice_music의 알려진 160개도 모두 1이다. unknown을 0으로 바꾸면 잘못된 정답이 된다. 현재 구현이 unknown을 마스킹하는 것은 맞지만, 이 데이터로 Voice Presence AUC나 완전한 공식 총점을 검증할 수는 없다.

| 현재 평가 부분집합 | 파일 수 | REAL / FAKE | 독립 연결 성분 |
|---|---:|---:|---:|
| E6 core Music test | 127 | 23 / 104 | 24 |
| core Voice test | 48 | 24 / 24 | **1** |
| E5 짧은 음성 위조 test | 144 | 72 / 72 | **1** |

Voice EER 0이나 partial 144행이라는 숫자만으로 음성·짧은 위조 문제를 해결했다고 판단할 수 없다. 그룹이 1개인 이유는 공통 화자·부모·파생 관계가 이어지기 때문이다. 여러 seed는 같은 원본에서 최적화 변동을 보는 것이며 독립 화자를 추가하는 효과가 없다. [데이터 감사](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/precision_audit_20260918/sections/data_lineage.md), [통계 감사](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/precision_audit_20260918/sections/evaluation_statistics.md)

### 3.3 평균 특징으로 사라지는 정보

실제 최신 학습 cache 1,322개를 모두 열었다. 전부 **8초, 3 windows**이며 3,966 windows 모두 audible이다. 무음 window나 padding을 포함한 학습 예는 없다. 배포는 4초에서 1 window, 60초에서 29 windows를 평균한다.

현재 head는 평균 이후의 벡터만 보므로 위조 구간의 위치·길이·시간별 변화를 복원할 수 없다. 저장 embedding 두 개를 실제 head에 넣은 수치 반례에서 fake-like 단독은 약 0.999992였지만 real-like 28개에 fake-like 1개를 섞어 평균하면 약 0.000147이었다. 실제 오디오를 합성해 평가한 결과는 아니므로 검출률로 해석하지 않았다.

또 `head(mean(features))`를 `mean(head(window))`로 바꾸는 것은 단순한 동일 연산이 아니다. 실제 학습 특징에서 두 방식의 확률 차이는 최대 **0.3897524774**, 0.01을 넘는 파일은 **148개**였다. 새 pooling은 별도 학습·검증 대상이다. 특히 L12 특징은 base 음악에만 있고 E3 codec·swap 분석은 L24를 사용했다. **기존 강건성 실험을 최신 MERT12의 강건성 검증으로 대신할 수 없다.**

### 3.4 음성·음악 혼합과 File 결합

six의 4개 실제 음성 probe는 단독일 때 Voice Fake 약 0.0052–0.0102였고 실제 음악을 섞은 대응 파일에서는 약 0.7804–0.9435였다. 같은 실제 음성을 유지한 상태에서 큰 변화가 관측됐다. 다만 화자 1명의 4쌍이며 정답은 기존 메타데이터에 근거한다. 이번 감사에서 독립 청취로 라벨을 확정하지 않았고, 이를 전체 대회의 오탐률로 확대하지 않았다.

최신 File은 `max_t(VoicePresent_t) × max_t(VoiceFake_t)`를 사용한다. 두 최대값이 서로 다른 구간에서 나올 수 있다. 실제 음성이 있는 구간의 presence와 다른 구간의 분리 잔여물에서 나온 fake 점수가 결합될 여지가 있다. `max_t(VoicePresent_t × VoiceFake_t)`와는 다르다. 과거 V6에는 시간 정합 결합이 있었지만 최신 MERT12에는 없다. 시간 정합 방식이 반드시 더 좋다는 결과는 아직 없으며 실제 window별 출력으로 확인해야 한다.

현재 File의 비중은 총점의 0.45다. 음악 EER만 좋아져도 혼합 음성이나 File 결합에서 손해가 나면 전체 개선이 작을 수 있다. 새 실험은 Music 단일 수치 외에 File·Voice·Presence를 함께 기록해야 한다. [혼합 분석 원본](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/six_h100_20260917/analysis.json), [최신 결합 함수](/Users/choiwuseck/Documents/ChatGPT/데이콘/mert12_submission.py:102)

### 3.5 작은 수치·선정 차이가 해석을 바꾼 사례

| 항목 | 재확인한 사실 | 해석·조치 |
|---|---|---|
| E3 consistency | 원본 831개·pair 1,662개 중 앞 500 pair, 원본 250개에만 penalty 적용 | consistency 자체의 효과를 일반적으로 기각할 수 없음. pair 노출을 균등하게 관리 |
| E3 집계 | seed 평균 EER 23.293%→22.408%, 확률 ensemble EER 21.927%→22.408% | 제출 방식인 확률 평균의 지표로 dev 선정 |
| E6 layer | L12 seed 평균 8.995%, ensemble 9.156%; L24 17.349% | 유용한 로컬 신호지만 역사적으로 노출된 127개·24성분의 결과이며 sampler 결함 영향도 포함 |
| E5 그룹 수 | `FINAL_REPORT.metrics.json` test components=4, 실제 test=1 | 전체 648행의 4개와 test 144행의 1개를 분리. EER 자체는 재현됨 |
| bootstrap | 양성 그룹이 하나인 재현 사례에서 1,000회 중 무효 355회를 버리고도 improvement 판정 | 무효 비율·class별 그룹 수 기준 필요. 저장 E6는 1,000회 모두 유효하여 같은 실패를 관측한 것은 아님 |
| six 후보 선택 | 새 음악과 혼합의 dev 개선량이 정확히 동점, tuple 이름 순서로 mixture 선정 | 동점 규칙 사전 고정. 혼합이 더 우수했다는 증거 없음 |
| six 학습량 | B / 새 음악 / 혼합은 459 / 476 / 493 updates | 같은 epoch를 데이터 구성만 바꾼 비교로 해석하지 않음 |
| temporal batch | 선택 checkpoint는 batch16 126 updates, batch128 32 updates | batch 효과와 optimizer 횟수 효과가 섞임 |
| fullFT | encoder gradient·실제 parameter update 연결 확인 | 학습 미연결 주장은 근거 없음. tune BCE 악화와 EER 개선은 별도 관찰 |
| B global weight | minibatch 재정규화 후 REAL 계수 비중 약 47.7–48.0%, 의도 약 50% | global weighted objective와 다른 추정. 최신 MERT12 문제와 분리 |
| threshold | E1/E3/E4/E6의 test recall/FPR은 해당 test EER점에서 threshold 결정 | 사전 고정 운영 threshold의 성능으로 쓰지 않음 |
| V4 중복 | dev/sealed 각각 210행 중 고유 PCM 190개; splice 20쌍 동일 | 과거 전체 집계에서 splice 비중 증가. 최신 eight에는 이 중복 없음 |
| spectrum | cache448 = residual384 + band64, E4는 실제로 앞384 사용 | 보고서384와 cache448은 오류가 아님 |

자세한 수치와 코드는 [평가](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/precision_audit_20260918/sections/evaluation.md), [통계](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/precision_audit_20260918/sections/evaluation_statistics.md), [학습](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/precision_audit_20260918/sections/training.md), [temporal/fullFT](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/precision_audit_20260918/sections/training_temporal.md)에 보존했다.

### 3.6 경계 입력에서 확인한 사항

최종 QA의 실제 4초 무음은 Music Fake 0.9999964237, Music Present 0.1139623746, File Fake 0.1139619671이었다. 음악이 없을 때 조건부 Music Fake가 높다는 사실 자체는 Music EER 오류가 아니다. 그러나 존재 확률을 거쳐 File로 전달되는 현상은 확인됐다. 0.5 미만이므로 괜찮다고 판단할 수도 없다. EER은 고정 0.5 정확도가 아니기 때문이다. 음성-only·무음·약한 음악을 포함한 라벨 패널에서 File ranking을 측정해야 한다.

`L=x, R=-x`인 역상 stereo 입력은 각 채널에 신호가 있어도 mono 평균이 0이 되어 현재 분리기의 무음 경로로 들어갔다. 공식 데이터에 얼마나 존재하는지는 모른다. 4초 MERT 입력은 600 samples를 zero-pad하며 processor 이전에 붙여 실제 길이 mask는 사라진다. 최소 4초 조건에서는 약 0.93% padding이므로 이를 큰 점수 손실로 단정하지 않았다. exact-tail window는 전체 구간을 덮고 중복 start가 없었다. 겹치는 샘플의 가중치는 균등하지 않지만 규칙 자체가 잘못 구현된 것은 아니다.

## 4. 문제가 없거나 이미 고쳐진 부분

의심되는 모든 부분을 버그로 분류하지 않았다.

- **평가 산식:** 실제 예측을 graph/v2/eight/sklearn에 넣은 356개 비교의 최대 차이는 2.22e-16이었다. fake-positive 방향, 정답 presence 기반 성분 마스크, unknown 처리도 확인했다.
- **현재 데이터 무결성:** eight 7,101개, six 839개, voice_music 568개 파일을 다시 읽어 hash·sample rate·channel·duration을 대조했고 불일치는 없었다. 기록된 lineage 기준 split 간 겹침도 없었다. 미기록된 복제·모방 관계까지 배제하는 음향 유사도 검사는 아니다.
- **OOF:** 저장된 A/B 6개 seed 산출물과 checkpoint 30개를 대조했다. 각 1,469행은 한 번씩 held-out으로 예측됐고 fitting/early-stop/held 그룹이 분리됐다. OOF 평균도 일치했다. 이 결과가 모든 원본 1,696개나 새 독립 모집단을 평가한 것은 아니다.
- **head·cache 연결:** 최신 1,322개 NPZ와 manifest hash가 일치했고 mean/std 재계산 차이는 0이었다. label·ID 뒤바뀜이나 잘못된 layer 선택 근거는 찾지 못했다.
- **실제 제출 ZIP:** 6,268,222,843-byte 전체 파일을 다시 hash했다. 제출 메모·로컬 receipt의 `e7bebbded6fef06517db0d42421a575a0440849248de99856796ffc3e5eb5ee5`와 일치한다. 메모 해시는 사용자가 적은 값이며 서버가 계산한 hash로 표현하지 않았다.
- **decoder 수리:** 초기 raw/cache 음악 확률 차이 0.00019127128483675815는 최종 3.971516315814938e-08로 줄었다. torchaudio Demucs 입력 고정은 이미 반영됐다. 이 과거 결함을 최신의 미수정 오류로 재기재하지 않았다.
- **출력 순서·독립성:** 저장된 최종 원본/역순/부분집합의 공통 ID 확률 차이는 0이었다. 중복·누락 ID, NaN, 범위 밖 확률, 잘못된 열도 실제 함수에서 거부됐다.
- **공식 실행:** 최신 제출은 실제 채점 완료·19분 12초다. A100 smoke만으로 L4 성공을 추정한 것이 아니다. 다만 공식 peak memory나 설치 상세 log는 확보하지 않았다.

패키지 상세와 과거 M330의 별도 CPU/A100 차이, B decoder 미고정 가능성은 [패키징 감사](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/precision_audit_20260918/sections/packaging.md)에 구분했다.

## 5. 데이터셋은 어떻게 다시 구성할 것인가

**분할 비율보다 먼저 원본·라벨·평가 질문을 바꿔야 한다.** 기존 데이터는 버리지 않고 개발·학습 자료로 유지하되, 이미 열어 본 test를 새 독립 확인셋으로 부르지 않는다.

1. **최소 메타데이터를 정리한다.** parent/source recording, artist/speaker, provider/수집 batch, generator/버전, 원본·PCM hash, 길이·채널·codec, 다섯 target, 라벨 검토 상태와 근거, 생성·편집 구간을 파일별로 연결한다. `source` 하나에 provider·generator·진위가 섞여 있으면 별도 열로 풀어낸다.
2. **원본을 먼저 분할하고 파생본은 그 뒤에 만든다.** 원곡·화자·reference·mixture의 모든 부모를 연결한 component가 split을 넘지 않게 한다. 같은 화자를 반복 섞어 test 전체가 하나로 연결되지 않도록 split 안에서도 독립된 부모 묶음별로 조합한다. generator는 전부 하나의 거대 component로 연결하지 않고 별도 holdout 축으로 관리한다.
3. **평가를 세 질문으로 나눈다.** 같은 도메인의 새 원본 평가, 학습하지 않은 generator 평가, 수집 출처 자체가 다른 최종 확인 평가를 별도로 둔다. generator가 다르더라도 codec·길이·수집 batch가 정답을 대신 알려주면 안 된다. 양 class를 같은 후처리 조건으로 교차 배치한다.
4. **빠진 라벨과 사례를 확보한다.** 확인된 음성 부재·음악 부재, 사람 노래·AI 노래·말소리, REAL/REAL·REAL/FAKE·FAKE/REAL·FAKE/FAKE 혼합, 동시 혼합·시간 삽입을 포함한다. 출처만으로 애매한 노래의 성분 진위를 확정하지 않는다. 부분 검토는 unknown을 유지하고 완전 라벨 평가와 분리한다.
5. **길이·변환·삽입 비율을 교차한다.** 4/8/16/30/60초, codec/telephone, mono/stereo, 삽입 위치와 길이를 정답 및 source와 독립적으로 배정한다. 같은 부모의 변형은 하나의 의존 그룹으로 센다. 합성 경계만 탐지하지 못하도록 REAL→REAL·FAKE→FAKE sham edit도 같은 crossfade·gain 처리로 만든다.
6. **test를 열기 전에 선택 규칙을 잠근다.** 주 지표는 완전 라벨에서 공식 가중 점수와 성분별 지표, 미완전 라벨에서는 정의된 부분 지표다. 최종 사용 방식의 ensemble을 dev에서 선택하고 threshold는 calibration에서 고정한다. 동일 원본의 paired 비교와 class별 component 수, 무효 bootstrap 비율을 함께 기록한다.

독립 원본 몇 개가 충분한지는 구분하려는 개선량과 class·source·의존 구조에 따라 달라진다. 300개 같은 보편적 합격선을 두지 않는다. 현재 부족한 class·source를 먼저 채우고, 목표 정밀도를 만족하는지 pilot의 그룹별 변동으로 산정하는 것이 맞다. group 분할의 의미는 [scikit-learn 공식 문서](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)를 따른다.

## 6. ViT와 새로운 방법론에 대한 판단

**ViT를 검토할 가치가 있다.** 다만 AST는 이미 spectrogram Transformer이고, 프로젝트에서도 AST 마지막 block·전체 backbone과 EAT 전체 미세조정을 수행했다. 기존 fullFT의 gradient도 연결돼 있었다. 따라서 “Transformer를 아직 안 써서 막혔다”거나 “end-to-end 학습이 처음이다”라는 진단은 맞지 않는다. 기존 AST 제출 실패가 모든 ViT 방식의 실패를 뜻하지도 않는다.

시간 구조와 부분 편집도 처음 시도하는 것은 아니다. 과거 V4 P4는 temporal bag loss와 partial-edit를 함께 적용했고, 이후 temporal head와 B의 구간 loss도 있었다. 이번 제안은 그 이력을 버리고 같은 시도를 새 이름으로 반복하는 것이 아니라, 확인된 sampler·데이터 문제를 정리한 조건에서 patch 감독과 집계 효과를 분리해 다시 비교하는 것이다. [기존 P4 기록](/Users/choiwuseck/Documents/ChatGPT/데이콘/EXPERIMENTS_ALL_2026-09-17.md:191)

새 실험의 차이는 **어떤 구간을 감독하고, 무엇을 평균하며, 어떤 출처에서 검증하느냐**에 둔다. 아래 우선순위는 이번 감사의 판단이며 성능 측정 결과가 아니다.

| 순서 | 비교할 방법 | 기존과 다른 질문 | 채택 근거 |
|---|---|---|---|
| A | 수정된 sampler의 MERT12 + window별 head / 사전 지정 MIL pooling | 평균 전에 fake evidence를 보존하면 긴 파일·부분 위조가 좋아지는가 | 같은 encoder를 고정해 시간 집계 효과를 분리 |
| B | **spectrogram ViT의 patch/window 구간 감독** | 시간·주파수 patch가 위조 구간을 직접 학습하면 유리한가 | 파일 label만 복제하지 않고, 알려진 삽입 구간의 loss와 file loss를 함께 평가 |
| C | **multi-resolution STFT/log-mel CNN** | 작은 국소 주파수 단서가 Transformer보다 잘 일반화하는가 | B와 같은 입력·라벨·pooling으로 구조 차이를 비교 |
| D | SSAST / BEATs 표현 교체 | 사전학습 목표가 다른 encoder가 보완 단서를 갖는가 | 동일 split·window·head·선택 예산에서 먼저 frozen 비교 |
| E | raw waveform 또는 wavelet 보조 경로 | mel/semantic 특징에서 잃은 단서가 있는가 | 단독 대비 추가 정보와 미학습 generator 이득 확인 후 결합 |
| F | codec contrastive / 구간 일관성 loss | codec 변화와 위조 단서를 구분할 수 있는가 | pair 전체 노출을 보장하고 원본·변환 양쪽 순위 성능 확인 |

ViT의 구체적인 최소 실험은 audio-pretrained AST/SSAST를 초기화해 log-mel patch를 읽고, 시간 구간별 authenticity와 file authenticity를 함께 학습하는 것이다. 실제 경계가 알려진 합성 삽입에서는 구간 label을 사용하고, 파일 label만 있는 곡은 모든 patch를 fake로 강제하지 않는다. 원곡 split을 유지한 뒤 변형을 생성하고 sham edit도 포함해야 생성한 경계의 흔적만 외우는 문제를 줄일 수 있다. raw/stem 두 경로, 새 encoder, 새 loss, 새 pooling을 한 번에 모두 바꾸지 않는다.

SSAST는 AST 구조에 masked patch의 생성·판별 사전학습을 적용한 방법이고, BEATs는 acoustic tokenizer가 만든 이산 목표를 학습한다. 서로 다른 사전학습이 forensic 단서를 보존할지는 이 대회 데이터에서 확인해야 한다. 음성 anti-spoofing의 RawNet2/AASIST·contrastive 성과를 음악 성능으로 그대로 옮길 수 없다. [SSAST 논문](https://arxiv.org/abs/2110.09784), [BEATs 논문](https://arxiv.org/abs/2212.09058), [RawNet2 논문](https://arxiv.org/abs/2011.01108)

공식 코드의 고정 commit, 후보별 장단점과 근거 한계는 [신규 방법 조사](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/precision_audit_20260918/sections/new_methods.md)에 있다. 모델 가중치별 사용·재배포 조건과 L4 실행 비용은 후보 선정 후 실제 artifact로 확인할 사항이다. 아직 측정하지 않은 학습 비용이나 예상 공식 점수는 제시하지 않는다.

## 7. 다음 실험의 실행 순서와 중단 기준

| 단계 | 작업 | 결정에 필요한 결과 |
|---|---|---|
| 1 | sampler root cause 수정, per-cell cursor 또는 동등한 독립 순회 구현 | class 비율 유지, source/generator/parent별 반복 노출, seed 재현성, 학습 전후 전체 기간에서 특정 원본이 배제되지 않음 |
| 2 | 기존 split에서 동일 조건 MERT12 대조 | 노출 수정이 학습 fit와 진단 dev/test에 미치는 영향. 기존 test는 역사적 진단으로 표시 |
| 3 | 새 원본·완전 라벨 확인 패널 확보·봉인 | 출처/화자/부모 분리, 성분별 양 class, 길이·혼합·부분 위조·부재 포함 |
| 4 | 시간 집계 A와 새 구조 B/C 비교 | 동일 split·decoder·입력 구간·seed·선택 예산, optimizer updates와 실제 sample exposure를 함께 기록 |
| 5 | 새 구조를 잠근 뒤 독립 확인 | 공식 가중 지표 또는 사전 정의 부분 지표의 paired 차이, 그룹별 일관성, 불확실성 확인 |
| 6 | 가중치·runtime 패키지 검증 후 공식 제출 | 실제 오디오 parity, ID/순서/부분집합 독립성, 공식 offline L4 자원 조건 |

새 architecture가 dev에서만 좋아지고 source/generator holdout에서 나빠지면 확대하지 않는다. window recall이 좋아져도 File EER가 악화되면 전체 개선으로 채택하지 않는다. 효과가 특정 한 그룹에만 있거나 불확실성이 큰 경우에는 결론을 유보하고 부족한 독립 사례를 추가한다. 고정 update는 exposure까지 동일하다는 뜻이 아니므로 둘 다 보고하며, 동점 규칙은 결과를 보기 전에 정한다.

당장 큰 GPU 학습을 다시 시작할 근거는 부족하다. sampler와 비교 조건을 먼저 바로잡은 뒤 좁은 대조부터 시작하면, 구조 변경의 실제 효과를 판별할 수 있다. 이 감사에서는 운영 코드 수정·재학습·GPU 시작·재제출을 하지 않았다.

## 8. 감사 범위와 남은 미확인 사항

현재 제출의 입력→decode→분리→특징→head→결합→CSV→ZIP→공식 결과 경로를 추적했다. eight/six/voice_music의 원시 manifest·예측, 최신 head·feature 전수, 이전 V4·temporal·fullFT·redesign OOF의 주요 경로도 검사했다. 루트 Python 246개를 파싱해 import inventory를 남겼지만, 이 숫자가 모든 파일을 줄마다 검토했거나 모든 과거 실험을 다시 실행했다는 뜻은 아니다.

| 영역 | 이번 증거 | 한계 |
|---|---|---|
| 최신 제출 | 인증된 공식 화면, 전체 ZIP 새 hash, 내부 runtime·head | 서버의 hidden 정답·성분 EER·전체 출력은 미공개 |
| 데이터 | 세 현행 cohort 파일 hash/format 전수 대조, lineage·label 집계 | 모든 음원 독립 청취, fingerprint 기반 숨은 중복 검사 미실시 |
| 학습 | 실제 schedule digest/count 재현, 최신 cached head 전수 실행 | 수정 sampler 재학습·새 모델 학습 미실시 |
| 추론 | 실제 작은 함수 실행, 저장된 GPU raw QA 재계산, head 수치 반례 | 전체 encoder를 새 환경에서 전 데이터 재추론하지 않음 |
| 평가 | 저장 예측의 지표 재계산, group·OOF·선정 검토 | 이미 노출된 test로 새 일반화를 입증할 수 없음 |
| 방법론 | 기존 코드·이력과 1차 논문·공식 구현 대조 | 음악·이 대회에서 후보의 효과 및 비용 미측정 |
| 원격 자원 | 기존 종료 receipt 확인, 이번 감사에서 시작하지 않음 | 현재 VESSL live 상태를 새로 조회한 것은 아님 |

아직 답할 수 없는 핵심 질문은 공식 ADS 손실의 성분별 비중, 부분 위조·혼합·출처 변화 중 실제 hidden test 병목, 라벨 자체의 음향적 정확성, 수정 sampler의 일반화 효과다. 이번에 확정한 코드 결함과 관측 현상은 위 가설들을 구분할 후속 실험의 근거로 삼는다.

## 9. 상세 근거와 재현

- [데이터·라벨·분할](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/precision_audit_20260918/sections/data_lineage.md)
- [학습·sampler·최신 head](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/precision_audit_20260918/sections/training.md)
- [temporal·fullFT](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/precision_audit_20260918/sections/training_temporal.md)
- [전처리·추론·File 결합](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/precision_audit_20260918/sections/inference.md)
- [평가 산식·후보 선정](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/precision_audit_20260918/sections/evaluation.md)
- [통계·OOF·bootstrap](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/precision_audit_20260918/sections/evaluation_statistics.md)
- [패키지·실행 계약](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/precision_audit_20260918/sections/packaging.md)
- [ViT와 신규 방법 조사](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/precision_audit_20260918/sections/new_methods.md)
- [범위·판정 기록](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/precision_audit_20260918/AUDIT_LEDGER.md), [소스 inventory](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/precision_audit_20260918/source_inventory.json)
- [종합 보고서 독립 검토](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/precision_audit_20260918/REVIEW.md)

프로젝트 루트에서 CPU로 핵심 수치를 재현한다. 기존 artifact를 읽으며 모델을 다시 학습하지 않는다.

```sh
PYTHONPATH=. .venv/bin/python results/precision_audit_20260918/training_reproduce.py
.venv/bin/python results/precision_audit_20260918/evaluation_recompute.py
```

학습 데이터 오류와 공식 점수, fixed-model bootstrap과 독립 재현, package parity와 품질을 분리해 읽어야 한다. 운영 코드 변경은 이 감사의 결과에 포함하지 않았다.

<!-- HUMANIZE-SUMMARY: Conservative Korean final style pass applied after drafting. Facts, numeric literals, identifiers, uncertainty and evidence scope preserved; no new claims introduced by the style pass. -->
