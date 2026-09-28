**제 결론은 “데이터의 구성과 실제 학습 비중을 고쳐야 한다”입니다.** 부족한 조합과 가창 도메인은 분명합니다. 다만 그것만으로 S8의 하락 원인이 밝혀진 것은 아닙니다. 남은 시간은 **S7 기반의 재가중 실험 하나, 작은 데이터 보강 실험 하나, FILE 변경의 대조 제출 하나**에 쓰겠습니다. S8을 새 기준 모델로 삼지는 않겠습니다.

**A. DB–LB 차이와 S8 실패의 원인**

**[VERIFIED] 단순한 행 수보다 심각한 문제가 loss 비중에 있습니다.** `loss_weights`는 source별이 아니라 `(class, source, generator)` cell별로 균등화합니다. CSV와 수식을 재계산하면, 원래 train의 라벨이 있는 fake voice에서 MLAAD+ASVspoof는 행 수의 **76.48%**지만, cell은 **347/358개**여서 fake-class 기본 loss의 **96.93%**를 차지합니다. generator 이름이 많은 데이터셋의 영향력이 오히려 커지는 구조입니다. 단, 이 수치는 **L·cache 필터·duration 보정·P1 병합 이전**입니다. 제출 모델의 최종 비중이라고 해석하면 안 됩니다. ([full_bank_batches.py:27](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/full_bank_batches.py:27), `data/bank_trainval.csv` 재집계)

**[VERIFIED] 데이터 공백도 실제입니다.** `realV+fakeM`은 train **5**, val **5**, test **107**입니다. 하지만 P1은 train+test로 학습하므로 **최종 패키지도 5개만 학습했다는 설명은 틀립니다. P1에서는 112개입니다.** 한국어 반주 포함 real singing의 메타데이터상 부재, SVC/SVS 가창 위조의 부재는 별도의 중요한 공백입니다. unknown language가 많으므로 “한국어 실제 노래가 음향적으로 한 곡도 없다”까지 확정할 수는 없습니다. ([브리핑:57](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/FULL_BRIEFING_20260928.md:57), [train_backbone_heads.py:254](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/train_backbone_heads.py:254))

**[INFERRED] 가장 유력한 구조적 설명은 출처·가창·제작 방식과 진위 라벨이 얽힌 상태에서 생긴 분포 이동입니다.** raw mix가 추가되면 음악 생성 흔적뿐 아니라 보컬·믹싱·마스터링 특성도 이용할 수 있습니다. 그러나 **[UNKNOWN] LB에 어떤 실제 음악이나 위조 계열이 많았는지는 모릅니다.** 새로운 real 도메인의 오탐뿐 아니라, 새로운 fake 도메인의 미탐도 동등하게 조사해야 합니다.

“불균형이 원인”이라는 주장에 대한 반증도 있습니다.

- EER은 real/fake **개수 비율 자체**에 직접 좌우되는 지표가 아닙니다. 중요한 것은 각 클래스 안의 도메인 구성입니다.
- 같은 데이터에서도 S6→S7은 LB가 **+0.0154** 개선됐습니다. 불균형이 모든 표현 개선을 막는 것은 아닙니다.
- SingMOS 등 기존 증량의 개선은 작았습니다. 빈칸을 채운다는 명분만으로 성능 향상을 보장할 수 없습니다.

**가장 위험하게 무시할 경쟁 설명은 FILE/CDF와 DB→P1 재학습의 변화입니다.** 실제 제출 가중치에 맞춘 DB 비교는 `0.91352−0.90982=+0.00370`입니다. S8에서 `0/9/1`로 바꾸면 DB ADS가 **+0.00126**지만, 이것이 LB 효과의 상한은 아닙니다. 브리핑의 “가중치로는 약 0.0077의 역전을 설명할 수 없다”는 결론은 성립하지 않습니다. ([diag_s8.json:41](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/results/diag_s8.json:41), 같은 파일 98–114행)

또한 HAIM MUSIC FPR은 **0.089→0.089**입니다. “모든 source에서 감소”, “DB에는 실패 도메인이 없다”는 설명을 고쳐야 합니다. 한 threshold의 real FPR만으로 fake recall이나 전체 ROC를 판단할 수 없습니다. 반복 선택한 held-out의 bootstrap도 새로운 source로의 이동을 보장하지 않습니다. 과거의 `SD≈0.006/1,200-file draw` 역시 **고정된 공식 test에서 두 모델을 비교한 paired 차이의 표준오차가 아닙니다.**

**B. 남은 약 15시간의 우선순위**

아래 수량은 **[ESTIMATE] 전체 수집 목표가 아닌 이번 실험의 상한**입니다. 같은 곡을 잘라 수량을 채우지 말고 독립 곡·가수·생성 계열을 확보해야 합니다.

| 순위 | 할 일 | 규모와 제한 |
|---|---|---|
| 1 | 실제 학습 비중 집계 + 재가중 후보 `S7-R` | 모든 필터와 duration 보정 **이후** source-family·generator·가창/발화·조합별 loss 합계를 확인. 기존 가중치와 `class→source-family→generator` 계층 가중치를 **50:50으로 혼합한 한 후보**만 비교. encoder·L·FILE 계수는 유지 |
| 2 | 새 도메인 검증 세트를 먼저 확보 | 독립 real 완성곡·가창, SVC/SVS, speech+music을 포함한 **600–1,000개**. 학습에 편입하기 전에 분리 |
| 3 | 데이터 후보 `S7-D` | real 완성곡/가창 **200–400개**, 기존 bank에 없는 SVC/SVS **200–400개**, 아래 통제 혼합 **400–800개**. 하나의 고정된 보강안만 학습 |
| 4 | AI-Hub 조건부 편입 | 국내 처리 경로가 첫 **1–2시간** 안에 확보될 때만 이번 학습 일정에 포함. 확보되지 않으면 다른 후보를 기다리게 하지 않음 |

재가중 비율 `50:50`은 최적값이라는 주장이 아닙니다. 확인된 cell 편중을 완화하는 **단일 사전 지정 실험값**입니다. 새로운 작은 source를 무조건 큰 source와 동등하게 만들거나, 5개 희소 조합을 반복 복제하는 방식은 피하겠습니다.

AI-Hub는 다음처럼 쓰겠습니다. **대회 사용 허용은 전제로 인정합니다.**

- **465:** 이미 로컬에 있는 **256 train + 55 dev**부터 점검합니다. real Korean singing과 혼합용 vocal donor로 가장 빠릅니다. 가수·원곡으로 다시 묶어야 하며 반주 포함 여부는 파일별로 확인합니다. 공식 구성은 18명, 4,681개입니다. [AI-Hub 465](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=465)
- **473:** **200–400곡**, 가능하면 **20명 이상**을 우선합니다. 92명 규모여서 가수 다양성 보완에 유리하지만, 가이드보컬을 완성된 상업 음원과 동일시하면 안 됩니다. [AI-Hub 473](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=473)
- **71767:** **200–300개** 악기·loop를 보강용으로만 사용합니다. 120,000 WAV는 완성곡 120,000곡이 아닙니다. 마스터링된 real song 공백의 직접적인 대체재가 아닙니다. [AI-Hub 71767](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71767)
- **71314:** 실제 음원 접근이 확인되면 **100–200개** 방송 발화/BGM 구간만 추립니다. 프로그램에 BGM이 있다는 정보로 모든 clip을 `MUSIC_PRESENT=1`로 만들지 않습니다. 전체 7,000시간 수집은 제외합니다. ([데이터 조사:281](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/PUBLIC_DATASET_SURVEY_20260927.md:281))

**[INFERRED] 합성 혼합은 두 종류의 공백을 구분해야 합니다.** `real vocal+AI accompaniment`와 `SVC fake vocal+real accompaniment`는 다른 과제입니다. 같은 donor 묶음으로 네 진위 조합을 만들고 codec·gain·mixing 처리를 동일하게 적용하십시오. 먼저 donor를 split에 배정한 뒤 합성해야 합니다. 이미 작동하는 SVC 경로가 없다면 오늘 새로 설치·학습하지 말고, 확보된 SVS/SVC 자료를 우선합니다. 새로운 ACE-Step prompt만 늘리는 일도 후순위입니다.

**[VERIFIED: 작업 제약] AI-Hub 원음·stem·feature·그 파생 합성물은 국내에서 처리해야 합니다.** EU VESSL이나 cloud LLM에 보내지 않습니다. `deneb-kr`의 실제 위치·사용 가능 여부와 Mac 가용성은 **UNKNOWN**입니다. 국내에서 feature만 뽑아 EU로 옮겨 head를 학습하는 우회도 이번 제약에 맞지 않습니다. ([브리핑:99](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/FULL_BRIEFING_20260928.md:99))

**[ESTIMATE] 비용은 신규 1,000–2,000행의 전체 pipeline에 H100x4 1–3시간, head 비교·P1에 추가 약 1시간, 패키지 검증·전송에 최소 3시간을 잡겠습니다.** 첫 50개로 처리량을 측정해 수정해야 합니다. 기존 stem의 새 encoder 추출 15–20분을 신규 데이터 전체 처리 시간으로 쓰면 안 됩니다. 9.74 GB를 2.4 MB/s로 받는 데만 **약 68분**, 3.3 MB/s로 올리는 데 **약 49분**입니다. 기존 단가 추정으로 EU GPU 작업은 **$40–50 안에 제한**하고, 국내 GPU 비용·환경 구성 시간은 별도 UNKNOWN으로 둡니다.

지금 중단할 것은 **새 encoder 탐색 확대, full fine-tuning, 대규모 TTS 증량, 전체 AI-Hub 다운로드, 실패한 held-out 뒤의 가중치 재탐색**입니다.

**C. S8 실패를 잡을 검증 설계**

**“이 설계였으면 반드시 S8을 걸렀다”는 주장은 할 수 없습니다.** 새 검증에서도 S8이 이길 수 있습니다. 그때는 해당 실패 가설을 낮춰야지, S8이 질 때까지 자료를 바꾸면 안 됩니다.

제안하는 사전 규칙은 다음과 같습니다.

1. **실제 S7·S8 P1 패키지를 새 독립 음원에서 비교합니다.** 기존 test는 P1 학습에 들어갔으므로 패키지의 독립 평가가 아닙니다. 신규 자료는 원곡·가수·donor 계보로 학습/dev/audit을 먼저 분리하고, audit은 P1 재학습에서도 제외합니다.
2. 완성곡, 가창 위조, 발화+음악, instrumental 도메인의 구성을 **점수를 보기 전에 고정**합니다. ADS와 component EER을 함께 보고, 한 클래스뿐인 조합에서는 EER 대신 dev에서 고정한 threshold의 FPR/FNR을 봅니다.
3. **[INFERRED: 제안 gate]** 새 audit의 S7 대비 `ΔADS≥+0.005`, lineage-group bootstrap `P(Δ>0)≥0.90`, 기존 DB `ΔADS≥−0.002`. 미리 정한 핵심 조합의 FPR/FNR 악화는 각각 **3%p 이내**로 제한합니다. 작은 표본에서 이는 운영상 기준이지 일반화 보증은 아닙니다.
4. `S7-R`, `S7-D`를 dev에서 결정한 뒤 audit을 한 번만 엽니다. S8 결과를 보고 만든 이번 audit은 **S8에는 사후 진단**, 새 후보에는 사전 평가입니다. random group K-fold나 기존 unseen-generator 제외 분석으로 새 도메인 검증을 대체하지 않습니다.

**D. 09-29 세 제출**

Round 9의 `S7-noL` 우선 권고는 수정하겠습니다. 이번 진단에서는 **실제로 바뀐 FILE 계수를 먼저 분리할 가치가 더 큽니다.**

| 순서 | 후보 | 판단과 fallback |
|---|---|---|
| **1 — 00:00 이후** | **S8의 FILE 계수만 `0/9/1`로 변경** | S8의 head·자체 CDF tables·전처리는 그대로 유지. S7 수준으로 회복하면 FILE 변경이 순하락을 설명할 수 있다는 직접 증거가 됩니다. 회복하지 않아도 raw-mix만의 책임이 확정되는 것은 아닙니다 |
| **2** | **S7-R: 재가중만 변경** | 새 gate 통과 시 제출. 실패하거나 준비되지 않으면 검증된 `S7-noL`을 탐색 후보로 사용 |
| **3 — 08:00 이전 준비 완료** | **S7-D: 작은 데이터 보강** | 새 audit까지 통과한 후보. 미완료이면 기존 lock을 통과하고 패키지 검증까지 끝난 S9를 탐색 제출로 사용. S9도 없으면 남아 있는 검증된 fallback만 사용 |

S8 5-seed에는 자동으로 슬롯을 주지 않겠습니다. seed 평균화가 이번 도메인 역전을 해결한다는 근거가 없습니다. 제출 결과를 보고 밤새 FILE 가중치를 여러 번 바꾸는 것도 하지 않겠습니다.

**[UNKNOWN] 이 중 어느 것도 +0.02 ADS를 보장하지 않습니다.** S7에서 ADS 0.86까지는 **+0.02260**이 필요합니다. 지금 가장 근거 있는 도전은 확인된 학습 비중 편중과 빠진 도메인을 함께 조사하되, **변경은 분리해서 평가하는 것**입니다. S7은 계속 best-ever로 남습니다.

