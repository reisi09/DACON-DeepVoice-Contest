**[INFERRED] 제안은 S7 인코더와 head 구조를 유지하고, music 가중치·WavLM head·FILE 결합만 제한적으로 비교하는 통합 후보 1개입니다. 다만 현재 `--final` 경로는 E1/E2를 학습에 편입하므로 그대로 사용하면 안 됩니다.**

아래에서 **VERIFIED는 제공된 코드·기록에서 확인한 사실**, **INFERRED는 제 판단과 제안**, **UNKNOWN은 제공 자료로 확인하지 못한 사항**입니다. 학습이나 추론을 새로 실행하지 않았으며, CSV 합계와 기록된 수치는 재계산했습니다. 파일 경로는 별도 표시가 없으면 `code_r14/` 기준입니다.

## 먼저 바로잡아야 할 전제

1. **[VERIFIED] 최종 학습 코드가 문서의 평가 원칙과 충돌합니다.**  
   `train_backbone_heads.py:349`는 기존 test를 train에 합칩니다. `:369–372`는 `NEW_VAL=1`일 때의 newval만 제외하고 새 행 전부를 train에 넣으므로 **newtest도 학습에 들어갑니다**. `NEW_TRAIN=0`도 이 경로에서는 적용되지 않습니다. `pkg_chain.sh:12` 역시 “held-out 6,450”의 학습 편입을 명시합니다.  
   **결정:** DB 학습으로 선택한 체크포인트를 그대로 패키징하고, 평가 후 재학습하지 않습니다.

2. **[VERIFIED] “새 학습 데이터 40,214행”은 현재 자료로 입증되지 않습니다.**

   | CSV 재계산 | 행 수 |
   |---|---:|
   | train | 23,955 |
   | newval | 3,841 |
   | newtest | 12,426 |
   | 전체 | 40,222 |

   근거는 `new_data_summary.csv:2–13`입니다. DB 경로는 train만 추가합니다(`train_backbone_heads.py:76–86`). **[UNKNOWN] 40,214와의 8행 차이 및 실제 학습 투입 수**는 실패 목록·manifest·체크포인트 `train_ids`가 있어야 확인됩니다. 캐시가 없으면 행을 건너뛰는 코드는 존재합니다(`:61–73`).

3. **[VERIFIED] AT-ADD singing 수치가 문서와 다릅니다.**  
   전체 E2의 singing 3,500행에서 W_new의 FILE EER는 **0.0940**, VOICE EER는 **0.0542**입니다(`new_eval_newtest.json:291–298`). 문서의 “모든 구성에서 FILE EER 0.13–0.15”는 맞지 않습니다. 다만 이 수치는 **unlinked subset만의 결과는 아닙니다**.

4. **[VERIFIED] E1/E2는 이미 여러 구성의 비교에 사용됐습니다.**  
   `new_eval_newtest.json`과 `leak_check.txt:2–5`에 구성별 결과가 있습니다. **[INFERRED] 이번 후보를 잠근 뒤 한 번만 확인할 수는 있지만, 이를 처음 사용한 독립 최종 평가라고 부를 수는 없습니다.**

## (A) 통합 모델 설계

### 구성별 기본값과 후보

아래 선택은 모두 **[INFERRED: 제안]**입니다.

| 구성 | 기본값 | 비교할 대안 | 결정 이유 |
|---|---|---|---|
| 인코더·전처리 | S7 그대로 | 없음 | 새 인코더·추가 raw-mix forward 없이 진행 |
| Music head | 기존 3840-d WindowMIL, s7 가중치 | W 가중치, 두 구성의 확률 50:50 평균 | 구조 변경 없이 데이터 가중치 효과와 상보성 비교 |
| DF-Arena voice | s7 가중치, MLP, rule L | 없음 | voice 전체에 W를 적용하는 위험을 제한 |
| AntiDeepfake voice | s7 가중치, ResMLP, rule L | 없음 | 초기 classifier와 residual 구조 유지 |
| WavLM voice | s7 가중치, MLP | W 가중치 head로 교체 | W의 다른 도메인 개선을 한 member에서 시험 |
| Voice fusion | DF/AD/WavLM = **1/0.5/1** | 아래 V1·V2만 | 가중치 연속 탐색 방지 |
| FILE | **0/9/1** | **0/1/0**, **1/1/1** | noisy_or 단독 및 현재 규칙의 비교 |
| Presence | 기존 head 고정 | 없음 | direct CPS 개선 여지가 작고 FILE 효과와 혼동 방지 |
| Seeds | **41/42/43**, 확률 평균 | 없음 | seed 선택 자체를 추가 탐색으로 만들지 않음 |

Voice는 정확히 세 구성만 비교합니다.

- **V0:** s7 DF/AD/WavLM, fusion **1/0.5/1**.
- **V1:** s7 DF/AD + W WavLM, fusion **1/0.5/1**.
- **V2:** s7 DF/AD/WavLM, fusion **1/1/1**.

WavLM head 교체와 fusion 변경을 동시에 시험하지 않아 효과를 구분합니다.

**[VERIFIED] 현재 runtime은 head별 확률을 구한 뒤 평균합니다.** Music은 head별 log-mean-exp → sigmoid, voice는 head별 window/segment 최대 fake probability → 평균입니다(`full_bank_submission.py:213–245`). 문서의 “fake odds”보다 코드의 **fake probability**가 정확합니다.

### 왜 music과 voice의 가중치를 분리하는가

**[VERIFIED: 저장된 E1 결과]**

| 구성 | FILE EER | VOICE EER | MUSIC EER | ADS |
|---|---:|---:|---:|---:|
| s7_ref | 0.07410 | 0.06116 | 0.13910 | 0.90899 |
| s7_new | 0.06640 | 0.06224 | 0.12984 | 0.91540 |
| W_new | 0.07116 | 0.07913 | 0.12094 | 0.91231 |

근거: `new_eval_newtest.json:17`, `:267`, `:543`.

W_new의 s7_ref 대비 ADS 변화는 반올림된 EER로 계산하면 다음과 같습니다.

- FILE 기여: **+0.001470**
- VOICE 기여: **−0.003594**
- MUSIC 기여: **+0.005448**
- 합계: **+0.003324**

**[INFERRED] W music을 따로 시험할 이유는 충분합니다. W voice 전체를 기본값으로 삼을 근거는 부족합니다.** 단, W music과 s7 voice를 결합했을 때의 FILE EER는 별도로 계산해야 합니다. 위 숫자를 합쳐 새 모델 성능으로 제시하면 안 됩니다.

### 데이터와 학습

**[INFERRED: 기본값]**

- 기존 DB train + 새 데이터의 **train 행만** 사용합니다.
- `newval` 3,841행, `newtest` 12,426행, E1, E3는 학습에 넣지 않습니다.
- 추가 OpenBMAT 400행은 이번 기본 후보에서 제외합니다. 이미 공개된 약점에 맞춘 추가 변경까지 겹치지 않기 위해서입니다.
- optimizer와 구조는 유지합니다. Music **lr 1e-3, 50 epochs/patience 5**, voice **lr 1e-4, 150/10**, AdamW **weight decay 0.01**, batch **256**입니다.
- rule L은 기존 DF/AD 적용 범위를 유지하고, WavLM으로 확대하지 않습니다.

**[VERIFIED]** optimizer·early stopping은 `run.py:213–254`, voice 구조와 rule L 적용은 `voice_data.py:34–54`, `:103–128`에서 확인됩니다. 반면 localization loss **0.25**와 gradient clip **1**의 실제 구현은 호출 대상 `full_bank_train.py`가 이 스냅샷에 없어 **UNKNOWN**입니다.

W music은 현 구현을 그대로 사용합니다. Class별 목표 질량 **N/2**, cell 빈도에 따른 **max(n,20)^−1/2**, row cap **class 평균의 5배**, group cap **class 질량의 2%**입니다(`weighting.py:84–135`).

**[VERIFIED] 주의할 점은 W의 duration 보정이 항상 [0.5,2]가 아니라는 것입니다.** 조건에 따라 [0.1,10]으로 확대됩니다(`weighting.py:124–130`). 최종 weight의 실제 source 비중과 유효 표본 수를 기록해야 합니다.

### 구성 간 head ensemble은 할 가치가 있는가

**[INFERRED] 조건부로 있습니다. 기본 채택은 하지 않습니다.**

Music 두 구성의 seed 평균을 각각 \(p_{s7},p_W\)라 하면 후보는 다음 하나뿐입니다.

\[
p_M=0.5p_{s7}+0.5p_W
\]

각 head의 mean/scale을 유지하고 **최종 확률을 평균**합니다. 파라미터 평균이나 pooled logit 평균으로 바꾸지 않습니다.

**[VERIFIED] 현재 loader는 seed별 `music_{seed}.pt`만 읽습니다**(`full_bank_submission.py:61–78`). 따라서 6개 music head를 쓰려면 loader·패키지 목록·checkpoint lock·parity 경로를 함께 수정해야 합니다(`package_datav4_submission.py:29–30`, `:50–66`).

**[INFERRED: 채택 기준]**

- 더 좋은 단일 music 구성 대비 val ADS **+0.002 이상**.
- paired group bootstrap의 차이 **95% CI 하한 > 0**.
- 같은 캐시를 사용하는 두 구현의 최종 출력 최대 차이 **≤1e-6**.
- 실제 ZIP **≤10,000,000,000 bytes**.
- 기존 encoder forward 수가 늘지 않을 것.

**[UNKNOWN] 추가 head의 정확한 용량·메모리·시간은 체크포인트와 실행 측정 없이는 확정할 수 없습니다.**

## (B) 실험 계획

### 후보 수와 val 사용법

**[INFERRED] 전체 조합 27개를 돌리지 않고, 순차 비교 7개로 제한합니다.**

| 단계 | 고정하는 부분 | 비교 |
|---|---|---|
| 0 | 전체 기본값 | baseline 1개 |
| 1 | voice V0, FILE 0/9/1 | music 대안 2개 |
| 2 | 선택된 music, FILE 0/9/1 | voice V1·V2 |
| 3 | 선택된 music·voice | FILE 대안 2개 |

학습할 head는 **6종 × 3 seeds = 최대 18개**입니다. s7 DF, s7 AD, s7 WavLM, W WavLM, s7 music, W music입니다.

Val 4,931행은 lineage group을 보존해 한 번만 나눕니다.

- **Vfit 약 70% ≈3,452행:** epoch 선택, voice CDF 및 FILE CDF 적합.
- **Vselect 약 30% ≈1,479행:** 위 7개 비교.
- 정확한 행 수는 group 경계를 우선합니다. 각 component의 mask 적용 후 real/fake 수를 따로 기록합니다.
- Vselect를 확인한 뒤 split·candidate grid·threshold를 바꾸지 않습니다.
- 선택 후 전체 val로 epoch/CDF를 다시 맞추지 않고 **선택 당시 모델 그대로** 패키징합니다.

**[INFERRED] 이 분리는 이번 실험의 추가 과적합을 줄이는 장치입니다. 과거에 이미 사용한 val을 새 독립 평가로 되돌리지는 못합니다.** 기존 전체-val early stopping head를 재사용하면서 Vselect를 미사용 검증이라고 주장해서도 안 됩니다.

후보 선택 규칙은 다음으로 고정합니다.

- Vselect ADS 최대.
- 기본/현재 선택 대비 개선이 **0.002 미만**이거나 paired group-bootstrap **95% CI 하한 ≤0**이면 변경하지 않음.
- VOICE·MUSIC EER가 각각 **0.005 초과 악화**하면 탈락.
- Bootstrap **2,000회**, source·label 구성을 고려한 lineage group 재표집.
- class가 하나인 slice에는 EER를 계산하지 않음.
- 동률이면 기존 구성, 그다음 head가 적은 구성 선택.

이 기준은 **[INFERRED: 운영상 제안한 보수적 기준]**이며, 대회 성능을 보장하는 검증된 임계값은 아닙니다.

### E1/E2/E3 확인 기준

**[INFERRED] 최종 후보와 비교 기준을 잠근 뒤 한 번만 평가합니다.** 기준은 동일한 DB 학습·전처리의 s7_ref로 고정하고, 이번 기본 후보 대비 변화도 함께 기록합니다.

| 평가 | 사전 고정할 통과 기준 |
|---|---|
| E1, 4,822행 | ΔADS ≥0, paired 95% CI 하한 ≥−0.003; VOICE EER 악화 ≤0.005 |
| E1 natural, 3,816행 | ΔADS ≥−0.003 |
| E2 unlinked, 12,351행 | ΔADS ≥0, CI 하한 ≥−0.005 |
| E2 open, 2,361행 | ΔADS ≥0 |
| E2 AT-ADD, 9,990행 | ΔADS ≥0 |
| E3, 문서상 257행 | 적용 가능한 FILE/VOICE FPR 각각 ≤0.04, 기준 모델보다 false positive 증가 ≤2건 |

E3 threshold는 후보별 **Vfit fake score의 5% quantile**로 고정합니다. 기존 코드도 val TPR-95 방식으로 threshold를 정합니다(`new_eval.py:74–90`). E3에서 threshold를 조정하지 않습니다.

**[VERIFIED: 산술]** 257행 전부가 해당 지표의 대상이라면 FPR ≤0.04는 **최대 10건**, 10/257=**3.891%**입니다. 11건은 **4.280%**로 탈락합니다. Mask 때문에 분모가 달라지면 그 분모로 다시 계산합니다.

**[INFERRED] 탈락하면 E 결과를 보고 weight를 고쳐 재평가하지 않습니다. 새 후보를 중단합니다.** E1/E2를 이미 봤다는 한계도 결과에 남깁니다.

### 어느 구성에서 개선됐는지 확인하는 방법

**[INFERRED] Vselect에서 다음 순서의 ADS·FILE/VOICE/MUSIC EER 차이를 기록합니다.**

1. 기본 → music만 교체.
2. 그 결과 → voice만 교체.
3. 그 결과 → FILE만 교체.

Head가 바뀌면 CDF도 달라지므로, 각 단계에서 **기존 CDF 고정 결과**와 **Vfit에서 다시 적합한 결과**를 함께 남깁니다. 이를 각각 head 교체 효과와 calibration을 포함한 배포 효과로 구분합니다.

최종 선택 후에는 music·voice·FILE 변경을 하나씩 되돌리는 **최대 3개 val 진단**만 추가합니다. 선택을 다시 열지 않습니다. 이렇게 해도 평가 구성은 **최대 10개**입니다.

**[UNKNOWN] 현재 기록만으로는 W_new의 singing 개선을 WavLM, DF, AD 중 어느 member가 만들었는지 분리할 수 없습니다.**

### 비용과 일정

**[VERIFIED: 문서에 제시된 단가로 계산]**

- $19 / $2.98 = **6.376 H100 GPU-hours**.
- H100×1 **5시간 = $14.90**, 6시간 = **$17.88**.
- H100×4 **1.5시간 = $17.88**. 이 작업에는 검증·수정 여유가 너무 적습니다.

**[INFERRED] H100×1만 사용하고, idle을 포함해 총 6시간을 상한으로 둡니다.**

| 배정 | H100 시간 |
|---|---:|
| 준비·캐시 확인 | 0.25 h |
| 최대 18개 head 학습 | 2.50 h |
| 잠근 후보의 캐시 평가 | 0.50 h |
| 패키지 CUDA 확인 | 0.75 h |
| 수정·재확인 | 0.50 h |
| idle 여유 | 0.50 h |
| 계획 합계 | **5.00 h / $14.90** |
| 추가 예비 | **1.00 h / $2.98** |

**[UNKNOWN] 18개 head가 2.5시간에 끝나는지는 실행 기록이 없어 확정할 수 없습니다.** 첫 head들의 실측으로 예상 시간을 갱신하고, 초과하면 사전에 정한 순서대로 **W WavLM → W music·blend** 후보를 제외합니다. 기본 구성의 12개 head를 먼저 확보합니다.

문서의 **09-28 16:50 KST**를 시작점으로 잡은 제안 일정입니다.

| KST | 완료할 작업 |
|---|---|
| 16:50–17:30 | split·실제 캐시 행·학습 provenance 확인, 후보·gate 고정 |
| 17:30–20:00 | head 학습 |
| 20:00–21:00 | val 선택·기여 분석, 후보 lock |
| 21:00–21:30 | E1/E2 단회 확인, E3는 Mac에서 병행 |
| 21:30–23:00 | 평가한 head 그대로 패키징, CPU·parity 확인 |
| 23:00–00:00 | CUDA 실행, offline·순서 독립성 확인 |
| 00:00–03:00 | 결함 수정과 필요한 재검증 |
| 03:00–04:30 | 최종 archive hash·검증 기록 확정 |
| 04:30–06:00 | 예비 시간, 새 실험 금지 |

**[UNKNOWN] 이 일정으로 실제 L4 검증까지 끝낼 수 있는지는 L4 접근 가능 여부에 달려 있습니다.** H100 CUDA 확인만 끝났다면 완료 표시는 “CPU/CUDA 검증 완료, L4 시간·메모리 미검증”이어야 합니다.

## (C) 위험과 하지 않을 일

**1. [VERIFIED] 누수 제거는 ID 비교만으로 충분하지 않습니다.**  
E2에는 연결된 **75행: AT-ADD 10, OpenBMAT 65**가 기록돼 있습니다(`leak_check.txt:1`). OpenBMAT의 현재 group은 첫 music track만 사용합니다(`open_rows.py:122–129`). 여러 track을 공유하는 연결 관계를 모두 표현하지 못합니다.

**[INFERRED]** 전체 parent·source track 연결요소로 train/val/E 사이 겹침을 확인합니다. 이미 알려진 75행만 제외했다고 모든 누수가 제거됐다고 선언하지 않습니다.

**2. [VERIFIED] E2 전체 개선은 AT-ADD 비중이 큽니다.**  
Unlinked E2의 **80.884% = 9,990/12,351**가 AT-ADD입니다. W_new의 개선도 open **+0.0058**, AT-ADD **+0.0562**로 다릅니다(`leak_check.txt:4–5`).

**[INFERRED]** pooled E2 하나로 선택하지 않습니다. 이 개선을 대회 전체나 한국어 fake의 개선으로 환산하지 않습니다.

**3. [VERIFIED] 현재 parity 스크립트의 PASS는 자동 실패 차단이 아닙니다.**  
`w_runtime_parity.py:39`는 tolerance 초과 시 `FAIL`을 출력하지만 nonzero exit를 만들지 않습니다. FILE·presence도 비교하지 않습니다. CUDA 검증은 순서 독립성과 plain-max와의 차이를 확인하지만 reference와의 수치 일치를 검사하지 않습니다(`verify_s8_vessl.py:63–72`).

**[INFERRED]** 패키지 통과 조건에는 다음을 포함합니다.

- 같은 캐시 입력에서 research 경로와 packaged 경로의 **5개 출력 최대 차이 ≤1e-6**.
- 기존 오디오 parity의 VOICE **≤0.02**, MUSIC **≤0.01** 기준을 실제 실패 조건으로 연결.
- FILE·presence도 reference와 비교하고 허용치를 실행 전에 명시.
- Silence, 긴 다중-window 파일, speech-over-music을 포함한 실제 입력 검증.
- CUDA 성공과 L4 **60분·22.4 GiB** 충족을 별도로 기록.

**4. [VERIFIED] “runtime-exact features”라는 주석만으로 수치 동등성을 보장할 수 없습니다.**  
캐시는 MERT/EAT/SSLAM/WavLM/AD feature를 fp16으로 저장합니다(`dump_features.py:51–65`). Runtime은 해당 feature를 fp32로 head에 넘기는 경로가 있습니다(`full_bank_submission.py:193–216`, `:237–240`).

**[INFERRED]** 막판 precision 변경이나 quantization을 추가하지 않습니다. 현재 차이가 head·CDF·FILE 출력에 미치는 영향을 측정합니다.

**5. [UNKNOWN] 한국어 fake 일반화와 0.07 DB–LB 차이의 원인은 확인되지 않았습니다.**  
E3 real FPR이 낮아도 한국어 fake 탐지율은 알 수 없습니다. S8은 raw-mix feature와 FILE 규칙을 함께 바꿨으므로 LB 하락 원인을 한쪽으로 단정할 수도 없습니다.

**[INFERRED]** 새 한국어 fake 생성, 인코더 fine-tuning, layer sweep, pooling temperature·hidden width 탐색, 대규모 stacking, leaderboard probing은 이번 범위에서 제외합니다.

**6. [UNKNOWN] 데이터 사용·재현 제출 조건은 해결됐다고 볼 수 없습니다.**  
`LICENSES.md:14–16` 자체가 MUSDB·AT-ADD·파생 데이터의 불확실성을 기록합니다. 이 문서만으로 허용 여부를 확정하지 않습니다. 특히 해당 행을 나중에 빼고 재학습한 모델은 원래 제출 모델과 동일한 재현물이 아닙니다.

**최종 판단은 [INFERRED] 조건부 진행입니다.** 가장 먼저 학습 split을 보존하는 패키징 경로를 확보하고, 그다음 최대 7개 선택 비교로 후보 1개를 잠그는 것이 맞습니다. 성능 개선과 실제 L4 적합성은 현재 **UNKNOWN**입니다.

