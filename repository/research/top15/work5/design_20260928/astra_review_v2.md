**판정: X1을 먼저 진단하는 순서는 유지하되, 현재 문서 그대로는 후보 채택과 G6 통과를 결정할 수 없습니다.** 아래 사항을 실험 전에 고정해야 합니다. 수치는 직접 재계산한 경우만 VERIFIED로, 감사자가 재현했다고 보고한 수치는 REPORTED로 구분했습니다.

1. **F2는 의도된 동작이며, 결함 여부는 아직 HYPOTHESIS입니다.**

   **[VERIFIED]** `max_nopresence`는 코드에 명시된 “presence 오류에 대한 방어”입니다. `noisy_or`는 이미 `Pm·Fm`을 사용합니다. 따라서 F2는 구현 오류라기보다 **음악 부재 시 오탐과 작은 음악을 놓칠 위험 사이의 설계 문제**입니다. 음악 없는 real 파일의 Fm 분포와 오라클 개선은 REPORTED 진단 근거이며, 실제 Pm 게이트의 개선을 입증하지 않습니다.  
   근거: `code/decision_heads.py:70`, `round2/code_audit.md:154`.

   **[HYPOTHESIS·설계 권고]** 세 후보는 적절한 최소 비교군입니다. 다만 세 번째 규칙이 FILE 전체를 대체하는지, 마지막 10% 항만 바꾸는지 명확히 해야 합니다. 후자로 고정하는 것이 효과를 분리하기 좋습니다.

   - A: 현행 `0.9·CDF(noisy_or) + 0.1·CDF(max(Fv,Fm))`
   - B: `CDF(noisy_or)` — `0/10/0`
   - C: `0.9·CDF(noisy_or) + 0.1·CDF(max(Fv,Pm·Fm))`

   **[VERIFIED]** C에는 변경된 함수로 적합한 별도 CDF가 필요합니다. 기존 `max_nopresence` 표를 그대로 사용하면 함수와 표의 의미가 달라집니다. A까지 newval에서 CDF를 다시 적합한다면 현행 모델과의 차이에 재보정 효과도 포함됩니다. 변경하지 않은 현행 A를 채택 판단의 기준으로 남기십시오.  
   근거: `round2/DESIGN_KO.md:173`, `code/decision_heads.py:88`, `code/data_v4/file_ensemble.py:37`.

   **[VERIFIED]** DB `s7_new`가 사용하지 않은 newval에서 적합·선택하는 것은 **개발 목적에는 유효**합니다. 다만 같은 행으로 적합하고 선택한 결과는 독립 성능 증거가 아니며, newval은 이전 mix 선택에도 사용됐습니다. 적합과 선택의 분리를 주장하려면 계보 그룹으로 분리해야 합니다. 최종 P1 `s7_new`는 newval까지 학습하므로 DB용 CDF의 이전 방법과 P1용 재적합 자료도 미리 정해야 합니다.  
   근거: `code/data_v4/train_backbone_heads.py:76`, `code/data_v4/train_backbone_heads.py:369`, `code/research/top15/work5/newdata/select_newval.py:39`.

   **[ESTIMATE·실험 전 고정할 보호 기준]**

   | 보호 층 | 측정과 허용 악화 |
   |---|---|
   | 음악 없는 real 음성 | 후보별 calibration TPR 95% 문턱에서 FILE FPR 증가 ≤ 0.01 |
   | 실제 음악이 있는 방송형·혼합 | FILE EER 증가 ≤ 0.010; 저-Pm 부분집합의 FILE FNR 증가 ≤ 0.01 |
   | realV+fakeM | FILE·MUSIC FNR 및 VOICE FPR 증가 ≤ 0.01. 이 층 자체의 EER은 계산하지 않음 |
   | 음악만·노래·전화·순차 혼합 | 해당 라벨이 있는 FILE·성분 EER 증가 ≤ 0.010 |

   **[ESTIMATE]** 저-Pm 구간은 예를 들어 `Pm≤0.1`, `0.1<Pm≤0.5`로 먼저 고정하십시오. 실제 저음량·짧은 음악은 Pm만으로 정의하지 말고, 가능한 경우 음악/음성 active-RMS 비와 음악 점유율로 별도 구분해야 합니다. **[UNKNOWN]** 현재 패킷만으로 이 음량·시간 구간의 표본 수는 확인할 수 없습니다. 후보마다 같은 calibration 자료와 TPR 목표로 문턱을 계산해야 하며, 동일한 숫자 문턱을 모든 후보에 적용하면 척도 변화가 오탐 변화로 섞입니다.  
   근거: `round2/DESIGN_KO.md:131`, `code/research/top15/work5/newdata/new_eval.py:74`, `code/research/top15/work5/newdata/open_rows.py:128`.

2. **실험 순서는 유지하되, G1–G7의 다음 부분은 수정해야 합니다.**

   **[REPORTED / VERIFIED 계산]** 오라클 FILE EER 개선 `0.0741→0.0716`은 ADS 약 `+0.00125`에 해당합니다. FILE만 바꾸는 X1이 G1의 `+0.005`를 만족하려면 FILE EER이 `0.010` 개선돼야 합니다. 오라클 수치를 모든 규칙의 엄밀한 상한으로 볼 수는 없지만, 현재 근거는 X1의 큰 개선을 뒷받침하지 않습니다. **[HYPOTHESIS·권고]** X1은 저비용 진단으로 먼저 수행하되, 결과를 보고 G1을 낮춰 채택하지 마십시오.  
   근거: `round2/code_audit.md:159`, `round2/DESIGN_KO.md:129`, `history/research/top15/work/lane1/official_text.txt:134`.

   **[VERIFIED]** 다음은 실제 수용 결정을 바꾸는 미정 사항입니다.

   - **G1·G2:** “자연 분포”의 행 목록·가중치와 대조 모델을 고정해야 합니다. X1/X2의 대조는 문서대로 DB `s7_new`로 명시하십시오. 혼합 비율 격자의 간격과 중첩 층 처리도 없습니다. EER은 비선형이므로 층별 ADS를 단순 가중평균하지 말고, 각 격자의 행 가중치로 ROC/EER을 다시 계산해야 합니다.
   - **G3·G4:** CSV를 재집계해 **newval·E2의 realV+fakeM은 모두 0행**임을 확인했습니다. OpenBMAT도 VOICE_PRESENT·VOICE_FAKE가 모두 미상이라 이를 대체하지 못합니다. 핵심 보호 층의 “판정 불가”는 **통과가 아니라 채택 보류**로 명시해야 합니다.
   - **G5:** X1은 결정적 규칙입니다. 동일 앙상블을 세 번 실행하는 검사가 아니라, 기존 세 seed 각각에서 같은 멤버·보정 절차를 적용한 대조/변경의 paired 차이를 확인해야 합니다.
   - **G6:** 아래 3번처럼 점추정치 조건을 CI 조건과 구분해야 합니다.
   - **G7:** §5.1은 CUDA 검증을 요구하지만 X1 표와 §9는 코드 변경 때만 요구합니다. 최종 산출물에 적용할 조건을 하나로 통일해야 합니다.

   근거: `round2/DESIGN_KO.md:38`, `round2/DESIGN_KO.md:129`, `round2/DESIGN_KO.md:180`, `round2/DESIGN_KO.md:192`; CSV 재집계와 일치하는 기록: `round2/code_audit.md:200`.

   **[VERIFIED·X2 실행 전 차단 사항]** X2 표의 조건은 `방송형 J −0.02`, `새 패널 CI >0`, `E1·E2 ΔADS≥−0.002`인데, §9는 X2에도 G1–G5를 요구합니다. `J`의 정의, 새 패널의 역할, 두 조건 집합을 모두 요구하는지가 불명확합니다. **선택에 사용하는 새 패널은 locked set일 수 없습니다.** 이를 먼저 확정해야 합니다.  
   근거: `round2/DESIGN_KO.md:181`, `round2/DESIGN_KO.md:189`.

3. **제안된 locked set은 후보로는 타당하지만, 현재 상태로 G6에 충분하지 않습니다.**

   **[VERIFIED / UNKNOWN]** 모든 사용 음악 트랙을 제외하는 것은 기존 “첫 트랙만 사용” 문제를 개선합니다. 그러나 이것만으로 방송 원본·음성·화자·다른 donor와의 독립성이 확보되지는 않습니다. 실제 공유 여부는 UNKNOWN입니다. 제외 대상은 bank뿐 아니라 **newtrain/newval/E2, 추가 400행, 이전 진단·선택 자료, X2의 모든 donor 및 파생물**까지 포함해야 합니다. 새 remix가 잠근 자료와 연결되지 않도록 이 제외 목록을 X2 생성에도 적용해야 합니다.  
   근거: `round2/DESIGN_KO.md:48`, `round2/DESIGN_KO.md:187`, `code/research/top15/work5/newdata/openbmat_extra.py:17`, `code/research/top15/work5/newdata/open_rows.py:191`.

   **[VERIFIED]** OpenBMAT만으로는 VOICE EER과 전체 ADS를 산출할 라벨이 부족합니다. archive 행을 추가하더라도 각 성분의 두 클래스, 보호 층, 독립 계보 그룹이 실제로 남는지 먼저 확인해야 합니다. ID 해시뿐 아니라 오디오·라벨·계보·제외 목록을 함께 고정해야 합니다. **[UNKNOWN]** 아직 만들지 않은 셋의 적정 표본 수와 CI 폭은 확인되지 않았으며, “클래스별 20그룹”은 `0.002` 비열등 마진을 판정할 수 있다는 근거가 아닙니다.  
   근거: `round2/DESIGN_KO.md:45`, `round2/DESIGN_KO.md:52`, `round2/DESIGN_KO.md:57`, `code/research/top15/work5/newdata/open_rows.py:128`.

   **[VERIFIED·통계적 구분]** `ΔADS≥−0.002`라는 **점추정치만으로 비열등성이 입증되지는 않습니다.** G6를 비열등성 관문으로 유지한다면, 사전 고정한 paired 계보 bootstrap에서 다음을 요구하십시오.

   - S7과 s7_new **각각에 대해** ΔADS의 95% CI 하한 ≥ −0.002.
   - 필수 보호 층의 EER 악화 95% CI 상한 ≤ 0.015.
   - 단일 클래스·표본 부족·정의되지 않는 CI는 통과 처리하지 않음.

   **[HYPOTHESIS·권고]** 남은 자원으로 이를 측정할 수 없다면 G6를 “제한된 신규 자료에서의 회귀 점검”으로 낮춰 명명하고, 비열등성 입증이라고 보고하지 마십시오. 최종 후보 하나만 고정하고 두 기준 패키지와 한 번 비교하는 절차 자체는 유지할 만합니다.  
   근거: `round2/DESIGN_KO.md:134`, `round2/DESIGN_KO.md:191`.

4. **추가 BLOCKING 사항은 최종 검증 경로와 제출 정책입니다.**

   **[REPORTED]** 새 parity 기록은 s7_new의 FILE 최대 오차 `0.00011`, W_new의 `0.0002`, 각각 `n=10 PASS`를 보여 줍니다. 따라서 기존의 “FILE·표본 수 기록 부재” 문제는 해소됐습니다.  
   근거: `round2/runtime_parity_s7_new_rerun.txt:11`, `round2/runtime_parity_W_new_rerun.txt:11`.

   **[VERIFIED]** 다만 현재 parity는 런타임의 Pv/Pm을 오프라인 FILE 계산에 그대로 넣습니다. 따라서 X1이 의존하는 **캐시 presence와 런타임 presence의 일치**까지 검증하지 않습니다. 또한 공유 특징은 fp16으로 저장되므로 10파일의 확률 오차 허용치가 G6의 ADS `0.002` 마진을 보증하지 않습니다. **[HYPOTHESIS·권고]** 최종 후보의 실제 패키지에서 낮은 Pm·짧은 음악 사례를 포함해 presence와 FILE 경로를 확인하고, 수치 차이로 G6 판정이 바뀔 수 있는 경우 실제 런타임 예측으로 판정해야 합니다.  
   근거: `code/research/top15/work5/w/w_runtime_parity.py:46`, `code/research/top15/work5/newdata/dump_features.py:53`, `round2/DESIGN_KO.md:64`.

   **[UNKNOWN·제출 전 차단 사항]** 팀 최고 점수와 마지막 제출 중 무엇이 최종 반영되는지 확인되지 않았습니다. 이 상태에서 “LB 적응은 순위에 해가 없다”거나 ①②③을 순서대로 제출해도 안전하다고 결론 내릴 수 없습니다. 이는 읽기·CPU 진단의 차단 사항은 아니지만, 실제 제출 전에 해소해야 합니다.  
   근거: `round2/DESIGN_KO.md:17`, `round2/DESIGN_KO.md:193`, `round2/DESIGN_KO.md:200`.

**[NOT RUN]** 이번 검토에서는 X1/X2, 모델 추론, locked set 생성, CUDA/L4 검증을 실행하지 않았습니다. 폴더 안의 코드·기록을 대조하고 CSV만 읽기 전용으로 재집계했습니다.