### (a) 반대 사항과 최소 수정

- **[INFERRED] P-a는 채택합니다.** 기존 P1 head 조합은 가능합니다. DB 결과는 **recipe의 근거**, P1 패키지는 **추가 데이터를 학습한 최종 모델**로 구분하면 됩니다. P1의 E1/E2 점수를 독립 성능으로 제시하면 안 됩니다. 제 이전 **val 재분할·2.5 GPU시간 재학습 제안은 철회**합니다.

- **[VERIFIED] P-b의 4후보 비교에는 문제가 있습니다.** W_newB는 **newval 3,841행을 early stopping에 사용**했습니다(`ROUND16_CROSS.md:29`). 따라서 newval을 미사용 선택 데이터로 삼으려면 후보를 **① s7_new ② s7_new voice/W_new music**, 두 개로 줄여야 합니다. W_newB와 그 혼합형은 제외합니다. 실제 lineage 겹침·캐시 누락을 제거한 공통 평가 행 수를 확정해야 합니다.

- **[INFERRED] 선택은 위 두 DB 모델의 newval 결과로만 합니다.** 제안 기준은 혼합형의 **ΔADS ≥+0.002**, paired group bootstrap **2,000회, 95% CI 하한 >0**입니다. 미충족이면 s7_new 유지. Fusion **1/0.5/1**, FILE **0/9/1**, CDF 적합 데이터는 old-val로 고정합니다.

- **[INFERRED] 이미 본 E1/E2로 선택하는 것은 개발 판단으로는 가능하지만, P-b의 guard가 선택 편향을 제거하지는 않습니다.** 이번에는 확인·보고만 하고 후보 순위를 바꾸지 않는 편을 권합니다.  
  **[VERIFIED: 문서 수치]** W_newB 혼합형의 s7_new 대비 차이는 E1 **약 +0.0039**, E2-all **약 +0.0104**입니다. **[UNKNOWN] 이 두 후보 간 paired CI**는 제시되지 않았습니다. s7_ref 대비 CI를 대신 사용할 수 없습니다.

- **[INFERRED] P-c는 제출 순서만 수정합니다.** **s7_new → W_newB → 검증을 마친 혼합형**. LB 결과로 구성을 수정하지 않습니다. 혼합형이 탈락하면 **2회만 제출**하고 중복 후보로 슬롯을 채우지 않습니다.

### (b) 혼합 패키지에 필요한 검증

1. **[VERIFIED] `--reuse-voice`만으로 DF 출처가 고정되지는 않습니다.** `--voice-arch copy`이면 DF는 `--ours`에서 복사합니다(`code_r14/train_backbone_heads.py:438–440`). **`--ours`도 s7_new P1로 명시**하고, DF/AD/WavLM **9개**와 music **3개**, 총 **12개 checkpoint hash**를 원본과 대조해야 합니다.

2. **[INFERRED] 각 head의 mean/scale·feature 순서·fake index를 보존**하고 혼합된 P1 출력으로 old-val CDF를 다시 맞춥니다. FILE 계수 **0/9/1**은 명시적으로 고정합니다(`train_backbone_heads.py:530–550`). 동일 입력에서 voice는 s7_new, music은 W_new 원본 경로와 **최대 차이 ≤1e−6**인지 확인합니다.

3. **[VERIFIED] parity 수정은 확인했지만 receipt 검사는 아직 선택 사항입니다.** `w_runtime_parity.py:20`은 `RECEIPT`가 있어야 검사하며, `pkg_chain.sh:16`은 이를 전달하지 않습니다. **RECEIPT 전달을 필수화**하고 offline 계산값에도 finite 검사를 추가해야 합니다. 현재 finite 검사는 runtime 출력만 대상입니다(`w_runtime_parity.py:42–46`).

4. **[INFERRED] 새 혼합 archive 자체를 재검증합니다.** VOICE/MUSIC/FILE 오차 **≤0.02/0.01/0.01**, 정확한 ID 집합·중복 없음·실패 nonzero·offline CUDA·순서 독립성을 확인합니다. FILE parity는 runtime presence를 재사용하므로 presence 일치도 별도로 확인합니다(`w_runtime_parity.py:44`).  
   **[UNKNOWN] 기존 3개 패키지의 통과가 새 혼합형이나 L4 시간·메모리까지 증명하지는 않습니다.**

### (c) 최종 계획

**[INFERRED] 재학습 0시간, runtime 구조 변경 0건, 신규 패키지 최대 1개.**

- **17:15–18:30:** 두 DB 후보의 newval 평가·선택. 후보와 판정 기준 고정.
- **18:30–20:00:** 선택 recipe의 기존 P1 heads 조합, CDF 재적합, 패키징·CPU parity.
- **20:00–21:00:** 새 archive의 CUDA 검증. **21:00 완료 목표**.
- **04:00 수정 종료, 06:00 최종 마감**. 실패하면 기존 검증 패키지 사용.

**[INFERRED: 비용 추정]** H100 **0.2시간 × $2.98 = $0.596**, CPU **$0.30** 가정 시 총 **약 $0.90**. 예비 포함 상한은 GPU **1시간** + CPU **2시간**, 총 **$3.58**입니다. **[UNKNOWN] 실제 소요 시간과 L4 적합성은 실행 전 확정할 수 없습니다.**