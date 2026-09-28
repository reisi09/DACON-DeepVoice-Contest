1. **두 기각을 유지하는 것이 타당합니다. [VERIFIED: 판정식 대조]**

   - **X1:** 잘못된 N2 판정은 제외해야 합니다. 그래도 목표 FPR 개선은 `−0.0042`로 요구치 `−0.01`에 못 미치고 CI가 0을 포함합니다. 저-Pm FILE FNR도 `+0.1549` 악화해 별도로 탈락합니다. (`round4/x1_gates.json:60`, `round4/x1_gates.json:130`)
   - **X2:** T1 `ΔJ=+0.05797`, 세 seed 모두 양수로 T1·S가 실패합니다. N1·N2도 실패하므로 T2 개선만으로 채택할 수 없습니다. (`round4/x2_gates.json:17`, `round4/x2_gates.json:139`)
   - **[UNKNOWN]** 행별 예측이 없어 EER·bootstrap 자체를 독립 재계산하지는 못했습니다. 확인 범위에서 기각을 뒤집을 오류는 보이지 않습니다.

2. **잠금셋을 채점하지 않고 보존하는 결정은 맞습니다.** 채택 자격을 얻은 후보가 없으므로 이를 후보 구제나 재선택에 사용하지 마십시오. 보고 문구는 **“이번 DB 개발 평가에서 사전 등록한 채택 기준을 충족한 후보가 없었다”**가 정확합니다. T2 등 일부 지표는 개선됐으므로 “어떤 개선도 없었다”는 표현은 피해야 합니다. (`round4/x2_gates.json:32`, `round4/ROUND4_DECISION.md:11`)

3. **보고서에는 다음을 남기십시오.**

   - **[VERIFIED]** X1은 최종 평가기 수정 전 결과이며, N2를 기각 근거에서 제외했음을 명시합니다. (`round4/ROUND4_DECISION.md:6`)
   - **[REPORTED]** X2 T1의 CI는 0을 포함합니다. 따라서 “유의한 악화 입증”이 아니라 “개선 기준 미충족”입니다. N1도 “비열등성 미확인”입니다. (`round4/x2_gates.json:7`, `round4/x2_gates.json:17`)
   - **[VERIFIED]** E3는 저장된 집계 FPR이 동일합니다. 행별 예측 동일성까지 뜻하지 않습니다. (`round4/e3_s7_new_vs_s7_bc_aggregate.json:9`, `round4/e3_s7_new_vs_s7_bc_aggregate.json:34`)
   - 슬롯 3의 W_new 유지는 **기존 계획으로 복귀**한 것이며, 이번 실험으로 W_new의 우월성을 검증한 것은 아닙니다. 잠금셋은 **NOT RUN**으로 기록하십시오. (`round4/ROUND4_DECISION.md:11`)