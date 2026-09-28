**(a) 슬롯**

- **[VERIFIED]** S7 대비 E1/E2 변화: `s7_new +0.00641/+0.02851`, `W_new +0.00332/+0.04508`, `W_newB +0.00322/+0.04697`. `W_ref −0.00284/−0.00865`는 제외합니다. ([측정표](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/ROUND13_DATA_SCALE.md:8))
- **[INFERRED] 순서:** **① s7_new → ② W_newB → ③ 아래 검증을 통과한 소규모 데이터 보강 후보**. 세 번째가 없으면 `W_new`. `W_newB`는 `W_new`보다 E2 **+0.00189**, E1 **−0.00010**입니다.

**(b) AI-Hub 규모**

- **[INFERRED] 주력 후보에는 추가 0행.** 별도 pilot 상한은 **71544 100곡, 473 100개, 465 100개**입니다. **71767·71470·123/571/100/87은 이번 추가 0행**으로 둡니다.
- **[VERIFIED]** E3에서 71544 FPR은 모든 후보 **0**, 465는 새 데이터 후보 모두 **0**입니다. 473은 `s7_new 0.027`, `W_new/B 0.040`입니다. 대규모 real 증량을 뒷받침하는 오류량은 아직 작습니다.
- **[INFERRED] 5,000개 remix는 보류합니다.** 한국어 real vocal에 AI 반주를 붙여도 **VOICE는 real**입니다. 외국어 SVC vocal에 한국어 반주를 붙여도 한국어 fake voice가 되지는 않습니다. **real mass 25% cap만으로 언어 shortcut을 해결할 수 없습니다.**
- pilot을 한다면 **각 head의 최종 real loss 중 AI-Hub 합계 ≤10%**로 제한합니다. 한국어 fake가 필요한 검증에는 실제 한국어 SVC/TTS를 사용해야 합니다.
- **[UNKNOWN]** LB 효과와 Mac 학습 시간. 측정 속도 기준 **300×30초 추출은 약 8.3분/H100**, **5,000×30초는 약 139분**이며 전송·설정·학습은 별도입니다.

**(c) 추가 데이터 우선순위**

- **[INFERRED] 우선순위:** **기존 후보 패키징 → AI-OpenBMAT 보강 → 조건부 AI-Hub pilot**.
- AI-OpenBMAT은 미사용 계보의 **real/AI 200쌍=400행**만 우선 추가합니다. 현재 FILE EER **0.336–0.354**로 가장 큰 잔여 약점입니다. 동일 방송 원본·음악 donor의 기존 newtest 연결을 제외합니다.
- **AT-ADD eval pool 추가 학습 0행.** 현재 unseen-generator 평가를 유지합니다. 추가한다면 train/dev의 미사용 계보만 사용합니다.
- **HAIM B3/B4/B6 추가 0행:** 현재 fake TPR **≥0.995**입니다.
- **[UNKNOWN]** 시간당 LB 이득은 수치화할 근거가 없습니다.

**(d) 슬롯 전 추가 확인**

- **[VERIFIED] E2의 `10,000/12,426=80.5%`가 AT-ADD입니다.** E2 전체 개선을 모든 새 도메인의 개선으로 해석하지 않습니다.
- **[INFERRED] 기존 예측으로:** ① source-family별 결과, ② AT-ADD 제외 결과, ③ 계보 단위 paired bootstrap을 계산합니다. 한 클래스만 남는 component EER은 **UNKNOWN** 처리합니다.
- **[INFERRED] 추가 후보 gate:** 대응 기준 후보 대비 **E1 ΔADS≥0**, 유효한 **E2 비AT-ADD ΔADS≥0**, 핵심 source의 **EER 악화≤0.01 / real FPR 증가≤3%p**.
- **[INFERRED] ID·원곡·가수·donor의 train/newval/newtest 교차 중복 0**을 확인하고, 새 audit은 P1에서도 제외합니다.
- **[VERIFIED]** E3는 real-only입니다. **E1+E3 ADS 상승이나 E3 FPR=0으로 한국어 fake 탐지 개선을 주장할 수 없습니다.**