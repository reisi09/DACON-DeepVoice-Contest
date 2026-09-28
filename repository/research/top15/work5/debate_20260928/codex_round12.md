**(a) 차단 항목과 최소 수정**

1. **[VERIFIED] §1–2: 최종 cap 적용 후 class·stratum 합계 보장이 빠졌습니다.** 단순 clipping은 목표 질량을 깨뜨립니다. **class 합·stratum 합·row/group cap을 동시에 만족**시키고, 50회 안에 지정 오차로 수렴하지 않으면 실패 처리하십시오. C에도 class 합 제약이 필요합니다.  
   또한 원래 DB train은 real voice **7,452행**, real `constructed_mix`는 최대 **59행**이므로 row cap 아래 SM 최대 질량은 `5×59/7452=3.96%`입니다. **SM 목표 5%는 필터 전부터 불가능합니다.** 실제 admitted rows에서 확인하고, 불가능하면 이미 정한 C fallback을 실행하십시오. ([SYNTHESIS_R3.md:14](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/SYNTHESIS_R3.md:14))

2. **[VERIFIED] §1: FILE의 “no refit”이 모호합니다.** voice-member CDF와 FILE candidate CDF는 별개입니다. 최소 수정은 **“FILE 계수만 `0/9/1`로 고정하고, FILE candidate CDF는 새 head의 val 예측으로 기존 절차대로 재적합”**이라고 명시하는 것입니다. ([10–11행](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/SYNTHESIS_R3.md:10))

3. **[INFERRED] §4: source 이름과 분리된 stem만으로 D1의 component 라벨을 보장할 수 없습니다.** fake 파일의 vocal이 반드시 fake인 것은 아니며, 분리 누출 때문에 `fakeV+realM`의 음악에도 AI 반주가 남을 수 있습니다. **component mask·생성 provenance로 donor를 선별하고, partial/미상·비가창·반대 성분 누출이 있는 donor는 제외하거나 해당 라벨을 mask 처리**하십시오. ([42–45행](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/SYNTHESIS_R3.md:42))

4. **[VERIFIED] §5: 제출 분기가 겹치고 일부 후보가 정의되지 않았습니다.** `+0.001…+0.003`은 두 조건에 해당하며, slot 1이 D1 fallback이면 “other W/C”가 불명확합니다. **서로 겹치지 않는 구간과 slot-1 후보별 대응표를 고정**하고, “extension of the best”는 사전 정의한 유한 후보 목록으로 바꾸십시오. 모든 fallback에도 §3 gate를 적용해야 합니다. ([51–52행](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/SYNTHESIS_R3.md:51))

**(b) 필수 누락**

- **[VERIFIED] D1의 DB/P1 경계:** DB 학습 donor는 original train만 사용하십시오. original test donor의 remix는 **DB 모델 평가용**이며, train+test로 학습한 P1의 독립 검증이 아닙니다. 새 donor audit이 없으면 **“독립 검증 없음”**으로 기록하면 됩니다.
- **[VERIFIED] 단일 클래스 지표:** scout 302행의 `realV+fakeM VoiceEER`는 정의되지 않습니다. **고정 val threshold의 VOICE FPR**로 바꾸고, 모든 subset에서 클래스 부족을 자동 통과로 처리하지 마십시오.
- **[UNKNOWN] 조건부 자료의 원자료별 학습·2차 제출 권한:** 다운로드 완료나 “무혼합”만으로 해결되지 않습니다. 사용자가 확인한 범위만 편입하고, **D1 생성 파일·donor 계보·재현 정보 보존**을 §4에 추가하십시오.