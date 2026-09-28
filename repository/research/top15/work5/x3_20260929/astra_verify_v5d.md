**BLOCKING — 5c의 2번·4번은 아직 완전히 해결되지 않았습니다.**

1. **[VERIFIED] P1 PGID 기록이 잘못될 수 있습니다.** `cd … && setsid … &`의 `$!`은 중간 셸 PID일 수 있습니다. 같은 셸 구조에서 기록 PID와 실제 새 PGID가 다른 것을 재현했습니다. 새 세션 내부에서 실제 PGID를 기록해야 합니다. 현재 `kill -- -$g`는 P1 종료에 실패할 수 있습니다. (`round5d/after_db_x3.sh:8`; `round5d/x3_mac_chain.sh:11`)

2. **[VERIFIED] 예산 조회 실패가 통과로 처리됩니다.** `bal()`은 실패하면 `999`를 반환하며, 이때 budget 검사도 PASS함을 재현했습니다. 또 대기 조건이 즉시 성공하면 예산·기한 검사를 건너뛰고, E3·locked scoring·패키징 중에는 상한을 감시하지 않습니다. 조회 실패 시 중단하고 장시간 단계에도 예산·시간 제한을 적용해야 합니다. (`round5d/x3_mac_chain.sh:9`, `:22`, `:43`, `:62`, `:69`)

3. **[VERIFIED] 해시 기록과 원래 잠금 검증이 아직 다릅니다.** 음원은 현재 CSV의 `audio_sha256`과만 비교하고, 그 값과 `source`는 기존 LOCK의 검증 대상에 없습니다. 따라서 음원과 CSV 해시가 함께 바뀌거나 `source`가 바뀌어도 기존 잠금 검사를 통과합니다. 채점 시 생성하는 기록만으로는 이를 검출하지 못합니다. 사전에 고정한 manifest와 ID별 음원·특징 연결을 대조해야 합니다. (`round5d/score_locked.py:35`, `:125`, `:127`, `:130`; `round4/LOCK_v2.json:38`)

**[VERIFIED] 해결 확인:** G6의 FAIL→HOLD→PASS 우선순위는 분리 실행으로 확인했습니다. 정확한 PASS만 진행하는 조건, 주요 단계 종료 코드 확인, CPU 검증 실패 중단, silent 기준 fusion도 수정됐습니다. (`round5d/score_locked.py:113`, `:66`; `round5d/x3_mac_chain.sh:67`; `round5d/pkg_chain.sh:15`)

**[UNKNOWN]** 실행 중이던 v1이 실제로 v2로 재시작됐는지는 제공 자료로 확인할 수 없습니다. 또한 “이전 fusion 오류는 반드시 FAIL만 만들었다”는 주장은 성립하지 않습니다. 작은 차이는 허용오차 안에서 PASS할 수 있으므로 수정본의 parity 확인이 필요합니다. (`round5d/ROUND5D_VERIFY.md:5`, `:21`; `round5d/w_runtime_parity.py:52`)