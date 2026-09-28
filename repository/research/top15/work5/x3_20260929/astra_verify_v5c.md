**BLOCKING — 잠근 셋 채점 전에 수정이 필요합니다.**

1. **[VERIFIED] G6의 실패·판정 불가가 채택 경로를 통과합니다.** 한 층의 CI가 undefined이면 다른 층이나 ADS의 명백한 실패도 `regression check only`로 덮습니다. Mac 체인은 `FAIL` 문자열만 차단하므로 undefined뿐 아니라 결과 읽기 실패로 생긴 빈 값도 통과합니다. 분리 실행으로 재현했습니다. **정의된 실패→FAIL, 나머지 undefined→HOLD, 두 기준 모두 PASS일 때만 진행**하도록 바꿔야 합니다. (`round5c/score_locked.py:130`; `round5c/x3_mac_chain.sh:35`; `round5c/design_locked_excerpt.md:14`)

2. **[VERIFIED] ‘한 번만, 후보 고정 후’가 보장되지 않습니다.** 출력 파일 존재 여부만 검사하므로 다른 출력명·동시 실행·결과 저장 전 실패 후 재실행이 가능합니다. `LOCK=NONE`도 잠근 행을 거부하지 않습니다. 후보 head 내용과 `source`·음원·NPZ 연결도 잠금 검증에 포함되지 않습니다. **후보와 입력 해시를 고정하고, 출력명과 무관한 평가 사용 기록을 예측 전에 원자적으로 생성**해야 합니다. (`round5c/score_locked.py:29`, `:100`, `:108`, `:136`)

3. **[VERIFIED] 실패가 정상 완료처럼 처리됩니다.** 평가 실패 후에도 `after_db done`을 출력하며 기존 gate JSON을 재사용할 수 있습니다. locked scorer와 CUDA 검증의 종료 코드도 강제하지 않아 실패 후 `X3 chain complete`까지 도달합니다. 실행별 새 상태·결과를 사용하고, 각 단계의 실제 종료 코드와 완료 증거를 확인해야 합니다. (`round5c/after_db_x3.sh:9`; `round5c/x3_mac_chain.sh:14`, `:34`, `:40`)

4. **[VERIFIED] 무한 대기와 정지 실패가 가능합니다.** 두 대기 루프에 마감 시간이 없고, SSH 장애나 완료 표시 없는 프로세스 종료를 충분히 처리하지 않습니다. P1의 일반 문자열 `done`도 성공 증거로 부족합니다. pause 명령 실패는 `tail`에 가려진 채 종료 코드 0으로 끝납니다. **기한·정확한 작업 상태·실제 paused 상태를 확인**하고, 종료 대상도 작업 소유 PID로 제한해야 합니다. (`round5c/x3_mac_chain.sh:7`, `:12`, `:32`)

5. **[VERIFIED] VOICE 예측에 runtime과 다른 분기가 있습니다.** scorer는 `fv==0`이면 fusion을 생략하지만 runtime은 `silent`일 때만 생략합니다. 비무음 파일에서 DF 확률이 0이면 AD/WavLM 기여가 사라집니다. `0. if silent else fuse(...)`로 맞춰야 합니다. parity 스크립트에도 같은 오류가 있습니다. (`round5c/score_locked.py:60`; `round5c/w_runtime_parity.py:41`; `runtime_pkg/DataV4_s7_new/model/runtime/full_bank_submission.py:232`)

**[NOT RUN]** 잠근 셋·학습·원격 체인은 실행하지 않았습니다. 제공된 E2 132행 결과는 실제 패키지와의 예측 일치를 입증하지 않습니다. 수치 오차로 판정이 바뀔 경우 실제 `script.py` 예측을 사용한다는 설계 조건도 지켜야 합니다. (`round5c/design_locked_excerpt.md:12`)