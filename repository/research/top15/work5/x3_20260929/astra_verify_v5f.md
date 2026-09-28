**BLOCKING — 1번은 코드상 해결됐고, 2·3번은 보완이 필요합니다.**

1. **[VERIFIED] 특징 비교가 잘못된 PASS를 허용합니다.** `presence=[0.2,0.4]`와 `[0.4,0.8]`은 최대 차이가 0.4인데도 cosine 기준을 통과합니다. NaN도 비교에서 거부되지 않고 `min_cosine=1.0`으로 남는 경로를 재현했습니다. 유한값 검사와 사전 지정한 절대·상대 오차 검사가 필요합니다. (`round5f/locked_recheck.py:43`, `:45`)

2. **[VERIFIED] 재검증 기록이 실제 NPZ 내용에 연결되지 않습니다.** 기록에는 비교한 NPZ 해시가 없고, scorer는 PASS와 manifest 해시만 확인합니다. 따라서 캐시가 바뀌어도 이전 PASS를 사용할 수 있습니다. 비교한 원본·재추출 NPZ 해시를 기록하고 채점 직전에 대조해야 합니다. **[UNKNOWN] 실제 캐시 변경 여부는 확인하지 못했습니다.** (`round5f/locked_recheck.py:47`; `round5f/score_locked.py:131`)

3. **[VERIFIED] 잔액 조회 중 기한이 지나면 늦게 시작할 수 있습니다.** 기한 검사 뒤 `budget()`을 기다리고 바로 실행합니다. 시작 부분을 분리해 확인한 결과, 지연된 잔액 검사 후 기한을 넘겨 자식이 실행됐습니다. `budget()` 반환 후, 실행 직전에 기한을 다시 확인해야 합니다. (`round5f/x3_mac_chain.sh:39`)

**[VERIFIED] 해결 확인:** 별도 세션과 그룹 전체 TERM/KILL, guarded 다운로드, 잔액 조회의 30초 timeout은 반영됐습니다. 제공된 재검증 기록의 manifest·30행 표본 ID 해시도 재계산 결과와 일치합니다. (`round5f/x3_mac_chain.sh:11`, `:36`, `:44`, `:65`; `round5f/locked_recheck.json:2`)

**[NOT RUN]** 원격 체인·특징 재추출·잠근 셋 채점은 실행하지 않았습니다.