**BLOCKING**

- **[VERIFIED] 선택량 부족을 막지 않습니다.** ZIP 매칭·필터링 결과가 0개여도 CSV를 쓰고 학습으로 넘어갑니다. 메타데이터 `filename`과 ZIP의 `stem`을 같은 형식으로 맞추고, 알고리즘별 750개·총 1,500개·고유 lyrics ID를 확인한 뒤 부족하면 중단해야 합니다. (`round5/sonics_rows.py:32`, `:37`, `:53`; `round5/run_x3_db.sh:32`)
- **[VERIFIED] 크롭의 보컬 존재가 검증되지 않습니다.** 곡 전체의 `no_vocal=False`만으로 30% 지점의 20초 구간까지 `VOICE_PRESENT=1`이라고 확정할 수 없습니다. 실제 크롭에서 확인하거나, 불확실한 행은 제외하거나 해당 라벨을 unknown으로 처리해야 합니다. **[UNKNOWN] 실제 오라벨 규모는 음원이 없어 확인하지 못했습니다.** (`round5/sonics_rows.py:33`, `:63`, `:66`)
- **[VERIFIED] 현재 평가셋 그대로라면 X3 채택은 불가능합니다.** singing 보호 슬라이스의 real 그룹은 11개여서 FILE·VOICE 모두 계속 HOLD이고, 최종 판정은 이를 거절로 처리합니다. 실행 전에 진단 실험으로 한정하거나 독립 real 그룹을 확보해야 합니다. 결과를 본 뒤 HOLD를 PASS로 바꾸면 안 됩니다. (`round4/x2_gates.json:112`; `round5/x2_gates.py:249`, `:341`)

1. **코드·특징 — [VERIFIED]** 크롭 시작점 계산은 맞지만, `.wav` 경로에 FLAC을 저장하므로 확장자와 `codec_chain`이 틀립니다. 출력명을 `.flac`으로 맞추십시오. 또 runner는 `presence`만 검사하므로 학습에 쓰는 encoder 특징의 누락·비유한값을 통과시킬 수 있습니다. 필요한 특징 키와 값도 검증해야 합니다. (`round5/sonics_rows.py:61`; `round5/open_rows.py:47`, `:64`; `round5/run_x3_db.sh:19`)

2. **누수 — [VERIFIED]** bank 제외 정규식은 기존 SONICS 1,600행 모두에 매칭됩니다. DB 경로도 `train`만 추가하므로 E1/E2 행을 직접 학습에 넣는 코드는 발견하지 못했습니다. **[UNKNOWN] 간접 중복 배제는 미확인입니다.** ID를 문자열 그대로 비교하므로 bank의 `02205`와 metadata의 `2205`는 서로 다르게 취급됩니다. 실제 metadata·선택 목록이 없으므로 ID 정규화 후 bank/E2 계보와 교집합이 없는지 확인해야 합니다. 현재 생성 행의 원본·음원 해시는 비어 있습니다. (`round5/sonics_rows.py:29`, `:42`; `code/data_v4/train_backbone_heads.py:81`; `round5/open_rows.py:42`)

3. **T3 — [VERIFIED]** 개발 실험의 사전 지정 목표로 타당합니다. 구현은 E1의 **voice+music 전체**를 사용하며, 순수 노래만의 평가가 아닙니다. `candidate−control ≤ −0.01`, 그룹 CI 상한 `<0`, 클래스별 20그룹, seed 3개 중 2개 음수의 방향도 맞습니다. 실행 전에 슬라이스 ID·그룹 수·대조군·판정 규칙을 고정하십시오. 반복 활용한 E1에서의 개선을 독립 최종 일반화 증거로 해석해서는 안 됩니다. (`round5/x2_gates.py:265`, `:272`, `:275`, `:328`, `:339`)

**[NOT RUN]** X3 학습·추론·음원 검증은 실행하지 않았습니다.