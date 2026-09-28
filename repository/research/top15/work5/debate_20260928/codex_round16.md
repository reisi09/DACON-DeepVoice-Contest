**(a) 판단**

- **INFERRED — P-a 채택.** 기존 P1 heads 재사용, 재학습 **0회**, runtime 변경 **0개**. DB로 recipe를 평가한 뒤 P1으로 최종 학습하는 것은 허용 가능. 다만 **P1 자체의 E1/E2 일반화 성능은 미측정**. fresh reviewer의 DB heads 강제 패키징·**2.5 GPU시간 재학습은 철회 권고**.
- **VERIFIED — 내 L 누수 지적은 실제 실행에 대한 결론을 철회.** 새 기록의 stage A는 DF/AD+L, stage B는 `label_completion=False`로 WavLM/music 학습. 공통 함수만 보고 실제 호출 조합을 놓쳤음 (`ROUND16_CROSS.md:13–14`).
- **BLOCKING · VERIFIED — P-b의 newval 독립성 주장이 틀림.** `W_newB`와 이를 포함한 hybrid는 **newval을 early stopping에 사용**했음 (`ROUND16_CROSS.md:29`). **최소 수정:** 미사용 newval 비교는 `{s7_new, mix:s7_new/W_new}` **2개로 제한**. W_newB 계열은 이미 본 자료를 이용한 개발 비교로 별도 표기.
- **INFERRED — E1/E2를 이용한 후향적 개발 선택은 가능.** 그러나 **CI 하한 >0이 선택 편향을 제거하지는 않음**. newval만 사용해야 하는 절대 규칙은 없지만, 이번에는 위 2개를 newval로 선택하고 E1/E2는 고정된 veto로 사용 권고.
- **VERIFIED — 현재 hybrid B의 직접 개선점은** s7_new 대비 E1 **+0.0039**, E2 전체 **+0.0104**. 표의 CI **[+0.0036,+0.0162]는 s7_ref 대비**이므로 s7_new 대비 유의성 근거로 사용할 수 없음 (`ROUND16_CROSS.md:20–24`). **직접 paired CI 재계산 필요.**

**(b) hybrid 검증에 추가할 것**

- **VERIFIED — SHA 검사가 아직 선택사항.** `w_runtime_parity.py:20`은 `RECEIPT`가 있을 때만 검사하고, `pkg_chain.sh:16`은 이를 전달하지 않음. **필수 인자로 변경**, 검증 device도 assert.
- **INFERRED — 12개 heads의 출처를 고정:** DF/AD/WavLM **9개는 s7_new**, music **3개는 선택한 W**. 원본·작업 폴더·ZIP 내부 checkpoint SHA를 각각 대조. 기존 폴더 재사용으로 남은 optional JSON도 배제.
- **VERIFIED — FILE 강제 고정 필요:** `--file-weights 0,9,1`; 없으면 재탐색함 (`train_backbone_heads.py:548–550`). fusion **1/0.5/1**, feature 순서 **MERT6_12→EAT16→SSLAM5**, WavLM **h8**, presence 동일성 확인.
- **INFERRED — 독립 기준값:** 같은 cached features에서 원본 P1 heads를 직접 조합한 출력과 패키지의 **5개 출력 차이 ≤1e−6**. 오디오 CPU/CUDA parity는 **V≤0.02 / M≤0.01 / F≤0.01**.
- **VERIFIED — reference finite 검사가 빠짐.** `w_runtime_parity.py:43`은 runtime만 검사함. **reference의 fv/fm/ff도 finite assert**. FILE reference가 runtime presence를 재사용하므로 presence parity는 별도 확인 (`:44`).
- **UNKNOWN:** H100 PASS만으로 L4 **60분·22.4 GiB** 충족이 새로 입증되지는 않음.

**(c) 최종 계획 — INFERRED**

- **17:30–19:30:** CPU로 위 **2개** newval 비교. hybrid 채택 기준 **ΔADS≥+0.002**. E1/E2-unlinked 각각 **ΔADS≥0**, E2-open **≥−0.002**, E3 FPR 악화 **≤0.02**를 veto로 고정. 탈락 후 다른 hybrid 순차 탐색 금지.
- **19:30–22:00:** 선택 recipe의 기존 P1 heads 조립·CDF 재계산·CPU 검증.
- **22:00–23:00:** CUDA 검증. 계획 **0.2 GPU시간/$0.596**, CPU workspace **약 $0.30**, 합계 **약 $0.90**. 수정 포함 GPU 상한 **0.5시간/$1.49**.
- **제출:** 통과하면 **hybrid → W_newB → s7_new**. 탈락하면 **s7_new → W_newB**, 세 번째는 이미 검증한 `s7_new_ob`가 준비된 경우만 사용. **04:00 변경 종료, 06:00 최종 검증 마감.**