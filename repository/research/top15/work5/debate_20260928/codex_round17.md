**(a) INFERRED — 현재는 mix 탈락. 누락 presence를 완성한 뒤 같은 규칙으로 한 번 판정하는 것은 허용.**

- **VERIFIED:** ΔADS **+0.0013**, CI **[−0.0020,+0.0057]**로 두 조건 모두 미충족. E2 개선으로 예외 승인하지 않음 (`ROUND17_FINAL.md:10–13`).
- **VERIFIED:** 평가 대상 3,802행 중 **1,521행, 40.01%** 누락. 데이터 종류에 따른 누락이므로 현재 결과를 전체 newval 결과로 간주할 수 없음.
- **INFERRED:** 원래 대상인 **3,802개 ID**, 두 후보, CDF·threshold·bootstrap·gate를 고정하고 presence만 backfill. 중간 점수 확인 없이 전부 완료한 뒤 **ΔADS≥+0.002, CI 하한>0** 재적용. 이는 불완전한 검사의 완성이며, 별개의 독립 검증은 아님.
- 비용 추정 **0.1–0.2 H100시간 = $0.30–0.60**. 다시 실패하면 종료. `W_newB` mix로 교체하거나 기준 완화 금지.

**(b) INFERRED — 기본 제출 순서**

1. **s7_new:** E1 개선 **+0.0064**, voice 악화가 작은 기준 후보.
2. **W_newB:** E2-unlinked **+0.0467**, AT-ADD singing FILE EER **0.094**인 다른 학습 recipe.
3. **s7_new_ob:** CUDA 통과 조건. 추가 OpenBMAT **400행**, E2-unlinked **+0.0363**. 미완료면 이미 검증된 **W_new**.

완성된 newval 검사를 통과하고 패키지 검증까지 끝난 mix는 **3번을 대체**. W_new와 W_newB는 다른 checkpoints이므로, 세 번째 슬롯을 무조건 비우자는 fresh reviewer의 제안에는 반대. **LB 추가 개선 여부는 UNKNOWN**.

**(c) INFERRED — 대회 후 기록할 통합 설계와 실험**

- **Voice:** DF MLP / AD ResMLP / WavLM h8 유지. L은 DF/AD만. fusion **1/0.5/1**, seeds **41/42/43**.
- **Music:** 3840-d WindowMIL 유지. **s7 / W / 확률 50:50 평균**의 3개만 비교. 현재 W music의 최종 채택은 보류.
- **FILE:** **0/9/1 vs 1/1/1**, music과 합쳐 **6개 구성**. Presence는 우선 고정하고 broadcast 오류 기여를 분리 측정.
- **데이터:** 독립 학습용 한국어 speech **real 2,000 + fake 2,000**, fake는 최소 **4개 TTS/VC 계열**. 별도로 clean selection **6,000**, 최종 test **6,000** 확보. 각각 speech **2,000**, music/song **2,000**, VOICE/MUSIC 진위 4조합 **500개씩**. speaker·원곡·generator 계열을 split 간 분리.
- **평가:** early stopping용 val, selection, final test를 분리. final test는 **1회**. raw-mix·FILE 효과를 동시에 바꾸지 않음.
- **GPU 추정:** feature extraction **3–5시간**, head 학습 **2–4시간**, package 검증 **0.5시간**, 합계 **약 6–10 H100시간**. 데이터 확보·합성 시간과 충분한 독립 lineage 확보 여부는 **UNKNOWN**.

**(d) 남은 BLOCKING**

- **VERIFIED:** receipt 필수·device 확인·reference finite·실패 nonzero 수정 확인 (`code_r14/w_runtime_parity.py:20–22`, `:47`, `:53`; `pkg_chain.sh:16`). 이 지적은 해소.
- **INFERRED:** 기존 검증된 **s7_new / W_new / W_newB를 막을 추가 BLOCKING은 이번 자료에서 발견하지 못함**.
- **BLOCKING:** mix는 완성된 선택 검사와 새 패키지 검증 전 제출 불가. `s7_new_ob`는 CUDA 검증 완료 전 제출 불가.
- **UNKNOWN:** H100 검증과 별개로 L4 전체 **60분·22.4 GiB** 충족 여부는 이 자료만으로 확정할 수 없음.