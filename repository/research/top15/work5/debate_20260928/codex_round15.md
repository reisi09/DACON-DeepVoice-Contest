**(A) INFERRED — 통합 후보는 `s7_new` voice + 선택한 music heads + 고정된 FILE 규칙으로 구성. W를 모든 head에 적용하지 않음.**

**VERIFIED:** W_new의 E1 VOICE EER는 **0.06116 → 0.07913**, ADS 기여 **−0.00359**. 반면 누출 연결 75행을 제외해도 E2 이득은 유지됨. 따라서 **데이터 추가는 유지하고, W의 music 효과를 분리해 검증**하는 것이 우선임.  
근거: `code_r14/new_eval_newtest.json`, `code_r14/leak_check.txt:1–5`.

| 구성 | 기본값 | 허용할 후보 |
|---|---|---|
| Encoders | S7 그대로 | 추가·raw-mix forward 없음 |
| Voice | `s7_new` DF/AD/WavLM, 각 3 seeds | architecture·pooling·weights 변경 없음 |
| Voice fusion | **1 / 0.5 / 1**, old-val CDF | 탐색 없음 |
| Music | 기존 **3840-d WindowMIL**, localization **0.25**, 기존 optimizer | `s7_new`, `W_new`, 두 recipe의 **50:50 확률 평균** |
| FILE | **0/9/1** | **1/1/1**만 비교 |
| Presence | 기존 head 유지 | 재학습 없음 |
| Seeds | **41/42/43** | 5 seeds 확장 없음 |

- **INFERRED — music head ensemble은 시도할 가치 있음.**  
  \[
  F_m=\tfrac12\operatorname{mean}_{s}(p_{\text{s7},s})+
      \tfrac12\operatorname{mean}_{s}(p_{\text{W},s})
  \]
  **각 head의 mean/scale과 pooling을 유지한 뒤 확률을 평균**. checkpoint parameter 평균 금지. 현재 runtime도 head별 확률 평균 구조임 (`code_r14/full_bank_submission.py:213–231`).
- 변경은 **music 전용 checkpoint 목록·가중치 로더**로 제한. 전역 `SEEDS`를 늘리면 voice까지 바뀜 (`full_bank_submission.py:64`). packager의 checkpoint lock·manifest에도 추가 heads 포함 (`package_datav4_submission.py:50–67`). **추가 archive 예산 ≤100 MB**, encoder 재계산 **0회**.
- **INFERRED — 데이터:** DB에는 새 데이터의 **train split만** 추가. **40,214는 전체 가용 행 수이지 DB 추가 학습 행 수가 아님**. 원래 summary의 새 train은 **23,955행**, 실제 수는 실패 행을 제외한 manifest로 확정 (`new_data_summary.csv:2–13`, `train_backbone_heads.py:76–86`). 추가 OpenBMAT **400행은 이번 통합 탐색에서 제외**해 가중치·FILE 효과와 분리.
- Voice는 기존 s7의 **class 균형 → source/generator cell 균등 → duration balance** 유지. Music의 W 후보만 현재 구현인 **`max(n_cell,20)^−0.5`**, duration balance, **row 5× / group 2% cap** 사용. 새 가중치 공식은 추가하지 않음 (`weighting.py:84–134`).

**선행 수정 — VERIFIED:** 문서는 “L은 DF/AD만”이라고 하지만 P1은 L을 적용한 공통 rows를 WavLM에도 전달함 (`train_backbone_heads.py:355`, `:474`). DB WavLM 경로에는 L 적용이 없음 (`run.py:261–268`). **DF/AD용 completed rows와 WavLM용 original rows를 분리하고, 재사용 heads의 실제 학습 설정도 확인해야 함.**

**(B) INFERRED — 최대 6개 구성, 최종 후보 1개만 확인 평가.**

1. **17:00–18:00: recipe와 평가 행 고정, GPU 0시간.**  
   기준은 `s7_new` DB 모델. train/val/newval/E1/E2의 ID·labels·features hash 고정. 누출 연결 **75행 제외한 E2 12,351행** 사용. 기존 P1 heads로 E1/E2를 평가하지 않음.

2. **18:00–20:00: val 탐색, GPU 최대 1시간 = $2.98.**  
   Music **3개 × FILE 2개 = 6개**만 비교. 이미 저장된 DB heads 재사용; 없는 newval 예측만 계산. Voice predictions·fusion은 모든 구성에서 동일하게 유지.

   선택 점수:
   \[
   J=0.5\,ADS_{\text{old-val}}+0.5\,ADS_{\text{newval}}
   \]
   CDF tables는 **old-val만으로** 구성별 재계산. checkpoint 선택은 기존 old-val 방식 유지. newval은 추가 구성 선택용으로 사용하며 학습에 넣지 않음.

   채택 조건은 기준 대비 **ΔJ ≥ +0.002**, old-val·newval 각각 **ΔADS ≥ −0.001**. val의 MUSIC EER **악화 ≤0.005**. source 하나씩 제외한 비교에서도 **ΔJ ≥ −0.002**. 성립하지 않으면 기본 구성 유지. **J 차이 <0.001이면 heads가 적은 구성, 다음으로 0/9/1 우선.**

   **UNKNOWN:** 실제 평가 가능한 old-val 수는 4,931보다 작을 수 있음. frozen/cache 교집합과 component별 유효 label 수를 별도로 기록.

3. **20:00–20:30: 선택한 후보 하나의 E1/E2/E3 확인.**  
   다음을 **모두** 만족해야 통합 후보 채택. 기준은 `s7_new`.

   - E1: **ΔADS ≥ −0.001**, **ΔMusicEER ≤ +0.005**.
   - E2 unlinked의 AT-ADD / non-AT-ADD 각각: **ΔADS ≥ −0.002**.
   - **E1 또는 E2 non-AT-ADD에서 ΔADS ≥ +0.002**. AT-ADD만 좋아지면 통합 후보로 채택하지 않음.
   - 유효 label **50개 이상**인 real-source별 FPR 악화 **≤0.03**. threshold는 후보별 old-val TPR95에서 고정.
   - E3 FILE: 추가 오탐 **≤5/257**, 총 오탐 **≤15/257**. VOICE는 유효 label 행에서 FPR 악화 **≤0.02**.
   - Voice를 고정했으므로 DB VOICE predictions 차이 **≤1e−7**. 초과하면 recipe/행 정렬 오류부터 확인.

   group bootstrap **2,000회**로 paired CI도 보고하되, 반복 사용한 E1/E2의 CI를 신규 독립 검증으로 해석하지 않음. 실패 후 runner-up을 같은 held-out에 순차적으로 대입하지 않음.

4. **20:30–23:00: 선택 recipe의 P1 refit, 최대 2.5 GPU시간 = $7.45.**  
   recipe·FILE 계수는 변경 금지. 최종 heads에 맞춰 old-val CDF만 재계산. P1에서 E1/E2를 학습에 포함한다면 독립 평가는 종료된 것으로 기록 (`train_backbone_heads.py:349–372`).

5. **23:00–04:00: 패키징·CPU parity·CUDA 검증.**  
   CUDA **1시간 $2.98**, 수정 예비 **0.5시간 $1.49**. **04:00 수정 종료, 06:00 검증 완료**. 목표 총 **5 GPU시간/$14.90**, 절대 상한 **6시간/$17.88**. 시간은 계획값이며 초과 시 이미 검증한 후보로 복귀. H100x4 사용 없음.

**Attribution:** val에서 `music만 변경`, `FILE만 변경`, `둘 다 변경`의 차이를 분리. interaction은  
**Δjoint − Δmusic-only − ΔFILE-only**로 보고. held-out ablation 추가 금지.

**제출:** **00시 이후 s7_new → W_newB → 통합 후보**. 모두 완성 모델로 제출하고 LB에 따라 recipe를 다시 조정하지 않음. 통합 후보가 탈락·미완료면 사전에 검증을 끝낸 `s7_new_ob`를 대안으로 두며, 그것도 미충족이면 중복 제출로 슬롯을 채우지 않음.

**(C) 수정할 해석과 중단할 작업**

- **VERIFIED:** 누출 제거는 효과를 없애지 않았음. W_new E2 **+0.0451 → +0.0442**, s7_new **+0.0285 유지**. 다만 OpenBMAT는 **65/161 = 40.37%**가 연결 행이었음. 전체 제거율 **0.60%**만 보고 누출이 작다고 결론 내리면 안 됨 (`leak_check.txt:1–5`).
- **VERIFIED:** E2 unlinked도 AT-ADD **9,990/12,351 = 80.88%**. **UNKNOWN:** 이 비중이 DACON과 유사한지, +0.0442가 LB에 전이되는지.
- **VERIFIED:** E1/E2/E3 결과를 이미 여러 번 확인했으므로 문서의 “held-out once”는 **이번 후보의 추가 확인 1회**만 뜻함. 미사용 test로 되돌릴 수 없음 (`ROUND15_INTEGRATED_DESIGN.md:40–49`).
- **VERIFIED:** “Korean FAKE 없음”은 코드와 충돌. HAIM B6 `kr`에 **language=ko, VF=1, MF=1**을 부여함 (`open_rows.py:76–79`). **한국어 fake speech 부족**과 **한국어 fake 전체 부재**를 구분해야 함.
- **INFERRED:** CPS 포화만으로 presence를 무시하면 안 됨. presence는 FILE의 `Pv*Fv`, `Pm*Fm`에도 직접 들어감 (`code/decision_heads.py:71–73`). 다만 오늘은 재학습 대신 위 FILE 2개 비교로 제한.
- **UNKNOWN:** 새 문서는 패키지 검증 완료라고 하지만 제공된 `w_runtime_parity.py:39`는 여전히 실패를 출력만 함. 최종 artifact의 **SHA/device/IDs/finite/FILE 포함, 실패 nonzero** 검증 증거가 필요함.
- **INFERRED:** raw-mix 재확장, encoder fine-tuning, attention-MIL, max→mean 변경, 5 seeds, 대규모 합성·신규 다운로드, FILE GBM 재탐색은 중단. 남은 **ADS 약 +0.0226**을 이번 조합이 확보한다는 근거는 **UNKNOWN**.