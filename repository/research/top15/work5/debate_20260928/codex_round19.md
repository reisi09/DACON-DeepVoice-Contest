**(a) INFERRED — mix가 없으면 `s7_new_ob`.** CUDA 통과 조건. `s7_new`와 비교해 **OpenBMAT 400행 추가 효과**를 확인할 수 있음. 실패·미완료면 `W_new`. LB 우월성은 **UNKNOWN**.

**(b) INFERRED — LB 구간으로 recipe 효과를 추정하지 않고, 모든 구간에서 같은 우선순위 적용.**

③은 왼쪽부터 **선택 gate·패키지 검증을 모두 통과한 후보**를 사용.

| s7_new LB ADS | ② | ③ |
|---|---|---|
| ≥0.845 | W_newB | mix → s7_new_ob → W_new |
| 0.838 이상, 0.845 미만 | W_newB | mix → s7_new_ob → W_new |
| 0.834 이상, 0.838 미만 | W_newB | mix → s7_new_ob → W_new |
| <0.834 | W_newB | mix → s7_new_ob → W_new |

**INFERRED:** 낮은 첫 점수도 다른 완성 후보의 실패를 입증하지 않음. 최고점 유지 조건에서 제출 중단·새 recipe 재검토로 연결하지 않음.

**(c) section 4.0의 인과 해석은 삭제 필요.**

- **VERIFIED:** AT-ADD **34,000행 = fake 21,500 + real 12,500**, 전체 추가 행의 **84.55%**. “최신 fake 생성기만 추가한 실험”이 아님 (`code_r14/new_data_summary.csv:2–4`).
- **UNKNOWN:** LB **+0.02**가 최신 생성기 위주임을, **0 이하**가 제3의 분포임을 증명하지 않음. 데이터 추가는 real 구성·duration·학습 가중치·heads·CDF에도 영향을 줌. 첫 제출이 측정하는 것은 **추가 데이터 recipe의 순효과**.
- **VERIFIED:** **0.845−0.8373968254 = +0.0076031746**. 표의 “+0.008 이상”은 수정.
- **INFERRED:** section 4.1의 “②−①=가중치 효과”도 수정. `W_newB−s7_new`는 **가중치+early stopping 변경 효과**. `s7_new_ob−s7_new`는 **400행 추가 효과**, `mix−s7_new`는 **music recipe 교체 효과**로 읽어야 함. **③−②로 단일 효과를 분리할 수 없음.**