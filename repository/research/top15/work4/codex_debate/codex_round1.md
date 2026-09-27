**S6 우선 제출은 합리적이지만, S7의 규칙 위반을 긍정적인 CI로 정당화하거나 FILE 평균 차이를 결함으로 단정하는 데에는 반대합니다.**

VERIFIED는 제공 파일에서 확인했다는 뜻이며, 원 실험을 재실행했다는 뜻은 아닙니다. 학습·추론은 실행하지 않았습니다. 아래 ‘보고서’는 `MODEL_ARCHITECTURE_ENSEMBLE_DATA_AUDIT_20260928.md`입니다.

1. **S7: 사전 규칙은 실패, 탐색 제출은 별도 판단**

   **VERIFIED:** `locks/s6_encoders_lock.json.rule.C`는 `dADS >= +0.005`와 **모든 seed의 component EER 개선**을 동시에 요구합니다. 기준선도 S6가 아닌 S5-equivalent입니다. `results/rt_s6enc_report.json.boot`의 개선은 각각 `S5+both vs S5 = +0.00934`, `S6+both vs S6 = +0.01044`, 후자의 CI는 `[0.00427, 0.01523]`입니다.

   보고서:23은 seed 42 실패를 인정합니다. 다만 정확한 `+0.0079`는 seed별 원본이 없어 독립 확인은 **UNKNOWN**입니다. [rt_s6enc.py:37](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/rt_s6enc.py:37)은 seed 예측을 먼저 평균하며, 이후 bootstrap합니다. 이 CI는 seed 안정성이나 P1 재학습의 불확실성을 검증하지 않습니다.

   **INFERRED:** 최고점 유지 조건에서는 S7을 *규칙 미통과 탐색 후보*로 제출할 수 있습니다. 예외 기록은 정직한 설명이지 규칙 통과의 대체 증거가 아닙니다. 결과를 본 뒤 seed를 늘려 “5개 중 4개”로 바꾸는 보고서:304의 제안도 사후 기준 변경입니다.

   **CPU 검사 ≤30분:** 기존 seed별 예측으로 동일 ID·mask에서 EER 차이를 재계산하고 원래 C를 그대로 적용합니다. seed 42가 실제로 개선된다면 실패 기록이 오류이고, 악화한다면 C 실패가 확정됩니다.

2. **제출 순서: S6 → S5 → S4, 다음 날 S7은 조건부**

   **INFERRED:** `2026-09-28`에는 **S6 → S5 → S4**를 권합니다. 준비된 최대 구성부터 공식 성능과 실행 가능성을 확인하되, 같은 날 L을 제외한 S5와 EAT도 제외한 S4를 확보하는 순서입니다. 세 개가 모두 제출된다면 순서 자체의 가치는 제한적입니다.

   **VERIFIED:** S5−S4는 RT `+0.01216`, CI `[0.00484, 0.01971]`입니다(`results/rt_check.json.delta_S5_vs_S4.RT`). S6−S5는 `+0.00204`, CI `[-0.00074, 0.00584]`로 불확실합니다(`results/report2__base-L-S-LS.json.delta_vs_base_test.L.RT`). 따라서 Claude가 S6 우선을 **확정적인 성능 우위**로 설명한다면 과장입니다.

   `2026-09-29` 첫 슬롯은 실제 런타임 연결과 패키지 검증을 마친 S7에 배정합니다. 나머지 두 슬롯은 실패 제출 수정이나 별도로 검증한 후보용으로 남깁니다. 동일 패키지 반복 제출로 여섯 칸을 채울 이유는 없습니다.

   **CPU 검사 ≤30분:** L의 잠금 조건을 JSON에 그대로 적용합니다. 이번 확인에서는 AC는 통과하지만 RT는 `val_completed.voice_eer: 0.03354 → 0.03456` 때문에 실패했습니다. S6를 유일한 안전 후보로 취급할 근거가 약해집니다.

3. **FILE: 평균 차이는 결함 증거가 아닙니다**

   **VERIFIED:** MUSIC은 sigmoid 점수 평균이고, FILE은 후보별 val CDF 값의 평균입니다([decision_heads.py:70](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/decision_heads.py:70)). 따라서 `0.711`과 `0.41~0.44`는 직접 비교할 척도가 아닙니다. FILE에 `0.5`라는 정답 판정 임계값도 보장되지 않습니다.

   **INFERRED:** 보고서:154의 “음악 증거의 순위를 제대로 물려받지 못한다”는 결론은 평균만으로 도출할 수 없습니다. 실제 순위 손실은 **UNKNOWN**입니다. 고정된 presence에서 현재 후보들은 MUSIC 증가에 따라 감소하지 않지만, 파일마다 다른 presence가 순위를 해칠 가능성은 남습니다.

   최소 변경 후보는 이미 있는 `max_nopresence` 하나입니다. `FILE=max(FILE,MUSIC)`처럼 척도가 다른 값을 직접 합치거나 새 OOF 결합기를 만드는 단계로 바로 넘어가지 않겠습니다.

   **CPU 검사 ≤30분:** 기존 val 예측에서 현재 FILE과 고정한 `max_nopresence`를 비교합니다. 전체 FILE EER과 realV+fakeM 양성 대 고정된 real 음성군의 순위를 측정하고, 전체 성능이 악화하거나 해당 순위가 개선되지 않으면 기각합니다. **양성 86개만으로 EER은 계산할 수 없습니다.** val의 해당 양성이 5행뿐이라는 보고서:281의 한계도 유지해야 합니다.

4. **가장 불신하는 것은 선택 이후 CI와 LB 전이입니다**

   **VERIFIED:** AC의 낙관 편향은 상쇄되는 상수가 아닙니다. `results/rt_check.json.results`에서 FILE EER의 RT−AC 차이는 S4 `0.00372`, S5 `0.00165`입니다. S5의 개선 방향은 유지되지만, “모두 같은 영향을 받았다”는 설명으로 delta의 정확성을 보장할 수 없습니다.

   더 큰 문제는 동일 held-out을 반복해 선택에 사용했다는 보고서:180과, P1이 그 데이터를 학습에 합친다는 [train_backbone_heads.py:260](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/train_backbone_heads.py:260)입니다.

   **INFERRED:** 고정된 예측을 재표집한 CI는 후보 탐색, 사후 예외, 새 생성기 분포, P1 재학습까지 포함하지 않습니다. 특히 작은 L 개선과 S7의 LB 개선량 외삽을 불신합니다. `~0.09` 차이만으로 “LB 데이터가 더 낯설다”고 원인을 확정한 보고서:62도 근거가 부족합니다.

   **CPU 검사 ≤30분:** 기존 예측에서 unseen generator를 하나씩 제외하고 고정 후보의 delta를 재계산합니다. 특정 생성기 하나를 제외했을 때 부호가 뒤집히면 광범위한 일반화 개선 주장을 기각합니다. 이는 새 후보 선택이 아닌 민감도 감사로만 사용합니다.

5. **단일 최고 기대가치 행동: 준비된 S6의 공식 제출**

   **INFERRED:** 지금은 S6를 제출해 실제 점수와 실행 결과를 확보하는 것이 우선입니다. 추가 로컬 실험보다 남은 시간의 의사결정을 직접 바꿉니다.

   top-15 달성 가능성은 **UNKNOWN**입니다. 단순히 `0.09` 차감 가정을 적용해도 S7의 `0.90921`은 `0.81921`이며 목표 ADS `0.86`에 못 미칩니다. 이는 예측값이 아니라, 현재 개선량만으로 목표 도달을 주장할 수 없다는 산술입니다.

   새 미세조정, 교차 스템 Transformer, held-out을 보며 seed 채택 규칙을 바꾸는 작업은 중단하겠습니다. **VERIFIED:** `results/ftC1_gate.json.boot_vs_S4["R2|RT"]`는 `−0.004`, CI `[-0.00627, -0.00051]`입니다. Claude의 후속 실험 목록은 이 마감에서 실행 우선순위로 쓰기에는 넓습니다.

   **CPU 검사 ≤30분:** S6 ZIP의 manifest·실제 멤버 해시·설정·의존성 명세를 대조해 제출물 정체와 누락을 확인합니다. 실패하면 제출 우선 후보를 완전한 패키지로 바꿉니다. 이 검사는 L4 시간·메모리를 인증하지 않습니다.

6. **보고서에서 바로 고칠 사실·논리 오류**

   **VERIFIED:** 가장 큰 준비 상태 문제는 S7 런타임입니다. trainer는 SSLAM·WavLM을 지원하지만, [full_bank_submission.py:120](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/full_bank_submission.py:120)은 MERT+EAT만 결합하고, `:144`는 `ours/ad`만 전달합니다. **INFERRED:** 그대로 S7 head를 넣으면 음악 특징 차원 불일치가 예상되며, WavLM은 `component_fusion.fuse`의 누락 멤버 제외 동작 때문에 조용히 빠질 수 있습니다. 패키징 외에 런타임 구현이 필요합니다.

   **VERIFIED:** 보고서:56의 “처음 보는 생성기 1,135개”는 잘못된 단위입니다. `unseen_generators_in_test`는 **6종**, `1,135`는 fake FILE 행 수입니다. DB 결과도 전체 `6,450`행이 아니라 `rows.test=4,822`인 부분집합입니다.

   보고서:197의 “HuBERT는 B 단계 전 탈락”도 `results/ads_report_S6encoders.json.val`의 HuBERT 가중치 `0.25/0.5/1` 평가와 모순됩니다. 보고서:302의 E1은 parity·용량 검사로 “S7 P1이 S6보다 낫다”를 검증하려 하지만, 그 검사는 탐지 품질 우위를 입증하지 못합니다.

   **CPU 검사 ≤30분:** trainer가 내보내는 설정과 런타임 소비 경로를 정적으로 대조합니다. 이번 AST 확인에서는 `music_sslam_layers.json`, `voice_wavlm_layer.json` 소비가 모두 없었습니다.

원본 행별·seed별 예측과 ZIP은 이 폴더에 없어, 이를 요구하는 추가 검사는 제안으로 남겼습니다.

