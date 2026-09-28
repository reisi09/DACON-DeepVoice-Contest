**제 판단은 새 CV 때문에 S7 제출을 미루지 않고, E2′는 폐기하며, 이번 P1 패키지의 FILE 가중치는 `0/9/1`로 유지하는 것입니다.** 다만 새 CV가 원래의 per-seed lock 실패를 지워주지는 않습니다. `0/9/1`도 비열등성이 입증돼서 유지하는 것이 아니라, 이미 정한 val 선택 절차의 결과라서 유지합니다.

아래 **VERIFIED**는 제공된 파일·코드에서 확인했다는 뜻입니다. 새 브리핑의 실험 수치를 원시 예측으로 재계산하지는 않았습니다.

## A. CV 설계

### 세 설계는 서로 다른 질문에 답합니다

| 설계 | 과학적으로 답할 수 있는 질문 | 답할 수 없는 질문 | 판정 |
|---|---|---|---|
| **(1) Purged leave-source-family-out** | 특정 family를 학습에서 제외해도 S7의 이득이 남는가? | 임의의 미래 generator나 LB에서 같은 이득이 나는가? | **우선 추천하는 source-shift 스트레스 평가** |
| **(2) Lineage group K-fold** | 관측된 source 구성 안에서 group 배치를 바꿔도 이득이 남는가? | 새로운 source로의 전이가 안정적인가? | **split sensitivity 진단으로 유효** |
| **(3) 기존 held-out + generator exclusion + seeds** | 현재 split에서 이득이 특정 generator·초기화에 집중됐는가? | 다른 train/test 분할에서도 이득이 남는가? | **마감 전 의사결정에는 충분히 선택 가능한 방안** |

**[INFERRED] “S7 gain이 split luck인가?”를 좁게 해석하면 (2)가 가장 직접적입니다.** 단, 한 번의 K-fold는 fold별 이질성까지 섞어 측정합니다. 분할 배치 자체의 민감도를 보려면 서로 다른 사전 고정 partition이 필요합니다. 새로운 source에 대한 우연한 적합이 걱정이라면 (1)이 더 관련 있습니다.

이번 E2′가 구체적인 반례입니다. held-out 내부 GBM은 **5개 fold 모두 FILE EER가 개선**됐지만, val에서 학습한 배포 방식은 ADS가 **−0.0057**이었습니다. 내부 group CV의 일관된 양수 결과만으로 source 전이를 주장할 수 없습니다. [briefing_round6.md:20](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/results/briefing_round6.md:20)

### 83% component를 정확히 해석해야 합니다

**[VERIFIED] `26,016/31,270`은 source들을 연결한 component입니다. 단일 lineage group이 전체의 83%라는 뜻은 아닙니다.** 실제 cross-source lineage group은 `1,376/14,786`입니다. [briefing_round5.md:30](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/results/briefing_round5.md:30), [briefing_round6.md:14](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/results/briefing_round6.md:14)

따라서 family A의 test row와 lineage를 공유하는 B의 row는 purge하되, 그 연결 하나 때문에 **B 전체와 B에 연결된 모든 source 전체**를 제거할 필요는 없습니다. 그 방식이 원래 CV를 붕괴시켰습니다.

설계 (1)에서는 다음 경계를 지켜야 합니다.

- **Family 정의:** source 이름과 generator family는 다릅니다. `mlaad`를 제외했다고 모든 관련 generator를 학습에서 제거한 것은 아닙니다. source holdout인지 generator holdout인지 명시하고, alias·파생 mix의 처리도 metadata로 고정합니다.
- **Purge 범위:** test와 fixed real pool에 연결된 전체 lineage group을 **학습과 inner-val 모두**에서 제거합니다. 해당 family의 real row를 학습에 남겨두면 corpus 단서까지 생소한 조건은 아닙니다.
- **학습량 변화:** purge 뒤 학습량·class·source 구성이 크게 달라지면, 결과는 family 배제와 학습 지원 감소가 합쳐진 효과입니다. S6/S7에 동일한 잔여 데이터를 써도 이 해석상 한계는 남습니다.
- **Fixed real pool:** EER는 fake family와 어떤 real 분포를 비교하느냐에 의존합니다. speech-only real pool로 music family를 평가하면 source·presence 구분이 authenticity 성능처럼 보일 수 있습니다. real speech, real instrumental, real vocal music의 출처·길이·채널 구성을 미리 고정해야 합니다.
- **미정의 지표:** fake-speech family + real pool에는 MUSIC_FAKE 양성이 없을 수 있습니다. 이때 MUSIC EER와 ADS는 **미정의**입니다. `0`으로 채우거나 나머지 항의 가중치를 재정규화하면 공식 ADS가 아닙니다.
- **반복 real 사용:** 같은 real pool을 쓰는 fold들은 상관됩니다. bootstrap에서도 동일 real group의 재표집 가중치를 모든 fold에 함께 적용해야 합니다. fold 수를 독립 반복 횟수로 세면 안 됩니다.

### 내가 새로 사전 등록할 규칙

**[INFERRED — 제안] 이번에는 광범위한 CV 재설계보다, 아래의 작은 paired source-shift 평가를 등록하겠습니다.** 기존 `cv_lock.json`은 폐기 사유를 남기고 새 버전으로 구분합니다.

1. **평가 모집단:** 원래 R2 `train+val`만 사용하고 기존 held-out은 추가 편입하지 않습니다. 그래도 S7 자체가 과거 val/held-out 결과로 선택됐으므로, 새 독립 검증이라고 부르지는 않습니다.

2. **ADS를 계산할 수 있는 세 panel:** 예를 들어  
   `mlaad + SONICS`,  
   `speechfake + FakeMusicCaps`,  
   `asvspoof + HAIM + Echoes/generated_music`.  
   이는 단일-family fold에서 **명시적으로 변경한 설계**입니다. 실제 source alias, real-pool ID, purge 후 ID, inner-val ID를 성능 계산 전에 확정합니다. family별 component 지표도 따로 보고합니다.

3. **실행 가능성 선검사:** 각 panel의 outer-test에서 세 component 모두 양·음성이 있어야 하고, purge 뒤 학습·inner-val에도 필요한 class가 남아야 합니다. 잔여 row/group/class 수를 먼저 공개합니다. 조건이 안 맞으면 해당 ADS 설계는 `INFEASIBLE`입니다. 결과를 본 뒤 panel을 합치거나 빼지 않습니다.

4. **학습·평가:** seeds `41/42/43`, S6/S7에 동일한 split과 학습 recipe를 적용합니다. normalization, 학습된 presence, component heads, early stopping, CDF, FILE fit 모두 outer-test를 제외해 적합합니다. L은 학습 row에만 적용하고 outer metric은 원래 label/mask를 씁니다. RT로 모든 필요한 member를 계산합니다.

5. **주 판정:** 각 panel에서 3-seed ensemble의  
   \[
   \Delta_j=ADS(S7)_j-ADS(S6)_j
   \]
   를 계산합니다. **panel 평균 `≥ +0.003`, 세 panel 중 최소 두 개 양수, 어떤 panel도 `< −0.003`이 아니면 추가 지지 근거로 인정**합니다. 이 수치는 새 운영 기준이며 통계적 상수가 아닙니다. 실패는 “추가 지지 불충분”이지 곧바로 S7 열등의 증명은 아닙니다.

6. **불확실성:** panel별 지표를 먼저 계산하고 평균합니다. 서로 다른 모델의 OOF 점수를 합쳐 하나의 EER를 주 지표로 삼지 않습니다. shared real group을 보존한 paired bootstrap을 붙이되, 그 CI는 고정된 panel·학습 결과에 조건부인 값입니다. per-seed 결과는 모두 공개하고 원래 lock 실패 기록을 유지합니다.

원래 lock은 이미 알려진 S6/S7 결과 뒤에 작성됐고, `train+val+test`를 합치는 설계였습니다. 어떤 대체 CV도 지금까지의 모델 선택 과정을 소급해 독립적으로 만들 수 없습니다. [cv_lock.json:3](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/locks/cv_lock.json:3)

**제출은 이 평가를 기다리지 않겠습니다. [INFERRED]** 약 `11,900`개 voice-only row의 EAT feature 누락은 실제 RT 전제조건입니다. MUSIC label이 없어도 FILE 계산에는 music score가 필요하므로 zero-fill이나 누락 row 제거로 우회할 수 없습니다. 기존 예측에 fold 번호만 붙여도 새 OOF 예측은 생기지 않습니다.

`6–8 folds / CPU 1–1.5 h`는 **ESTIMATE**입니다. 매 fold의 전체 재적합과 feature 보충 시간을 포함하는지 확인되지 않았습니다. S7의 패키지 검증을 늦출 만큼 확실한 비용 추정은 아닙니다. 브리핑 시점에는 CPU offline verification도 아직 진행 중이므로, 그 완료 여부와 새 CV의 필요성을 구분해야 합니다. [briefing_round6.md:28](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/results/briefing_round6.md:28)

## B. FILE 변경은 멈추고 `0/9/1`을 유지합니다

**[VERIFIED] E2′의 배포 방식은 실패했습니다.** GBM은 FILE EER `0.07531 → 0.08672`, logistic은 `0.07531 → 0.07574`입니다. GBM의 ADS 손실은 `0.5 × 0.01141 = 0.005705`입니다.

다만 브리핑의 “in-held-out gain was source-specific”는 **INFERRED**로 낮춰야 합니다. 모든 fold에서 개선됐으므로 pooled OOF calibration만으로 이전 개선을 설명할 수는 없습니다. 그렇다고 source 차이만 원인으로 확정되지도 않습니다. 학습 표본 구성·유효 표본 수·regularization·입력 calibration 차이도 후보입니다.

**새 learned FILE combiner 탐색은 중단하겠습니다.** 지금 가치가 있는 작업은 기존 예측으로 배포 계산과 source별 실패 위치를 확인하는 것입니다. 그 과정에서 구체적인 계산 오류가 나오면 고치되, held-out에서 좋은 weight나 변환을 더 찾지는 않겠습니다.

`0/9/1` 유지 이유는 다음과 같습니다.

- **[VERIFIED] 동일한 선택 알고리즘의 정상적인 결과입니다.** `file_ensemble.fit`은 val의 **FILE EER** 개선이 `≥ 0.002`일 때 grid를 택합니다. DB의 `0.0019`는 equal, P1의 `0.0033`은 grid 조건을 충족합니다. `0.002`는 ADS 기준이 아니며, ADS로는 `0.001`에 해당합니다. [file_ensemble.py:37](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/file_ensemble.py:37)
- **[INFERRED] 지금 equal로 덮어쓰는 것도 새로운 선택입니다.** DB에서 평가한 숫자와 맞추겠다는 이유만으로, 다른 학습 결과에 val 규칙을 적용한 P1 선택을 취소할 근거는 부족합니다.
- **[VERIFIED] `+0.00061 [−0.00033,+0.00252]`는 “not worse”를 입증하지 않습니다.** 사전 비열등성 margin이 없고 CI에 손실이 포함됩니다. “이 진단에서 뚜렷한 악화는 관찰되지 않았다”가 적절합니다.
- **[INFERRED] 그 비교는 패키지 성능 검증도 아닙니다.** weight만 DB 예측에 적용했다면 P1 heads와 P1 CDF를 검증하지 않았습니다. 더구나 P1은 원래 test row를 학습에 넣습니다. 그 heads의 val 예측으로 선택한 weight를 옛 DB test에 가져오는 경로도 완전히 독립적이지 않습니다. [train_backbone_heads.py:260](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/train_backbone_heads.py:260)

`0.002` 경계가 불연속적이라는 비판은 맞습니다. 그러나 이번에 바꿀 대상은 결과를 본 weight가 아니라, 다음 실험에서 사전 등록할 선택 절차입니다.

## C. Val에서 적합한 요소들의 source-shift 위험

**[INFERRED] 우선 의심할 대상은 supervised 선택을 반복한 부분입니다.** FILE grid와 early stopping의 위험이 크고, component weight는 낮은 차원이지만 반복 탐색 이력이 중요합니다. CDF는 label을 직접 쓰지 않아도 source 구성에 민감합니다. L은 진짜 label을 복원하는 규칙의 타당성과, 그 규칙을 채택한 성능 근거를 분리해 검토해야 합니다.

| 요소 | 현재 위험과 근거 | 기존 예측으로 할 수 있는 구체적인 점검 |
|---|---|---|
| **CDF tables** | 각 member/candidate의 marginal 분포가 val source 비중과 fake 비중에 영향을 받습니다. 개별 CDF 뒤 평균하면 ranking도 달라집니다. | val source를 하나씩 제외해 CDF만 다시 계산하고 **weight는 고정**합니다. DB 예측의 순위 역전, clipping·tie 비율, source별 ΔEER를 측정합니다. held-out으로 CDF를 fit하거나 좋은 버전을 고르지 않습니다. |
| **VOICE weights `1/0.5/1`** | 고정됐다는 사실이 source 일반성을 보장하지는 않습니다. encoder/layer/weight를 여러 번 선택한 전체 탐색량도 고려해야 합니다. | 원래 후보 grid와 member 예측이 있으면 val group bootstrap 및 leave-val-source-out에서 선택 빈도와 차선과의 성능 차이를 봅니다. held-out은 현재 weight와 사전에 정한 member 제거 비교만 진단합니다. |
| **Early stopping** | component EER, 동률이면 BCE로 epoch를 선택합니다. 최종 ensemble ADS나 unseen-source 성능을 직접 선택하지 않습니다. | **epoch별 예측이 저장돼 있을 때만** source별 최적 epoch와 선택 민감도를 계산합니다. 최종 checkpoint 예측만 있으면 대체 stopping 규칙의 효과는 **UNKNOWN**입니다. source별 오류 집중은 볼 수 있어도 과적합 시점을 복원할 수는 없습니다. |
| **FILE weights** | 세 candidate일 때도 0.1-step simplex에는 66개 조합이 있습니다. 최솟값 탐색과 `0.002` 절단점이 val의 작은 변동을 증폭할 수 있습니다. | P1 val 예측으로 `0.0033` 및 `0/9/1`을 재현하고, group bootstrap에서 grid/equal 선택 비율과 source별 기여를 계산합니다. 배포 JSON으로 저장 FILE 점수를 재현해 실제 계산 차이도 확인합니다. |
| **L 규칙 선택** | provenance 기반 규칙 자체의 오류와, 특정 source에서 유리해서 채택한 선택 편향이 별개입니다. AI song의 VOICE label 공백 때문에 기존 held-out은 핵심 효익을 완전히 측정하지 못합니다. | 이미 있는 L/no-L 예측을 같은 ID·mask로 비교합니다. original labels와 completed-label 진단을 분리하고, real singing 및 `realV+fakeM`에서 false positive 증가를 확인합니다. unknown label을 real로 취급하지 않습니다. |

근거 코드는 CDF 및 FILE 계산의 [decision_heads.py:78](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/decision_heads.py:78), member CDF 결합의 [component_fusion.py:25](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/component_fusion.py:25), epoch 선택의 [train_backbone_heads.py:167](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/train_backbone_heads.py:167)입니다.

여기서 세 가지는 특히 구분해야 합니다.

**첫째, monotone이라는 이유로 CDF 전체가 안전한 것은 아닙니다. [VERIFIED/INFERRED]** 최종 단일 score에 strictly monotone 변환만 적용하면 ranking과 EER가 유지됩니다. 현재 코드는 여러 score를 각각 CDF로 바꾼 뒤 평균하고, VOICE는 다시 `ours` scale로 되돌려 FILE 계산에 사용합니다. component ranking이 유지되는 마지막 역변환도 FILE에서는 `Pv*Fv`의 상대 크기를 바꿀 수 있습니다. clipping으로 tie가 생기는 경우도 따로 확인해야 합니다.

**둘째, L은 이미 RT 진단에서 불편한 결과가 있습니다. [VERIFIED]** `variants.base.RT.val_completed.voice_eer = 0.03354`, `variants.L.RT.val_completed.voice_eer = 0.03456`으로 나빠졌습니다. 원래 L gate의 “completed-val Voice EER 개선” 조건과 맞지 않습니다. completed-val은 early stopping에 쓰지 않았지만, **L 채택 여부를 판단하는 데 사용했으므로 완전히 미사용 평가셋도 아닙니다.** [report2__base-L-S-LS.json:16](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/results/report2__base-L-S-LS.json:16), [voice_data_lock.json:14](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/locks/voice_data_lock.json:14)

**셋째, 예측만 고정한 재평가는 full-pipeline robustness를 측정하지 못합니다. [INFERRED]** CDF나 weight의 민감도는 확인할 수 있지만, source가 바뀔 때 heads가 어떻게 학습되는지까지 검증하지는 않습니다. 같은 이유로 새 CV에서는 기존 presence head의 학습 출처도 확인해야 합니다. outer-test label에 노출된 presence를 그대로 쓰면 full-pipeline CV가 아닙니다.

패키지에는 별도로 확인할 만한 **구체적인 recipe 위험**도 있습니다. **[VERIFIED]** trainer에서 `--label-completion`은 공용 `rows['train']`을 바꾸고 WavLM도 그 rows를 사용합니다. S7의 의도인 “S6 voice heads 재사용 + WavLM original labels”를 지켰는지는 실제 실행 설정으로 확인해야 합니다. `reuse_voice`와 `label_completion`의 동시 사용을 막는 assertion은 있지만, 제공된 브리핑만으로 실제 설정까지 확인되지는 않습니다. **[UNKNOWN] 현재 패키지가 이 문제를 겪었다는 증거는 없습니다.** [train_backbone_heads.py:261](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/train_backbone_heads.py:261), [train_backbone_heads.py:367](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/train_backbone_heads.py:367)

이 점검들은 **정해진 recipe를 제대로 평가·포장했는지 확인하는 데** 쓰겠습니다. 여기서 얻은 held-out slice를 다시 최적화 대상으로 삼으면, E2′에서 확인한 문제를 더 작은 형태로 반복하게 됩니다.

