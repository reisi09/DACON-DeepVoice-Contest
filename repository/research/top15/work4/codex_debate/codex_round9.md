**[VERIFIED] 차단 버그는 runner의 실패 처리입니다. EAT/MERT 추출 수식에서는 현재 runtime과의 불일치를 찾지 못했습니다.** S9는 후보군을 줄여 진행하되, standalone EER로 layer를 먼저 고르는 방식은 바꾸겠습니다.

**A. 코드 결함과 일치 여부**

- **P1 — 실패해도 exit code가 0입니다.** `run_s9.sh:20`의 `... || echo FAILED >...`는 마지막 `echo`가 성공하므로 worker·validator 실패를 호출자에게 전달하지 않습니다. 이전 `DONE`도 남습니다. **lock 획득 후 이전 상태 표시를 지우고, 명시적인 실패 분기에서 `exit 1`** 해야 합니다. ([run_s9.sh:8](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/run_s9.sh:8), 같은 파일 18–20행)

- **P2 — 패키지 runtime을 import한다는 보장이 없습니다.** `sys.path[:0]=[code_root, runtime]`이므로 `/root/ft/code`의 동명 모듈이 우선합니다. **runtime을 먼저 두고 실제 import된 `__file__`과 코드 hash를 기록**해야 합니다. **[UNKNOWN]** 현재 두 사본이 실제로 다른지는 이 코드만으로 확정할 수 없습니다. ([extract_s9.py:71](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/extract_s9.py:71))

- **P2 — resume·validator가 의미상 잘못된 cache를 통과시킬 수 있습니다.** 기존 파일은 이름만 보고 건너뛰고, validator는 shape·finite·samples만 확인합니다. 같은 shape의 다른 layer 순서나 다른 checkpoint 산출물도 통과합니다. `eat_layers==(4,8,12,16,20,24)`, `mert_layers==(3,6,9,12,15,18,21,24)`와 추출 코드·checkpoint·입력 manifest identity를 확인해야 합니다. 손상된 기존 파일은 재실행해도 계속 skip되고, 남은 `*.tmp.npz`는 계속 실패를 일으킵니다. lock 아래에서 실패한 ID만 재추출하고, 종료된 작업의 임시 파일만 정리하십시오. ([extract_s9.py:91](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/extract_s9.py:91), [validate_s9.py:17](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/validate_s9.py:17))

**[VERIFIED] 수치 경로는 일치합니다.** raw의 peak cap `0.6`, window 배치·padding, EAT fbank 정규화와 CLS 제외 평균, MERT의 window별 `16000→24000` `soxr_hq` resampling·processor·hidden 평균이 runtime과 같습니다. 저장 feature만 fp16입니다. **새 runtime도 raw는 fp32로 계산하고 결과 feature에 `.half().float()`를 적용**해야 합니다. 정상적인 shard 분할·원자적 rename에서는 추가 race를 찾지 못했습니다. ([extract_s9.py:28](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/extract_s9.py:28), 97–105행; [full_bank_submission.py:172](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/full_bank_submission.py:172), 238–259행)

**B. 제안하는 S9 lock — [INFERRED: 의사결정 제안]**

**standalone layer 선별을 생략하고, S8에 추가했을 때의 val RT ADS를 직접 비교**하십시오. 이미 standalone에서는 block 5가, 조합에서는 block 4·7이 유리했습니다. 다만 조합 MUSIC EER 우위가 ADS 우위를 뜻하지는 않습니다. ([s8_choice.json:10](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/results/s8_choice.json:10))

후보는 다음 **10개**로 제한하겠습니다.

| 계열 | S8 대비 변경 |
|---|---|
| EAT, 6개 | `+mixeat{4,8,12,16,20,24}` |
| MERT, 3개 | `+mixmert6`, `+mixmert12`, `+mixmert6_12` |
| SSLAM, 1개 | 기존 raw `mixsslam5`를 `mixsslam4_5`로 **교체** |

마지막 후보에서 block 5를 중복 append하면 안 됩니다. `4_5`는 이미 본 val 결과에서 나온 탐색 후보라는 사실을 lock에 기록하십시오.

- **Control:** 정확히 같은 GPU S8 DB recipe, seeds `41,42,43`, fp32·TF32 off. split·label mask·voice·presence를 고정하고 FILE fitting도 같은 절차를 사용합니다. identity가 같은 기존 control은 재사용합니다.
- **Step A:** 10개 후보 각각을 control에 추가해 비교합니다. 선택 기준은 3-seed ensemble의 **val RT ADS**, 진입선은 `Δ≥+0.002`.
- **Step B:** EAT·MERT의 최상위 후보가 각각 진입선을 넘었을 때만 둘의 joint를 한 번 평가합니다. joint는 최상위 단독 후보보다 다시 `+0.002` 이상 좋아야 채택합니다. SSLAM과의 추가 조합은 열지 않습니다. 동점은 추가 forward가 적은 쪽을 택합니다.
- **Held-out:** 최종 한 후보만 평가합니다. S8 대비 RT `ΔADS≥+0.002`, paired group bootstrap `P(Δ>0)≥0.90`, unseen-generator `ΔADS≥−0.002`, `fakeV+realM` MUSIC FPR 증가 `≤0.03`. FPR은 기존 평가와 같은 **R2 대상 214개 ID**, 각 모델의 val `TPR95` threshold로 비교합니다. ([s8_heldout.json:91](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/results/s8_heldout.json:91))
- 실패 후 layer·seed·slice를 바꿔 재도전하지 않습니다. 이 held-out은 이미 반복 사용했으므로 **독립적인 새 검증이라는 주장은 금지**합니다.

**S8 LB가 S7 이하라는 이유만으로 완료된 S9를 자동 기각하는 데는 반대합니다.** S8에는 P1 재학습·FILE 변화도 있어 raw-mix만의 인과효과가 아닙니다. 다만 그런 결과라면 **새로운 고비용 raw-mix 확장은 중단하고**, 이미 완료한 후보와 추가 forward가 없는 `4_5`까지만 보겠습니다. S9가 로컬에서 S7도 넘는다는 조건은 별 도움이 없습니다. 로컬 S8이 이미 S7보다 높기 때문입니다.

**C. 제출 슬롯 — [INFERRED]**

기본 순서는 **S9 → S8 음악 5 seeds → S7-noL**입니다.

1. **S9:** 위 gate와 패키지 검증을 모두 통과한 경우.
2. **S8 음악 5 seeds:** `41–45`를 전부 사용하고, 3-seed 대비 val·held-out ADS가 모두 비열화이며 unseen `Δ≥−0.002`일 때. seed 선별이나 즉석 `S9+5 seeds`는 하지 않습니다.
3. **S7-noL:** 나머지는 S7로 고정하고 voice의 label completion만 제거한 패키지. L의 외부 일반화 여부를 직접 확인하는 탐색 슬롯입니다.

S8 LB가 S7 이하이면 **S7-noL을 첫 번째로 올립니다.** 완료되어 gate를 통과한 S9는 best-ever 조건에서 여전히 탐색 가치가 있지만, S8 5-seed는 후순위입니다. 자격을 갖춘 후보가 없으면 슬롯을 억지로 채우지 않습니다. A1의 val `+0.00035`만으로 별도 슬롯을 배정하지 않겠습니다. 패키지는 **08:00까지 동결**하십시오.

**D. L4 위험**

**[VERIFIED]** 현재 MERT runtime은 전체 encoder를 실행합니다. 따라서 `mixmert6`을 골라도 자동으로 6층 비용이 되지는 않습니다. EAT는 선택한 최대 block까지 실행합니다. ([full_bank_submission.py:187](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/full_bank_submission.py:187), 254행)

**[INFERRED]** 기존 인스턴스를 순차 재사용하고 이전 hidden tensor를 해제해야 합니다. encoder 가중치 추가는 없지만 head가 커지므로 **zip 증가가 정확히 0은 아닙니다.** `mixsslam4_5`는 기존 block 5 forward에서 중간 feature를 함께 얻을 수 있어 가장 저렴합니다.

**[UNKNOWN]** 두 forward를 추가한 L4 시간·peak는 아직 모릅니다. 사용자 제공 H100 allocated `21.36 GiB`와 L4 총 `22.4 GiB`의 차이 `1.04 GiB`를 실제 여유 메모리로 보면 안 됩니다. **[ESTIMATE]** 시간은 `S8 약 31분 + raw EAT 추가 시간 + raw MERT 추가 시간`입니다. 같은 파일로 추가 비용과 긴 파일의 peak를 측정하고, 보수적으로 **45분 이내를 목표**로 잡겠습니다. 그 근거가 없으면 두 encoder joint보다 단독 후보를 우선합니다.

