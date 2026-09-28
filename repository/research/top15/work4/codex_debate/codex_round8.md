**선택된 S8의 feature 순서·raw 입력 경로에서 확정적인 제출 차단 버그는 찾지 못했습니다. 다만 검증 스크립트에는 제출 승인 전에 고쳐야 할 결함이 있습니다.** 실제 CUDA/CPU 검증 결과와 L4 자원 측정치는 제공되지 않아 통과 여부는 **UNKNOWN**입니다.

### 먼저 수정할 검증 결함

1. **P1 · VERIFIED — 이전 `TREE_OK`가 새 패키지 검증을 우회합니다.**  
   파일이 존재하면 재구성과 모든 member 해시 검사를 건너뛰지만, 결과에는 **현재 receipt의 SHA**를 기록합니다. 같은 `OUT`을 재사용하면 이전 tree의 실행 결과를 새 ZIP 결과로 잘못 표시할 수 있습니다.  
   **수정:** 매 실행마다 현재 receipt와 tree의 파일 집합·SHA를 재검사하십시오. 최소한 이번 검증은 새 `OUT`에서 실행해야 합니다. [verify_s8_vessl.py:14](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/verify_s8_vessl.py:14), [동일 파일:69](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/verify_s8_vessl.py:69)

2. **P1 · VERIFIED — parity가 실패해도 exit code는 성공입니다.**  
   마지막 줄은 `FAIL`을 출력할 뿐입니다. sample이 비어도 `PASS`이며, reference가 NaN이면 `max(0, nan) == 0` 때문에 오차가 숨을 수도 있습니다.  
   **수정:** 예상 sample ID의 비어 있지 않은 정확한 집합, 중복 여부, 모든 reference/runtime 값의 finite 여부를 검사하고, tolerance 초과 시 반드시 예외 또는 nonzero exit를 내야 합니다. [s8_runtime_parity.py:22](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/s8_runtime_parity.py:22), [동일 파일:42](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/s8_runtime_parity.py:42)

3. **검증 누락 · VERIFIED — 비교 대상과 FILE 계산을 충분히 확인하지 않습니다.**  
   reference head는 외부 `HEADS`, runtime 예측은 임의의 `VERIFICATION_JSON`에서 읽으며 둘을 같은 archive SHA에 연결하지 않습니다. 비교하는 출력도 VOICE/MUSIC뿐입니다. `FILE != plain max` 검사는 잘못된 FILE combiner도 통과할 수 있습니다.  
   **수정:** `TREE/model/trained`의 head/config를 사용하고 verification SHA·device·ID를 확인하십시오. packaged CDF·equal weights로 FILE도 재계산해 비교해야 합니다. [s8_runtime_parity.py:14](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/s8_runtime_parity.py:14), [verify_s8_vessl.py:67](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/verify_s8_vessl.py:67)

### 질문별 판정

**① Concatenation 순서 — 확인한 경로는 일치합니다. VERIFIED**

순서는 다음과 같습니다.

`MERT6 | MERT12 | EAT16 | backing SSLAM5 | raw-mix SSLAM5`

- P1 `items_for`: 기존 SSLAM 뒤에 extras 추가 — `diff_trainer_s8.patch:80–89`
- P1 `every_row`: 같은 순서 — `:153–159`
- RT `every_items`: 전달된 `spec` 순서대로 concat — `diff_runpy_s8.patch:44–56`
- runtime: 기존 backing feature 뒤에 mix 추가 — [diff_runtime_s8.patch:58](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/diff_runtime_s8.patch:58)

단, **screening의 기존 `new_items` 본문과 spec parser는 diff에 없습니다. 그 내부 순서까지 완전히 확인했다는 주장은 UNKNOWN**입니다. 선택 결과의 spec 문자열은 위 순서입니다. [s8_choice.json:169](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/results/s8_choice.json:169)

**② Raw·peak cap·`preserve_gain` — 정상 제출 경로는 일치합니다. VERIFIED**

추출은 `.6` cap 이후 fp32 raw에 windowing합니다. runtime도 `preserve_gain=False`가 기본값이고 정상 `run()`은 `engine.predict(path)`로 호출합니다. 새 mix branch에 stem용 `fp16_round`를 적용하지 않은 것도 맞습니다.

`preserve_gain=True`로 검사하면 peak가 `.6`을 넘는 파일은 학습 경로와 달라지지만, 현재 제출 호출에는 그 override가 없습니다. [extract_s8.py:104](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/extract_s8.py:104), [full_bank_submission.py:168](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/full_bank_submission.py:168), [동일 파일:312](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/full_bank_submission.py:312)

**③ fp16 cache와 fp32 runtime — 기존 block의 수치 차이는 남습니다. VERIFIED**

P1은 fp16 cache 값을 fp32로 읽습니다. 새 mix feature의 `.half().float()`는 그 저장 정밀도를 재현하므로 맞습니다. **SSLAM 계산 자체를 fp16으로 바꾼 것은 아닙니다.**

기존 MERT/EAT/backing SSLAM 출력은 runtime에서 같은 feature round trip을 거치지 않습니다. 따라서 전체 입력이 cache와 bit-identical하지는 않습니다. S7부터 있던 근사지만 새 S8 head의 민감도는 **UNKNOWN**이므로, 이번 head로 실제 score parity를 확인해야 합니다. 실패하면 block별 차이를 확인한 뒤 해당 출력의 rounding을 맞추는 것이 최소 수정입니다. [diff_trainer_s8.patch:33](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/diff_trainer_s8.patch:33), [diff_runtime_s8.patch:64](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/diff_runtime_s8.patch:64)

추가로 `s8_parity_features.py`는 reference를 **CPU에서 새로 추출**합니다. 원래 GPU cache와의 직접 비교는 아닙니다. CUDA runtime 결과에도 parity를 적용하고, reference의 코드·weight·audio provenance를 기록해야 검증 범위를 정확히 설명할 수 있습니다. [s8_parity_features.py:9](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/s8_parity_features.py:9)

**④ Rebuilt tree의 내용 동일성 — fresh 실행에서는 검사가 타당합니다. VERIFIED / 조건부**

`TREE_OK` 우회를 제거하면 추가 파일 삭제, missing 검사, member SHA 대조가 있습니다. 원본이 S7 ZIP인지 changed directory인지는 최종 byte가 맞으면 문제가 되지 않습니다.

다만 **Mac receipt가 최종 ZIP의 실제 내용과 일치한다는 연결**은 별도로 필요합니다. Mac에서 최종 archive SHA, 중복 없는 member 이름 집합, 각 member SHA를 receipt와 확인하십시오. VESSL은 새 ZIP 자체를 읽지 않으므로 receipt의 주장만으로 그 연결을 독립 검증하지는 못합니다. [verify_s8_vessl.py:23](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/verify_s8_vessl.py:23)

**⑤ 두 번째 SSLAM — 중복 메모리는 확실하고, L4 초과 여부는 UNKNOWN입니다.**

동일한 `sslam_pt`를 다시 로드해 GPU에 상주시킵니다. ZIP에는 같은 asset을 재사용하더라도 runtime 객체는 두 개입니다. **기존 backing SSLAM 객체를 raw forward에도 순차 재사용**하면 재학습 없이 중복 weight 메모리·로딩 비용을 줄일 수 있습니다. raw forward 연산 자체는 추가로 필요합니다. [diff_runtime_s8.patch:53](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/diff_runtime_s8.patch:53)

10-file H100/CPU smoke에는 peak GPU memory 기록이 없으며 L4 1,200파일 제한을 검증하지 않습니다. S7의 30분만으로 S8도 안전하다고 단정할 수 없습니다.

그 외 **P1의 GPU 학습 후 CPU 평가 전환은 정상**입니다. `train()`이 명시적으로 CPU로 반환합니다. WavLM byte-copy와 S6-L voice 재사용도 코드·run identity가 일치하며, FILE equal 선택도 `0.0008 < 0.002` 규칙에 맞습니다. 현재 확인된 우선 수정 대상은 모델 recipe보다 **검증 결과가 잘못 승인될 수 있는 경로**입니다.

