## A. 먼저 고칠 버그

**추출 전체를 중단하고 다시 만들어야 할 전처리 수식 오류는 발견하지 못했습니다. 그러나 완료 판정·재실행·resume에는 실제 결함이 있습니다.** 아래 VERIFIED는 파일·코드 확인을 뜻하며, 실행 중인 GPU의 상태를 직접 확인했다는 뜻은 아닙니다.

| 심각도 | 판정·근거 | 최소 수정 |
|---|---|---|
| **P1 — 실패한 실행도 `DONE` 처리** | **VERIFIED.** `run_s8.sh:15`의 인자 없는 `wait`는 각 worker의 실패를 집계하지 않습니다. `:16`의 개수에는 `.tmp.npz`도 포함됩니다. deadline 종료도 정상 종료이므로 성공 여부와 전체 완료 여부가 다릅니다. | PID별 종료 코드를 회수하고, 임시 파일을 제외한 **예상 ID 집합과 실제 유효 ID 집합의 일치**를 확인한 뒤 완료 처리합니다. [run_s8.sh:12](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/run_s8.sh:12) |
| **P1 — 겹친 재실행이 쓰기 중 파일을 훼손할 수 있음** | **VERIFIED, 발생 여부 UNKNOWN.** launcher `:7`은 공용 출력의 모든 `.tmp.npz`를 삭제합니다. extractor `:124`는 같은 ID에 항상 같은 임시 파일명을 씁니다. 중복 ID나 겹친 launch에서는 경쟁합니다. | 현재 worker가 살아 있는 상태에서 launcher를 재실행하지 않습니다. 단일 실행 잠금, ID 중복 검사, worker별 임시 이름을 사용합니다. atomic rename만으로 동시 writer 문제는 해결되지 않습니다. [extract_s8.py:124](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/extract_s8.py:124) |
| **P1 — 누락·잘못된 캐시를 정상 resume로 취급** | **VERIFIED.** `extract_s8.py:94–95`는 파일 존재만 보고 건너뜁니다. `:100–113`은 입력 누락·길이 불일치를 출력하고 계속합니다. manifest의 중복 ID도 `:70`에서 덮어써집니다. | 누락·오류 ID를 명시적으로 집계하고 screening 전에 실패 처리합니다. 캐시는 필요한 feature, finite 값, layer 목록뿐 아니라 **입력·checkpoint·전처리 identity**를 확인합니다. 올바른 캐시까지 다시 추출할 필요는 없습니다. |
| **P2 — feature 저장 정밀도와 runtime 입력 정밀도가 다름** | **VERIFIED.** 추출기는 embedding을 fp16으로 저장하지만 runtime의 `eat_features`·`wavlm_features`는 fp32 출력을 head에 넣습니다. stem fp16 문제와는 별개의 차이입니다. `extract_s6.py:53,65`, `extract_s8.py:40` ↔ `full_bank_submission.py:206,216`. | S8의 최종 score parity를 확인하고, 필요하면 **새 branch의 embedding에도 fp16→fp32 round trip**을 적용합니다. 이미 검증한 S7 branch를 일괄 변경하지 않습니다. |
| **배포 차단 — S8 runtime 경로가 아직 없음** | **VERIFIED.** 제공된 runtime은 VOICE에 `ours/ad/wavlm`, MUSIC에 기존 backing MERT/EAT/SSLAM만 연결합니다. `mixwavlm`, `mixsslam`, `d2v`, `usat`, `encodec`는 없습니다. | 선정된 member만 해당 입력 경로·head·fusion에 연결합니다. 추출 smoke 성공을 S8 배포 검증으로 계산하면 안 됩니다. [full_bank_submission.py:174](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/full_bank_submission.py:174) |

전처리 항목별 판단은 다음과 같습니다.

- **Decode·peak cap — 코드상 일치, decoder 수치 parity는 UNKNOWN.** extractor `:104–107`과 `Engine.predict:131–134`는 같은 `base.load_audio` 호출과 `.6` cap을 사용합니다. 다만 `pin_torchaudio_loader`는 `load_track`만 교체합니다. `load_audio` 구현·라이브러리·fallback decoder까지 같다는 증명은 아닙니다. 같은 길이도 같은 waveform의 증거가 아닙니다. [mert12_submission.py:109](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/mert12_submission.py:109)
- **Window·padding — 불일치 없음. VERIFIED.** 공용 `window_layout/windows`를 사용합니다. `64600`, hop `32300`, 중복 없는 exact tail, 우측 zero padding이 같습니다. 짧은 파일만 runtime에서 valid-sample normalization이나 attention mask를 새로 적용하면 달라집니다. [mert12_submission.py:47](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/mert12_submission.py:47)
- **fp16 stem / fp32 mix — 의도된 구분이며 유지해야 합니다. VERIFIED.** `d2v/usat`는 fp16 저장 voice, EnCodec은 fp16 저장 backing입니다. `mixwavlm/mixsslam`은 cap 이후 fp32 raw입니다. 미래 runtime에서 stem branch만 `fp16_round`하고 raw branch에는 적용하지 않아야 합니다. S7에서 이 차이가 VOICE `0.064` 오류를 실제로 만들었습니다. [briefing_round7.md:9](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/results/briefing_round7.md:9)
- **SSL normalization — 불일치 없음. VERIFIED.** `extract_s6.py:63`과 runtime `:215` 모두 window별 평균 제거, `unbiased=False` 분산, `1e-7`입니다. padding도 이 계산에 포함됩니다.
- **Layer truncation — 확인된 오류 없음.** 선택 index는 `(4,6,8,10,12,16)`, 유지 block은 17개입니다. 선택한 hidden state를 중간 출력으로 유지하려는 구현이며, 단순 off-by-one으로 고칠 이유가 없습니다. 브리핑의 full/truncated 차이 `0.0`은 해당 검사 결과이고, 실제 파일·패키지 parity까지 증명하지는 않습니다. [extract_s8.py:74](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/extract_s8.py:74)
- **EnCodec — latent 선택 오류 없음. VERIFIED.** `model.encoder`의 pre-RVQ continuous latent를 사용합니다. **16 kHz window를 먼저 만든 뒤** torchaudio로 24 kHz resample하고 mean/std를 붙입니다. runtime도 같은 순서와 std 정의를 써야 합니다. MERT의 `soxr_hq`나 전체 파일 resample 후 windowing으로 대체하면 불일치입니다. [extract_s8.py:34](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/extract_s8.py:34)
- **SSLAM fbank — 계산 방식의 불일치 없음. VERIFIED.** mean 제거, 128-bin fbank, hanning, dither `0`, frame shift `10`, 16-frame 단위 절삭, normalization과 CLS 제외가 runtime과 같습니다. 입력만 raw mix로 바꾸는 실험입니다. `extract_s6.py:42–53` ↔ runtime `:195–206`.
- **Silent flag — 저장 방식은 일치합니다. VERIFIED.** `dump_stems.py:59`는 fp16 저장 전 voice RMS로 flag를 계산하며 runtime도 원래 fp32 voice로 판단합니다. 추출기가 silent 파일의 feature를 계산하는 것은 버그가 아닙니다. 평가·runtime은 해당 flag에서 최종 VOICE를 `0`으로 만드는 기존 정책을 유지해야 합니다. raw mix RMS로 gate를 바꾸면 별도 실험입니다.

현재 캐시가 실제로 훼손됐다는 증거는 없습니다. **유효한 추출은 유지하면서 완료·캐시 검증을 바로잡는 쪽**을 권합니다.

## B. S8 lock은 수정한 뒤 등록해야 합니다

### 후보 수와 control부터 명확히

**[VERIFIED] 현재 코드의 단일-member 후보는 cached HuBERT까지 31개입니다.** d2v/usat는 `4..16` 전체가 아니라 각각 여섯 layer입니다.

- voice: HuBERT 6 + d2v 6 + usat 6 + mixwavlm 6 = **24**
- music: mixsslam 6 + EnCodec 1 = **7**
- 모두 3-seed면 control·concat 실험 전부터 **93 head fits**

여기에 greedy weight 선택과 반복 FILE fit이 붙습니다. “약 40개 head”로 탐색량을 요약하면 부정확합니다. [extract_s8.py:23](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/extract_s8.py:23), [extract_s6.py:18](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/extract_s6.py:18)

**[INFERRED] CPU voice를 그대로 고정하고 신규 voice만 GPU에서 학습하는 비교는 괜찮습니다.** 기존 member를 바꾸지 않고 추가하는 효과를 측정하기 때문입니다. MUSIC을 GPU에서 재학습하는 경우에는 아래 세 결과를 구분해야 합니다.

1. 기존 S7 DB CPU recipe
2. 같은 MUSIC recipe를 GPU에서 재학습한 control
3. 신규 MUSIC feature를 추가한 GPU candidate

`candidate − GPU control`은 representation 효과, `GPU control − 기존 S7`은 backend·재학습 효과입니다. 최종 패키지의 실용적 개선은 `candidate − 기존 S7`로 확인합니다. **이 두 개선량을 모두 공개**하십시오.

`cuda:0` 문자열은 FP32의 증명이 아닙니다. 제공된 기존 trainer는 `T.step/predict`에 `'cpu'`를 하드코딩합니다. 실제 GPU용 entry point와 내부 autocast 경로가 이 폴더에 없으므로, GPU 학습의 유효 dtype은 **UNKNOWN**입니다. [train_backbone_heads.py:166](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/train_backbone_heads.py:166)

### Step A/B의 수정

**[INFERRED] 단독 EER 최적 layer를 하나 고르는 방식은 계산 절약용 근사로 받아들일 수 있습니다.** 다만 기존 S7과 오류가 겹치는 layer를 고를 수 있으므로, “encoder의 최선의 ensemble 효과를 검사했다”고 주장하면 안 됩니다. 지금은 layer×weight 전체 탐색을 늘리지 않고 이 한계를 명시하겠습니다.

Step B에는 다음을 추가해야 합니다.

- 모든 후보의 **동일 R2 row·split·label mask**, seeds, head recipe를 고정합니다. 신규 voice member의 label completion 적용 여부도 명시합니다.
- voice greedy의 기존 weight는 고정하고 새 member만 추가합니다. 동률 처리도 고정합니다.
- MUSIC의 `+0.002`가 **val RT ADS**임을 명시하고, FILE은 각 구성에서 같은 규칙으로 val refit합니다.
- voice와 music이 각각 통과해도 joint에서 상쇄될 수 있습니다. **joint도 실제 val RT ADS로 최종 확인**합니다.
- held-out을 보기 전에 패키지에 들어갈 수 있는 후보만 남깁니다. raw WavLM/SSLAM은 기존 weight를 공유할 수 있지만 forward 비용은 늘고, d2v/usat/HuBERT는 새 weight가 필요합니다.
- 최종 **한 개 recipe**를 잠급니다. held-out에서 joint가 실패한 뒤 voice-only/music-only를 골라 구제하지 않습니다.

ZIP 여유는 현재 **264,931,181 B**입니다. deflate로 약 `1.1 GB`를 확보한다는 것은 **ESTIMATE**이며, 실제 archive 전에는 두 신규 encoder가 들어간다고 가정할 수 없습니다. 원본 weight의 fp16 변환도 별도 확인 없이 lossless라고 부르면 안 됩니다. [briefing_round7.md:17](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/results/briefing_round7.md:17)

### Step C: `+0.002`는 탐색 제출 기준으로 유지하되 guard를 보강

**[INFERRED — 제안] `+0.002`를 유지하겠습니다.** val에서 여러 후보를 비교했다고 독립적인 단일 최종 test에 무조건 31배 보정을 적용하는 것은 적절하지 않습니다. 실제 문제는 이 held-out이 이미 여러 연구의 선택에 영향을 줬다는 점입니다. 이제 CI를 붙여도 완전한 확증 검정은 아닙니다.

제가 등록할 탐색 제출 조건은 다음과 같습니다.

1. 최종 joint의 held-out RT `ΔADS ≥ +0.002` 대 기존 S7.
2. paired group bootstrap의 **`P(ΔADS > 0) ≥ 0.90`**. 고정된 예측에 조건부인 보조 근거로만 해석합니다.
3. 고정 unseen-generator subset의 `ΔADS ≥ −0.002`.
4. raw-mix voice를 쓰면 `realV+fakeM`의 VOICE FPR 증가 `≤ 0.03`.
5. **raw-mix music을 쓰면 `fakeV+realM`의 MUSIC FPR 증가도 `≤ 0.03`.** 초안에서 빠진 핵심 조건입니다.
6. 선택한 실제 package의 offline load, score parity, ZIP·메모리·시간 조건 충족.

FPR guard에는 각 모델의 **val positive TPR 95%에 맞춘 threshold**를 고정하겠습니다. 같은 숫자의 threshold를 두 모델에 적용하면 CDF scale 차이가 섞입니다. 각자의 val-EER threshold를 쓰는 초안도 하나의 정책이지만, 모델마다 sensitivity가 달라져 성분 오염을 비교하기에는 덜 명확합니다.

107개에서 `0.03`은 순증 false positive **3개**, 286개에서는 **8개**까지입니다. group 의존성까지 있으므로 작은 차이에 대한 강한 보증은 아닙니다. ensemble guard와 함께 seed별·group별 변화도 공개하되, 결과를 보고 조건을 바꾸지는 않습니다.

**Best-ever는 낮은 점수의 직접 손실을 제한합니다. 따라서 제출 기준은 논문식 확증보다 느슨해도 됩니다. 하지만 평가·배포 요건까지 느슨하게 할 이유는 없습니다.** `+0.002` 미만을 제출하려면 지금 별도의 탐색 슬롯 규칙으로 선언해야 하며, Step C 실패 후 예외를 만들면 안 됩니다.

Step C가 사실상 무의미해지는 경우는 분명합니다. P1처럼 test를 학습에 포함한 head를 평가하거나, 캐시 누락으로 어려운 row를 제외하거나, GPU control 없이 backend 효과를 encoder 개선으로 계산하거나, held-out 결과를 보고 member·weight·threshold를 다시 고르는 경우입니다.

## C. 작은 대안 다섯 개

**S8 본 실험과 분리된 이름·후보 목록·판정으로 진행하고, S8 recipe가 잠기기 전에는 이 결과를 끼워 넣지 않겠습니다.** head 학습은 사용자 제약대로 VESSL에서 진행하는 계획입니다.

| 우선순위 | 실험·판정 | 공정한 비교와 배포 경로 |
|---|---|---|
| **1** | **WavLM `h6+h8+h12` concat — 실행 가치 높음** | 현재 stem cache에 모두 있습니다. 기존 h8 member를 대체하고 fusion weight·pooling은 고정합니다. GPU h8 control과 같은 seeds로 비교합니다. 새 encoder weight 없이 head 입력 폭만 늘어납니다. runtime은 현재 단일 layer만 읽으므로 layer-list와 concat 경로를 추가해야 합니다. |
| **2** | **Voice mean vs max — 가장 싼 진단** | 먼저 동일 head의 window 확률을 재집계해 pooling만 비교합니다. per-window 예측이 없으면 cached feature로 head만 실행합니다. mean에 맞춰 재학습하는 실험은 별도로 구분합니다. sparse fake·부분 조작·긴 real 파일을 함께 보고, mean이 fake 구간을 희석하는지 확인합니다. |
| **3** | **Music gated attention-MIL — 한 가지 고정 구성만 실행** | S7 MUSIC feature와 GPU LME control을 그대로 사용하고 pooling만 바꿉니다. seeds·epoch cap·early stopping을 동일하게 둡니다. attention 폭·temperature sweep은 하지 않습니다. runtime `:154–156`은 LME를 직접 계산하므로 checkpoint만 바꾸면 attention이 실행되지 않습니다. |
| **4** | **Raw-mix FILE head — 탐색 가치는 있으나 통합 비용 큼** | 미리 정한 하나의 feature, 예컨대 `mixsslam5`로 **train의 `FILE_FAKE`**를 학습합니다. VOICE/MUSIC label을 대신 쓰지 않습니다. component 점수는 그대로 두고 FILE candidate로만 추가합니다. raw mix의 성분 혼합은 FILE target에는 자연스럽습니다. |
| **5** | **5 seeds vs 3 — finalist를 정한 뒤에만 실행** | `41/42/43` 대 `41/42/43/44/45`, 모두 포함하고 좋은 seed만 고르지 않습니다. 같은 GPU backend로 비교해야 seed 수 효과와 CPU/GPU 차이를 분리할 수 있습니다. 모든 layer에 두 seed를 더 돌리는 것은 낭비입니다. |

**[INFERRED] 이 다섯 가지 중 LB 전이 가능성이 가장 높은 선택은 WavLM multi-layer입니다.** 지금까지 성과가 있었던 frozen representation 변화와 가깝고, 새 encoder·raw component 오염·큰 ZIP 증가를 피합니다. 실제 개선 근거는 아직 **UNKNOWN**입니다.

FILE head는 E2′와 구분할 필요가 있습니다. E2′는 기존 scalar들을 val에서 학습한 combiner였고, 이번 것은 raw representation에서 train label로 새 정보를 배우는 head입니다. E2′ 실패만으로 폐기할 수는 없습니다. 다만 네 번째 candidate를 추가하면 현재 0.1 grid는 **66개에서 286개 조합**으로 늘고 equal fallback도 3-way에서 4-way로 바뀝니다. 이를 그대로 두면 “새 representation”과 “확장된 FILE 선택”이 섞입니다. 첫 실험은 기존 세 weight의 비율을 고정하고 신규 weight만 사전 지정한 작은 집합에서 고르는 편이 낫습니다. [file_ensemble.py:37](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/file_ensemble.py:37)

5-seed도 단순 파일 추가로 끝나지 않습니다. `SEEDS=(41,42,43)`가 runtime 전체에 공유되므로 전역 변경하면 DF/AD/WavLM까지 추가 checkpoint를 요구합니다. 어느 head만 5개인지 명시하고 그 목록으로 로드해야 합니다. [mert12_submission.py:32](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/mert12_submission.py:32), [full_bank_submission.py:46](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/code/full_bank_submission.py:46)

추가 실험은 모두 train/val에서 평가한 뒤 **배포 가능한 한 개 finalist만** 별도 held-out 평가 대상으로 잠그겠습니다. 나머지는 val 결과만 남겨도 사용자가 원한 실험 자체는 수행한 것입니다. 이름을 분리해도 같은 held-out을 다시 보면 독립성이 생기지는 않습니다.

## D. Fine-tuning은 이번에도 하지 않겠습니다

**[INFERRED] 남은 약 29시간에서 encoder fine-tuning에 배정할 예산은 0입니다.** “더 오래 학습하면 해결된다”는 가설을 지지하는 프로젝트 증거가 없습니다.

**[VERIFIED]** ftC1은 member VOICE EER가 val `0.02147`로 좋아졌지만 test는 `0.12784`였습니다. 선택한 RT package는 S4 `0.88457 → 0.88057`, ADS `−0.0040`입니다. [ftC1_gate.json:10](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/results/ftC1_gate.json:10), [ftC1_gate.json:136](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/results/ftC1_gate.json:136)

Claude의 결론에는 동의하지만 근거를 과장하면 안 됩니다. 예전 EAT/FullFT8은 데이터·학습 recipe가 다른 실패이며, 모든 LoRA·부분 FT가 실패한다는 증거는 아닙니다. 새로운 FT를 정당화하려면 다른 가설과 source를 제외한 검증 설계가 필요합니다. 지금은 그 설계·재학습·패키지 검증까지 추가하는 것보다, 이미 추출 중인 frozen 후보와 작은 head 비교의 기대가치가 높습니다. bounded FT를 하나 더 제안할 만큼 근거가 없습니다.

## E. 슬롯 계획

**[VERIFIED: 브리핑 기록] S7 ADS는 `0.8373968254`, L4는 30분입니다.** `0.86`까지는 `+0.0226031746`이 필요합니다. 과거 LB/DB 비율 `1.48`을 S8의 `+0.002`에 곱해 예상 LB 상승량을 만들지는 않겠습니다. [briefing_round7.md:4](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/results/briefing_round7.md:4)

| 슬롯 | 제안 |
|---|---|
| **09-28 남은 1회** | `~12:00`까지 **screen 통과뿐 아니라 최종 패키지 검증까지 끝난 S8**이 있으면 제출합니다. 없으면 **S7−L**을 우선 대안으로 둡니다. S5의 no-L voice heads에 S7의 SSLAM MUSIC·WavLM을 유지하고 val CDF/FILE을 refit한 구성입니다. 이것도 준비되지 않았다면 기존 **S5**를 사용합니다. |
| **09-29 첫 번째** | 오늘 못 낸 S8이 준비됐으면 우선 제출합니다. 오늘 S8을 냈다면 C에서 val로 선정하고 별도 gate를 통과한 finalist를 제출합니다. 제 사전 우선순위는 WavLM multi-layer입니다. |
| **09-29 두 번째** | 아직 제출하지 않은, 검증된 별도 후보를 사용합니다. FILE raw head나 attention이 후보가 될 수 있지만, 단독 실험 통과를 이유로 자동 결합하지 않습니다. 결합 구성도 val에서 먼저 잠급니다. |
| **09-29 세 번째** | 앞선 제출의 구체적인 배포 오류 수정 또는 이미 검증한 최종 결합용으로 남깁니다. 늦어도 `08:00`에 후보를 고정하고 검증이 끝난 archive를 확보합니다. |

**[INFERRED] S7−L/S5 fallback은 top-15를 위한 유력한 도약으로 보지는 않습니다.** 다만 새 encoder 구현 없이 L의 LB 기여를 구분할 수 있고, expiring slot을 써도 기존 S7 점수는 유지됩니다. 오늘 S8의 held-out 결과를 본 뒤 임의로 member를 빼서 만든 변형보다, 지금 지정해 두는 대안이 낫습니다.

동일 예측을 내는 재압축 S7에는 슬롯을 쓰지 않겠습니다. 이번 검토에서는 파일 읽기와 작은 정적 계산만 했으며, GPU 실행·학습·추론·파일 수정은 하지 않았습니다.

