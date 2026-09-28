# 추론·전처리·성분 의미 감사

작성 범위는 최신 `submissions/MERT12_full1322_submit.zip`의 실제 소스, 그 학습 feature cache, 기존 최종 QA 출력, V6 및 B 추론 경로다. 생산 코드를 수정하지 않았으며 GPU 실행이나 대형 encoder 추론을 새로 수행하지 않았다. 로컬에서는 실제 checkpoint의 작은 head, 저장된 embedding, decoder·분할·결합 함수를 실행했다.

## 핵심 판단

최신 MERT12에서 class polarity 반전, 학습·추론 표준화 불일치, 잘못된 layer 선택, 파일 순서 오염을 뒷받침하는 근거는 발견하지 못했다. 반면 **8초·3-window·모든 stem이 audible인 학습 범위와 4–60초 배포 범위의 차이**, **무음의 높은 Music Fake가 File로 전달되는 실제 출력**, **시간이 다른 Voice·Presence 최대값이 결합될 수 있는 구조**는 확인했다. 이것이 공식 점수 손실의 원인이라는 뜻은 아니다. 각 현상의 범위와 검증 수준을 아래에서 구분한다.

## 실제 제출 흐름

`script.py:5–8` → `mert12_submission.run` → ZIP 내부 manifest·hash 확인 → baseline의 입력 ID 검사 및 sample 순서 정렬 → 원음 PANNs Presence → HTDemucs 분리 → vocal DF-Arena Voice → backing MERT330의 `hidden_states[12]` → 시간 평균 → window 평균 → 저장된 mean/std → seed 41/42/43 MLP sigmoid 평균 → File `max(vp*voice, mp*music)` → 10자리 반올림 → 유한성·범위 검사 → atomic CSV 저장.

ZIP의 `mert12_submission.py`, `eight_heads.py`, `v6_core.py`, `v6_runtime.py`, baseline `script.py`는 현재 루트 파일과 byte 단위로 일치했다. 따라서 아래 루트 파일의 줄 번호가 이 ZIP의 해당 member에도 적용된다. 학습 cache가 기록한 `eight_features.py`, `eight_feature_schema.py`, `eight_feature_identity.py` hash도 현재 루트와 일치한다. 근거: `inference_numeric.json/source_parity`, `extractor_identity`.

| 구간 | 실제 동작 | 소스 |
|---|---|---|
| 원음 decode | librosa, 16 kHz, mono, float32; 빈 값·비유한 입력 거부 | `baseline_original/script.py:144–150` |
| 분리기 decode | 최신 제출은 `torchaudio.load` + `demucs.audio.convert_audio`로 고정 | `mert12_submission.py:109–120,155` |
| 분리기 scaling | stereo의 mono 평균으로 mean/std 계산, 표준화 후 Demucs, 각 stem을 std/mean으로 복원; vocal 및 나머지 3개 stem 합을 mono로 바꾸고 16 kHz resample | `baseline_original/script.py:264–308` |
| Voice | 64,600 samples, full-window stride+exact tail; 짧으면 반복; 파일 RMS가 `1e-5` 미만이면 0, 아니면 segment별 spoof softmax의 최대 | `baseline_original/script.py:157–175,340–354` |
| Presence | 원음의 같은 baseline windows를 32 kHz로 resample; 지정 class 및 window 전체의 최대 | `baseline_original/script.py:206–225` |
| Music | 64,600 samples, hop 32,300, exact tail; 짧으면 zero-padding; 24 kHz soxr_hq; layer 12의 시간 평균 후 모든 window 평균 | `mert12_submission.py:47–64,87–99,137–146` |
| 학습·추론 normalization | 학습 file mean으로 fitted mean/std를 저장; std `<1e-6`만 1로 치환; 제출이 같은 값으로 표준화 | `mert12_full_refit.py:70–87,195–198,225–228`; `mert12_submission.py:74–98` |
| 성분·File 의미 | Voice/Music에는 Presence를 곱하지 않으며 File에서만 gate 적용 | `mert12_submission.py:102–106` |

MERT processor는 `do_normalize=false`, `return_attention_mask=true`다. 이를 무심코 바꿀 근거는 없다. 현재 훈련과 제출 모두 이 processor를 사용한다. DF-Arena config의 `spoof=0`, `bonafide=1`을 읽고 softmax의 spoof index를 선택한다. MERT head는 `MUSIC_FAKE=1`로 BCE 학습한 sigmoid 출력이므로 class 방향이 일치한다.

## I-1. 8초 학습과 긴 파일의 mean pooling 사이에 검증 공백이 있다

**분류:** 현재 MERT12의 확정된 분포 차이 및 품질 미검증. 배포 구현 오류로 단정하지 않음.

실제 `_workspace/mert12_a100_20260918/upload/features.zip`의 1,322개 NPZ와 manifest를 모두 읽었다. 1,322개가 전부 8.0초이며 각 3 windows, 총 3,966 windows다. padding은 하나도 없고 `stem_mask=0`인 window도 없다. 제출 함수로 계산한 4초는 1 window, 60초는 29 windows다. 즉 학습된 head는 29-window 평균이나 무음·padded window를 포함한 평균을 이 학습에서 보지 못했다. 근거: `inference_cache_masks.json`, `mert12_full_refit.py:87`, `mert12_submission.py:92`.

실제 ZIP의 세 head를 로드해 두 저장 embedding을 산술 혼합했다. real-like embedding 단독은 `0.0000867836182199729`, fake-like 단독은 `0.9999916950861613`이었다. real-like 28개와 fake-like 1개를 평균하는 비율 `1/29`에서는 `0.0001465825189370662`, fake 비중 0.25에서는 `0.0061436172885199385`였다. 이는 **실제 학습 head가 평균으로 희석된 fake-like evidence를 낮게 점수화하는 수치 반례**다. waveform을 합성하거나 partial spoof를 실제 추론한 결과가 아니므로 partial detection EER로 해석하면 안 된다. 근거: `inference_numeric.json/embedding_mixture_probe`.

`eight_feature_schema.py:13–22`에서 MERT layer 12는 core music base에만 존재한다. 기존 partial profile은 DF/AntiDeepfake 음성 probe이며 MERT12의 짧은 AI 음악 삽입 대응을 검증하지 않는다. window 안의 시간 평균도 먼저 일어나므로 window 내부의 짧은 삽입은 또 다른 검증 대상이다.

**가장 강한 반증 검사:** 학습·선택에 쓰지 않은 음악 source를 4/8/16/30/60초로 만들고 AI 구간 길이·위치를 고정한 실제 waveform을 같은 최신 ZIP에 입력한다. 같은 encoder/window 출력을 사용해 현재 file mean과 사전에 정한 대체 aggregation을 비교한다. 길이별 성능이 유지되면 이 위험의 실질적 중요성이 낮아진다. 현재 synthetic embedding 수치로 개선량을 약속할 수 없다.

## I-2. 무음의 Music Fake가 File score로 전달되는 출력이 실제로 존재한다

**분류:** 최신 패키지의 저장된 최종 QA에서 관측. 점수 영향은 미측정.

최종 supplemental CSV의 `qa_silence_4s`는 다음과 같다.

| File Fake | Voice Fake | Music Fake | Voice Present | Music Present |
|---:|---:|---:|---:|---:|
| 0.1139619671 | 0.0 | 0.9999964237 | 0.0563407838 | 0.1139623746 |

근거: `results/mert12_a100_20260918/downloaded/evidence/output/final_supplemental_qa/submission.csv:2`. 입력 생성은 `mert12_supplemental_qa.py:82–85`의 실제 zero WAV다. `MUSIC_FAKE_PROB`는 조건부 진위 출력이므로 음악이 없는 파일에서 높은 값이 나온 사실만으로 Music EER 오류라고 해석하면 안 된다. `graph_scoring.py:125–141`도 정답 Presence가 0이면 해당 성분 EER에서 제외한다. 다만 nonzero PANNs Music Presence와 곱해져 File이 0.1139619671이 된 현상은 관측된 결과다. EER은 0.5 threshold 지표가 아니므로 이것을 곧바로 최종 오분류라고 부를 수도 없다.

현재 Music은 silence 또는 `stem_mask`를 사용하지 않는 locked recipe다 (`mert12_full_refit.py:240`, `mert12_submission.py:137–145`). 기존 supplemental QA가 통과한 이유는 IDs·유한성·범위·실행 메모리를 검사했기 때문이며, 보고서도 `quality_claim=null`이라고 명시한다 (`mert12_supplemental_qa.py:125–146`). 이 QA는 품질이 좋다는 주장이 아니다.

**가장 강한 반증 검사:** 별도 음성-only·무음·약한 배경음 파일에서 실제 absence label과 File label을 확보해 이 현상이 전체 File ranking에 미치는 영향을 측정한다. 잘못된 hard gate가 약한 AI 음악을 지울 수 있으므로 무음 한 건만 보고 Music 출력을 일괄 0으로 바꾸면 안 된다.

## I-3. Voice Fake와 Voice Presence의 서로 다른 시간대 최대값이 곱해질 수 있다

**분류:** MERT12와 B의 확정된 aggregation 성질. 데이터에서 오류 빈도·점수 손실은 미측정.

MERT12는 Presence를 파일 전체의 최대값으로 먼저 만든다 (`baseline_original/script.py:220–225`). Voice 역시 파일 전체의 최대값이다 (`baseline_original/script.py:340–354`). 이후 두 수치를 곱한다 (`mert12_submission.py:105`). 예를 들어 두 windows의 `(presence,fake)`가 `(1,0)`, `(0,1)`이면 현재 reducer는 1, aligned `max(presence_t*fake_t)`는 0이다. 실제 `combine`으로 전자를 확인했다 (`inference_numeric.json/file_fusion_counterexample`). 이는 한 window의 실제 음성과 다른 window의 separation 잔여물/무음에서 생긴 높은 fake score를 결합할 수 있다는 의미다.

Voice silence gate는 파일 전체 RMS에만 적용되므로 일부 window가 silent여도 파일 전체 RMS가 `1e-5` 이상이면 모든 window가 DF-Arena를 통과한다. Music 경로는 window별 fake score를 사용하지 않고 file embedding mean을 사용하므로 현재 다섯 확률만으로 시간 정합을 복원할 수 없다.

과거 V6의 `v6_core.fusion_features`는 window별 product를 먼저 만든 뒤 max한다 (`v6_core.py:23–25`). 그러나 최신 MERT12는 `v6_runtime.baseline_module`만 import한다 (`mert12_submission.py:28`). `Engine`, `Heads`, V6 learned fusion은 실행하지 않는다. ZIP에 그 파일이 포함돼 있다는 이유로 제출에 temporal alignment가 있다고 말하면 틀린다.

**가장 강한 반증 검사:** 정답 음성이 있는 구간과 없는 구간이 분리된 clip에서 실제 per-window DF·PANNs 출력을 저장하고, 큰 score가 같은 window에 있는지 확인한다. 두 최대값의 위치가 거의 항상 같거나 기존 File ranking이 더 좋다면 대체 aggregation의 근거가 약해진다.

## I-4. 역상 stereo는 유효한 채널 신호를 무음으로 만들 수 있다

**분류:** 실제 decoder와 production 함수로 재현한 입력 edge case. 공식 데이터에 존재하는지는 미확인.

4초 stereo float WAV에 `L=x, R=-x`를 넣고 실제 librosa·torchaudio decode와 최신 고정 loader를 실행했다. 각 채널 RMS는 `0.14142135681755297`, Demucs stereo RMS는 `0.14142002165317535`였지만 raw mono RMS와 Demucs mono std가 모두 0이었다. `separate_voice_and_music`는 model inference 이전 silence branch로 들어가 vocal/backing을 모두 0으로 반환했다. 근거: `inference_audio_edge.json`; `baseline_original/script.py:145–146,268–275`.

이는 최신 MERT12에도 남는 mono downmix의 정보 손실이다. 기존 60초 stereo QA는 source가 mono이면 채널을 복제한다 (`mert12_supplemental_qa.py:78–81`). 따라서 그 QA 통과가 서로 다른 L/R phase에 대한 보장은 아니다.

**가장 강한 반증 검사:** 원래 stereo 데이터에서 channel correlation 및 mono/stereo RMS 비율을 측정해 이 조건이 실제 분포에 있는지 확인한다. 수치 반례가 존재한다는 사실과 공식 점수를 설명한다는 주장은 구분해야 한다.

## 짧은 입력·mask·tail의 판단

4초 입력은 64,000 samples라 MERT에 600 samples가 추가된다. 실제 processor 출력에서 resample 후 길이는 96,900, attention mask의 0 개수는 0이었다. padding을 processor 전에 붙여 원래 길이 정보가 사라졌고, model 호출에도 mask를 전달하지 않는다 (`mert12_submission.py:58–64,143–145`). 이 부분을 masked pooling이라고 설명하면 틀리다. 다만 최소 4초 규칙에서 padding은 약 0.93%이며, 이를 심각한 품질 오류로 입증한 결과는 없다. B·baseline은 이 경우 반복 padding을 사용하므로 두 recipe를 섞으면 안 된다.

4.0/4.0375/4.1/6/60초의 실제 window layout은 전 구간을 덮고 exact tail을 중복 start 없이 포함했다. 60초는 29 windows이며 일부 sample의 겹침 수가 3이다. exact tail이 추가되는 구조의 결과로 window별 동일 가중 평균은 sample별 동일 가중 평균이 아니다. 4.1초도 거의 같은 두 windows를 사용한다. 동일 규칙이 학습 feature 생성에도 쓰이므로 자체적으로 train/inference mismatch는 아니다.

MERT stem은 raw decode 길이에 맞춰 끝에서 truncate/pad한다 (`mert12_submission.py:186`). 1-sample resampler rounding 차이를 처리하며 waveform의 실제 시간 shift를 추정·보정하는 코드는 없다. 서로 다른 codec decoder의 timing 차이 전체가 검증됐다는 뜻은 아니다. 반면 최종 4개 raw-audio/cache parity는 정상이고, 최신 known FFmpeg/torchaudio 불일치는 아래처럼 수리됐다.

## 수리된 항목과 통과한 검사

- **Demucs decoder 분기:** baseline의 `demucs.separate.load_track`는 FFmpeg를 먼저 시도하고 torchaudio로 fallback한다. 최신 MERT12는 이를 명시적으로 대체한다. 이 문제를 최신 제출의 미수정 버그로 재기재하면 안 된다. 최종 smoke는 4개 원음에서 cache/raw Music 최대 차이 `3.971516315814938e-08`을 기록했다. 이번 로컬 실제 head/cache replay와 최종 CSV의 차이는 최대 `1.218368105189438e-09`였다.
- **학습 head와 normalization:** 세 seed의 mean/scale은 서로 같고 모든 parameter가 finite다. 최소 scale `0.9823168516159058`; scale 0 또는 비유한 normalizer 문제는 발견하지 못했다. 실제 저장 head를 strict load해 위 parity를 재현했다.
- **순서·부분집합 독립성:** 기존 최종 raw 실행은 4개 원본·permutation 및 2개 subset에서 모든 다섯 확률 차이 0을 기록했다. 현재 소스도 test-set 통계·cross-file normalization을 사용하지 않는다. 이는 검증한 네 파일에서의 증거이며 모든 입력에 대한 수학적 보장은 아니다.
- **입력 integrity:** duplicate audio stems, 누락·추가 IDs, duplicate submission IDs, 잘못된 column 순서를 거부한다 (`baseline_original/script.py:93–95,116–141`, `mert12_submission.py:167–169`). 출력 NaN/Inf 및 범위 밖 값은 clipping으로 숨기지 않고 실패시킨다 (`graph_scoring.py:48–76`). 실제 NaN float WAV도 `librosa ParameterError: Audio buffer is not finite everywhere`로 거부됐다.
- **atomic output:** 완성한 CSV를 `.csv.partial`에 쓴 뒤 replace한다. 중간 모델 실패로 불완전 CSV를 최종 파일로 남기는 경로는 없다 (`mert12_submission.py:193–196`).
- **평가 경계:** 현재 제출은 MERT330의 layer 12를 쓰지만 24-layer model 전체를 실행해 hidden states를 반환한다. 12-layer encoder라고 표현하면 부정확하다. 불필요한 후반 layer 연산을 줄일 여지는 있으나 이번 감사는 성능 변경을 하지 않았다.

## B 및 과거 V6와의 구분

B `submit.zip`은 현재 로컬 `submissions/B_full1696`에 없으며 README·JSON·SHA 파일만 남아 있다. 현재 B source들이 retained `submit.json` member hash와 일치하는 것은 확인했지만, 이 감사에서 B archive 자체를 다시 연 것은 아니다.

| 항목 | 최신 MERT12 | B full1696 | 과거 V6 |
|---|---|---|---|
| Music feature | MERT330 layer 12, backing | MERT330 last layer, raw+backing | MERT95 last layer, backing |
| 분할 | half-hop+exact tail, zero-pad | baseline full-hop+exact tail, repeat | baseline full-hop+exact tail, repeat |
| Music head | file embedding mean → normalized MLP | temporal stem head + gated raw residual | normalized linear window head |
| audible mask | 사용 안 함 | stem audible을 raw/stem 양쪽에 적용 | window music audible 적용 |
| Voice | DF-Arena vocal max | frozen DF-Arena vocal max | ONNX XLSR-AASIST vocal |
| File | separate maxima 기반 max | separate maxima 기반 noisy-OR | aligned products 및 raw disagreement의 learned fusion |

B에 남은 별도 위험은 두 가지다. 첫째, `redesign_features_runtime.py:124`가 baseline decoder 분기를 그대로 사용하고 `redesign_features_cache.py:73–79` receipt는 실제 선택된 decoder를 기록하지 않는다. MERT12의 수리가 자동으로 B에 전파된 것은 아니다. B 영향은 decoder별 실제 stem/feature/score 비교가 필요하다.

둘째, B raw residual도 `valid & stem_audible`에 묶인다 (`redesign_submission.py:91`, `redesign_models.py:77–82`). separator가 모든 stem을 inaudible로 만들면 populated raw feature가 있어도 empty bag logits `[0,-100]`으로 Music Fake가 약 `3.7835e-44`가 된다 (`redesign_models.py:43–49`). synthetic 작은 head에서 이 경로를 재현했다. 학습도 같은 mask를 쓰므로 배포 불일치는 아니지만, raw branch가 separator의 모든 실패를 구제한다는 설명은 틀린다. 실제 음악 label을 가진 all-inaudible stem 사례의 수를 확인해야 중요도를 판단할 수 있다.

B의 `redesign_voice.py`, `redesign_file.py` learned correction은 제출 runtime에 사용되지 않는다. B의 frozen voice/presence 및 noisy-OR recipe와 일치한다. 과거 H100 B smoke가 기록한 parity는 packaging 검증이며, 최신 MERT12 품질·공식 L4 실행의 증거로 옮겨 붙이면 안 된다.

## 근거 파일과 재검증 범위

- `inference_numeric.json`: ZIP/root parity, 실제 checkpoint head replay, normalization, duration layout, processor padding mask, analytic fusion·embedding counterexamples.
- `inference_cache_masks.json`: 실제 1,322 feature NPZ의 전체 duration/window/padding/audibility 집계.
- `inference_audio_edge.json`: 실제 WAV decoder·resampler·silence branch 및 NaN rejection 실행.
- `results/mert12_a100_20260918/downloaded/evidence/output/final_submission_smoke.json`: 기존 최종 GPU raw/cache 및 순서·subset parity.
- `results/mert12_a100_20260918/downloaded/evidence/output/final_supplemental_qa/`: 기존 4초 무음·60초 stereo·6초 MP3의 실제 출력 및 실행 receipt.

우선순위는 새로운 대형 모델보다, 현재 ZIP을 그대로 쓰는 **미사용 source의 길이·부분 삽입·성분 부재 평가**다. 이 감사에서 확정한 것은 실행 경로, 저장된 출력, 수치 성질과 검증 공백이다. 공식 점수 하락 원인이나 특정 개선안의 이득은 아직 입증하지 않았다.

<!-- HUMANIZE-SUMMARY: Korean report received a conservative final style pass after technical drafting; facts, numeric literals, paths, identifiers, uncertainty and evidence boundaries preserved. Self-check 6/6. No new factual content introduced by polishing. -->
