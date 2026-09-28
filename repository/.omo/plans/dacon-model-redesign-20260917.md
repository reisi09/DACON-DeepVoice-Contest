# dacon-model-redesign-20260917 - Work Plan

## TL;DR (For humans)
<!-- Fill this LAST, after the detailed plan below is written, so it summarizes the REAL plan. -->
<!-- Plain English for a non-engineer: NO file paths, NO todo numbers, NO wave/agent/tool names. -->

**What you'll get:** 공식 리더보드 1위를 목표로, 현재 최고 모델을 대조군으로 둔 재설계와 실행 순서다. 새로운 독립 평가, 작은 File 보정, 혼합 오디오를 학습한 music 보정, singing·telephone voice 검증을 차례로 거쳐 통과한 구성만 결합한다.

**Why this approach:** 선두와의 점수 차이는 대부분 진위 판별에서 생긴다. 기존 실험에는 새 mixture를 music head가 학습하지 않은 경우와 voice 정답이 없는 경우가 있어, 더 큰 모델부터 늘리면 같은 판단 오류를 반복할 수 있다.

**What it will NOT do:** 이번에는 설계만 완료한다. 학습·GPU 결제·제출을 실행하거나, 평가 파일로 학습·보정하거나, 현재 근거로 1위 성능을 보장하지 않는다.

**Effort:** XL — 전체 실행 기준. 첫 비교는 frozen encoder와 작은 head로 제한한다.
**Risk:** High — 독립 natural-real·singing 라벨 부족, 선두의 미공개 구조, 실제 L4 비용이 남아 있다.
**Decisions to sanity-check:** raw 추가는 L4 선측정 후 결정한다. 충분한 새 데이터와 재현된 이득이 없으면 구조 확대를 멈추고 현재 최고 모델을 유지한다.

Your next move: 설계 검토 뒤 실행 범위와 GPU 예산을 정한다. 아래에는 구체적인 구현·검증 조건을 적었다.

---

> TL;DR (machine): XL / High / design-only; 8 implementation todos + 4 final verifiers; independent data → OOF File → matched music residual → voice → accepted-only package; leaderboard #1 not achieved.

## Scope
### Must have

이 문서는 **공식 리더보드 1위**를 최종 목표로 삼는 모델 재설계안이다. 이번 작업의 완료 조건은 실행 가능한 설계와 근거 검토다. 1위 달성, 새 모델 구현, 학습, 유료 GPU 사용, 제출은 이번 작업에 포함하지 않는다. 아래 수량·허용 오차·승격 기준은 별도 표시가 없으면 **실행 전에 고정할 제안값**이며, 이미 달성한 수치가 아니다.

작업 기준 경로는 `/Users/choiwuseck/Documents/ChatGPT/데이콘`이다. 본문의 저장소 상대 경로는 모두 이 경로를 기준으로 한다. `E=.omo/evidence/model-redesign-20260917`, `R=results/redesign_20260917`, `D=_workspace/redesign_20260917/data`로 표기한다. `E`의 네 감사 문서와 `gap-calculation.json`은 이번 설계의 고정 근거이고, 실행 결과는 `R`에 새로 쓴다.

#### 현재 위치와 해결해야 할 문제

2026-09-17 02:01 KST에 확인한 공식 1위는 `55`, 0.88450이고, 우리 팀 `killerpjw`는 175위다. 최고 제출 `M330_MLP_full1206_submit.zip`의 정확한 점수는 0.7396465503, 사용자가 S2로 식별한 2026-09-16 23:06:28 제출 `submit.zip`은 0.6378679788이다. 순위와 선두 점수는 이후 바뀔 수 있다. [공식 리더보드](https://dacon.io/competitions/official/236749/leaderboard), [제출 기록](https://dacon.io/competitions/official/236749/mysubmission), `E/live-evidence.md`, `E/gap-calculation.json`.

공식 점수는 다음과 같다.

`S = .45(1−FileEER) + .18(1−VoiceEER) + .27(1−MusicEER) + .05 VoicePresenceAUC + .05 MusicPresenceAUC`.

현재 ADS 0.71189, CPS 0.98939이고, 표시된 선두와의 차이 0.1448534497 중 약 99.4%가 ADS 차이다. CPS를 1로 올려도 최대 이득은 0.001061이다. 현재 CPS를 유지해 표시된 선두 점수에 도달하려면 ADS 약 0.872846이 필요하다. 이는 필요한 변화의 크기를 나타내는 계산이지, 로컬 실험에서 예측할 수 있는 점수가 아니다. File EER 0.01 감소도 공식 점수로는 0.0045에 해당하므로, 작은 File 보정 하나를 1위 해법으로 취급하지 않는다. 선두의 모델과 File/Voice/Music별 EER는 확인되지 않았다.

핵심 결정은 **M330·DF-Arena·PANNs·HTDemucs를 대조군으로 유지하면서, 독립 평가 데이터와 File 관계식부터 바로잡고, 같은 라벨과 노출 조건에서 raw/stem music 보정의 효과를 확인한다**는 것이다. 그다음 singing·telephone voice를 별도 검증한다. 작은 실험을 통과한 구성만 합친다.

#### 결과를 읽는 기준

| 근거 묶음 | 확인된 결과 | 설계에 반영할 결정 |
| --- | --- | --- |
| 공식 제출 | M330 0.7396465503, C08 0.7314822646, baseline 0.6909110688, C3 0.6843536931, V5 B1 0.6776485873, V6 0.6727822646, S2 0.6378679788 | M330 ZIP과 SHA-256을 보존한다. 로컬 Music EER 개선을 공식 총점 개선으로 바꾸어 말하지 않는다. |
| V5 B0/B1/B2 | 같은 230행 패널 Music EER .542222/.368333/.457778. B2 presence AUC 악화, 학습 노출도 달랐다. B3/B4는 준비 시간 초과로 미실행, B4_joint 실행 기록 없음 | 증강·fusion 실패를 단정하지 않는다. 다만 과거의 대형 V5 구조를 기본안으로 재가동하지 않는다. |
| V6/V7/C3 | normalization·창 샘플링·NSynth 비중 변경으로 재사용 dev Music EER 개선. C3 공식 점수는 baseline보다 낮았다 | 한 번에 바뀐 요인을 분리하고, 오래 재사용한 dev/test를 최종 확인에 쓰지 않는다. |
| temporal·augmentation·compact | A1이 tune 기준 선택, A2는 historical test가 더 낮음. C08도 tune 기준 선택. R1/R8 sigmoid는 사실상 동률로 최종 R1 선택 | test를 본 뒤 winner를 바꾸지 않는다. 8-bin이나 attention 추가 자체를 새 가설로 포장하지 않는다. |
| MERT95/330·cosine·methods·AFWhisper | 동일 재사용 분할에서 M330의 근거가 강함. AFWhisper 크기 증가가 우위를 보장하지 않음. 48-fit methods 등 많은 선택이 이미 이뤄짐 | frozen M330를 출발점으로 쓰고 foundation-model 탐색을 먼저 반복하지 않는다. |
| full1206 M330 | 5개 fold, seed41–45, 각 파일은 네 head 학습에 포함. train/dev/sealed 전체를 사용 | 기존 full1206 ensemble을 새로 나눈 과거 holdout의 공정한 대조군으로 쓰지 않는다. 같은 train으로 control을 다시 학습하거나 완전히 새로운 외부 패널에서만 비교한다. |
| S0/S1/S2 fullFT | S0 MERT frozen/fullFT test .266667/.319444. S2 EAT .167603이지만 real 12개, music 중심·seed41 한 번. S2 공식 점수 하락 | fullFT가 실제 encoder까지 학습됐다는 사실은 인정한다. 음악 부분 성능으로 전체 제출을 선택하지 않는다. |
| separator adaptation | 모든 41,984,456 trainable parameter에 gradient, S0 EAT .300000→.252778 | 고정 teacher stem과의 일치 및 한 번의 historical 개선이다. true stem 품질·S2 개선·공식 개선으로 확장하지 않는다. |
| 최신 fast H100 | stem/raw Music ensemble test .206564/.210564. raw의 개별 seed 평균은 더 낮아 ensemble 순위와 다름 | seed별 EER 평균·SD와 probability-ensemble EER를 따로 보고한다. |
| 최신 File MLP | test File .110063→.150943, natural .093023→.162791, mixed .200000→.116667. mixed validation은 .100000→.300000으로 악화 | synthetic test 일부의 이득으로 채택하지 않는다. 2,051차원 temporal mean MLP를 되풀이하지 않는다. |
| 최신 학습 노출 | music head는 새 mixture를 전혀 학습하지 않음. 별도 raw/stem head를 상수 가중 평균했음 | mixed 학습과 실제 raw conditioning을 2×2로 구분해 검증한다. |
| 데이터 감사 | train real music은 FMA·NSynth에 편중, fake는 Echoes에 편중. duration-only shortcut도 강함. dedup은 평가 가중치 변경 | 독립 natural-real group, codec/duration 대조, 출처 누락 검사를 모델 확장보다 먼저 한다. |

수치는 `E/local-evidence.md`의 원본 경로·행 번호와 `E/architecture-evidence.md`의 실행 경로를 따른다. 서로 다른 분할의 EER를 한 순위표로 합치지 않는다. 최신 로컬 총점은 필요한 voice 라벨이 없어 모두 `null`이다. natural voice EER 0.0은 speech 48개에 한정되며 singing 검증이 아니다.

#### 기본 아키텍처

1. **입력과 분리.** 파일은 독립적으로 처리한다. 공식 입력은 16 kHz, mono/stereo, 4–60초다. PANNs·raw MERT용 mono 경로와 HTDemucs의 native channel 경로를 구분한다. HTDemucs는 기존 `baseline_original/script.py:264–308`을 재사용해 stereo 정보를 보존하고 vocals와 accompaniment를 얻는다. 모든 expert의 실제 입력 sample rate·채널 변환·resampler를 cache identity에 넣는다. 16 kHz를 24/32/44.1 kHz로 올려도 원본 8 kHz 이상 정보가 복원되지는 않는다.
2. **전체 구간.** `get_segment_starts`/`extract_segment`의 64,600 sample, 4.0375초 grid와 마지막 창을 유지한다. 60초 파일도 빈 내부 구간이 없어야 한다. 짧은 파일의 기존 padding/tiling을 기록하고 valid-time mask로 새 loss의 중복 가중치를 막는다. C08/M330에는 원래 tail 누락이 없다. S2의 8×5초 sampling은 60초에서 20초를 놓치므로, S2를 challenger로 쓸 때만 all-window 경로와 고정 weight 비교를 한다.
3. **공유 frozen music encoder.** 한 MERT330 weight로 각 창의 raw와 accompaniment를 인코딩한다. 두 view 모두 frame-token mean으로 `E_s,E_r∈R[B,T,1024]`, `start/end∈R[B,T,2]`, valid/audibility mask를 저장한다. raw view를 추가하면 weight는 공유하지만 MERT forward 작업은 늘어난다. 8-bin은 기본안에 넣지 않는다.
4. **작은 music residual.** 기존 compact `TemporalMusic`의 LayerNorm→1×1 projection→3-layer residual temporal block으로 `H_s,H_r∈R[B,T,64]`를 얻는다. 두 view의 projection/block weight를 공유한다. 창별 보조값 `r_t∈R6`은 raw·stem·vocal log-RMS, raw/stem audible indicator, 실제 valid-duration 비율이다. 이 값은 encoder cache에서 얻고 source ID·codec ID·파일 길이를 별도 분류 feature로 넣지 않는다. `u_t=[H_s,H_r,H_r−H_s,r_t]∈R198`, `δ_t=W2 GELU(W1 u_t)`는 198→32→64, `g_t=sigmoid(w_gᵀ[H_s,r_t]+b_g)`는 scalar다. `W2,b2`는 0으로 초기화하고 `H*=H_s+g_tδ_t`를 기존 masked max→64→32→2 MLP에 넣는다. 초기 출력은 동일 train으로 fit한 stem control과 정확히 같아야 한다. raw가 없거나 cache가 잘못됐을 때 조용히 다른 모델로 바꾸지 말고 오류를 낸다.
5. **창별 증거와 bag 출력.** 기존 MLP output을 각 `H*_t`에도 적용해 conditional music logit `m_t`를 얻되, 제출의 `MUSIC_FAKE_PROB`는 기존 bag pooling을 거친 probability다. 창별 score는 위치 라벨의 보조 loss와 File 특징에 쓴다. 기존 8-bin 실험과의 차이는 시간 해상도 자체가 아니라 **같은 창의 raw/stem 상호작용·맞는 mixed 라벨·sparse-event 보조 감독**이다. V5의 단순 시작점 정렬·큰 Transformer와도 이 점을 구분한다.
6. **voice와 presence.** 첫 music/File 비교에서는 DF-Arena voice와 PANNs presence를 고정한다. 이후 voice 과제에서 `VoiceExpert.forward`의 frozen 1,280차원 pre-classifier embedding `V∈R[B,T,1280]`을 vocals의 같은 64,600-sample 창에서 읽는다. 기존 fake logit에 1280→32→1 zero-init residual을 더하고, 기존 max window aggregation을 유지한다. fake class 방향은 baseline 출력과 실제 parity로 확인한다. singing/telephone의 real/fake component 라벨로만 보완하며 PANNs는 기본적으로 고정한다. 별도 raw voice encoder는 기본안에 없다.
7. **File 관계.** `a=vp·vf`, `b=mp·mf`, `f0=max(a,b)`를 남긴다. 1차 비교는 f0, 비학습 `1−(1−a)(1−b)`, OOF에서 학습한 12차원 offset logistic residual이다. z feature의 scalar logit 입력에만 `clip(p,1e−5,1−1e−5)`를 적용한다. 고정 feature 순서는 `[logit(a),logit(b),logit(vf),logit(mf),logit(vp),logit(mp),|a−b|,max_t(vp_t·vf_t),max_t(mp_t·mf_t),max_t max(vp_t·vf_t,mp_t·mf_t),top2mean_t max(vp_t·vf_t,mp_t·mf_t),mean_t I(vp_t>.5 and mp_t>.5)]`다. 창이 하나면 top2mean은 그 창 값이다. PANNs가 이미 처리하는 각 창의 class probability를 읽고 같은 시간 support에 맞춘다. `0<f0<1`에서는 안정적인 logit으로 `f=sigmoid(logit(f0)+wᵀz+c)`를 계산하고 `f0=0→f=0`, `f0=1→f=1`은 그대로 보존한다. train-OOF에서만 표준화·L2 fitting하며 endpoint를 포함해 `w=c=0` control이 f0와 같아야 한다. 이 관계는 자유로운 MLP도, 확률 독립성이 보장된 union도 아니다. EER와 natural-real 오탐으로 채택 여부를 정한다.
8. **출력 의미.** `VOICE_FAKE_PROB=P(fake voice | voice present)`, `MUSIC_FAKE_PROB=P(fake accompaniment | accompaniment present)`를 presence와 분리한다. 두 컬럼에 presence를 다시 곱하지 않는다. 5개 제출 column과 범위·ID 순서 계약은 `graph_scoring.validate_predictions`를 재사용한다.

File relation의 train data에는 underlying music/voice head가 해당 group을 보지 않은 OOF 예측만 들어간다. historical full1206 ensemble은 OOF 생성에 사용할 수 없다. 외부 validation/sealed 파일의 예측을 fitting하거나 그 파일 집합의 통계로 표준화하지 않는다.

#### 손실과 학습 노출

각 binary target에는 known mask를 붙인다. authenticity loss는 `known authenticity AND true presence=1`인 위치에서만 계산한다. presence unknown, 생성된 노래 속 vocal provenance unknown, inaudible 구간은 정답 0으로 채우지 않는다. 완전한 target이 없는 whole-song label은 File 또는 허용되는 bag loss에만 쓴다.

손실 항의 기준 가중치는 `L=.45 L_file+.18 L_voice+.27 L_music+.05 L_vp+.05 L_mp+.10 L_event`다. 기본안은 staged training이므로 music 단계는 music/event, voice 단계는 voice/event, OOF File 단계는 File/L2 항만 최적화한다. File meta loss로 upstream을 다시 학습하지 않는다. 각 항은 eligible group/class 안에서 정규화한 BCE 또는 두-class CE이고, frozen·미확인 항은 0이다. 이 surrogate가 EER/AUC와 같다고 주장하지 않는다. 최종 선택은 아래의 공식 가중 metric으로 한다. `L_event`는 실제 interval truth가 있는 창에만 적용한다. fake 구간과 창이 일부 겹치지만 구간 전체의 authenticity를 정하기 어려우면 그 창의 event loss를 mask하고 whole-bag supervision만 쓴다. partial-fake 파일을 random crop한 뒤 모든 crop을 fake로 지정하는 경로는 금지한다.

mixing은 `(real voice,real accompaniment)`, `(fake voice,real accompaniment)`, `(real voice,fake accompaniment)`, `(fake voice,fake accompaniment)` 네 상태를 균등한 목표 mass로 만든다. 검증된 instrumental source만 accompaniment 단독 라벨을 부여한다. 일반 자연 노래는 singing voice와 accompaniment가 함께 있는 mixed다. speech-only/music-only/neither는 presence 평가용으로 따로 확보한다. real-real 편집 control을 같은 crossfade·splice·codec·gain 경로로 만들고, 일반 denoise·gain·enhancement 때문에 fake 라벨을 붙이지 않는다.

실험 A/B/C/D는 `(A) stem+pure`, `(B) residual+pure`, `(C) stem+mixed`, `(D) residual+mixed`다. source group·원본 target accompaniment·전처리·seed41/42/43·초기 control weight·optimizer update 상한·목표 component가 노출된 초·유효 창 수를 맞춘다. mixed slot의 pure control은 동일 accompaniment 구간을 간섭 성분 없이 쓴다. 추가 voice interference가 data factor이며 원본 음악 구간을 바꾸지 않는다. 기본 batch64, head LR1e−4, weight decay.01, dropout.1, 최대300 epoch/patience20을 출발점으로 고정하고, 네 arm의 update cap·sampler draw list를 기록한다. 조기 종료 시 실제 update 수를 함께 보고하고 checkpoint 선택 차이를 추가 원인으로 남긴다. A→B는 pure에서 conditioning, A→C는 mixed 학습, C→D는 같은 mixed 감독에서 raw 기여를 검증한다. 가짜 `raw=zeros` arm만으로 용량 효과를 제거했다고 주장하지 않는다. 같은 residual 용량에 `raw=stem`을 넣는 control을 한 번 추가한다.

#### 데이터와 검증 집합

기존 reused dev/sealed/historical test는 오류 분석용이다. 새 source 자료의 original song, artist, singer, speaker, 동일 생성물, stem, mix, splice, 재인코딩, 편집 부모를 연결한 component group으로 8:1:1 train/validation/sealed를 만든다. generator/provider 전체를 하나의 거대 group으로 union하지 않는다. provider/generator는 분할 층화와 별도의 사전 지정 leave-generator/provider-out stress에 사용한다. 최대 두 held-out domain을 실험 전에 고르고 그 결과로 다음 domain을 반복 선택하지 않는다. 출처를 연결할 수 없는 항목은 격리한다. PCM hash만으로 near-duplicate가 없다고 결론내리지 않는다.

빠른 screening의 제안 최소치는 새로운 validation에서 relevant head마다 real/fake 각 25 독립 group이다. 해당 조건이 안 되면 gradient·경로·runtime 확인까지만 하고 성능 우위를 판정하지 않는다. 최종 후보 승격의 제안 목표는 validation과 sealed **각각** File/Voice/Music의 eligible real/fake 각 100 독립 group, presence positive/negative 각 100 group이다. natural-real accompaniment는 NSynth를 제외하고 집계한다. singing·telephone 핵심 strata는 각 패널에서 eligible real/fake 각 30 group, 독립 provider는 전체 pool에서 class별 3개 이상을 목표로 한다. 각 head에 같은 group이 중복 집계될 수 있으므로 이 숫자를 단순 합산한 파일 수로 보고하지 않는다. 8:1:1에서 이 목표는 head별 class마다 약 1,000개 eligible parent group을 확보해야 할 수 있다. 실제 라이선스·출처·일정으로 확보하지 못하면 목표 미달을 그대로 기록한다.

이 숫자만으로 검정력을 보장하지 않는다. train-OOF에서 얻은 paired error 분포로 독립 group 수를 늘렸을 때의 bootstrap CI 폭을 추정하고, 목표 `Δweighted=.01`의 부호를 구분할 수 있는지 사전 확인한다. derivatives를 늘려 group 수가 늘어난 것처럼 계산하지 않는다. final validation/sealed의 열람·요약 접근은 기록하고, sealed는 모든 architecture/recipe/ensemble/file calibration이 잠긴 뒤 한 번만 연다. 실패한 sealed를 다음 tune으로 재사용하지 않는다. 새 확인이 필요하면 새로운 source group 패널을 만든다.

duration 4–10/10–30/30–60초, mono/stereo, codec/container, 전화 대역, real/fake source, speech/singing/instrumental, simultaneous/sequential, fake span 0.5/1/2/4초와 위치를 기록한다. 양 클래스에 같은 crop·codec·gain 처리를 적용하고 원본 natural-real도 유지한다. group-OOF duration/codec/source probe와 label-only nuisance classifier를 두어 shortcut 위험을 측정한다. NSynth는 기본 주평가에서 제외하고 악기 단음 stress로 별도 보고한다. training mass는 초기 0으로 두되, 같은 데이터를 쓰는 모든 arm에 동일하게 적용한다.

#### 하나의 승격 기준

모든 후보에 같은 기준을 적용한다. baseline 비교는 동일 train으로 다시 fit한 control 또는 아직 어떤 기존 모델도 보지 않은 external panel로 한정한다. checkpoint는 train 내부 early-stop partition에서 known-label loss로 정하고, architecture·recipe 선택은 고정 validation으로만 한다. 고정 validation을 epoch마다 inspect하지 않는다.

- 5개 metric이 전부 식별되면 `graph_scoring.official_metrics`의 S를 primary로 삼는다. File-only 수정은 `Δ=.45(EER_old−EER_new)`, Music+File만 수정하면 `Δ=.45ΔFile+.27ΔMusic`, Voice까지 수정하면 확인된 항을 더한다. 이 partial delta는 비교한 파일 집합과 고정된 나머지 출력이 정확히 같을 때만 계산한다. unknown head가 변하면 absolute S도 partial delta도 대신 만들어내지 않는다.
- seed별 metric 평균·population SD와 확률을 평균한 ensemble metric을 모두 낸다. seed ensemble은 사전에 정한 같은 구성의 41/42/43만 사용한다. 5-fold 모델 5개를 5-seed 반복 실험이라고 부르지 않는다.
- 2,000회 paired connected-group bootstrap으로 candidate−control Δ의 95% CI를 낸다. 클래스가 사라진 resample은 제외 수를 보고하며 5% 이상 무효면 해당 CI로 승격하지 않는다. primary Δ point estimate ≥.005, CI lower bound >0을 제안 승격 gate로 둔다. screening은 후보 제거용이고 final claim은 충분한 fresh 패널에서 한다.
- natural-real이 포함된 natural panel의 File 또는 component EER 악화 >.01, singing/telephone 핵심 stratum 악화 >.02, 또는 presence AUC 악화 >.005이면 결합하지 않는다. stratum 표본 수가 부족하면 통과로 취급하지 않는다. frozen presence가 같으면 그 항은 numerical identity로 확인한다.
- validation 동률은 `Δ` 차이 <.002 또는 CI가 겹치고 신뢰할 이득이 없을 때 기존 control을 우선한다. 비교 가능한 개선 후보 사이에서는 더 낮은 L4 시간→더 적은 encoder pass→더 작은 package 순으로 고른다. historical/sealed/LB 수치로 tie-break하지 않는다.
- alignment shuffle/wrong-time raw에서 성능·event consistency가 변하지 않으면 raw 상호작용 근거가 약하므로 stem-only C를 우선한다. raw branch가 학습 gradient를 받는다는 사실만으로 성능 기여를 인정하지 않는다.

#### 조건부 확장과 1위 도전 경로

기본 screen은 File 보정과 A/B/C/D만으로 좁힌다. voice 작업은 데이터 확보와 frozen 경로 검증을 병행하고, 필요한 singing/telephone 라벨이 갖춰졌을 때 residual 학습으로 이어진다. Voice EER는 공식 18%, File EER는 45%이므로 음악 개선만으로 전체 목표를 마무리하지 않는다.

조건부 후보는 **한 번에 하나**만 연다. (a) 충분한 complete voice×music state와 correct OOF가 있는데도 File 관계가 포화되면 `v5_fusion.joint_probabilities`/`joint_partial_label_loss`를 재사용한 9-state를 같은 M330·DF features에서 비교한다. absent/real/fake×absent/real/fake의 softmax mass에서 conditional authenticity를 분모로 나누는 기존 의미를 유지한다. 모든 상태마다 train 30 독립 group, relevant validation 양 클래스가 확보될 때만 연다. (b) residual 오류가 짧은 forensic event에 집중되고 MERT 창 평균에서 사라진다는 측정이 있으면 16 kHz log-mel의 작은 3-block CNN 하나를 쓴다(25 ms Hann, 10 ms hop, 64 mel, 0–8 kHz, 32/64/64 channel, max+mean pooling). 기존 torch/torchaudio로 구현하며 별도 dependency를 추가하지 않는다. (c) frozen 최선 모델과 같은 데이터에서 제한된 마지막 block tuning을 먼저 확인하고, reproducible external gain이 있을 때만 fullFT로 확장한다. 세 선택을 동시에 기본 ensemble로 만들지 않는다.

현재 S2 EAT는 새 학습 없이, 기존 native-channel voice 경로를 고정한 music-only challenger로 검사할 수 있다. 8개 crop 대 full coverage, preprocessing 변경 대 encoder 변경을 분리한다. separator 재학습은 실제 separation failure와 authenticity 손상 측정이 확보될 때까지 보류한다. pseudo-stem L1 감소만으로 열지 않는다.

2026-09-17~18은 데이터·label·runtime gate, 19~20은 File 및 A/B/C/D screen, 21~23은 새 real·singing·telephone 확보와 voice/최선 music 재확인, 24~26은 앞 gate가 지목한 조건부 후보 하나와 OOF 재학습, 27일은 후보 동결·sealed·ZIP QA, 28~29일은 사용자가 실행을 요청한 경우에만 고정 후보의 공식 확인에 배정한다. 종료는 9월 29일로 확인됐지만 정확한 마감 시각은 실행 단계에서 다시 확인한다. 하루 3회 제한은 상한이며 채워야 할 목표가 아니다. 예정 checkpoint는 `file-only`, `accepted music+voice`, `final accepted package` 세 개뿐이고 각각 로컬·L4 gate 통과 후 제출 여부를 정한다. Public 100%=Private이므로 공식 점수로 blending weight를 반복 탐색하지 않는다.

1차 개선이 Δ.005 정도에 머물면 그것은 유용한 국소 개선일 뿐 0.14485 공식 격차를 설명하지 못한다. 9월 23일 checkpoint에서 신규 natural-real·singing·generator stress의 실패 유형을 다시 분류하고, 데이터 부족이면 데이터 확보에 남은 자원을 우선한다. 데이터가 충분하고 residual error가 특정 표현에 남을 때만 위 조건부 branch를 연다. 9월 26일까지 독립 패널 이득이 재현되지 않으면 구조 확대를 중단하고 검증된 후보 또는 기존 M330를 유지한다. 로컬 Δ를 공식 점수 예상치로 환산하지 않으며, #1 목표는 실제 공식 순위가 확인될 때까지 미달성이다.

#### Kaggle 1위 사례에서 가져올 것

| 실제 1위와 1차 출처 | 가져올 원리 | 그대로 가져오지 않을 것 |
| --- | --- | --- |
| Freesound Audio Tagging 2019, Ruslan Baikulov. [주최 측 Kaggle 순위표](https://dcase.community/challenge2019/task-audio-tagging-results), [우승자 코드·설명](https://github.com/lRomul/argus-freesound/blob/4faf8f192035b413e8946bda3555474cb9ad8237/README.md) | curated/noisy 라벨을 구분한 학습, CV에서 검증한 stacking, 잘못된 train 라벨의 선별 검토 | random fold 누출, 다수 MLP·7-model blend, tagging MixUp을 fake component label에 적용하는 것 |
| BirdCLEF 2024, Team Kefir. [Kaggle 1st-place write-up](https://www.kaggle.com/competitions/birdclef-2024/writeups/team-kefir-1st-place-solution) | 목표 domain과 train/inference window의 일치, 작은 모델과 전처리 대조, duplicate 관리 | DACON 평가 파일 pseudo-labeling, bird CE/teacher .05 계수, 다른 대회의 augmentation을 그대로 이식하는 것 |
| DFDC 2020, Selim Seferbekov. [Meta 공식 결과](https://ai.meta.com/datasets/dfdc/), [우승자 코드·설명](https://github.com/selimsef/dfdc_deepfake_challenge/blob/89c6290490bac96b29193a4061b3db9dd3933e36/README.md) | 국소 forensic evidence와 파일 내 집계, benign transmission 변형에 대한 검증 | image backbone/32-frame threshold, 비디오 metric과 DACON EER의 혼용, 이 대회에서 Public/Private 순위 변동을 예측하는 것 |

Freesound는 lwlrap, BirdCLEF는 조류 탐지, DFDC는 video log loss를 다룬다. 이 결과가 DACON 개선을 보장하지 않는다. 관련 음악 연구 [SONICS](https://arxiv.org/abs/2408.14080v4)와 [공식 repository](https://github.com/awsaf49/sonics/blob/9156ffad151f797c71556923c4a02fa01fa8fc91/README.md)는 temporal forensic 가설의 참고다. whole-song 생성 라벨을 검증 없이 각 stem의 라벨로 쓰지 않는다. software MIT와 dataset CC BY-NC 4.0 및 Suno/Udio 추가 조건을 [고정 license](https://github.com/awsaf49/sonics/blob/9156ffad151f797c71556923c4a02fa01fa8fc91/LICENSE)에 따라 따로 기록한다. 현 1위의 구조를 재현한다고 주장하지 않는다.

#### GPU와 배포 예산

확인된 환경은 L4 22.4 GiB VRAM, RAM 28 GB, CPU 6개, 1,200파일/60분, ZIP 10 GB/압축 해제 32 GB, 설치 10분 이내다. **50분·VRAM 20 GiB·RAM 24 GB**를 사전 목표로 제안한다. 3초/파일은 hard 평균 상한, 2.5초/파일은 50분 목표에서 나온 산술값이지 측정 속도가 아니다. `E/live-evidence.md`와 [공식 평가 환경](https://dacon.io/competitions/official/236749/overview/evaluation)을 따른다.

기존 fast H100은 2,086파일의 shared feature 추출에 약 60분, 작은 head fit은 수초 수준이었다. 이를 같은 duration/window 수의 frozen 추출 계획에만 참고한다. `t_extract≈60분×(새 valid window 수/기존 valid window 수)×(새 measured sec/window/기존 sec/window)`로 pilot 후 고친다. 파일 수만 비례시키거나 H100 수치로 L4를 예측하지 않는다. head가 빨라도 raw MERT pass·separator·DF-Arena는 무료가 아니다.

| 단계 | 예정 자원과 잠정 범위 | hard stop과 산출물 |
| --- | --- | --- |
| CPU 준비 | GPU 0시간. source/license/hash·압축·업로드·bucket/object 재다운로드 hash 확인 | 준비·다운로드 중 GPU를 켜 두지 않는다. 승인된 실제 저장소를 쓰고 기존 bucket을 임의로 삭제하지 않는다. |
| L4 pilot | 현 package와 raw 추가 후보를 합쳐 GPU .5–1.5시간, 96파일 duration/channel/codec 층화 | actual decode/separation/encoder/출력별 시간, peak GPU/RAM, cold start를 잰다. 1,200파일 예상 >50분 또는 peak >20 GiB면 raw rollout 보류, 60분/22.4 GiB 초과는 탈락. |
| 첫 frozen screen | H100×1, 기존과 비슷한 2,000–4,000파일 범위에서 1–3시간 feature 추출을 초기 추정; head·OOF .25–1시간 | 100파일 pilot 후 window당 속도·남은 시간으로 재산정. stage cap 4 GPU시간, 남은 예산이 export 15분을 남기지 못하면 새 trial 금지. |
| 데이터 확대·voice 보완 | cache reuse 후 H100×1 2–8시간의 계획 범위. 새 group/창 수로 재산정 | stage cap 8 GPU시간. 독립 확인 없이 encoder/seed 수를 늘리지 않는다. cache 변경 시 모든 관련 feature 재생성 비용을 포함한다. |
| 조건부 adaptation 1개 | 마지막 block 2–6시간, fullFT 필요 시 별도 pilot와 별도 상한 결정 | fullFT 비용은 현재 자료로 확정할 수 없다. optimizer state까지 포함한 VRAM·step time을 먼저 재고, 사전 gate를 못 넘으면 시작하지 않는다. |
| 최종 L4 package | 최소 실제 1,200파일 end-to-end 1회, cold start 포함 50분 목표 | full worst-case 60초 파일 panel의 extrapolation도 함께 보고한다. 실제 duration 분포를 모르면 그 한계를 숨기지 않는다. |

위 범위는 예산 승인이나 결제액이 아니다. 처음에는 H100 1대로 충분하며 다중 GPU의 효율을 가정하지 않는다. 향후 실행 시 업로드·hash 검증 뒤 GPU를 시작하고, 15분마다 stage/ETA/예산/최근 checkpoint를 확인한다. 완료 또는 idle 15분이면 checksum을 붙인 checkpoint·로그를 다운로드·검증한 뒤 인스턴스를 중지한다. 실패 시 export 여유 15분을 남기고 종료한다. 이 설계 단계에서는 VM·bucket·automation·학습·제출을 생성하지 않는다.

### Must NOT have (guardrails, anti-slop, scope boundaries)

- 평가 데이터의 학습, pseudo-label, test-time update, 다른 평가 파일의 score/statistics를 이용한 보정. 파일 내 segment 집계만 허용한다. [공식 규칙](https://dacon.io/competitions/official/236749/overview/rules).
- unknown vocal truth를 real/absent로 치환하거나 자연 노래를 instrumental로 간주하는 라벨 확장. synthetic frame의 일부 fake를 모든 crop의 fake로 확장하는 처리.
- 새 분할에서 historical full1206 head의 in-sample 점수를 대조군으로 사용하거나, meta head fitting에서 underlying head가 본 group의 예측을 OOF라고 부르는 것.
- sealed를 반복 열어 architecture·blend·hyperparameter를 고르는 것. 기존 test에서 좋아 보이는 seed만 선택하는 것. subgroup 하나의 이득을 전체 개선으로 말하는 것.
- 대형 backbone·separator joint training·9-state·log-mel·EAT를 한꺼번에 넣는 기본 구성, 불필요한 framework/dependency, shallow schema/shape/file-exists test.
- 기존 최고 제출 ZIP, checkpoint, 원본 results 변경. 최신 raw blend나 temporal-mean File MLP를 이름만 바꿔 반복. 현재 제공된 design-only 범위에서 유료 실행 또는 제출.

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: **이번 설계는 none(근거·코드·계획 대조 검토)**. 실행 단계는 tests-after이며 실제 자료·checkpoint를 쓰는 소수의 runnable verification command를 각 과제 안에 둔다. 기존 torch/numpy/scikit-learn만 사용하고 새 test framework를 도입하지 않는다.
- Evidence: 이번 설계는 `E`의 원본 ledger와 계획 review다. 실행 시 `attemptDir=.omo/evidence/redesign_20260917/<run-id>`에 task receipt를 남기고 큰 prediction/cache/checkpoint는 `R/<run-id>`에 둔다. ulw-loop로 실행하면 그때 `omo ulw-loop status --json`의 `currentAttemptDir`를 사용한다. 아래 CLI는 **향후 각 task가 구현할 계약**이며 지금 실행된 명령이 아니다.
- 모든 check는 특정 실패를 막아야 한다. 핵심은 lineage 누출, 조건부 label 오염, sparse-fake 누락, 무의미한 raw branch, OOF 오염, numerical ROC 오류, cross-file coupling, package/runtime 불일치다. test count·coverage·shape·파일 존재만으로 통과 처리하지 않는다.
- 이미 재현된 latest 110 EER 값의 max difference 0.0은 metric 계산 근거다. 새 score implementation을 만들지 않고 `graph_scoring.official_metrics`를 그대로 호출한다. 새로운 score coverage와 component eligibility는 실제 labels/predictions로 확인한다.

## Execution strategy
### Parallel execution waves
> Target 5-8 todos per wave. Fewer than 3 (except the final) means you under-split.

Wave 1은 1(평가 계약), 2(L4 선측정), 3(데이터)이다. 3의 수집·라이선스 확인은 병행하고 최종 split 쓰기는 1의 계약을 기다린다. Wave 2는 4(File), 5(music), 6(voice)이며 같은 immutable manifest/cache API를 읽고 각기 다른 module/result 경로만 소유한다. Wave 3은 7(후보 결합·동결), 8(package)로 이어진다. package 최종 검증은 후보 동결 뒤에만 가능하므로 개수를 맞추려고 병렬 과제로 쪼개지 않는다. GPU는 한 번에 한 추출/학습 stage만 실행하고 cached head 작업은 CPU로 병행할 수 있다.

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| 1 | 없음 | 3의 split 확정, 4–8 | 2, 3의 source 조사 |
| 2 | 없음, 기존 package 사용 | 5의 raw rollout, 7–8 | 1, 3 |
| 3 | 1의 계약 | 4–7 | 2, 1의 scorer |
| 4 | 1, 3, 같은 train의 control/OOF | 7 | 5, 6 |
| 5 | 1–3, fitting 전 4의 control/hidden API | 7 | 4의 File fitting, 6 |
| 6 | 1–3, singing/telephone 라벨 gate | 7 | 4, 5 |
| 7 | 4–6 완료 또는 근거 있는 gate 탈락 | 8 | 없음 |
| 8 | 2, 7 | F1–F4 | 없음 |

## Todos
> Implementation + Test = ONE todo. Never separate.
<!-- APPEND TASK BATCHES BELOW THIS LINE WITH edit/apply_patch - never rewrite the headers above. -->
- [ ] 1. 평가 계약과 대조군을 고정한다
  - What to do / Must NOT do: `R/protocol.json`에 source/group 정책, known masks, candidate budget, split seed, seeds41/42/43, loss·승격 gate·tie-break·sealed open 횟수·hash를 고정한다. `redesign_evaluate.py` 하나에 기존 scorer 호출, paired group bootstrap, per-seed/ensemble·stratum report만 구현한다. metric을 새로 근사하지 않는다. 현재 package hash `1f6cfd6c9feacead01d4ebcfd217d10c1796969b1f863ba4b0283d46e7432c7e`와 구현 시작 시 `git rev-parse HEAD`의 `base_git_commit`을 control receipt에 보존한다. control 재학습은 task 4/5의 same-train refit을 사용하고 외부 holdout에서는 기존 package도 추가 비교한다.
  - Ownership: `redesign_evaluate.py`, `R/protocol.json`, `R/control_receipt.json`만 소유한다. 원본 `graph_scoring.py` 및 과거 results는 읽기 전용이다.
  - Parallelization: Wave 1 | Blocked by: 없음 | Blocks: 3의 확정 split, 4–8.
  - References: `graph_scoring.py:81–165`(`_roc_metrics`, `official_metrics`); `temporal_full_refit.py:22–64`; `submissions/M330_MLP_full1206/M330_MLP_full1206_submit.json:2–61`; `E/local-evidence.md`, `E/live-evidence.md`, `E/gap-calculation.json`.
  - Acceptance criteria: `python3 redesign_evaluate.py verify --protocol results/redesign_20260917/protocol.json --historical results/fast_h100_20260917/downloaded/unpacked --output results/redesign_20260917/evaluation_contract.json`이 현재 저장된 ensemble/seed metric을 tolerance1e−12로 재현하고 unknown 때문에 총점이 null임을 보존한다. File-only에 동일 나머지 출력을 넣었을 때 partial delta가 `.45*ΔFile`와 일치하고, unknown voice 출력을 변경하면 delta 계산을 거부한다.
  - QA scenarios: happy는 위 명령으로 실제 latest predictions와 labels를 대조한다. failure는 `python3 redesign_evaluate.py verify-failures --protocol results/redesign_20260917/protocol.json --output results/redesign_20260917/evaluation_failures.json`으로 labels가 없는 voice 변경, 단일 class EER, train 참여 head를 OOF로 표시한 receipt를 넣어 승격/stack fitting을 막는지 확인한다. 단순 hard-coded mapping 검사가 아니라 실제 scorer·선택 경로의 잘못된 승인 여부를 검사한다. Evidence: `<attemptDir>/task-1-dacon-model-redesign-20260917.json`에 입력 hash·명령·결과를 기록한다.
  - Commit: Y | `feat(eval): lock independent redesign selection protocol`.

- [ ] 2. 현재 package와 raw 추가 경로의 L4 비용을 먼저 잰다
  - What to do / Must NOT do: `benchmark_redesign.py`로 기존 packaged entrypoint를 실제 L4에서 구동하고 cold start, decode, PANNs, HTDemucs, DF-Arena, MERT, heads, CSV 시간을 분리한다. 라이선스가 확인된 96파일을 duration/channel/codec별로 고정한다. dual music pilot은 이미 있는 `fast_h100_ablation.extract`의 raw/stem 처리와 기존 control head만 사용한다. 아직 구현하지 않은 residual에 의존하지 않으며 전체 residual 비용은 task 8에서 실측한다. H100 throughput을 L4 수치로 바꾸어 쓰지 않는다. 4/60초·mono/stereo·telephone 입력과 기존 실패 가능 codec을 포함하고 sequential event coverage도 기록한다.
  - Ownership: `benchmark_redesign.py`, `R/runtime_pilot/`만 소유한다. 기존 ZIP·runtime은 변경하지 않는다.
  - Parallelization: Wave 1 | Blocked by: 향후 실행 권한과 L4 이용 가능, source 허용 확인 | Blocks: 5의 dual 추출 확대, 7–8.
  - References: `c08_submission.py:38–108`; `baseline_original/script.py:144–175,206–308,340–364`; `fast_h100_ablation.py:113–195`; `submissions/M330_MLP_full1206/README.md:14–20`; `E/architecture-evidence.md`의 Deployment; `E/live-evidence.md`의 runtime.
  - Acceptance criteria: `python3 benchmark_redesign.py --package submissions/M330_MLP_full1206/M330_MLP_full1206_submit.zip --manifest results/redesign_20260917/runtime_pilot/manifest.csv --views stem,raw --output results/redesign_20260917/runtime_pilot`이 actual device name, stage time, files/seconds/windows, peak reserved/allocated VRAM·host RSS·예상 상한을 남긴다. 50분/20 GiB/24 GB 목표를 넘으면 raw를 pending/rejected로 표시하고 stem-only path를 남긴다. 실제 60분/22.4 GiB/28 GB 초과는 hard fail이다.
  - QA scenarios: happy는 실제 stereo 파일의 native-channel separation과 동일 음악 score 경로를 측정한다. failure는 `python3 benchmark_redesign.py coverage-check --durations 4,10,30,60 --event-seconds .5,1,2,4 --output results/redesign_20260917/runtime_pilot/coverage.json`으로 event가 기존 S2의 5–10/20–25/35–40/50–55초 공백에 들어가면 sampled S2가 못 보지만 full grid는 본다는 입력 support를 검증한다. 측정과 모델 감지 성능은 따로 보고한다. 손상 audio는 filename과 stage가 있는 오류를 내고 완성 CSV로 승격하지 않아야 한다. Evidence: `<attemptDir>/task-2-dacon-model-redesign-20260917.json`.
  - Commit: Y | `feat(bench): measure L4 pipeline cost and coverage`.

- [ ] 3. 새로운 component-label 데이터와 잠긴 분할을 만든다
  - What to do / Must NOT do: `redesign_data.py`에서 기존 manifest reader/hash·group 검사를 재사용하고 parent graph, provenance, license, weak/known label, event interval, channel/codec/duration을 추가한다. `D` 안에 immutable train/validation/sealed manifests와 source ledger를 쓴다. synthetic 네 조합·simultaneous/sequential·real-real edit·전화/codec·4–60초를 만들되 짧은 부모를 무조건 `np.resize` 반복하지 않는다. 필요한 길이가 부족하면 허용 crop 길이를 줄이거나 그 조합을 제외한다. 서로 다른 split 부모로 mixture를 만들지 않는다. whole-song fake label은 근거 없으면 vocal/accompaniment 양쪽으로 확장하지 않는다.
  - Ownership: `redesign_data.py`, `D/`, `R/data_audit/`만 소유한다. 기존 `v4_validation`/`v6_data.py`는 유지한다.
  - Source acquisition: 아직 없는 `D/sources.csv`는 이 과제의 산출물이다. `_workspace/v2_sources/source_inventory.json`, `_workspace/v2_discovery/echoes_fma_unique.json`, `_workspace/v2_discovery/archive_probes/echoes_manifest.csv`, `v4_validation/protocol.json:96–126`부터 읽고 `prepare_v4_data.py:175–249`, `prepare_recovery_data.py:130–176`의 FMA/Echoes/MLAAD URL·revision·fetch 경로를 재사용한다. 이미 쓴 artist/song/speaker는 새 holdout에서 제외한다. singing은 공개 MUSDB18·MedleyDB·SVDD 2024의 공식 data card와 SONICS provenance를 우선 조사하는 **미확보 후보**다. 라이선스, 공개 시점, 재배포/대회 허용, singer identity, 실제 생성 component를 확인한 자료만 ledger에 승인한다. 1차 조사 상한은 source 후보 6개·CPU 작업 4시간이다. 확인되지 않으면 다운로드/라벨 확장을 멈추고 부족량을 보고한다. 이 계획이 해당 자료의 사용 허가나 component-label 품질을 확인했다는 뜻은 아니다.
  - Parallelization: Wave 1 | Blocked by: 1의 protocol 확정 | Blocks: 4–7.
  - References: `v6_data.py:13–84`(`sha256`, `keys`, `assert_disjoint`, `read_manifest`, `split_groups`); `v5_dataset.py:184–240`(`training_crop`, `compose_mixture`, `real_real_edit`); `temporal_augmentation.py:32–64`; `results/data_audit_20260915/{summary.json,source_counts.csv,source_profiles.csv}`; `E/local-evidence.md`의 Dataset trust; 공식 description/rules 링크.
  - Acceptance criteria: `python3 redesign_data.py build --protocol results/redesign_20260917/protocol.json --sources _workspace/redesign_20260917/data/sources.csv --output _workspace/redesign_20260917/data`와 `python3 redesign_data.py audit --data _workspace/redesign_20260917/data --output results/redesign_20260917/data_audit`가 source licenses, 분할 count, independent groups, PCM/near-duplicate/parent connected overlap, unknown label count, nuisance probe를 기록한다. train/validation/sealed overlap은 0이어야 하고 최소 표본 gate는 충족/미충족을 구분한다. 로컬 기존 자료를 새 external source라고 표시하지 않는다.
  - QA scenarios: happy는 실제 두 부모의 four-state mixture와 real-real splice를 만들고 audio interval·File OR·known component truth가 같은 provenance를 가리키는지 검토한다. failure는 `python3 redesign_data.py verify-lineage --data _workspace/redesign_20260917/data --inject-cross-split-parent --inject-unknown-vocal --output results/redesign_20260917/data_audit/rejection.json`으로 cross-split 부모를 거부하고 unknown vocal의 authenticity loss mask가 0으로 남는지 실제 training loader까지 확인한다. 0.5초 fake interval이 없는 crop은 fake target으로 들어가면 실패다. Evidence: `<attemptDir>/task-3-dacon-model-redesign-20260917.json`.
  - Commit: Y | `feat(data): build component labels and independent source splits`.

- [ ] 4. 저차원 OOF File 보정을 검증한다
  - What to do / Must NOT do: `redesign_fusion.py`에 명시된 12개 feature, max/noisy-OR control, offset logistic residual만 구현한다. `music_temporal.fit_temporal_head`로 same-train stem control과 train parent-group 5fold OOF를 먼저 생성해 task 5에 제공한다. each fold의 early stopping도 train 내부에서 한다. meta regularization은 outer train 안의 nested group split 두 값 `λ∈{.01,.1}`만 비교하고 external validation은 건드리지 않는다. sklearn의 기존 LogisticRegression 사용 패턴을 참고하되 offset은 torch의 작은 linear objective로 구현한다. full1206 ensemble이나 validation 예측으로 scaler·meta weight를 학습하지 않는다.
  - Ownership: `redesign_fusion.py`, `music_temporal.py`의 `encode_hidden` method 추출, `R/file_relation/`을 소유한다. 기존 forward 결과·checkpoint keys를 유지하고 `encode_hidden` 및 기존 `output`으로 per-window music logit을 얻어 task 5에 공유한다. 이미 실행하는 DF/PANNs window output도 동일 support로 저장한다. 다른 task는 `build_file_features`, `fit_file_relation`, `predict_file_relation`과 serialized scalers/weights를 사용한다. 별도 registry/framework는 만들지 않는다.
  - Parallelization: Wave 2 | Blocked by: 1, 3; 같은 train의 stem control OOF 준비 | Blocks: 7 | Parallel: 5, 6.
  - References: `v6_train.py:66–108`의 fitting·audit 패턴; `v6_core.py:14–26`; `temporal_full_refit.py:22–64`의 fold split; `c08_submission.py:27–35`; `E/architecture-evidence.md`의 Objective mismatch; `results/fast_h100_20260917/downloaded/unpacked/metrics.json:459–468,1221–1303`.
  - Acceptance criteria: `python3 redesign_fusion.py run --data _workspace/redesign_20260917/data --protocol results/redesign_20260917/protocol.json --output results/redesign_20260917/file_relation`에서 모든 meta row가 underlying head의 train group과 disjoint임을 증명한다. frozen component 출력이 동일하고 validation File gain과 `.45ΔFile` CI가 shared gate를 통과할 때만 accepted로 쓴다. natural 악화·mixed validation 악화면 File=max를 유지한다.
  - QA scenarios: happy는 max-preserving zero weight에서 원래 File probability와 absolute tolerance1e−7 이내 일치한 뒤 실제 OOF fitting 결과를 평가한다. failure는 `python3 redesign_fusion.py verify --run results/redesign_20260917/file_relation --swap-fold-receipt --output results/redesign_20260917/file_relation/verification.json`으로 학습 group이 섞인 receipt를 거부한다. 입력 파일 순서와 이웃 파일을 바꿔도 해당 파일의 probability가 tolerance1e−6 안에서 같아야 한다. Evidence: `<attemptDir>/task-4-dacon-model-redesign-20260917.json`.
  - Commit: Y | `feat(fusion): fit file correction on group out-of-fold evidence`.

- [ ] 5. 같은 감독 조건에서 music raw/stem conditioning을 검증한다
  - What to do / Must NOT do: `redesign_music.py`에 198→32→64 residual과 event loss를 구현한다. task 4가 제공하는 `TemporalMusic.encode_hidden`·control checkpoint를 읽고 기존 forward나 checkpoint keys를 바꾸지 않는다. A/B/C/D의 draw plan·41/42/43·valid-window count를 고정한다. 기존 fast extractor의 frozen MERT330 loading/batching/cache receipt를 가져와 새 cache key에 waveform path·separator·channels·resampling·checkpoints·starts·view·dtype를 모두 넣는다. pure/mixed별 fitting path가 실제 loader에서 사용되는지 검사한다.
  - Ownership: `redesign_music.py`, `R/music_ablation/`만 소유한다. task 4가 소유한 `music_temporal.py`나 task 6 module을 수정하지 않는다.
  - Parallelization: Wave 2 | Blocked by: 1–3, L4 dual gate; fitting은 4가 제공한 control receipt 이후 | Blocks: 7 | Parallel: 4의 File fitting, 6. feature 추출과 module 구현은 4의 control fit 중에도 진행한다.
  - References: `music_temporal.py:86–238`(`TemporalMusic`, `_batch`); `temporal_augmentation.py:32–64`; `fast_h100_ablation.py:113–195,259–285`; `c08_submission.py:74–91`; `results/fast_h100_20260917/downloaded/unpacked/selection.json:2–55`·`results/fast_h100_20260917/downloaded/unpacked/test_seed_probabilities.npz`; `E/architecture-evidence.md`의 Time coverage.
  - Acceptance criteria: `python3 redesign_music.py run --data _workspace/redesign_20260917/data --protocol results/redesign_20260917/protocol.json --seeds 41,42,43 --arms A,B,C,D --output results/redesign_20260917/music_ablation`는 L4 dual gate 통과 시 12fit와 seed41의 same-capacity control 1fit를 저장한다. task 2에서 raw runtime gate가 실패하면 같은 명령의 `--arms A,C`로 6fit만 수행하고, B/D 및 same-capacity control은 `skipped_runtime_gate`로 기록한다. 이 경로에는 12fit acceptance를 적용하지 않는다. 파일 수뿐 아니라 class/source/window/update 노출표를 낸다. zero residual parity ≤1e−7, 새 branch의 nonzero gradient와 optimizer update, all-window support를 확인한다. C→D의 weighted Music+File Δ와 natural/telephone/sequential gate가 통과한 경우에만 raw를 유지한다. 2의 runtime fail이면 A/C만 수행하고 그 이유를 기록한다.
  - QA scenarios: happy는 known fake interval을 0/중간/끝으로 옮긴 동일 parent bag에서 event detection·File ranking과 real-real control 오탐을 측정한다. failure는 `python3 redesign_music.py verify --run results/redesign_20260917/music_ablation --checks gradient,zero-residual,raw-shuffle,partial-fake,cache-mismatch --output results/redesign_20260917/music_ablation/verification.json`으로 raw 차단 시 gradient 경로, wrong-time raw의 성능 영향, fake가 없는 crop의 masked loss, 바뀐 separator hash의 stale cache 거부를 검사한다. gradient는 zero-init 직후와 다음 update 이후를 구분해 W2만 초기 gradient를 받는 정상 동작을 실패로 오판하지 않는다. Evidence: `<attemptDir>/task-5-dacon-model-redesign-20260917.json`.
  - Commit: Y | `feat(music): test aligned residual conditioning with mixed supervision`.

- [ ] 6. singing·telephone voice의 잔여 오류를 보완한다
  - What to do / Must NOT do: `redesign_voice.py`가 frozen DF-Arena의 기존 vocal-path probability/embedding을 함께 읽고 1280→32→1 residual을 학습한다. pure spoken, pure singing, simultaneous/sequential singing, 전화 대역 real/fake를 source-disjoint로 비교한다. whole-song 라벨만 있는 항목은 voice authenticity loss에서 제외한다. max aggregation과 frozen PANNs를 유지하고 clean speech regression을 별도 gate로 둔다. label gate가 미달이면 voice 후보를 provisional로 남기고 학습 실행을 강제하지 않는다.
  - Ownership: `redesign_voice.py`, `R/voice_residual/`만 소유한다. frozen wrapper를 사용하는 과정에서 `v5_experts.py`의 전역 trainability를 바꾸지 않는다.
  - Parallelization: Wave 2 | Blocked by: 1–3, singing/telephone known labels | Blocks: 7 | Parallel: 4, 5.
  - References: `v5_experts.py:48–94`(`VoiceExpert.forward`); `baseline_original/script.py:264–308,315–358`; `v5_training.py:363–490`; `E/local-evidence.md`의 latest voice unknown counts; `E/architecture-evidence.md`의 S2 separator 구분.
  - Acceptance criteria: `python3 redesign_voice.py run --data _workspace/redesign_20260917/data --protocol results/redesign_20260917/protocol.json --seeds 41,42,43 --output results/redesign_20260917/voice_residual`가 baseline native logits 방향·window score parity ≤1e−6, residual optimizer update, eligible group 수, clean/singing/telephone metric을 기록한다. Voice+File identifiable Δ의 CI와 shared gate를 통과할 때만 채택한다. clean speech File/Voice EER >.01 악화는 탈락이다.
  - QA scenarios: happy는 실제 real/fake singing과 전화 변형의 대응 parent에서 score와 conditional eligibility를 확인한다. failure는 `python3 redesign_voice.py verify --run results/redesign_20260917/voice_residual --checks unknown-label,real-enhancement,clean-regression,gradient --output results/redesign_20260917/voice_residual/verification.json`으로 unknown singer truth가 gradient를 만들지 않는지, real denoise/gain control이 fake 감독에 들어가지 않는지 확인한다. fixed voice와 residual voice가 같은 vocal signal을 받지 않으면 matched comparison을 거부한다. Evidence: `<attemptDir>/task-6-dacon-model-redesign-20260917.json`.
  - Commit: Y | `feat(voice): evaluate singing and telephone residual correction`.

- [ ] 7. 통과한 변경만 합치고 최종 후보를 잠근다
  - What to do / Must NOT do: `redesign_train.py`는 기존 task module을 호출하는 작은 runner로 한정한다. accepted music·voice 조합에서 train-only OOF를 새로 만들고 File relation을 다시 fit한다. 서로 다른 upstream의 OOF·scaler를 섞지 않는다. single head 대 41/42/43 probability ensemble을 정해진 validation 규칙으로 선택한다. 모든 주후보 gate가 끝난 후 Scope의 조건을 충족한 경우에만 9-state, 짧은 log-mel, limited adaptation 중 하나를 열고, 선택 이유·추가 budget·변하지 않는 대조 조건을 protocol addendum에 먼저 고정한다. 결과를 본 뒤 variant를 늘리지 않는다.
  - Ownership: `redesign_train.py`, `R/selection/`, 필요할 때만 `redesign_joint.py` 또는 `redesign_logmel.py` 중 한 파일. optional adaptation은 기존 head module의 허용 trainable-name 목록으로 구현한다. 앞 task의 완료 module 변경은 해당 소유자에게 돌리고 별도 commit으로 남긴다.
  - Parallelization: Wave 3 | Blocked by: 4–6 완료/탈락 receipt | Blocks: 8.
  - References: `v5_fusion.py:34–54,64–124`; `v5_fusion_training.py:121–239`; fullFT `results/fullft8_h100x1_20260916/downloaded/h100x1_fullft8_run/worker.py:351–485`; `E/kaggle-evidence.md`; 앞 tasks의 immutable prediction/checkpoint receipts.
  - Acceptance criteria: `python3 redesign_train.py select --protocol results/redesign_20260917/protocol.json --data _workspace/redesign_20260917/data --runs results/redesign_20260917/file_relation,results/redesign_20260917/music_ablation,results/redesign_20260917/voice_residual --output results/redesign_20260917/selection`이 전체 공식 S 또는 정확한 partial Δ, CIs, counts, source stress, L4 gate, chosen hashes를 `locked_candidate.json`에 쓴다. `python3 redesign_evaluate.py open-sealed --lock results/redesign_20260917/selection/locked_candidate.json --data _workspace/redesign_20260917/data --output results/redesign_20260917/selection/sealed`는 1회 사용 receipt를 만든다. sealed 실패면 후보를 실패로 기록하며 sealed를 이용해 재학습·재선택하지 않는다.
  - QA scenarios: happy는 upstream을 바꾼 조합의 OOF가 새 hash로 생성되고 natural/mixed/voice 누적 개선이 재현되는지 확인한다. failure는 `python3 redesign_train.py verify-selection --run results/redesign_20260917/selection --checks reused-oof,reopen-sealed,unknown-total,unsupported-escalation --output results/redesign_20260917/selection/verification.json`으로 old upstream OOF, 두 번째 sealed 열람, missing labels로 계산한 absolute score, 근거 없는 optional branch를 막는다. optional training은 실제 실행/gradient/update와 예산 종료 상태가 있어야 완료로 친다. Evidence: `<attemptDir>/task-7-dacon-model-redesign-20260917.json`.
  - Commit: Y | `feat(selection): lock independently validated redesign candidate`.

- [ ] 8. 독립 파일 추론과 최종 ZIP을 검증한다
  - What to do / Must NOT do: `redesign_submission.py`에서 기존 파일 discovery·sample CSV order·native separator·PANNs release·offline model loading·atomic CSV 저장을 재사용한다. `package_redesign_submission.py`는 `package_submission._model_members`와 기존 hash manifest 방식을 재사용해 선택된 weight와 필요한 runtime만 포장한다. 기존 `package_c08_submission.py`의 full1206 전용 assert를 느슨하게 바꾸지 않는다. inference에서 cache lookup은 해당 파일의 실제 waveform 처리 결과만 허용하고 training/evaluator ID 기반 prediction lookup을 금지한다.
  - Runtime manifest producer: task 8이 `benchmark_redesign.py`에 `build-runtime-manifest`를 추가한다. `python3 benchmark_redesign.py build-runtime-manifest --sources _workspace/redesign_20260917/data/train.csv --protocol results/redesign_20260917/protocol.json --count 1200 --seed 41 --output results/redesign_20260917/package_qa/1200_manifest.csv`로 라이선스·hash가 확인된 비평가용 train audio만 선택한다. sealed와 DACON 평가 파일은 사용하지 않는다. 4–10/10–30/30–60초를 각 400개로 고정하고 각 duration 구간 안에서 mono/stereo를 200개씩 배정한다. codec은 WAV/FLAC/MP3, clean/telephone, pure/simultaneous/sequential을 가능한 한 균등 배분한다. 부족한 셀은 train parent로 결정적인 crop·codec·mix 변형을 생성하고 부모 ID·변형·hash·실제 duration을 기록한다. 동일 원본을 반복한 것은 runtime 측정 표본일 뿐 독립 성능 표본으로 세지 않는다. 실제 stereo source가 없으면 mono 복제로 대체하지 말고 해당 runtime gate를 미충족으로 둔다. 1,200행 모두 서로 다른 output ID/파일을 가지며 실제 duration 합계를 timing과 함께 보고한다. 이 생성 규칙은 숨은 평가 분포를 추정했다는 뜻이 아니다.
  - Ownership: `redesign_submission.py`, `package_redesign_submission.py`, `submissions/redesign_20260917/`, `R/package_qa/`, task 2 완료 뒤의 `benchmark_redesign.py` manifest subcommand만 소유한다. 최고 M330 artifact는 보존한다.
  - Parallelization: Wave 3 | Blocked by: 2, 7 accepted lock | Blocks: F1–F4.
  - References: `c08_submission.py:38–108`; `package_c08_submission.py:31–105` 및 이어지는 ZIP manifest/verify 처리; `package_submission.py`의 `_model_members`; `graph_scoring.py:48–79`; `submissions/M330_MLP_full1206/verify_submission.py`; 공식 evaluation/rules.
  - Acceptance criteria: `python3 package_redesign_submission.py --lock results/redesign_20260917/selection/locked_candidate.json --output submissions/redesign_20260917` 후 `python3 benchmark_redesign.py verify-package --package submissions/redesign_20260917/redesign_submit.zip --manifest results/redesign_20260917/package_qa/1200_manifest.csv --offline --output results/redesign_20260917/package_qa`로 실제 L4 cold-start 1,200파일, ZIP/unpacked/install/time/VRAM/RAM 제한, reference 대비 max probability diff≤1e−5를 확인한다. tolerance 초과는 원인을 밝히고 score/순위 안정성을 별도로 확인하기 전까지 실패다. 설치 후 네트워크 없이 실행해야 한다. 학습 code/data provenance/HWP 보고서 재료의 index를 남기되 이번 설계에서 HWP를 미리 만들지는 않는다.
  - QA scenarios: happy는 최종 ZIP을 새 디렉터리에서 풀어 reference와 모든 ID·5개 probability를 비교한다. failure는 같은 A 파일을 단독, A→B, B→A, B 대신 다른 C와 실행해 A의 결과가 tolerance1e−6 안에서 같고 frozen eval state가 변하지 않는지 확인한다. weight 한 byte 훼손과 stale cache는 inference 전에 거부해야 한다. 60초 stereo·전화·손상 input 및 GPU OOM은 미완성 CSV를 final로 남기면 실패다. Evidence: `<attemptDir>/task-8-dacon-model-redesign-20260917.json`.
  - Commit: Y | `feat(package): verify offline independent L4 inference`.

## Final verification wave
> Runs in parallel after ALL todos. ALL must APPROVE. Surface results and wait for the user's explicit okay before declaring complete.

이번 설계의 검토는 파일·근거 대조로 완료한다. 위 scaffold의 실행 승인 문구는 이미 승인된 설계 작업을 멈추라는 뜻으로 적용하지 않는다. 향후 실행을 별도로 요청받았을 때 아래 네 검증을 수행한다. 모델 검증 완료와 공식 #1 달성은 별도로 보고하고, 요청 범위를 넘는 유료 실행·제출은 해당 실행 권한이 있을 때만 한다.

- [ ] F1. Plan compliance audit
  - Tool/inputs: `functions.exec`의 `tools.exec_command`로 `python3 redesign_train.py verify-selection --run results/redesign_20260917/selection --checks reused-oof,reopen-sealed,unknown-total,unsupported-escalation --output results/redesign_20260917/selection/final_verification.json`을 실행한다. 이어 같은 도구의 `cat`으로 `results/redesign_20260917/protocol.json`, `control_receipt.json`, `selection/locked_candidate.json`, data audit·각 task acceptance receipt를 읽고 8개 과제를 대조한다. 실제 attemptDir는 실행의 protocol에 기록된 값을 사용한다.
  - APPROVE: same-train control/OOF, label·license·split, CI 및 natural/voice gate, sealed 1회와 budget을 충족하고 모든 task가 완료 또는 사전에 정한 조건에 따라 탈락한 경우. package 후보는 승격 gate를 통과해야 하며, 탐색 결과만 있으면 구현 연구 완료와 후보 미확보를 분리해 기록한다. 근거 없는 fullscore·#1·속도 주장, 명령 실패, 누락 receipt는 REJECT.
  - Receipt: `<attemptDir>/final-F1-compliance.json`에 `verdict`, 실행 명령/exit code, 검토한 입력 hash, criterion별 근거와 blockers를 기록한다. 검토자가 직접 확인한 결과만 승인한다.
- [ ] F2. Code quality review
  - Tool/inputs: `functions.exec`의 `tools.exec_command`로 `python3 -c 'import json,subprocess; r=json.load(open("results/redesign_20260917/control_receipt.json")); subprocess.run(["git","diff","--no-ext-diff",r["base_git_commit"],"HEAD","--","*.py"],check=True)'`를 실행한다. task 1의 baseline commit부터 실제 변경 전체를 읽고, 해당 source와 task 4–6의 gradient/OOF/unknown-mask receipts, task 8의 package parity·file independence receipts를 대조한다. 빈 unstaged diff를 구현 검토로 대체하지 않는다.
  - APPROVE: forward에서 새 trainable branch가 실제 사용되고 optimizer/gradient가 연결되며, zero-init identity·unknown mask·OOF 경계·frozen eval state·atomic output·cache hash 계약이 구현과 증거에서 일치하는 경우. source만 있고 실측이 없거나 중대한 오류가 있으면 REJECT. 형식상 통과하는 shallow test와 중복 framework는 품질 근거로 인정하지 않는다.
  - Receipt: `<attemptDir>/final-F2-code.json`에 `verdict`, baseline/head commit, diff 범위, 확인한 source/receipt, actionable blockers를 기록한다.
- [ ] F3. Real manual QA
  - Tool/inputs: `functions.exec`의 `tools.exec_command`로 `python3 benchmark_redesign.py audit-package-run --run results/redesign_20260917/package_qa --package submissions/redesign_20260917/redesign_submit.zip --manifest results/redesign_20260917/package_qa/1200_manifest.csv --output results/redesign_20260917/package_qa/final_audit.json`을 실행한다. 이 read-only subcommand는 task 8에서 작성하며 실제 실행 로그의 device·cold start·1,200개 ID/파일·총 duration·시작/종료 시각·peak memory·offline·parity 기록을 원본 CSV/ZIP hash와 재대조한다. 별도 검증자는 task 8의 실제 stereo/telephone/sequential 입력과 실패 로그를 열어 이벤트 구간·atomic output 결과를 확인한다. GPU 실행을 감사용으로 무조건 반복하지 않는다.
  - APPROVE: 최종 동일 ZIP의 실제 L4 실행이 1,200파일 전체를 완료했고 정해진 시간·memory·설치·크기·parity·파일 독립성 gate를 통과한 경우. H100/A100 로그, extrapolation만 있는 기록, shape/import 검사, 다른 ZIP hash는 REJECT. 실제 평가 duration 분포가 알려지지 않았다는 한계는 남긴다.
  - Receipt: `<attemptDir>/final-F3-runtime.json`에 `verdict`, 명령/exit code, 실제 L4 run·ZIP·manifest identity, 실측값, 직접 연 입력과 blockers를 기록한다.
- [ ] F4. Scope fidelity
  - Tool/inputs: `functions.exec`의 `tools.exec_command`로 `python3 -m zipfile -l submissions/redesign_20260917/redesign_submit.zip`을 실행해 실제 packaged entrypoint와 manifest를 찾고, 동일 도구로 package runtime·calibration·모든 모델 loading 경로를 읽는다. `web.run`으로 `https://dacon.io/competitions/official/236749/overview/rules`와 `/overview/evaluation`을 다시 연 뒤 task 8의 file-neighbor independence·offline receipts 및 task 3 source/license ledger와 대조한다. AST/텍스트 금칙어 검색만으로 승인하지 않는다.
  - APPROVE: 평가 데이터로 학습·pseudo-label·상태 갱신·파일 간 보정이 없고 calibration이 사전 고정돼 있으며 conditional authenticity, 라이선스/출처, runtime/package 계약이 실제 경로에서 유지되는 경우. 기존 최고 M330 ZIP의 hash가 control receipt와 같아야 한다. 독립 검증자가 기존 ZIP을 SHA-256으로 다시 읽어 확인한다. 위반·미확인 실행 경로는 REJECT.
  - Receipt: `<attemptDir>/final-F4-scope.json`에 `verdict`, 확인 시각/공식 URL, 읽은 package members·hash, rule별 근거, 기존 M330 hash, blockers를 기록한다. 설계 완료·구현 완료·공식 순위 달성을 각각 구분한다.

## Commit strategy

이번 설계 파일과 근거 문서는 root가 검토 후 로컬 commit한다. 이 계획을 작성하는 담당자는 다른 파일을 변경하거나 commit하지 않는다. 향후 구현은 todo별 위의 작은 commit을 사용하고 테스트·실측 receipt를 같은 단위로 연결한다. `.git`이 있으면 변경을 로컬 commit하되 push·PR·배포를 자동으로 추가하지 않는다. 대형 audio·cache·ZIP·checkpoint는 기존 artifact 저장 관례에 따르고 Git에는 source·manifest·hash·재현 명령만 남긴다. 이미 추적 중인 user 변경은 되돌리거나 섞어 commit하지 않는다.

## Success criteria

이번 설계는 네 근거 문서와 실제 코드에 맞는 단일 기본안, 8개 구현 과제, 4개 최종 검증, shared metric/label/split/resource/stop gate를 갖추고 review를 통과하면 완료다. 구현·학습·L4 검증·제출·1위는 완료로 표시하지 않는다.

향후 모델 후보는 새로운 독립 패널에서 공식 총점 또는 식별 가능한 정확한 weighted delta의 승격 조건을 통과하고, natural/singing/telephone을 악화시키지 않으며, 실제 최종 ZIP이 L4 제한과 offline/file-independence 검증을 통과해야 accepted다. data gate가 부족하거나 CI가 넓으면 탐색 결과로만 남긴다. 모든 신규 후보가 탈락해도 실패 원인과 artifact를 기록하고 현 최고 package를 보존하는 것은 유효한 연구 결과다.

최종 목표의 완료 조건은 **공식 사이트에서 해당 제출과 팀의 리더보드 1위가 확인되는 것**이다. 점수 향상이나 로컬 우위만으로 이를 대체하지 않는다. Top 15 이후 심사와 최종 수상 순위는 별도 절차다.
