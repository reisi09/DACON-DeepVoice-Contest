# 다섯 출력·MTL·음원 전처리 제안 검토

2026-09-25. 현재 보존 DataV4 및 PANNs v1 코드, 실제 학습 이력, 공식 규정, 현재 D0 메타데이터와 CPU 점수 실험을 대조했다. 현재 보존 패키지가 사용자가 말한 리더보드 최고점 제출물과 동일하다는 증거는 아직 없다. 아래 현재 구조는 보존 패키지와 준비 중인 PANNs v1을 뜻한다.

## 현재 처리 여부

| 제안 | 실제 상태 | 판단 |
|---|---|---|
| 조건부 확률을 이용한 FILE 후처리 | presence를 곱한 max, noisy-OR, presence를 쓰지 않는 max를 각각 고정 CDF로 변환한 뒤 동일 가중치 평균 | 이미 적용. 새로운 상관관계 학습이나 교차 적합 결합은 별도 비교할 가치가 있다. |
| 하나의 backbone과 다섯 head MTL | AST 기반으로 실제 학습·과거 제출 이력 있음. 현재 보존 DataV4의 활성 경로에는 쓰이지 않음 | 처음 시도하는 해법이 아니다. 전문 표현을 유지한 shared fusion·9-state joint와 비교한다. |
| 300–3400 Hz·codec 증강 | offline 생성 경로와 실제 학습 대상 메타데이터에 포함 | 추가 유무보다 미지 codec 일반화와 강도·순서·조건별 효과를 비교한다. |
| 모노·샘플링레이트 통일 | decode에서 mono 16 kHz, 음성 16 kHz, MERT 24 kHz, PANNs 32 kHz | 이미 처리. 모델별 입력 규격은 유지한다. |
| 음량 정규화 | 추론에서 peak가 0.6을 넘으면 감쇠. 데이터 생성에는 RMS gain 변화도 있음 | LUFS 정규화와 다르다. 정규화 추가는 train/cache/inference를 맞춘 비교로 검증한다. |
| 음성 SSL | 이미 Wav2Vec2 XLS-R 1B 기반 DF-Arena + Conformer 사용 | 현재 없는 기능이 아니다. raw/stem 경로 및 일부 미세조정을 비교할 수 있다. |
| 음악용 AST/EfficientNet | 음악 fake는 이미 MERT + WindowMIL. AST 학습 이력도 있음 | 필수 구조로 단정할 수 없다. 음악 오류를 보완하는지 비교한다. |

코드 근거: [FILE 결합](../../../repository/decision_heads.py), [활성 추론 경로](../../../repository/full_bank_submission.py), [AST 다중과제](../../../repository/ast_multitask.py), [기존 학습 결과](../../RESULTS_V2.md), [전화 codec 생성](../../../repository/data_v4/build_manifest.py). 상세 감사·원문 출처 기록은 로컬 `_workspace/panns_cnn_v1/proposal_audit/`에 보존했다.

## 후처리에서 구분해야 할 확률

공식 Voice/Music EER 평가 대상은 **정답 presence=1인 파일**이다. 예측 presence로 평가 대상이 정해지지 않는다. [운영진 답변](https://dacon.io/competitions/official/236749/talkboard/417333), [평가식](https://dacon.io/competitions/official/236749/overview/evaluation).

현재 fake head는 성분이 존재하는 알려진 진위 라벨을 학습하므로 조건부 fake score로 해석한다. 공식 안내가 열마다 엄밀한 조건부 확률식을 명시했다는 뜻은 아니다. 이상적으로 보정된 확률이라면 다음을 구분한다.

- `pv = P(voice present | x)`
- `fv = P(voice fake | voice present, x)`
- `av = pv * fv = P(voice present and fake | x)`
- `P(file fake | x) = av + am - P(voice fake and music fake | x)`

마지막 교집합을 `av*am`으로 대체하는 noisy-OR에는 조건부 독립 가정이 추가된다. 한 성분 안에서 `pv*fv`를 계산하는 확률의 곱셈 법칙 자체에는 이 독립 가정이 필요하지 않다.

따라서 `fv<=pv` 제약이나 다섯 값의 합이 1이라는 제약은 맞지 않는다. `VOICE_FAKE_PROB=fv*pv`로 자동 대체하면 정답 음성이 있는 파일 안에서도 순위가 바뀐다. 현재 FILE의 CDF 값은 순위 기반 점수이며, 잘 보정된 사건 확률이라고 입증되지 않았다. 여기에 확률 일관성 제약을 강제로 투영하는 것도 별도 검증이 필요하다.

현재 FILE은 `max(pv*fv, pm*fm)`, `1-(1-pv*fv)*(1-pm*fm)`, `max(fv,fm)`의 고정 validation CDF 평균이다. 평가 batch 자체의 순위를 쓰는 방식은 아니다. E08은 이 학습된 presence/CDF 경로를 제외했고 fake head 등도 바꿨으므로, 당시 점수 하락을 한 가지 후처리의 탓으로 확정할 수 없다.

## 이번에 직접 비교한 결과

새 학습 없이, 사전에 정한 다섯 변환을 과거 예측값에 적용했다. 아래는 동일한 complete-label 개발 패널 6,183개, 부모 연결 집단 2,572개다. Voice EER은 5,139개, Music EER은 2,588개에서 계산했다. EER은 낮을수록 좋다.

| 변환 | FILE EER | Voice EER | Music EER |
|---|---:|---:|---:|
| 과거 고정 CDF 기준 | 0.10480029 | 0.12280055 | 0.17612018 |
| fake 출력 두 개에 presence 곱하기, FILE 유지 | 0.10480029 | 0.12433595 | 0.18431885 |
| FILE만 raw max로 변경 | 0.10351641 | 0.12280055 | 0.17612018 |
| FILE만 noisy-OR로 변경 | 0.10364943 | 0.12280055 | 0.17612018 |
| FILE만 presence 없는 max로 변경 | 0.11433303 | 0.12280055 | 0.17612018 |

fake×presence의 음성·음악 EER 변화는 각각 `+0.00153540`, `+0.00819866`였다. 부모 집단을 함께 재표집한 2,000회 bootstrap의 95% 구간은 각각 `[-0.00112809,+0.00208600]`, `[-0.00242054,+0.01148749]`로 0을 포함한다. Raw max와 noisy-OR의 작은 FILE 개선 구간도 0을 포함하고, 과거 val에서는 점 추정치가 반대로 움직였다. 이 결과로 승자를 고르지 않았다.

실행은 CPU 14.75초, 최대 RSS 약 0.947 GiB였다. 모델을 새로 추론하거나 학습한 결과가 아니다. 과거 CDF의 평가자료 노출, 현재 보존 패키지와 다른 fake checkpoint/CDF, 두 성분이 모두 없는 표본 부재라는 한계가 있다. 현재 PANNs v1 성능이나 공식 ADS로 환산하지 않는다. 전체 결과·입력 해시·재현 코드는 로컬 `_workspace/panns_cnn_v1/proposal_audit/conditional_scores/`에 보존했다.

양의 temperature나 양의 기울기의 Platt처럼 한 열에 적용하는 엄격히 증가하는 변환은 수치적 tie·clipping이 없다면 그 열의 ROC·AUC·EER을 바꾸지 않는다. 따라서 presence temperature만으로 CPS 순위 성능을 회복할 수는 없다. 보정한 여러 출력을 FILE에 결합할 때에는 FILE 순위가 달라질 수 있다. Brier/log loss 개선과 ADS/CPS 개선은 따로 확인해야 한다.

## MTL은 어디를 바꿀 것인가

AST 하나에서 5개의 Linear head를 내는 모델은 실제로 학습했다. AST_FROZEN과 AST_LAST_BLOCK은 개발 proxy가 올랐지만 음성 성능 조건을 통과하지 못해 원래 비교에서는 G0가 선택됐다. 이후 1,029개 파일로 재학습한 AST 패키지의 과거 제출 기록은 `0.6174819841`이다. 이는 보관된 기록이며, 지금 다른 데이터·학습 예산의 모델과 공정하게 비교한 수치는 아니다. MTL 전체가 실패한다고 결론 내릴 근거도 아니다.

현재 PANNs v1은 **presence 두 개의 loss만** 학습한다. Voice/Music fake는 그대로 복사하고 FILE만 새 presence로 다시 계산한다. 다섯 열을 출력한다고 다섯 과제가 PANNs로 gradient를 보내는 것은 아니다. 따라서 현재 실험만으로 Music fake EER이나 Voice fake EER이 개선되지는 않는다.

새 비교에서는 다음 세 조건을 같은 데이터·역할·예산으로 맞추는 것이 유용하다.

1. 기존 XLS-R·MERT·PANNs 표현과 독립 head.
2. 같은 표현을 결합한 shared trunk + masked 5-head loss.
3. voice/music 각각 `없음·REAL·FAKE`의 3×3=9개 결합 상태를 학습하고, 다섯 출력을 주변확률·조건부 비율로 계산.

3번의 구현은 `repository/v5_fusion.py`에 이미 있다. 완료된 품질 비교 결과는 없다. 구조가 확률 일관성을 만들 수는 있지만 일반화를 보장하지 않으며, 데이터의 성분 동시 출현 편향을 배울 위험도 비교한다. 입력 캐시의 역할·전처리·출처가 맞으면 작은 fusion 비교부터 시작할 수 있다. backbone 미세조정은 별도 A100 실험이다. 각 loss가 의도한 공유 층으로 실제 gradient를 보내는지, 전문 모델 대비 어느 지표가 나빠지는지 측정한다.

## 전화·codec 증강은 이미 얼마나 있는가

현재 D0 42,939개 전부에 codec_chain이 있고 누락은 0개다. 전화 codec은 8,749개(20.3754%), train에서는 5,831/28,315개(20.5933%)다. train의 진위별 전화 codec 비율은 다음과 같다.

| 정답 성분·진위 | 전화 codec / 해당 train 표본 | 비율 |
|---|---:|---:|
| voice present, FAKE | 3,189 / 15,480 | 20.6008% |
| voice present, REAL | 1,520 / 7,327 | 20.7452% |
| music present, FAKE | 914 / 4,395 | 20.7964% |
| music present, REAL | 706 / 3,365 | 20.9807% |

모르는 라벨은 REAL로 세지 않았다. 거친 class 단위 차이는 0.19 percentage point보다 작지만 source·생성기·codec 종류별 혼동까지 없다는 뜻은 아니다. 해시가 맞는 메타데이터 확인이며 이번에 모든 음원을 재생·대역폭 측정한 것은 아니다.

생성 코드는 300–3400 Hz 제한 → 8 kHz codec → 16 kHz decode를 수행한다. 현재 optimizer는 waveform 대신 캐시된 특징을 읽는다. 증강 코드만 추가하고 예전 캐시를 그대로 읽으면 학습 입력은 바뀌지 않는다. 새 증강 음원은 별도 ID·해시·부모 계보를 붙여 재추출하고, 기존 캐시는 원본 조건용으로 보존한다.

다음에는 기존 처리군을 대조군으로 두고 미지 codec·bitrate·bandwidth profile, 약한 혼합 성분, clean/channel 교차 평가를 비교한다. 원본 REAL의 비생성 codec 후처리는 공식 정의상 REAL이지만, 필터가 유일한 가청 성분을 제거하면 presence 라벨 보존은 별도로 확인한다. 전화 음질 증강이 Public/Private 격차를 줄일 것이라는 보장은 없다. 대회가 실제로 전화 채널을 포함하므로 타당한 강건성 실험이라는 판단이다. [공식 입력 설명](https://www.dacon.io/competitions/official/236749/overview/description).

## 정규화·SSL·음악 표현의 판단

모노 통합은 이미 수행한다. 대회 파일은 16 kHz이고 채널이 섞여 있다. 공통 16 kHz 이후 MERT와 PANNs는 사전학습 입력 규격에 맞게 24/32 kHz로 resample한다. 이는 사라진 고주파를 복원하는 처리가 아니다.

`librosa.util.normalize`는 기본 `norm=inf`에서 peak 정규화다. LUFS는 K-weighting·에너지·게이팅 등을 사용하는 별도 음량 측정이다. [librosa 공식 문서](https://librosa.org/doc/main/generated/librosa.util.normalize.html), [ITU-R BS.1770-5](https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1770-5-202311-I!!PDF-E.pdf).

현재 presence head의 입력에는 mix_rms_db도 있다. 추론에만 peak-to-1이나 LUFS 처리를 넣으면 학습 당시의 입력 분포가 바뀐다. 비교한다면 현재 peak cap, RMS 방식, LUFS 방식을 같은 train/cache/inference 조건으로 맞춘다. 무음·작은 잡음의 증폭, clipping, 원본과 분리 stem의 에너지 관계를 함께 측정한다. 순수 gain 증강 직후 peak 정규화를 하면 gain 차이가 사라질 수 있어 순서도 고정한다.

음성에는 이미 XLS-R 1B 기반 모델을 쓴다. wav2vec2 계열과 RawBoost가 ASVspoof에서 유효했던 논문 근거는 있으나, 모든 CNN보다 우월하거나 위상·언어 단서를 반드시 포착한다는 보장은 없다. 지금 Cnn14는 진위가 아닌 존재 여부를 다루므로 그 주장을 PANNs 교체 근거로 바로 적용할 수 없다. [SSL·증강 논문](https://arxiv.org/html/2202.12233v2).

음악에는 이미 음악 SSL인 MERT를 사용한다. AST/EfficientNet을 추가한다면 이름이나 모델 수로 채택하지 않고 MERT의 오류를 실제로 보완하는지 확인한다. raw mixture와 분리 backing의 차이, sung vocal 누설, generator·codec별 실패, 앙상블의 추가 추론 시간을 함께 본다. MERT나 AST의 원 논문도 이 대회의 생성 음악 탐지 성능을 보증하지 않는다. [MERT](https://arxiv.org/abs/2306.00107), [AST](https://arxiv.org/abs/2104.01778).

## 실행 후보와 판단 기준

| 순서 | 비교 | 필요한 자원·확인할 결과 |
|---|---|---|
| 1 | 고정 산술 후처리 대조 | 이번 CPU 진단 완료. 다음 학습 결합과 비교할 기준으로 보존. |
| 2 | 독립 head / shared fusion / 9-state joint | 유효한 동종 역할 캐시가 있으면 작은 head는 로컬 CPU부터. OOF가 필요하면 각 fold의 head를 다시 학습하며 최종 head의 train 예측을 OOF라 부르지 않음. |
| 3 | 기존 증강 / 미지 channel 조건 추가 | waveform 자료 준비는 CPU, 새 특징 추출·미세조정은 A100. clean 성능과 미지 codec 성능을 함께 비교. |
| 4 | 현재 peak cap / RMS / LUFS | 음원 통계·변환 준비는 CPU. 변환된 실제 인코더 출력과 학습·추론 일치 확인 후 채택. |
| 5 | XLS-R raw·stem 및 음악 MERT·AST 보완 | 추가 encoder·학습은 별도 GPU 예산. 지표별 회귀와 배포 추론 시간을 포함해 비교. |

평가식의 가중치 `FILE 0.45 / voice 0.18 / music 0.27 / presence 각각 0.05`는 우선순위를 판단하는 근거다. 이를 loss 가중치로 그대로 복사하는 것이 최적이라는 뜻은 아니다. 이 후보들은 고정된 PANNs v1을 몰래 바꾸지 않는 후속 실험이다. 새 데이터를 학습과 독립 평가에 동시에 쓰지 않고, 이미 노출된 점수 자료는 탐색으로 표시한다.

## Claude 토론 및 작업 상태

실제 Claude Code에서 `claude-opus-5-5 --effort xhigh`로 이번 제안을 두고 2회 토론했다. 소요 시간은 164.25초와 201.88초였고 두 호출 모두 정상 종료했다. 모델 이벤트를 확인했으며 도구·MCP는 비활성화했다. 프로젝트 코드·감사 결과·공식 출처·CPU 진단을 전달했고 음원·가중치·인증정보는 전달하지 않았다.

Opus도 기존 전문 표현을 유지한 shared fusion 및 9-state joint 비교, 추가 channel/gain 실험, 음악 보완 모델 비교를 제안했다. 다만 답변을 그대로 채택하지 않았다. 정답 presence 대입 결과를 개선의 상한으로 해석한 주장은 반박 후 철회됐다. (0,0) 표본이 없는 것은 과거 점수 패널이며 D0 전체가 아니다. 보정은 최종 FILE에만 적용한 것이 아니라 presence를 거쳐 결합한 실험이다. 서로 다른 패널의 calibration 행 수를 오류처럼 비교하거나, 고정 역할을 뒤집어 적합하거나, 복구 noise 전체를 train에 옮기자는 제안도 채택하지 않았다. 사용자의 전화음 제안은 학습 증강으로 평가했으며 전체 추론 입력을 강제 필터링하라는 제안으로 바꾸지 않았다. 모델의 동의는 성능 검증 근거가 아니다.

이전 설계 수정과 실행 준비는 `45fed8948fec47ecdf15dcdd8fc2a3852ad9638f`까지 커밋·원격 push를 확인했다. 다섯 검토와 별도 runtime 감사가 해당 SHA의 CPU 준비 범위를 통과했다. 새 분석은 현재 모델 출력·가중치를 변경하지 않았다. GPU/MPS 실행, 공식 제출, 새 모델 다운로드는 하지 않았다. 실제 CUDA 동작·본학습·독립 품질 검증은 아직 남아 있다.

## 실행 증거 식별자

코드 기준 SHA: `45fed8948fec47ecdf15dcdd8fc2a3852ad9638f`. 음원·모델·캐시는 Git에 추가하지 않았다. 이 문서는 분석이며 새로운 후보 채택이나 학습 완료를 뜻하지 않는다.

- CPU results_sha256: `51a8010f4e0a9661eecc328c1c09adc6f143a22604002468ba6f1378209dded0`
- CPU script_sha256: `e72a28f99533113394335b988a61601300c4c09a4c22d47275738c731b5944d5`
- CPU predeclared_sha256: `7b7990b545210e7510b2f79f3b299b694935d5e929bce6ba57f1931bdbfe85e0`
- D0 presence_rows SHA-256: `0e54b38495fb4d9acdfc311789ea3879936eee3abdf3e4c89e6280c0ef2ea691`
- Opus round 1 prompt SHA-256: `5cdad70ad13073ffb72f2b7076251eec624d3d386f8efb05596a99811920f07c`; response SHA-256: `19d0c412d7554c493e8e2ceaf6b6f05a772be34fa019600d48224b5c906f4933`
- Opus round 2 prompt SHA-256: `0e28199d3ed110891fe6d565d462522e311e2f5a832a4f54e30080d57e3ed807`; response SHA-256: `ccf9b6ad18d5826fd8d284c1ba1dceb2349012f3237f810aa6336b9d7434d81c`
