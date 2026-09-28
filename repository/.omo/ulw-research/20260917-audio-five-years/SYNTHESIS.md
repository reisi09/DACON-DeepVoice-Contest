# 최근 5년 오디오 딥페이크 연구와 상위권 해법 검토

조사 기준일: 2026-09-17. 범위: 2021-09-17~2026-09-17에 공개된 연구와 대회 자료. 논문·공식 결과·저자 코드·현재 실험 기록을 대조했다. 공개 연구 전체를 빠짐없이 읽었다는 뜻은 아니다. 초록만 확인한 자료, 코드가 없는 방법, 순위를 독립 확인하지 못한 해법은 출처 목록에서 구분한다. 논문의 수치는 저자 보고이며 여기서 재현한 성능이 아니다.

추가로 할 만한 실험은 있다. 먼저 비교할 순서는 **출처·생성기별 학습 비중 조절 → 같은 신호의 배경·처리 조건 대조 → 저비용 스펙트럼 특징 추가 → 필요한 경우에만 작은 범위의 적응 학습**이다. 이미 진행한 대형 모델 교체와 head 탐색을 반복할 근거는 약하다. 이 순서는 현재 실패 기록과 아래 문헌을 함께 보고 내린 제안이며, 공식 점수 상승이나 1등을 보장하지 않는다.

## 현재 실험에서 새로 확인한 사실

저장된 결과를 CPU에서 다시 계산했다. 검증 코드는 기존 그룹 연결 함수를 사용했고, 입력 해시와 결과를 `verify-single-window.json`, `verify-connected-groups.json`에 남겼다.

| 항목 | 확인 결과 | 다음 판단에 미치는 영향 |
|---|---|---|
| Anti max와 top-30% 평균 | 음성 관련 160개 입력 모두 실제 창이 1개, 두 집계값 차이 0건 | 이번 음성 패널로는 pooling의 우열을 판단할 수 없다. |
| 독립 그룹 106개라는 기록 | 106은 개별 출처 ID 수. 공유 화자·반주를 연결하면 568행은 93개 연결 성분 | 93개도 기록된 출처 관계에 따른 단위다. 완전한 독립성이 입증된 수는 아니다. |
| dev/test의 −6 dB 혼합 8개씩 | 각 패널이 공유 출처 때문에 연결 성분 1개로 묶임 | 0%→25% EER 변화는 해당 사례의 회귀 증거지만 일반화 차이를 정밀하게 추정할 표본은 아니다. |
| 분할 간 기록된 출처 연결 | 교차 연결 성분 0개 | 이번 확인에서 기록된 그룹의 분할 누출은 찾지 못했다. 미기록 관계·사전학습 중복은 별개다. |
| 음성 calibration | 8파일이 화자 1명에 해당 | 계수는 고정됐지만 화자·음질 이동에 대한 검증은 부족하다. |
| MERT와 DF의 layer 사용 | MERT는 마지막 층의 시간 평균. DF-Arena는 이미 여러 층의 학습된 집계 사용 | 중간층 실험은 MERT/Anti에 한정해 설계한다. 기존 캐시에는 새 층의 정보가 없다. |
| 새 음악 대조 | 같은 epoch지만 optimizer update는 425회와 442회 | 다음 데이터 구성 비교는 update 수도 같게 맞춘다. |

근거: [실행 결과](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/voice_music_h100_20260917_retry1/FINAL_REPORT.md), [현재 추출기](/Users/choiwuseck/Documents/ChatGPT/데이콘/redesign_features_runtime.py:66), [Anti 창 처리](/Users/choiwuseck/Documents/ChatGPT/데이콘/antideepfake_runtime.py:66), [그룹 연결 함수](/Users/choiwuseck/Documents/ChatGPT/데이콘/redesign_evaluation_common.py:111), [직접 재계산한 그룹 수](verify-connected-groups.json), [직접 재계산한 창 수](verify-single-window.json).

이미 실제로 수행한 항목에는 AF-Whisper, MERT/EAT full fine-tuning, 48개 head 설정·seed 조합, temporal bin 비교, coarse partial-fake music bag 학습, separator 적응, 음성 residual, Anti 결합, Platt, max/noisy-OR가 있다. 이들을 이름만 바꿔 새 실험으로 제안하지 않는다. S2 EAT의 16.76%와 다른 패널의 MERT 수치를 직접 비교해서도 안 된다. 상세 근거는 [로컬 감사](wave-1-local-audit.md)에 정리했다.

## 먼저 할 추가 실험

아래 E1~E8은 기존 계획과 다른 부분이 무엇인지 명시했다. 원천 수 확대, 생성기 홀드아웃, DF 단독+Platt 대조군, MP3 증강, L4 실측은 기존 계획에 포함돼 있으므로 새 아이디어로 세지 않는다.

| 순서 | 실험과 기존 계획에서 달라지는 점 | 최소 비교 | 채택 판단 | 추가 자원 |
|---|---|---|---|---|
| E1 | **출처×생성기별 sampling**. 단순히 자료를 더하는 대신 큰 출처가 학습을 독점하는지 확인 | 같은 자료·head·증강·REAL:FAKE 비율·update 수에서 행 균등 vs 출처/생성기 균등. 필요할 때만 비중 완화 1개 추가 | Music/File 개선이 새 REAL 출처와 홀드아웃 생성기에서도 유지되는가 | 기존 특징으로 작은 head 비교부터 가능 |
| E2 | **보컬 고정·반주 교체와 분리기 왕복 대조**. raw/stem 비교보다 원인을 좁힘 | 같은 보컬에 원래 반주/다른 실제 반주/노이즈. 이후 반주 진위만 교체. 깨끗한 보컬→혼합→분리 결과도 원본 보컬과 비교 | 보컬 정답이 같은데 Voice가 배경에 따라 흔들리는가. 반주 진위가 바뀌면 Music/File은 올바르게 반응하는가 | 새 혼합·분리·고정 모델 추론. 재학습은 아직 불필요 |
| E3 | **성분 보존 처리의 일관성 학습**. processed REAL을 넣는 것에서 한 단계 추가 | 동일 원본/처리 쌍, 같은 CE 학습량에서 CE만 vs CE+작은 일관성 손실. 진짜·가짜 모두 동일 처리 | 보지 않은 코덱·처리 순서에서 개선하면서 짧은 가짜 흔적을 지우지 않는가 | 고정 특징으로 시작. 파형 모델 적응은 성공 후 |
| E4 | **16 kHz 저대역 스펙트럼 흔적 분류기**. 기존 log-mel CNN/CLAP 제안과 달리 평균 스펙트럼의 주기적 잔차를 직접 사용 | 1–7 kHz fakeprint+로지스틱, MERT, 고정 결합. 처음에는 큰 CNN·복잡한 전문가 결합을 생략 | 출처·코덱만 맞히는가, MERT가 틀리는 홀드아웃 생성기를 보완하는가 | 특징 추출과 작은 분류기는 CPU부터 가능 |
| E5 | **실제로 여러 창이 생기는 짧은 가짜 구간 실험**. 이미 했던 가운데 3초 삽입과 구별 | 긴 음성에 짧은 가짜를 위치·점유율별 삽입. REAL→REAL 편집도 동일하게 생성. DF/Anti 창 위치·hop을 먼저 맞춤 | 짧은 가짜 recall과 긴 REAL 오탐을 함께 개선하는가 | 새 추론 필요. 이후 집계 비교는 점수 캐시 재사용 |
| E6 | **MERT/Anti 중간층 소규모 probe**. DF의 기존 다층 집계를 다시 만드는 실험이 아님 | 사전 지정한 얕은/중간/깊은 층, 같은 작은 head. 층 선택은 dev에서만 | 새 생성기 개선이 유지되는가. 새 층이 기존 최종층과 다른 오류를 줄이는가 | 층 특징 1회 재추출 필요. 현재 캐시만으로 불가 |
| E7 | **LoRA+기존 지식 보존 대조**. 전체 FT나 마지막 4층 FT와 구별 | 같은 학습량에서 frozen / LoRA CE / LoRA CE+reference KL | 목표 조건을 고치면서 기존 음성·가창·처리된 REAL을 악화시키지 않는가 | H100 학습 필요. gradient·저장/병합 동등성 확인 필수 |
| E8 | **진짜 음악만 이용한 이상도 보조 점수**. 일반 REAL/FAKE 분류와 다른 목적함수 | 먼저 기존 MERT 특징의 간단한 one-class 점수. 효과가 있으면 수정·검증한 MusicDET 1개 설정 | 새로운 실제 음악을 가짜로 오인하지 않으면서 새 생성기를 잡는가 | 첫 진단은 저비용. flow 학습은 조건부 |

**E1의 근거:** ACL 2026 DOSS는 REAL의 출처와 FAKE의 출처×생성기를 분리해 학습 비중을 조절한다. 논문은 적은 자료의 신중한 선택이 전체 자료 단순 합산보다 좋았던 결과를 보고하지만, 평균에 일부 in-domain 평가도 포함한다. 음악으로의 적용은 여기서 검증할 가설이다. 기존 Echoes 자료에는 공통 원곡 관계가 있어 그 단위로 묶어야 한다. 새 파일명을 새 독립 원천으로 세면 안 된다. [DOSS](https://aclanthology.org/2026.acl-long.796.pdf), [Echoes v3](https://arxiv.org/html/2603.23667v3)

**E2의 근거:** SingGraph 후속 연구는 보컬을 유지한 채 무관한 반주나 노이즈를 넣어도 이득이 남을 수 있음을 보였다. 보컬·반주의 조화를 이해해서 좋아졌다고 단정하기 어렵다. MixFake는 전경·배경 진위를 교차하지만 FMA/SONICS 출처 차이는 남는다. 구성 방식을 참고하되 같은 출처 편향까지 가져오지 않는다. 두 원천 풀을 먼저 분할하고 분할 안에서만 혼합한다. [반주 교체 연구](https://arxiv.org/html/2509.14675v2), [MixFake](https://arxiv.org/html/2605.23201v1)

**E3의 근거:** CLAD와 RTCFake는 처리 전후의 표현을 맞추는 학습을 다룬다. TFCL은 같은 clean/처리 자료를 학습한 대조군과 일관성 손실을 추가한 모델을 비교한다. 복잡한 대조학습·음소 정렬 전체를 재현하기 전에 단순 증강과 일관성 손실의 차이부터 확인할 수 있다. 교사 모델이 합법적인 음질 보정본을 가짜로 판단한다면 그 출력을 그대로 보존하지 않는다. 신뢰할 수 있는 라벨이 우선이다. [CLAD](https://arxiv.org/html/2404.15854v1), [RTCFake](https://arxiv.org/html/2604.23742v1), [TFCL](https://arxiv.org/html/2607.17761)

**E4의 근거:** Deezer 2025 Fourier 연구와 2026 robustness 후속 연구에는 16 kHz의 8 kHz 미만 대역을 이용한 실험이 있다. 현재 입력에서도 시험할 물리적 근거가 있다. 미학습 생성기·속도 변형에서 실패도 보고되므로 MERT를 즉시 교체하지 않고 보완 여부를 판단한다. 처리된 REAL, EQ·pitch·speed 변형을 양쪽 클래스에 동일하게 적용하고 원곡별 파생본을 같은 split에 둔다. [Fourier](https://arxiv.org/html/2506.19108v1), [Robust Fourier](https://arxiv.org/html/2607.27454v2)

**E5의 근거:** PartialSpoof와 SAL은 파일 정답과 구간 정답을 구분한다. 실제 사기 녹음 연구에서는 max가 mean보다 나쁜 경우도 있었다. 짧은 가짜를 놓치지 않으려는 max가 긴 REAL 파일에서 오탐을 쌓을 수도 있다. 이번 1창 결과로 어느 집계가 낫다고 판단할 수 없다. 4초 파일 전체가 FAKE인 경우와 30초 중 0.5초만 FAKE인 경우를 따로 봐야 한다. [PartialSpoof](https://arxiv.org/html/2204.05177v3), [SAL](https://arxiv.org/html/2601.21925v1), [Real Fraud](https://openaccess.thecvf.com/content/WACV2026W/SAFE-2026/papers/Gajewska_Audio_Deepfake_Detectors_vs._Real_Fraud_-_The_Fall_of_WACVW_2026_paper.pdf)

**E6·E7의 근거:** SSL 중간층 정보와 LoRA의 일반화 이점을 보고한 연구가 있다. 하지만 일부 층 선택 논문은 평가용 데이터로 최적 층을 고른다. 그 수치를 독립 평가 성능으로 가져올 수 없다. 먼저 적은 층을 고정해 비교하고, 학습이 필요할 때 ordinary LoRA를 대조군으로 둔다. 현재 Anti1B에서 q/v rank8 어댑터는 구조상 1,966,080개 파라미터로 계산되지만, 이것은 VRAM이나 시간 실측이 아니다. [NAACL 층별 분석](https://aclanthology.org/2025.findings-naacl.227.pdf), [MLDG-LoRA](https://arxiv.org/html/2502.10838), [선택 노출이 있는 2026 연구](https://arxiv.org/html/2606.30791)

**E8의 근거:** MusicDET는 실제 음악의 분포를 학습하는 별도 접근이다. 다만 공개 코드의 real-only는 train에만 적용되고, 혼합 dev가 checkpoint 선택에 쓰인다. 점수 방향도 평가 정답으로 고르는 코드가 있어 그대로 사용할 수 없다. 이번 코드 감사에서는 ActNorm log-determinant 누락도 작은 CPU 수치 실험으로 확인했다. 이는 현재 코드의 문제이며 논문 결과 전체가 틀렸다는 증거는 아니다. 수정 전 대규모 재현은 우선하지 않는다. [MusicDET](https://arxiv.org/html/2605.18072v1), [코드 감사](wave-1-music-code.md)

## 상위권 대회 해법에서 가져올 것

순위·트랙·지표가 다른 결과를 한 줄로 성능 비교하지 않았다. 아래는 해당 대회 안의 기록이다.

| 대회 | 확인된 결과 | 현재 작업에 유용한 부분 | 그대로 가져오면 안 되는 부분 |
|---|---|---|---|
| ASVspoof5 Track1 open | USTC T45 1위, 2.59% EER. 순위 기준은 minDCF | 이질적인 표현과 학습 자료, 독립 calibration | private 데이터와 큰 앙상블. open은 코드 공개를 뜻하지 않음 |
| ASVspoof5 같은 트랙 | Intema 2위, WavLM-base 평균+linear | 작은 backend도 경쟁력 있음 | 사전학습 모델에 head만 붙인 결과로 오해하면 안 됨. transformer FT와 여러 모델 평균 사용 |
| ADD2023 FG-D | USTC-NERCSLIP 1위, 12.45% WEER | 같은 내용의 REAL/합성 쌍으로 출처 차이를 줄이는 구성 | vocoder 출력의 FAKE 라벨을 대회 정의 확인 없이 복사 |
| SVDD2024 controlled | Fosafer 1위 1.65%, A14 제외 | 가창을 별도 평가하고 생성기별 실패를 공개 | A14까지 포함한 수치·다른 dev 평가를 공식 순위와 혼합 |
| MIREX2024 WildSVDD | UNIBS log-spectrum ResNet18이 두 test에서 최저 EER | 작은 spectral branch | 원본 44.1 kHz 이점을 16 kHz 입력에 그대로 기대 |
| SAFE2025 | ISPL private 3개 task 모두 1위 | 서로 다른 저수준 특징과 조건별 신뢰도 | fake에만 적용된 변형으로 얻은 결과를 양 클래스 강건성으로 해석 |
| DACON2024 236253 | SKKU AI 공식 1위 | 다중 해상도 mel, 혼합 조건과 보조 과제 | 음성 수 기반 File 억제. 이번에는 음성 없이 AI 음악만 있어도 FAKE |
| BirdCLEF2023~2026 | 공식 순위와 연결된 저자 해법·공개 노트북 확인 | 학습용 교사 증류, 오류가 다른 모델 조합, 특징 캐시 | 종·위치 사전확률, hidden-test pseudo-label, 코퍼스 통계 보정 |

근거: [ASVspoof5](https://www.isca-archive.org/asvspoof_2024/wang24_asvspoof.pdf), [Intema](https://www.isca-archive.org/asvspoof_2024/aliyev24_asvspoof.pdf), [ADD2023](https://ceur-ws.org/Vol-3597/paper20.pdf), [SVDD](https://arxiv.org/html/2408.16132v2), [MIREX 결과](https://music-ir.org/mirex/wiki/2024:Singing_Voice_Deepfake_Detection_Results), [SAFE 보고서](https://arxiv.org/html/2510.03387), [DACON2024](https://dacon.io/competitions/official/236253/leaderboard), [BirdCLEF 노트북 감사](wave-1-kaggle.md).

Kaggle에서는 현재 공개된 노트북 소스 5개를 읽었다. 4개는 순위가 확인된 해법과 연결된 추론 노트북, 1개는 순위와 무관한 딥페이크 교육용 노트북이다. 공개 버전이 실제 수상 당시 제출본과 같다는 보장은 없다. BirdCLEF 2022도 추가 조사에서 운영진 보고서의 KDL 1위와 저자 저장소의 공식 해법 연결을 확인했다. 다만 해당 해법 본문은 전부 읽지 못했다. [2022 운영진 보고서](https://ceur-ws.org/Vol-3180/paper-154.pdf)

증류는 L4 예산이 실제로 부족할 때 별도 후속 실험으로 남긴다. 교사 logits/embedding은 허용된 학습 자료에서만 만들고, supervised-only 대조군과 비교한다. 모든 distillation이 추론을 줄이지는 않는다. One-Class Knowledge Distillation처럼 추론 때 교사와 학생을 모두 쓰는 방법도 있다. [BirdCLEF2026 1위](https://www.kaggle.com/competitions/birdclef-2026/writeups/1st-place-solution-noisy-student-meets-distillati), [One-Class KD](https://arxiv.org/html/2309.08285)

## 이번 조사로 우선순위를 낮춘 방법

| 방법·주장 | 판단 이유 |
|---|---|
| 모든 신경망 코덱 재합성에 일괄 FAKE 또는 REAL 라벨 | 논문별 정의가 다르다. 공식 기준은 성분을 새로 생성했는지 여부이며 신경망 사용 자체가 아니다. |
| 44.1 kHz 고주파 ArtifactNet을 그대로 도입 | 실제 평가 입력은 16 kHz. 업샘플링으로 잃은 8 kHz 이상 정보가 복구되지 않는다. |
| Deezer2026 NMF를 테스트 전체에 fit | 다른 파일의 정보를 이용하는 transductive 방식이며 공식 독립 예측 규칙과 맞지 않는다. 해당 논문도 3–15 kHz를 사용한다. |
| MusicDET 공개 evaluator 그대로 사용 | 평가 정답으로 score 부호를 고른다. 방향은 dev에서 고정해야 한다. |
| GRPO를 다음 대형 실험으로 바로 실행 | 논문은 XLS-R-1B 결과를 보고하지 않음. 전체 모델 여러 개와 paper/code 차이 확인 필요. LoRA 대조가 먼저다. |
| CoMoE·AUDETER를 공개 구현이 완성된 방법으로 취급 | 조사 시점 저자 저장소는 README 중심. 논문 아이디어와 재현 가능한 release를 구분한다. |
| GenuVoice 1.82%를 SVDD 공식 unseen-attack 성적과 비교 | 저자 저장소에서 공식 dev를 test로 사용한다. |
| 노래 전체 가짜 탐지가 좋으면 가창 조작도 해결 | MIREX2025의 SONICS/WildSVDD 전이 결과가 반례다. |
| 음질 개선·무음 제거를 항상 적용 | 일부 조건에서는 탐지를 악화시킨다. 처리된 REAL과 FAKE를 함께 비교해야 한다. |
| noisy-OR, attention, checkpoint 평균을 무조건 채택 | 현재 max/noisy-OR EER 동률이며 문헌·수상 해법에도 learned fusion, EMA/SWA, SED가 악화한 사례가 있다. |

근거: [공식 정의](https://dacon.io/competitions/official/236749/overview/description), [공식 독립 예측 규칙](https://dacon.io/competitions/official/236749/overview/rules), [ArtifactNet](https://arxiv.org/html/2604.16254v2), [NMF](https://arxiv.org/html/2607.25530v1), [GRPO](https://arxiv.org/html/2603.02914), [GenuVoice 코드](https://github.com/aastha-sharma/genuvoice), [MIREX2025](https://music-ir.org/mirex/wiki/2025:Song_Deepfake_Detection_Results).

## GPU 비용을 줄이는 실행 순서

1. **GPU를 켜기 전:** 그룹 수·라벨·생성기별 학습 비중을 수정하고, 기존 점수로 오류 중복과 채택 안정성을 분석한다. 새 자료 다운로드·검수·변환·VESSL 업로드를 끝낸다. E1의 특징 기반 대조와 E4의 CPU 특징부터 시작한다.
2. **한 번의 추론 묶음:** E2·E5에 필요한 새 원본/분리 결과와 창별 점수를 저장한다. E6을 함께 하기로 정했다면 필요한 층만 추가 저장한다. audio hash, sample rate, 창 시작·길이, layer, 전처리 버전을 캐시 키에 넣는다.
3. **작은 비교:** 동일한 source split과 update 수로 E1·E3·E4를 한 요인씩 비교한다. 검증셋을 보고 계속 설정을 늘리지 않는다. 실패한 arm은 seed 추가나 full FT 없이 종료한다.
4. **조건부 적응:** 고정 모델로 해결되지 않는 오류가 남고 새 검증에서 이득이 확인될 때 E7을 진행한다. E8·증류·복잡한 구조는 해당 필요가 확인된 뒤다.
5. **최종 후보만 제출 환경 실측:** L4에서 디코드·분리·모델 로딩·모든 branch·집계·CSV 저장까지 1,200파일 전체 흐름을 재고 60분 제한을 확인한다. H100 시간으로 L4 시간을 단순 환산하지 않는다.

여기에 적은 자원 수준은 구현 경로에 따른 상대 추정이다. 새로운 학습의 H100/A100 소요 시간과 L4 추론 시간은 측정하지 않았다. 사전 준비 시간을 GPU 학습 시간과 섞어서 보고하지 않는다. 현재 요청에서는 GPU를 다시 시작하지 않았다.

## 다음 실험의 채택 기준

데이터 규모 기준은 파일 수보다 출처 연결 단위로 적용한다. 기존의 300원천·5생성기 기준은 충분성을 보장하는 통계 법칙이 아니라 실험 진입 조건이다. 음성·음악 존재와 진위가 모두 검수된 새 자료에서 판단하며, 이미 본 test는 진단 자료로 둔다.

주요 평가는 공식 산식 그대로 계산하되, 일부 정답이 없으면 총점을 만들지 않는다. 완전한 정답이 있는 경우 모델 간 차이는 다음처럼 분해해 원인을 본다.

`ΔScore = −0.45ΔFileEER −0.18ΔVoiceEER −0.27ΔMusicEER +0.05ΔVoiceAUC +0.05ΔMusicAUC`

각 EER/AUC는 0~1 단위다. REAL-only 처리 패널에서는 EER 대신 calibration에서 고정한 threshold의 FPR을 보고한다. 처리 전후 같은 원본의 점수 차이, 생성기별 miss, 음질·혼합 조건별 회귀를 함께 남긴다. bootstrap도 파생 파일이 아니라 연결된 출처 묶음을 재표집한다. 연결 단위 하나뿐인 패널에서는 일반화 신뢰구간을 만들지 않는다.

개선 폭·허용 회귀·최대 계산량은 새 test를 열기 전에 정한다. 양의 기울기 Platt 하나는 원칙적으로 순위를 바꾸지 않지만, 서로 보정한 두 모델을 합치면 ensemble 순위는 바뀔 수 있다. 결합 후 실제 File 성능을 확인해야 한다. 논문 성능, 로컬 비교, 공식 리더보드 점수는 끝까지 별도로 기록한다.

## 조사 범위와 남은 한계

음성 초기·최근 연구, 강건성/부분 조작, 음악, 가창, ASVspoof, ADD, SAFE/기타 대회, Kaggle, 데이터 타당성, 현 코드, 과거 실행, 음악 공개 구현, 독립 반례 검토로 나눴다. 1차 조사 뒤 두 차례의 표적 확장으로 중요한 반례·버전 차이·실제 코드 경로를 확인했다. 출처별 읽은 범위와 접근 실패는 `sources.csv` 및 wave 문서에 남겼다.

전체 arXiv와 모든 비공개 수상 코드를 망라한 조사는 아니다. 공개되지 않은 수상 구현, 논문별 데이터 중복, 일부 라이선스·가중치 배포 조건, 현재 계정 리더보드 위치, 새로운 방법의 실제 성능은 미확인이다. 코드 정적 분석은 target GPU 실행 증거와 구분했다. 유망한 방법이라도 데이터 정의·평가·구현의 세 조건을 통과한 뒤에만 실험 비용을 쓰는 것이 이번 조사의 제안이다.
