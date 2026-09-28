**(a) G2 통합 우선순위**

**VERIFIED — CSV 재집계 결과 `realV+fakeM` 117행은 train 5 / val 5 / test 107행입니다.** 전체 mixed 7,618행의 **1.54%**이며, 실제 학습 사례는 **5행**뿐입니다. (`data/bank_trainval.csv`, `data/bank_test.csv`)

**INFERRED — 실제 LB 위험 후보입니다.** 공식 유형에 해당하고, real voice 때문에 fake music을 놓치거나 voice까지 fake로 오판할 수 있습니다. 다만 LB 내 비중과 실제 오류율은 **UNKNOWN**입니다. (`official_text.txt:24–29`)

| 순위 | 데이터 항목 | 메울 공백 |
|---|---|---|
| **1** | **realV+fakeM 동시 혼합 400행＋대응 realV+realM 400행** | real speech·real vocals에 AI instrumental을 결합. 같은 real voice를 대조군에도 사용하되 같은 split에 배치 |
| **2** | **순차 혼합 800행** | RR/RF/FR/FF 각 200행, speech→music·역방향 각 절반. 현재 기록상 **0행** |
| **3** | **부분 fake 800행** | fake 삽입 400＋real-real 편집 대조 400. fake 길이 **0.5/1/2/4초**, 현재 121행의 위치·비율 공백 보충 |
| **4, 후순위** | **ASVspoof 5의 미보유 생성기·녹음환경 조합만 선별** | 기존 계열과 겹치지 않는다는 확인이 선행 조건. 일반 한국어 TTS·전화 증강은 제외 |

위 수량·순위는 **INFERRED — 설계안**입니다.

**라이선스:** [HAIM](https://huggingface.co/datasets/mippia/HAIM)은 데이터 **CC BY-NC 4.0·비상업 학술연구**, 코드 MIT를 구분합니다. [SONICS](https://huggingface.co/datasets/awsaf49/sonics)는 **CC BY-NC 4.0**입니다. 보컬 없는 원본·구간을 우선 사용하며, fake 보컬이 남은 반주에 real voice를 얹고 `VOICE_FAKE=0`으로 지정하면 안 됩니다. [AT-ADD](https://huggingface.co/datasets/xieyuankun/AT-ADD-Track2)는 비상업 연구·원천 조건·조건부 재배포 약정이 적용됩니다. [FakeMusicCaps 배포](https://zenodo.org/records/15063698)의 음원 라이선스는 이번 확인에서 **UNKNOWN**이므로 가공·제출 권리를 확정하기 전에는 제외합니다.

**(b) G1 통합 패널**

**INFERRED — 기본 2,400행으로 통합합니다.**

| 구간 | 수량·구성 |
|---|---|
| speech | **600**: 한국어 300＋기타 300, 각각 real/fake 균형 |
| singing only | **600**: real/fake 각 300 |
| instrumental music | **600**: real/fake 각 300 |
| mixed | **600**: 자연 완성곡 200＋동시 혼합 200＋순차 혼합 200. 통제 혼합은 네 성분 조합 각각 50 |

speech 600행에 전화·압축 대응본 **1,200행**을 추가해 **총 3,600행**입니다. 기본 계보를 선택/확인 **1,200행씩** 나누면 채널 대응본 포함 각각 **1,800행**입니다. donor·화자·원곡·파생본은 분리하지 않습니다.

**INFERRED — 신규 다운로드 0건으로 진단용 구성은 가능합니다.** S7 bank, AT-ADD speech/singing/music, GTSinger, HAIM·OpenBMAT와 자체 혼합을 사용합니다. 단, 원본 오디오·권리·미사용 계보가 VESSL에 충분히 있는지는 **UNKNOWN**입니다. 이미 학습·선택에 노출된 자료를 재분할해 독립 확인셋이라고 부를 수 없습니다. 새 혼합의 feature도 기존 feature의 평균으로 대체하면 안 됩니다.

공식 ADS, 구간별 FPR/FNR·성분 EER·CPS, 계보 단위 paired CI를 보고합니다. 단일 클래스 slice에는 EER를 계산하지 않습니다. 기존 **6개 LB 결과**와 순위·변화 방향을 비교하되 **회고 진단**으로 표시하고, 비중을 LB에 맞추지 않습니다.

**(c) 오늘 밤**

**INFERRED — 신규 데이터 작업 GPU는 0시간·$0을 유지합니다:** train 5행의 공백은 크지만 목표 **약 +0.022**의 달성 확률과 재학습·패키지 완료 시간이 **UNKNOWN**이므로, 기존 캐시로 해당 조합의 오류를 확인하는 데 그치며 이번 발견만으로 slot 3을 교체하지 않습니다.

**(d) 상대 답변의 오류·누락**

- **VERIFIED — 패널 총량이 불명확합니다.** 기본 **2,400행＋한국어 별도 400행＝2,800행**, 채널 복사본은 추가입니다. 총 2,400행 설계로 읽히지 않게 고쳐야 합니다. (`codex_round24.md:49`, `:59`)
- **VERIFIED — In-the-Wild의 CC BY-SA 표기는 HF 기준으로 맞지만 불일치 설명이 빠졌습니다.** [공식 사이트](https://deepfake-total.com/in_the_wild)는 Apache-2.0, [저자 HF](https://huggingface.co/datasets/mueller91/In-The-Wild)는 CC BY-SA 4.0입니다. (`codex_round24.md:27`)
- **INFERRED — CtrSVDD 1순위는 낮춰야 합니다.** AT-ADD singing **12,500행**이 추가된 상황에서는 무반주 fake singing 증량보다 성분 조합·순차 혼합 공백이 우선입니다. ND 변형·재배포 문제도 남습니다.
- **제 답변도 수정합니다.** 일반 전화 증강 1순위와 MeloTTS·OpenVoice·CosyVoice 추가 제안은 기존 보유 현황을 충분히 반영하지 못했습니다. 이번 통합안에서는 제외했습니다.