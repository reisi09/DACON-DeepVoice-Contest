**(a) G2 통합 우선순위**

[VERIFIED] `realV+fakeM` **117행은 train 5 / val 5 / test 107**이다. 두 bank CSV를 재집계했다. [INFERRED] 공식 평가가 성분별 진위를 구분하므로 **실제 학습 공백**이다. 다만 LB에서 이 조합이 차지하는 비율과 손실 기여도는 **UNKNOWN**이다.

| 순위 | 데이터 항목 | 메울 공백·범위 |
|---|---|---|
| **1** | **성분별 대조 혼합 1,600행** | `(VOICE_FAKE,MUSIC_FAKE)=00/01/10/11` 각 **400행**. `01`은 실제 발화 200＋실제 가창 200. 다른 조합에도 동일한 음량·혼합 처리를 적용한다. |
| **2** | **순차 speech↔music 480행** | 네 진위 조합×두 순서×길이 8/20/45초×20행. 공식 유형이지만 현재 기록된 학습 사례가 없다. |
| **3** | **부분 fake 320행** | fake 삽입 160＋같은 편집을 적용한 real→real 160. 기존 121행의 구간 길이·위치 범위를 넓힌다. |
| **4, 이후** | **CtrSVDD의 실제 미노출 SVS/SVC 계열만** | AT-ADD singing과 계열 대조 후 남는 부분만 채택. 단순 가창 수량 보강은 제외한다. 데이터는 **CC BY-NC-ND 4.0**이므로 remix 제출을 자동 허용으로 해석하지 않는다. [공식 논문](https://arxiv.org/html/2406.02438v1) |

수량·순위는 **[INFERRED] 설계안**이다. 일반 전화 증강·한국어 TTS 증량·대규모 외부 speech 다운로드는 제외한다.

1번의 반주는 **실제로 무보컬인 파일**을 우선 쓴다. AI 노래에서 분리한 backing에 fake 보컬이 남아 있으면 `VOICE_FAKE=0` 라벨이 틀어진다. 실제 보컬 파형을 그대로 섞은 데이터는 성분 조합을 학습시키지만, 자연스러운 AI-cover 완성곡과 동등하지는 않다.

[VERIFIED] FakeMusicCaps·SONICS는 **CC BY-NC 4.0**, HAIM도 공식 README에서 오디오를 **CC BY-NC 4.0**으로 명시한다. HAIM의 MIT는 코드·자체 metadata다. AT-ADD는 **비상업 학술연구·교육 및 조건부 재배포**다. 사용한 원본·변형 파일의 출처와 원천 약관을 함께 보존해야 한다. [FakeMusicCaps](https://github.com/polimi-ispl/FakeMusicCaps), [SONICS](https://huggingface.co/datasets/awsaf49/sonics), [HAIM](https://github.com/Mippia/HAIM_dataset), [AT-ADD](https://huggingface.co/datasets/xieyuankun/AT-ADD-Track2)

**(b) G1 통합안**

[INFERRED] 기본 **2,400행**으로 고정한다.

- speech **400**: 한국어 200＋기타 200, 각각 real/fake 균형.
- singing **400**, instrumental **400**: 각각 real/fake 균형.
- 동시 혼합 **600**, 순차 혼합 **600**: 각각 네 진위 조합 **150씩**.
- speech·singing 800행에 전화·압축 대응본을 각각 추가하면 **총 4,000행**. 기존 처리본을 clean이라고 부르거나 무조건 재압축하지 않는다.

모든 donor의 연결 계보를 먼저 분리해 **선택용/확인용 기본 1,200행씩** 배정한다. 합성과 채널 복제는 그다음이다. 동일 부모의 다른 조합도 양쪽에 걸치면 안 된다.

보고는 **기본셋 RT ADS**, 채널별 ADS, 성분 EER, val 고정 threshold의 FPR/FNR, natural-only 결과, 계보 bootstrap CI를 분리한다. 합성 비중이 큰 이 패널은 **스트레스 평가**이며 LB 분포 추정치가 아니다.

**[INFERRED] 새 다운로드 없이 스트레스 패널은 구성 가능하다. [UNKNOWN] 독립 확인셋까지 가능한지는 미확인이다.** VESSL에 아직 학습·선택에 쓰지 않은 부모와 생성기 계열이 충분한지 확인해야 한다. 이미 사용한 E1/E2를 재분할해도 독립성이 복구되지 않는다. 6개 LB 점수와의 상관은 회고 진단으로만 보고한다.

**(c) 오늘 밤**

**[INFERRED] 조건부 YES로 수정한다. train 5행인 성분 조합을 겨냥한 후보 하나는 top-15 목표에서 시도할 가치가 있지만, +0.022 도달 확률은 UNKNOWN이다.**

사용자 GPU 승인 후 진행할 **한 가지 계획**만 남긴다.

1. 먼저 CPU로 **30분 이내** donor 권리·계보·미사용 확인셋 확보 여부를 판정한다. 실패하면 **0 GPU·$0**로 종료.
2. **W_newB 레시피＋(a) 1번만** 학습한다. encoder·fusion·FILE 규칙은 고정하고 기존 네 component head를 재학습한다.
3. [INFERRED, 시간 추정] 추출 **1.5h**, DB 비교 **0.75h**, P1 2단계×3 seeds **0.75h**, CUDA 검증 **0.5h**, 총 **3.5h·$10.43**. 준비·재시작 포함 **절대 상한 4h·$11.92**.
4. 사전등록: W_newB 대비 선택셋 **ΔADS≥+0.005**일 때만 확인셋을 연다. 확인셋 **ΔADS≥+0.003 및 계보 CI 하한>0**, 기존 E1·natural-only 각각 **ΔADS≥−0.002**를 모두 요구한다. `realV+fakeM`의 MUSIC FNR은 **0.05 이상 감소**, VOICE FPR 상승은 **0.02 이하**로 제한한다.
5. **23:00까지 DB 판정**, **06:30까지 P1·패키지·CPU/parity/CUDA 검증 완료**. 하나라도 실패하면 기존 slot 3 후보 유지. 새 확인셋을 보고 비율·seed를 바꿔 재시도하지 않는다.

이 기준은 운영상 제안이며 LB 성공 보장이 아니다. 기존 두 제출 준비를 지연시키면 중단한다.

**(d) 상대 답변의 수정점**

- **[VERIFIED] 전화 증강 1순위는 철회해야 한다.** 한국어 fake `tel_*` **1,400행**, real **1,068행**이 이미 기록돼 있다(`fresh24_r10.md:29,41`).
- **[VERIFIED] MatchaTTS는 완전히 미지인 후보가 아니다.** 기존 MLAAD에 **25행**이 있다. 예: `data/bank_trainval.csv:13101`. 상대가 중복 확인을 조건으로 달았으므로 허위 단정은 아니지만, 이제 holdout 후보에서 제외해야 한다(`fresh24_r10.md:59`).
- **[INFERRED] AT-ADD singing 12,500행으로 완성곡 공백까지 완화됐다고 판단하면 과장이다.** vocals-only는 실제 보컬＋AI 반주의 상호 간섭을 다루지 못한다(`fresh24_r10.md:21`).
- **[VERIFIED] In-The-Wild의 라이선스 충돌 지적은 상대가 맞다.** 공식 사이트는 Apache-2.0, HF는 CC BY-SA 4.0이다. 내 이전 답변도 단일 라이선스처럼 요약한 부분을 수정한다. [공식 사이트](https://deepfake-total.com/in_the_wild), [HF 배포](https://huggingface.co/datasets/mueller91/In-The-Wild)