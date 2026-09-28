**(a) [INFERRED] YES는 유지하되, 후보를 ‘저음량·짧은 음악 구간’ 보강 하나로 좁힌다.** 일반 remix·순차 혼합·OpenBMAT crop 증량을 묶었던 이전 안은 철회한다.

- **[VERIFIED] 기존 D1은 이 조건을 충분히 다루지 않는다.** `voice/music` 비율이 **−12/−6/0/+6 dB**이므로 `music/voice`는 **+12/+6/0/−6 dB**다. 반주도 최대 10초 동안 이어진다(`code_r14/open_rows.py:177–186,227`). 내가 제안했던 `voice/music −6/0/+6 dB` 역시 같은 한계가 있었다.
- **[VERIFIED] AI-OpenBMAT 논문은 강한 발화와 짧은 음악에서 성능 저하를 확인했다.** 다만 CNN·SpectTTTra의 검증이며, 방송 실험 지표도 시간 단위 F1이다. **우리 frozen encoder＋head의 EER가 해당 증강으로 개선된다는 증거는 아니다.** [논문 §3–4](https://arxiv.org/html/2602.06823v1)
- **[UNKNOWN] 목표 증강의 효과와 LB +0.022 도달 확률은 아직 모른다.** 추가 OpenBMAT의 E2 공개 비연결 성능이 **+0.0058→+0.0049**였다는 결과도 효과를 뒷받침하지 않는다. 다만 기존 증강과 구별되는 조건을 시험할 근거는 있어, top-15 목표에서 제한된 한 번의 시도는 타당하다고 판단한다.

**(b) [INFERRED] 최종 단일 사양**

| 항목 | 고정안 |
|---|---|
| 기준 | **s7_new**. 기존 encoder·head 구조·seed·학습 가중치 산식·fusion·FILE 규칙 유지. 별도 **20% loss 할당 없음** |
| 추가 학습 | **1,200행**＝음악 real/fake 2종×music/speech **−20/−15/−10 dB**×음악 **2/6초**×100행 |
| 파형 | **20초 실제 발화** 안에 짧은 음악 삽입. 대응 real/fake 쌍은 동일 발화·위치·길이·fade 사용 |
| 비율 정의 | 음악이 겹치는 **발화 활성 구간**에서 RMS 비율을 계산. 전체 20초 평균으로 SNR을 맞추지 않음 |
| Donor | 이미 보유한 비AI-Hub 실제 발화＋권리·무보컬 여부가 확인된 real/AI instrumental 원본. 잔류 fake 보컬이 있는 분리 반주는 제외 |
| 라벨 | `VOICE_PRESENT=MUSIC_PRESENT=1`, `VOICE_FAKE=0`, `FILE_FAKE=MUSIC_FAKE`. 음악 구간 타임스탬프 보존 |
| 추가하지 않을 것 | 새 생성기, 순차 혼합, 추가 OpenBMAT crop, 새 encoder |

**계보·평가**

모든 발화·원곡·AI continuation의 부모 관계를 묶는다. 학습과 분리한 새 donor 패널 **360행**을 같은 격자에서 칸당 30행씩 만들고, **P1에도 편입하지 않는다**. OpenBMAT는 첫 음악뿐 아니라 모든 사용 음악·발화 계보로 중복을 제거한다.

비교 기준은 동일 DB 프로토콜의 `s7_new`; 다음을 결과 확인 전에 고정한다.

1. **선택:** 깨끗한 newval OpenBMAT에서 MUSIC EER **0.03 이상 감소**, FILE EER 악화 **≤0.01**.
2. **방송 확인:** E2-OpenBMAT 비연결 부분에서  
   `J=0.5×FILE_EER+0.3×MUSIC_EER`의 **ΔJ≤−0.02**, 각 EER 악화 **≤0.01**.
3. **새 donor 확인:** **ΔJ≤−0.02**, 계보 bootstrap으로 계산한 `J` 개선의 **95% CI 하한>0**. 기존 val threshold에서 VOICE real FPR 상승 **≤0.02**. 전부 real voice이므로 이 패널에 Voice EER·완전한 ADS를 계산하지 않는다.
4. **거부:** E1·E1 natural·전체 E2-unlinked 중 하나라도 **ΔADS<−0.002**, 또는 **ΔCPS<−0.002**이면 탈락.

E2는 **재사용한 개발 평가**다. 독립 확인 근거는 새 donor 패널이며, 이것도 LB 대표성을 보장하지 않는다.

**시간·중단**

- **19:00까지:** 권리·계보·donor·평가행 고정. OpenBMAT 확인셋이 각 클래스 **20행·10개 독립 원본 그룹** 미만이면 시작하지 않는다.
- **23:00까지:** DB 판정. **06:30까지:** P1 2단계×3 seeds와 패키지·CPU/parity/CUDA 검증 완료.
- **[INFERRED, 예산 상한]** 추출 1.5h＋DB 0.75h＋P1 1.25h＋CUDA 0.5h＝**최대 4 H100시간·$11.92**. 실제 완료 가능성은 착수 전 확인한다.
- 조건·예산·기한 중 하나라도 실패하면 **기존 slot 3 후보 유지**. 재탐색·두 번째 후보는 없다. GPU 실행은 사용자 승인 후다.