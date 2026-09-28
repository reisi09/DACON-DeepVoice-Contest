**[INFERRED] 추가할 만한 데이터는 있지만, 오늘 밤 새 학습을 시작할 근거는 부족하다. 먼저 데이터 공백을 잘못 짚은 부분부터 바로잡아야 한다.** 파일 열람·CSV 집계·웹 확인만 수행했다.

**(a) 외부 데이터: 전제 수정과 우선순위**

[VERIFIED] `data/bank_trainval.csv`, `data/bank_test.csv`를 재집계하면 다음과 같다.

| 항목 | 확인 결과 |
|---|---:|
| MLAAD·SpeechFake 한국어 fake speech | **6,830행** |
| 그중 `codec_chain=tel_*` | **1,400행**: train 1,004 / val 171 / test 225 |
| 한국어 real speech의 `tel_*`¹ | **1,068행** |
| `constructed_splice` | **121행 모두** `route=voice`, `VOICE_PRESENT=1`, `MUSIC_PRESENT=0`, `VOICE_FAKE=1` |

¹ Zeroth·FLEURS-ko·SpeechFake의 한국어 real speech 집계.

따라서 `ROUND24_DATA.md:33–35`의 “fake 전화 증강 없음”은 **manifest와 충돌**하며, “순차 speech→music 121행”도 잘못된 설명이다. [UNKNOWN] `tel_*` 기록대로 오디오를 변환했고 해당 오디오에서 feature를 추출했는지는 이 CSV만으로 확인할 수 없다.

[VERIFIED] 생성기도 전부 미상이 아니다. 한국어 MLAAD에는 **Chatterbox Multilingual 296행**, Qwen3-TTS 두 버전, Fish-S2-Pro, MOSS-TTS 등이 있다. SpeechFake 한국어 fake는 **EdgeTTS·MeloTTS·CosyVoice·SeedVC·OpenVoice 각 600행**이다. 버전·별칭을 정리하기 전에는 같은 계열을 “새 생성기”로 세면 안 된다.

아래는 **미사용 데이터셋 단위의 우선순위**다. 기존 AT-ADD 등에 재수록된 원본까지 미사용인지는 별도 확인이 필요하다.

| 순위 | 후보·규모·접근·라이선스 | 채울 공백과 위험 |
|---|---|---|
| **1** | **CtrSVDD**. [VERIFIED] 중국어·일본어, SVS/SVC **14개 시스템**. train **7.9 GB**, dev **3.4 GB**, 공개 다운로드. 데이터는 **CC BY-NC-ND 4.0**. [공식 배포](https://zenodo.org/records/10467648), [논문](https://arxiv.org/html/2406.02438v1) | [INFERRED] 생성기·가수 정보가 있는 fake singing 보강에 적합. **무반주 가창**이므로 완성곡 일반화를 보장하지 않는다. 일부 real 원본은 별도 복원이 필요하며, ND 데이터로 만든 remix의 2차 제출 가능 여부는 확인 전 **UNKNOWN**. |
| **2** | **ASVspoof 5**. [VERIFIED] 영어, 약 **2,000화자**, 다양한 녹음 환경의 TTS/VC. train archive 합계 약 **37.5 GB**, 전체 약 **142 GB**. 공개, **ODC-By**. [배포·크기](https://huggingface.co/datasets/jungjee/asvspoof5/tree/main), [공식 라이선스](https://zenodo.org/records/14498691/files/LICENSE.txt?download=1), [논문](https://arxiv.org/abs/2502.08857) | [INFERRED] 기존 studio speech보다 녹음 환경·공격 조합을 넓힌다. 한국어 공백은 못 채운다. 기존 AT-ADD와 생성기·원본 중복을 먼저 확인해야 한다. ODC-By만으로 개별 음원의 모든 권리를 확인했다고 해서는 안 된다. |
| **3** | **PartialSpoof**. [VERIFIED] 부분 합성 구간과 시간 라벨. train/dev 각각 **2.0 GB**, eval **5.8 GB**. 공개. README는 **CC BY 4.0**, 원천 ASVspoof 2019는 **ODC-By**를 명시한다. [배포](https://zenodo.org/records/5766198), [데이터 README](https://zenodo.org/records/5766198/files/README_v1.2?download=1) | [INFERRED] 부분 fake 비율·위치 일반화에 적합. **새 생성기 데이터는 아니다.** 기존 ASVspoof 2019와 원본 계보가 겹치므로 그대로 독립 평가셋으로 쓰면 안 된다. |
| **4** | **In-The-Wild**. [VERIFIED] 영어 **37.9시간**, fake 17.2 / real 20.7시간, 58명, **8.16 GB**. 저자 공개 배포, **CC BY-SA 4.0**. [저자 배포](https://huggingface.co/datasets/mueller91/In-The-Wild) | [INFERRED] 온라인 유통 음성의 평가용 가치가 학습용보다 크다. 생성기 계보·원본 영상 권리가 불명확한 항목은 따로 취급한다. 실제 전화망 데이터나 최신 한국어 생성기 데이터로 간주하면 안 된다. |
| **5** | **AT-ADD Track1**. [VERIFIED] train **49,575**, dev **49,734**, eval **146,346행**, 저장소 **25 GB**. 약관 동의 후 자동 접근. **비상업 학술연구·교육**, 조건부 재배포. [공식 카드](https://huggingface.co/datasets/xieyuankun/AT-ADD-Track1) | [INFERRED] `generator`, `generation_task`, `transformation`으로 기존 bank에 없는 조합만 선별한다. T2와 독립이라고 가정하지 않는다. fake의 원본·화자 ID가 대부분 비공개라 계보 검증에 한계가 있다. 한국어 신규분 규모는 **UNKNOWN**. |
| **6, 조건부** | **AI-Hub 71544 음악 유사성 판별**. [VERIFIED] 총 **20,000곡**, Regular Org **10,000곡**, 내국인 신청 방식. [공식 페이지](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71544) | [INFERRED] 실제 한국어 완성곡 후보. 우선 **Org 200곡**만 검토. 모든 Org가 한국어 보컬곡인지는 **UNKNOWN**. 다운로드 GB·해당 데이터 이용계약과 DACON 전달 범위도 **UNKNOWN**이므로 승인·약관 확인 전 투입하지 않는다. Mac에서만 처리한다. |

[UNKNOWN] 이번 조사에서 **기존 SONICS·HAIM·Echoes·AT-ADD와 중복되지 않고, 원본 파일 제출 권리까지 확인된 최신 상용 AI 음악**은 추가로 확보하지 못했다. 공개 링크가 있다는 이유로 목록을 채우지는 않겠다.

**(b) 자체 제작 데이터**

[INFERRED] 우선순위는 **성분 라벨이 확실한 혼합 데이터 → 부분 fake → 실제로 빠진 채널 조합**이다. 단순 한국어 TTS 증량은 뒤로 미룬다.

- **혼합 1,600행:** `realV+realM`, `fakeV+realM`, `realV+fakeM`, `fakeV+fakeM` 각각 **400행**. 각 조합을 동시 혼합 200, 순차 연결 200으로 나눈다. 순차 연결은 두 순서를 각각 100행씩 만든다. 모든 조합에 같은 음량·codec·접합 처리를 적용하고, 전체 길이는 **4–60초**로 맞춘다. 이것은 AI-cover의 성분 조합을 모사하지만 자연스러운 완성곡 자체를 대체하지는 못한다.
- **부분 fake 800행:** fake 음성 구간 **0.5초·1초·2초·4초**를 삽입한 400행과, 같은 접합 방식으로 만든 real→real 대조군 400행. 삽입 구간의 실제 발화 여부와 라벨을 확인한다. fake를 넣은 사실 대신 접합 흔적을 학습하지 않게 한다.
- **전화/codec:** 기존 `tel_*`의 실제 처리·캐시 반영을 확인한 뒤, 누락된 조건만 추가한다. real/fake에 동일한 처리 확률을 적용하고, 원본과 파생본은 같은 split에 둔다. 파생본 수만큼 해당 부모의 학습 가중치가 늘어나지 않게 나눈다.

[VERIFIED] 한국어 생성 모델로 **CosyVoice2-0.5B는 Apache-2.0이며 한국어 합성을 평가했다**. **Chatterbox Multilingual은 MIT이며 한국어를 지원하고 watermark를 삽입한다.** [CosyVoice 라이선스](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B), [한국어 평가](https://arxiv.org/html/2412.10117v1), [Chatterbox 라이선스](https://www.resemble.ai/learn/models/chatterbox), [언어·watermark](https://github.com/resemble-ai/chatterbox)

[INFERRED] **오늘 새로 생성할 가치는 낮다.** 이미 CosyVoice·Chatterbox 계열이 있으며, 버전 차이만으로 독립 family가 되지 않는다. 생성한다면 허용된 화자 reference만 쓰고, watermark나 특정 vocoder 흔적에만 의존하는지 구분해야 한다.

**가장 유망한 자체 제작 후보는 혼합 데이터다.** FILE과 두 성분 라벨을 함께 감독할 수 있고, 순차 혼합은 공식 평가 유형에도 명시돼 있다(`official_text.txt:24`). 다만 이것이 LB를 올린다는 증거는 아직 **UNKNOWN**이다.

**(c) G1: 더 나은 오프라인 채점기**

[INFERRED] 단일 데이터셋 하나보다 다음 **2,400행의 고정 패널**이 낫다. LB 재현이 입증됐다는 뜻은 아니다.

| 패널 | 구성 |
|---|---|
| speech 600 | ASVspoof 5·In-The-Wild의 real/fake 각 300 |
| singing 600 | CtrSVDD의 real/fake 각 300 |
| music 600 | 기존 학습과 원본·생성기 family가 겹치지 않는 instrumental real/fake 각 300 |
| mixed 600 | 자연 완성곡 300 + 네 성분 조합의 통제 혼합 각 75행 |

- 모든 패널에서 **원곡·화자·부모 원본의 파생본을 묶어 격리**한다. CtrSVDD의 A09–A14 전체를 새 family로 세면 안 된다. A09–A11·A13은 같은 Soft-VITS-SVC 계열의 변형이며, A12·A14는 별도로 볼 가치가 있다. [공식 설명](https://arxiv.org/html/2406.02438v1)
- 한국어는 **별도 400행(real/fake 각 200)**을 확보해야 한다. 기존 bank 재사용이면 독립 검증이 아니라 진단이다. 새 family와 권리를 확인하지 못하면 **한국어 독립 검증 미충족**으로 남긴다.
- 권리상 변형 가능한 speech 원본에 clean / 전화대역 / 압축 조건을 짝지어 만든다. bootstrap 단위는 변형 파일이 아니라 **부모 원본**이다.
- 공식 방식으로 pooled RT ADS를 계산하고, 패널별 성분 EER·val 고정 threshold의 FPR/FNR·CPS도 함께 보고한다. 한 클래스뿐인 slice에는 EER를 계산하지 않는다.
- 패널 비중과 채널 비중은 **모델 점수를 보기 전에 고정**한다. 전화 비율 **0/20/40%** 등의 민감도 분석은 가능하지만, 잘 나온 비율을 사후 선택하지 않는다.

[INFERRED] 기존 **6개 LB 점수**로는 예측력을 입증하기 어렵다. 동일 1,200파일에 대한 상관된 모델 결과이고, 이미 S8 실패를 알고 평가셋을 설계하기 때문이다. 고정한 채점기로 6개 실제 패키지의 순위상관·사전 지정된 모델 간 Δ 방향을 비교하되 **회고 진단**이라고 명시한다. S8을 떨어뜨리는 가중치를 찾는 행위는 검증이 아니다. 앞으로의 후보 순위를 LB 전에 기록해야 예측력을 새로 확인할 수 있다.

**(d) 오늘 밤과 이후**

**[INFERRED] 오늘 밤 새 데이터 학습은 하지 않는다. 추가 학습 GPU 0시간, $0.** 이유는 예산보다 전제와 검증의 불확실성이다. 전화 증강·한국어 생성기 부족이라는 핵심 가설이 CSV 확인으로 약해졌다.

[VERIFIED] 예산 **$18.79 / $2.98 = H100 약 6.31시간**이다. [INFERRED, 계산상 추정] 평균 10초인 2,400행의 feature 추출만 해도 `24,000/18 = 1,333초`, 약 **0.37 GPU시간·$1.10**이다. 다운로드·라벨 확인·분리 및 추출 범위의 차이·DB 비교·P1 2단계 3-seed·패키징·parity는 별도이며, 전체 완료 시간은 **UNKNOWN**이다.

오늘 할 일은 기존 예측과 manifest로 **채널×언어×생성기별 오류**, 실제 사용 파일·feature provenance를 확인하는 것이다. 새 archive 대량 수집은 미룬다. **이번 조사만으로 slot 3 후보를 교체하지 않는다.**

이후 혼합 데이터 한 변형만 시험한다면, 다음을 **모델 결과 전에** 등록하겠다.

- 고정한 새 평가 패널에서 **RT ΔADS ≥ +0.005**, 부모 단위 paired 95% CI 하한 **>0**.
- 기존 E1과 natural-only 평가에서 각각 **ΔADS ≥ −0.002**.
- 각 성분 EER 악화 **≤0.01**, CPS 악화 **≤0.002**.
- 새 평가를 본 뒤 비율·seed·threshold를 바꿔 재응시하지 않는다.

이 수치는 **[INFERRED] 운영상 채택 기준**이며 LB 향상 확률을 뜻하지 않는다. 하나라도 실패하거나, 원본 권리·계보·라벨·필수 패키지 검증이 끝나지 않으면 중단한다.

**(e) 하지 말아야 할 것**

- **manifest에 있는 전화 증강·생성기를 없는 것으로 간주해 다시 대량 생산하지 않는다.**
- `constructed_splice`를 speech→music으로 세거나, fake 완성곡이라는 이유만으로 두 성분을 모두 fake로 채우지 않는다.
- denoise·음량 조정·일반 codec 처리만 한 real을 fake로 바꾸지 않는다(`official_text.txt:27–30`).
- 새 데이터셋 이름, 다른 checkpoint, 다른 prompt를 새 generator family로 간주하지 않는다.
- 부모가 겹치는 remix·부분 fake·codec 변형을 train/test에 나누지 않는다.
- NC 허용을 모든 가공·재배포 허용으로 해석하지 않는다. **학습에 쓴 생성 파일도 제출 대상**이다.
- **AI-Hub 오디오와 feature를 Mac 밖으로 보내지 않는다.** 국내 VESSL도 이번 제약의 예외가 아니다(`ROUND24_DATA.md:40`).
- LB 6점에 맞춰 평가셋 가중치를 최적화하거나, 한 번의 LB 상승으로 “채점기는 최신 생성기 위주”라고 결론내리지 않는다.