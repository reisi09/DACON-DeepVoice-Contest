**INFERRED — 추가 데이터는 도움이 될 여지가 있지만, 오늘 밤 새 데이터로 제출 모델을 교체하는 것은 권하지 않습니다.** 우선순위는 한국어 real/fake에 동일하게 적용한 전화채널 변형, 순차 혼합, 부분 합성입니다. 효과의 크기와 LB 개선 여부는 **UNKNOWN**입니다.

**VERIFIED — 전제 두 가지를 바로잡아야 합니다.**

- 문서상 한국어 fake speech는 **3,830＋3,000＝6,830개**입니다. `generated_music-ko 222개`를 speech에 더하면 안 됩니다. 다만 제공된 inventory에는 언어·생성기 열이 없어 한국어 수량을 독립 재검산하지는 못했습니다. (`ROUND24_DATA.md:28`, `DATA_INVENTORY_20260928.txt:4`)
- ASVspoof 2021 LA는 원래 **real/fake 모두 전화망·코덱 조건**을 포함합니다. 현재 bank에는 해당 출처 fake **2,652개**가 있습니다. 따라서 “전화채널 fake가 전혀 없다”는 단정은 부정확합니다. 이 2,652개 중 실제 채널별 수량과 한국어 해당 여부는 **UNKNOWN**입니다. (`DATA_INVENTORY_20260928.txt:5`; [공식 ASVspoof 2021 논문](https://www.isca-archive.org/asvspoof_2021/yamagishi21_asvspoof.pdf))

**(a) G2 외부 데이터 우선순위**

아래 **6개**는 제공된 사용 목록에 없습니다. 순위는 **INFERRED**, 공개 조건·용량은 표기한 공식 자료 기준입니다. **새 데이터셋이라는 사실만으로 새 생성기 계열이라고 판정하지 않습니다.**

| 순위 | 데이터·접근·라이선스 | 다운로드 규모 | 메울 공백과 위험 |
|---|---|---|---|
| **1** | **DFADD** — 공개 HF, 데이터 카드 **MIT**. 원천 VCTK·LJSpeech 조건도 유지. [공식 설명](https://github.com/isjwdu/DFADD), [배포 파일](https://huggingface.co/datasets/isjwdu/DFADD/tree/main) | 수정본 ZIP 6개 합계 **약 13.86GB**. MatchaTTS＋real만 **5.18GB** | 영어 diffusion/flow TTS: GradTTS, MatchaTTS, NaturalSpeech2, PFlowTTS, StyleTTS2. **2025-04 이전 Matcha 라벨 오류** 주의. MLAAD·SpeechFake와 생성기 중복은 **UNKNOWN**. |
| **2** | **ASVspoof 5** — 공개 Zenodo, **ODC-By**, bona fide **CC BY 4.0**. [배포](https://zenodo.org/records/14498691), [공식 라이선스 설명](https://zenodo.org/records/14498691/files/README.txt?download=1) | 전체 **142.3GB**, train 약 **37.5GB**, dev 약 **20GB** | 영어 비스튜디오 음성, TTS/VC·코덱 조건. 한국어 공백은 못 메움. attack ID가 달라도 같은 모델 계열일 수 있으므로 계열 단위 대조 필요. [공식 논문](https://arxiv.org/html/2502.08857v1) |
| **3** | **PartialSpoof v1.2** — 공개 Zenodo, **CC BY 4.0**; 원천 ASVspoof 2019의 ODC-By도 명시. [배포](https://zenodo.org/records/5766198), [README](https://zenodo.org/records/5766198/files/README_v1.2?download=1) | train **2.0GB**＋라벨·protocol 약 **92MB**; 전체 **10.0GB** | 문장 일부만 fake인 상황과 구간 라벨. **새 생성기보다 새 편집 조건**을 보충함. 기존 ASVspoof 2019 원본과 계보 중복 위험이 큼. |
| **4** | **DSD-Corpus** — 공개 Zenodo, 저자 명시 **CC BY-NC 4.0**. [공식 배포](https://zenodo.org/records/13788455) | **17.4GB** | 공개 합성기와 ElevenLabs 음성. 원천에 **AI-Hub 포함**. 한국어 비중·생성기별 수량은 **UNKNOWN**. 비AI-Hub 계보를 확인하기 전에는 전체 자료를 VESSL에 올리지 않음. |
| **5** | **SpoofCeleb** — 영어, **CC BY 4.0 표기＋별도 제한 조건**. 소속기관 이메일로 신청, 처리 **2–3일**. [공식 배포](https://huggingface.co/datasets/jungjee/spoofceleb/tree/main) | **약 134.16GB** | 비스튜디오 실제 음성과 같은 환경에서 학습한 TTS. 접근 조건상 **오늘 밤 불가**. 누구나 접근 가능한 자원이라는 대회 요건 충족 여부와 세부 제한은 **UNKNOWN**. |
| **6** | **CodecFake** — 공개 Zenodo, **CC BY-NC-ND 4.0**. [공식 배포](https://zenodo.org/records/11171708) | train **첫 묶음만 32.9GB**; 전체 필요량 **UNKNOWN** | neural codec·audio language model 흔적. **일반 전화 코덱과 다른 공백**이다. 변형 음원 재배포 조건 때문에 자체 증강 학습에는 후순위. ND를 “학습 자체 금지”로 해석해서도 안 됨. [라이선스](https://creativecommons.org/licenses/by-nc-nd/4.0/) |

**INFERRED — 상용 AI 음악·한국어 노래는 당장 외부 수량을 더 늘릴 우선순위가 낮습니다.** 기존 Suno/Udio/Echoes와 신규 AT-ADD singing **12,500개**, GTSinger-ko 음성·노래 **1,640개**부터 생성기·곡 계보를 정리하는 편이 낫습니다. 한국어 방송·노래를 자유롭게 재배포할 수 있는 새로운 자료는 이번 확인 범위에서 확보하지 못했습니다. (`DATA_INVENTORY_20260928.txt:33`, `code_r14/new_data_summary.csv:2`, `:5`, `:11`)

**(b) G2 자체 제작**

아래 수량은 모두 **INFERRED — 후속 실험 설계안**입니다.

| 우선순위 | 제한된 제작안 | 판단 근거 |
|---|---|---|
| **1. 양쪽 클래스 전화채널 변형** | 비AI-Hub 한국어 원본 **real 400＋fake 400개**에 각각 ① 8kHz 전화대역＋G.711 ② MP3 압축을 적용해 **1,600개 추가**. 최종 16kHz. 클래스별 변형 확률·강도 동일. | 공식 평가에 전화채널이 명시됨. S2a의 **real만 800개 추가**한 결과로 이 설계의 효과를 부정할 수는 없음. |
| **2. 순차 혼합** | speech→music / music→speech, 구성요소 **RR·RF·FR·FF**, 길이 **8·20·45초**를 조합해 **480개**. real-real도 같은 연결·crossfade 처리. | 공식 평가의 “순차적으로”를 직접 다룸. 단순 연결 흔적이 fake 정답을 알려주지 않게 해야 함. |
| **3. 부분 합성** | real 문장 안 fake **0.5·1·2초** 삽입 **160개**, 같은 방식의 real-real 편집 대조군 **160개**. | 짧은 fake 때문에 전체 파일이 fake가 되는 조건. 삽입 구간·화자·원본 계보를 기록하고 심한 문맥 불일치는 제외. |

총 **2,400개 추가**입니다. 원본과 모든 변형은 반드시 같은 split에 둡니다. 파일 라벨은 구성요소의 OR로 정하되, 없는 구성요소를 real로 채우지 않습니다. denoise·음량 조정만 한 real은 real을 유지합니다.

**VERIFIED — 한국어 생성 모델 후보**

- [MeloTTS-Korean](https://huggingface.co/myshell-ai/MeloTTS-Korean): 한국어, **MIT**.
- [OpenVoice V2](https://github.com/myshell-ai/OpenVoice): 한국어 지원·voice cloning, **MIT**.
- [CosyVoice2-0.5B](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B): **Apache-2.0**. 한국어 지원은 [해당 버전 논문](https://arxiv.org/html/2412.10117v1)에서도 확인됨.

**INFERRED — 생성은 나중에, 계열 중복 확인 후에만 가치가 있습니다.** MeloTTS와 이를 사용하는 OpenVoice 경로를 완전히 독립적인 두 생성기 계열로 세지 않습니다. 권리가 확인된 비AI-Hub 참조음성을 쓰고, 생성 모델은 제출 ZIP에 넣지 않습니다. 현재 가장 유망한 선택은 **전화채널 변형**, 다음은 **순차 혼합**이며 실제 LB 개선은 **UNKNOWN**입니다.

**(c) G1 grader**

**UNKNOWN — 현재 자료만으로 “이 평가셋이 LB를 더 잘 예측한다”고 입증할 수는 없습니다.** 다음 구성을 먼저 고정하는 것이 합리적입니다.

**INFERRED — 기본 2,400개＋채널별 대응본**

| 구간 | 기본 수량 | 구성 |
|---|---:|---|
| 한국어 speech | **400** | real/fake 각 200, 학습에 쓰지 않은 화자·원본·생성기 계열 |
| 다국어·영어 speech | **400** | ASVspoof 5·DFADD에서 학습 제외 계열과 대응 real |
| music | **400** | 기존 보유 자료 중 미사용 곡·생성기 버전, real/fake 각 200 |
| 동시 혼합 | **600** | RR·RF·FR·FF 각 150 |
| 순차 혼합 | **600** | RR·RF·FR·FF 각 150, 방향 균형 |

각 기본 샘플에 **clean·전화채널·압축 3조건＝7,200행**을 만듭니다. 기본 계보를 선택용/최종 확인용 **1,200개씩** 나누고, 대응본은 함께 이동합니다. 길이 **4–60초**, mono/stereo·MP3/WAV/FLAC을 포함하되 비율은 **설계 가정**으로 명시합니다. fake speech 일부는 부분 합성으로 구성합니다.

- DFADD의 **MatchaTTS·PFlowTTS** 등을 계열 holdout 후보로 삼되 MLAAD·SpeechFake·AT-ADD 및 사전학습 detector의 노출 이력을 먼저 대조합니다. 이미 노출됐다면 “미지 생성기” 평가에서 제외합니다.
- music·혼합 구성요소도 원곡 단위로 분리합니다. 이미 모델 선택에 사용한 `newtest`를 다시 손대지 않은 최종 평가라고 부르지 않습니다.
- 공식 ADS와 구간별 성능을 함께 보고합니다. **7,200행을 독립 표본 7,200개로 계산하지 않습니다.**
- 실제 방송 음성 보조 평가에는 In-the-Wild가 맞지만, 공식 사이트는 **Apache-2.0**, 연결된 HF는 **CC BY-SA 4.0**, 용량 **8.16GB**로 표기가 다릅니다. 버전별 적용 조건을 확정한 뒤 사용합니다. [공식 사이트](https://deepfake-total.com/in_the_wild), [공식 연결 배포처](https://huggingface.co/datasets/mueller91/In-The-Wild/tree/main)

**LB 예측력 확인:** P1·S2a·S3·S6·S7·S8의 **동일 체크포인트 6개**를 고정 평가하고, 기존 DB와 새 grader의 Spearman/Kendall 및 **인접 변화 5회**의 방향 일치를 비교합니다. 원본 계보 단위 bootstrap으로 불확실성을 표시합니다. **6개 점수에 맞춰 평가셋 비중을 학습하지 않습니다.** 기존 **3/4회 일치**는 유망한 관찰이지만 예측력 검증은 아닙니다. 체크포인트별 새 평가값이 없으므로 현재 상관계수는 **UNKNOWN**입니다.

**(d) 오늘 밤과 이후**

**INFERRED — 오늘 밤에는 신규 데이터로 교체 모델을 만들지 않는 편이 낫습니다.** 할 만한 일은 생성기·계보·라이선스 manifest 정리이며, 추가 GPU는 **0시간·$0**입니다. 기존 제출 일정과 사전등록 규칙을 유지합니다.

문서의 속도·단가를 적용한 계산은 다음과 같습니다.

| 항목 | GPU시간·비용 |
|---|---:|
| 예산 전체 | **$18.79 ÷ $2.98＝6.31시간** |
| 자체 제작 2,400개, 평균 12초의 feature dump | 순수 처리 **0.44시간·$1.32** |
| grader 7,200행, 평균 12초의 feature dump | 순수 처리 **1.33시간·$3.97** |
| 후속 작업 상한 제안 | 학습용 dump **1시간·$2.98**, grader **2시간·$5.96** |
| 위 상한 사용 후 잔여 | **3.31시간·$9.85** |

순수 처리 시간에는 다운로드·시작·실패 재처리·생성·P1 **2단계×3 seeds**·패키징·검증이 빠져 있습니다. 남은 예산과 **07:30** 전에 전 과정을 끝낼 수 있는지는 **UNKNOWN**입니다.

**후속용 채택 규칙 제안:** 후보 **1개**와 비교 기준·평가셋을 결과 확인 전에 고정하고, 선택셋 ADS **Δ≥＋0.003**, 계보 bootstrap **95% CI 하한>0**, 주요 구간 하락 **0.005 이하**를 요구합니다. 통과 후보만 잠근 확인셋에서 한 번 확인하고 P1·CPU/parity/CUDA 검증까지 통과해야 채택합니다. 이 수치는 **설계 제안**이며 기존 슬롯 규칙을 대체하지 않습니다.

**중단 조건:** 권리·계보·라벨이 해결되지 않거나, 사전 상한 초과, 확인셋 실패, 또는 남은 시간이 실측 재학습·검증 시간＋**1시간 여유**보다 짧으면 중단합니다. GPU 실행은 별도 사용자 승인이 필요하며 이번 검토에서는 실행하지 않았습니다.

**(e) 하지 말아야 할 것**

- **라벨 오류 방치:** DFADD 구버전 사용, 영상 fake를 음성 fake로 간주, enhancement만 한 real을 fake로 변경하지 않습니다.
- **계보 누출:** 원본·crop·codec본·분리 stem·remix를 서로 다른 split에 넣지 않습니다. 혼합 음원은 **모든 donor**를 추적합니다.
- **생성기 중복 은폐:** dataset 이름·attack ID 차이를 새 계열의 증거로 쓰지 않습니다. 미상은 **UNKNOWN**으로 남깁니다.
- **채널 지름길 학습:** 전화채널은 real에만, 접합 흔적은 fake에만 넣지 않습니다.
- **AI-Hub 우회 반출:** DSD처럼 다른 데이터셋에 포함된 AI-Hub 유래 음성·features도 Mac 경계를 그대로 적용합니다.
- **재배포 조건 무시:** “비영리 사용 가능”과 “변형 학습 파일을 DACON에 제출 가능”을 같은 뜻으로 취급하지 않습니다. (`official_text.txt:94`, `:115`)
- **수량을 품질로 대체:** `constructed_splice` **121개**는 확인되지만 inventory에는 모두 `route=voice`로만 기록돼 있어 speech→music 순차 혼합이라는 의미까지 검증되지는 않습니다. (`DATA_INVENTORY_20260928.txt:26`, `:66`)