# 공개 음성·노래·음악 데이터 보강 조사

조사일: 2026-09-27 (KST)

대상: DACON 236749 딥보이스 범죄 대응 AI 탐지 모델

범위: 공개 배포 음성·노래·음악·부분 위조·실제 녹음과 현재 저장소의 데이터 보강 가능성

상태: 데이터 관련 96개 검토 항목과 대회 규칙·운영진 답변 대조 완료. 미확인 권리·접근 조건은 항목별로 남겼다. 신규 음원 다운로드·약관 동의·구매·학습·대회 제출은 수행하지 않았다.

## 1. 판단과 우선순위

공개 데이터만으로도 보강할 후보가 충분히 있다. 현재는 같은 생성기의 음성을 일괄 증량하기보다 **한국어 실제 가창, 새로운 합성 노래, 실제 보컬과 AI 반주의 혼합, 부분 위조, 처리된 실제 음원의 오탐 사례**에 우선순위를 둔다. 아래 순서는 예상 점수 순위가 아니라, 현재 데이터의 빈틈·확보 가능성·편입 비용을 고려한 실험 순서다.

| 순서 | 먼저 검토할 데이터 | 보강 목적 | 시작 전 확인할 사항 |
| --- | --- | --- | --- |
| 1 | GTSinger 한국어 및 기존에 없는 가창자 | 현재 CSD보다 실제 한국어 가창자·창법 다양성 확대 | **추가 약정 적용 범위 확인 전 편입 보류.** M4Singer를 지칭하는 문구·기관 권한 진술·면책 조건이 있다. 한국어 가창자 3명 확인. 전체 80.59시간을 한국어 분량으로 해석하지 않는다. [공식 설명](https://github.com/AaronZ345/GTSinger), [약정](https://raw.githubusercontent.com/AaronZ345/GTSinger/main/dataset_license.md) |
| 2 | SingMOS-Pro의 GT/SVS/SVC 중 출처가 명확한 부분 | 실제 노래와 합성·변환 노래 비교 | 7,981개·11.15시간은 전체 규모다. MOS가 아닌 생성 방식으로 진위 라벨을 만든다. [배포처](https://huggingface.co/datasets/TangRain/SingMOS-Pro) |
| 3 | 기존 Music Arena의 미편입 부분 + 배포 목록과의 ID 차집합 | 음악 생성기·버전 다양성 | 로컬 928개가 이미 있다. 동일 battle 분할 문제를 먼저 수정하고, 새 공개분이 실제로 얼마나 남는지 계산한다. [배포처](https://huggingface.co/datasets/music-arena/music-arena-dataset/blob/main/README.md) |
| 4 | Medley-solos-DB, Seoul Corpus | 실제 악기·보컬과 한국어 자연발화의 오탐 감소 | 곡·화자 단위 분할. 짧은 악기 클립을 완성곡이나 긴 대화의 대체재로 보지 않는다. [악기](https://zenodo.org/records/3464194), [한국어](https://www.openslr.org/113/) |
| 5 | HAIM C1/C2 및 미포함 hybrid 원본, Echoes의 빠진 생성기·ATA 조건 | 부분 AI 음악과 혼합·조건부 생성 사례 | **HAIM은 하위집합별 권리 설명이 달라 적용 조건 확인 전 추가 편입 보류.** 기존 원본·참조곡과의 차집합을 만든다. [HAIM](https://huggingface.co/datasets/mippia/HAIM), [Echoes](https://huggingface.co/datasets/Octavian97/Echoes) |
| 6 | DFADD 수정 ZIP, ODSS, MultiAPI-Spoof | 부족한 TTS·API 출력 다양성 | 기존 MLAAD/SpeechFake의 생성기·원발화와 비교. MultiAPI의 실제 Drive 파일 목록과 API별 메타데이터는 추가 확인한다. [DFADD](https://github.com/isjwdu/DFADD), [ODSS](https://zenodo.org/records/8370669), [MultiAPI](https://xuepingzhang.github.io/MultiAPI-Spoof-Dataset/) |
| 조건부 | MoM, CtrSVDD, AT-ADD Track 2, CodecFake+ CoSG | 신형 노래·음악 생성과 코덱 기반 생성 | 원본 권리, 약관, 재구성과 생성의 라벨 구분, 기존 데이터와의 계보를 확인한 부분만 사용한다. |
| 검증용 우선 | LlamaPartialSpoof, In-the-Wild의 독립 부분 | 새로운 출처·부분 위조에서 퇴행하는지 확인 | 학습·설정 선택에 쓴 부분을 독립 최종 평가라고 부르지 않는다. |

위 표는 보강 가치의 순서이며 규정 확인 완료 순서는 아니다. **권리 확인 부담까지 고려한 우선 준비 대상은 Music Arena, SingMOS-Pro, Medley-solos-DB, Seoul Corpus, DFADD, ODSS**다. GTSinger와 HAIM은 기술적으로 중요하지만 아래 보류 사유를 먼저 해소해야 한다. AI-Hub는 대회 사용을 운영진이 명시적으로 허용했으며, 해외 처리·제출 절차를 별도로 확인해야 한다.

권장 첫 실험은 기존 고정 인코더와 head 설정을 유지하고 한 종류의 데이터만 추가하는 비교다. 공개 데이터가 많다는 이유만으로 전체 미세조정이나 head 복잡도를 동시에 바꾸면 개선·악화의 원인을 구분하기 어렵다. 이 조사에서는 어느 후보의 리더보드 점수 상승도 실측하지 않았다.

## 2. 현재 데이터와 검증 한계

### 2.1 기준이 된 로컬 자료

이번 조사에서 다시 읽은 기준은 2026-09-23 결과물의 다음 두 CSV다. 이는 해당 실험의 저장된 구성이며, 실행 중인 모든 GPU 작업의 최신 구성을 확인했다는 뜻은 아니다.

- [bank_trainval.csv](../results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank/bank_trainval.csv)
- [bank_test.csv](../results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank/bank_test.csv)
- [수집 source 정의](../data_v4/sources.py)
- [추가 데이터 분할 코드](../data_v4/build_extra.py)
- [기존 권리 검토 기록](../_workspace/lb_admission_20260924/rights_decisions.json)

로컬 Git HEAD는 조사 시작 시 `f81e24ceb2d1bbe080f7c11c210c4a7695834791`였다. 미커밋 코드와 결과물이 다수 있어, 이 문서는 HEAD에 들어 있는 파일만을 분석한 보고서가 아니다. 원격 저장소의 최신 상태나 실행 중인 학습을 검증한 것으로 확대 해석하지 않는다.

| 항목 | CSV 재집계 결과 | 해석 |
| --- | ---: | --- |
| 전체 bank | 42,039 | 원본 train/val/test 합계 |
| 원본 train / val / test | 30,658 / 4,931 / 6,450 | test라는 이름만으로 미사용 독립 평가를 뜻하지 않는다. |
| P1 학습 구성 | 37,108 | 원본 train + test. val 제외 |
| P1 음성·음악 동시 존재 | 6,698 | 두 presence 값과 mask가 모두 1인 행 |
| 실제 음성 + 실제 음악 | 2,016 | 해당 진위 라벨이 알려진 행 |
| AI 음성 + AI 음악 | 1,817 | 동일 기준 |
| AI 음성 + 실제 음악 | 927 | 동일 기준 |
| 실제 음성 + AI 음악 | **112** | 우선 보강할 구체적 결핍 |
| 음성 진위 미상 + AI 음악 | 1,812 | 실제 음성으로 임의 변환 금지 |
| 음성 진위 미상 + 실제 음악 | 14 | 동일 |
| P1 `constructed_splice` | 103 | 해당 source 문자열의 수. 모든 부분 위조의 총수는 아니다. |
| P1 언어 미기재·unknown·und | 16,272 | 언어 다양성 평가가 불완전하다. |

위 수치는 clip 행 수다. 독립 가수·곡·원발화 수와 다르며, 부족한 조합이 곧 리더보드 오답의 주원인이라는 인과 증거도 아니다.

### 2.2 이미 사용한 데이터

전체 bank에서 확인한 주요 source는 MLAAD 8,775, SpeechFake 3,600, ASVspoof2021 LA 3,672/DF 2,875, ASVspoof2019 LA 2,400, Zeroth 3,000, FLEURS 한국어 1,500, OpenMIC 2,764, FakeMusicCaps 2,000, HAIM 1,492와 legacy 800, SONICS 800과 legacy 800, CSD 200, VocalSet 800이다. Echoes의 주 source는 **594개**이며, 여러 legacy 생성기별 source가 별도로 있다. 기존 설명의 600은 수집 목표·반올림과 구분한다.

FMA, JamendoMaxCaps, LibriSpeech, MUSAN 및 자체 혼합·편집도 존재한다. 따라서 이 문서에 다시 나온 이름이 모두 새 데이터라는 뜻은 아니다. `existing:...`와 새 source 문자열이 달라도 같은 원본을 재사용했을 수 있다.

## 3. 목록을 읽는 법

- **파일 공개**: 원 저자·기관의 배포처에 오디오 파일 또는 압축파일 목록이 확인됐다. 다운로드·디코딩·라벨 검사를 완료했다는 뜻은 아니다.
- **신청·동의**: 계정, 이용 동의, 소속 확인이나 수동 승인이 필요하다. 승인 기간을 보장하지 않는다.
- **링크·주석만**: URL·ID·CSV·JSON·논문·코드는 공개됐으나 전체 오디오 배포는 확인하지 못했다.
- **조건 미확인**: 실제 파일이 있어도 음원 이용 허락의 범위가 확인되지 않았다.
- **기존 계보**: 이미 사용 중인 자료 또는 다른 후보의 원곡·원발화를 재사용한다.

규모는 별도 표기가 없으면 **배포자·논문이 제시한 값**이다. Hugging Face의 row 수는 text label, 일부 config, 자동 변환 결과일 수 있어 usable audio 수와 같다고 가정하지 않는다. 서로 파생된 데이터셋의 시간·파일 수를 합산해 순증 규모로 제시하지 않는다.

대회는 공개 접근성과 최소 비영리 이용 허락을 요구하고, 2차 평가에서 출처와 학습 데이터 파일 일체를 제출하도록 명시한다. 그러므로 비상업 연구 이용이 가능하더라도 원음 전달이 제한되는 데이터는 별도 검토가 필요하다. 유료·gated·ND라는 이유 하나만으로 자동 허용·금지를 단정하지 않는다. [공식 규칙](https://dacon.io/competitions/official/236749/overview/rules)

### 3.1 이 대회의 공식 답변으로 확인한 허용 범위

다른 대회의 FAQ를 대신 적용하지 않고 **236749의 규칙과 DACON.GM 답변**을 대조했다. 아래 날짜는 운영진 답변일이다. 참가자 질문에 적힌 해석을 운영진 확답으로 인용하지 않았다.

| 쟁점 | 공식 확인 내용 | 이번 자료 선택에 적용할 판단 |
| --- | --- | --- |
| AI-Hub | **2026-09-21 10:28, 사용 가능**. 질문은 원데이터 제3자 전달 제한까지 언급한다. [답변](https://dacon.io/competitions/official/236749/talkboard/417844) | AI-Hub의 대회 사용 자격을 미확인·불허로 분류하지 않는다. 다만 짧은 답변에는 해외 GPU 반출과 구체적 파일 전달 절차가 없다. |
| NC 사전학습 모델 | **2026-09-21 10:26, 사용 가능**, 개별 조건 준수. 모델명·출처·license 기재 후 필요 시 추가 확인. [답변](https://dacon.io/competitions/official/236749/talkboard/417857?category=&dtype=&fType=&page=&ptype=) | 상금이 있다는 이유만으로 모든 NC 모델을 제외하지 않는다. 기관 한정·평가 전용 등 별도 제한까지 일괄 해제된 것은 아니다. |
| CC BY-NC-SA 데이터 | **2026-08-27 09:52, 사용 가능**, 공개 접근성·최소 비영리 허용·license 준수 조건. [답변](https://dacon.io/competitions/official/236749/talkboard/417212) | SA라는 이유만으로 CSD·OpenSinger 등을 제외하지 않는다. 출처 표시와 실제 배포·가공 조건은 지킨다. |
| 자체 생성 데이터 제출 | **2026-08-27 09:47**, 실제 학습에 사용한 생성 오디오 파일과 생성·재현 자료 모두 필요. 재생성 script만으로 파일을 대체할 수 없고 대용량 전달 수단 사용 가능. [답변](https://dacon.io/competitions/official/236749/talkboard/417198) | 사용한 파일·모델 revision·seed·prompt·후처리 정보를 보존한다. 생성했지만 폐기한 모든 후보까지 제출하라는 답변은 아니다. |
| 자체 검증 데이터 | **2026-09-19 23:56**, 순수 성능 검증에만 썼다면 제출 대상 제외. 결과로 모델·학습 설정을 선택·조정하면 학습 과정 활용으로 볼 수 있음. 상용 생성 출력의 순수 검증은 대회에서 제한하지 않음. [답변](https://dacon.io/competitions/official/236749/talkboard/417811?category=&dtype=recent&fType=&page=1&ptype=) | `dev`·`held-out`이라는 이름만으로 제출 의무에서 빠지지 않는다. 제공자의 자체 이용 조건도 여전히 적용된다. |
| 사전학습 원자료 | 같은 **2026-09-19** 답변에서 공개 가중치의 license를 따르면 원 개발자의 학습 데이터를 별도로 확인·제출할 필요가 없다고 안내. [답변](https://dacon.io/competitions/official/236749/talkboard/417811?category=&dtype=recent&fType=&page=1&ptype=) | 우리가 추가 학습에 넣은 자료와 원 개발자의 사전학습 자료를 구분한다. 후자를 전부 확보해야 한다는 요구를 만들지 않는다. |
| 파일 간 보정 | **2026-09-22 10:10**, 평가 파일 전체의 순위를 이용하는 보정은 불가. 학습 데이터에서 만든 고정 기준은 평가 자료를 사용하지 않고 파일 독립성을 지키는 조건에서 가능. [답변](https://dacon.io/en/competitions/official/236749/talkboard/417885?category=&dtype=recent&fType=&page=1&ptype=) | 새 데이터로 보정기를 만들더라도 비공개 평가 집합의 분포·순위를 계산해 다시 맞추면 안 된다. |

**구매·기관 심사 gate는 아직 일괄 허용을 확인하지 못했다.** NEXDATA나 DeepSpeak의 유료 계약은 학습 허용만으로 충분하지 않다. 누구나 접근 가능한 공개 자원 요건에 맞는 판매·접근 방식인지, 이 대회에서의 사용과 검증기관 전달이 계약상 가능한지를 함께 확인해야 한다. 유료라는 이유만으로 금지라고 단정하지도 않는다.

### 3.2 라벨과 라이선스에서 놓치면 안 되는 조건

**대회 라벨은 데이터셋의 real/fake 표시보다 우선한다.** 보컬은 voice, 반주·악기는 music이다. 둘이 함께 있으면 mixed이며 한 성분이라도 AI 생성이면 파일은 fake다. 실제 원천에 새 성분을 생성하지 않는 잡음 제거·음량 조정·품질 개선만 적용했다면 real이다. 따라서 HAIM의 AI mastering, ATDD의 auto-tune, codec 재구성을 이름만 보고 전부 fake로 바꾸면 안 된다. 해당 처리의 생성 여부를 확인할 수 없으면 진위 라벨을 보류한다. [공식 정의](https://dacon.io/competitions/official/236749/overview/description)

운영진은 **2026-09-16 17:58**에 voice와 music이 모두 없는 파일의 `FILE_FAKE=0`을 확인했다. 일부 구간·일부 원천이 fake이면 해당 성분도 fake이며, 생성 구간의 최소 길이는 공개하지 않았다. 같은 답변은 리더보드 프로빙을 권장하지 않고 사후 검증에서 불이익이 있을 수 있다고 안내한다. 점수만으로 숨은 평가 데이터의 구성·정답을 확정하거나 이를 역추정하는 제출 계획을 세우지 않는다. [운영진 답변](https://dacon.io/en/competitions/official/236749/talkboard/417702)

| 조건 | 이번 보고서의 해석 |
| --- | --- |
| BY | 배포 시 저자·원출처·license와 수정 사실을 함께 남긴다. HF 카드의 배지 하나가 제3자 권리까지 해결했다는 뜻은 아니다. [CC FAQ](https://creativecommons.org/faq/) |
| NC | 이 대회는 NC 모델·NC-SA 데이터를 허용했다. 그래도 별도 계약의 학술기관 한정, 사용 목적, 후속 상업 이용 제한까지 없어지는 것은 아니다. |
| ND 4.0 | 허용 목적의 비공개 변형과 변형물의 공유를 구분한다. 필요한 기술적 형식 변경은 그 자체로 adaptation이 되지 않는다고 명시한다. 반면 변형물 공유 제한은 실제 학습 파일 제출과 함께 검토해야 한다. **ND=모든 학습·리샘플링 금지**라고 쓰지 않는다. [법문 §2](https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.en) |
| SA | 변형물을 공유할 때 적용되는 조건을 확인한다. 학습했다는 사실만으로 모든 모델 가중치·코드를 같은 license로 공개해야 한다고 단정하지 않는다. [법문 §3](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en) |
| 평가 전용 | 제공자가 detector evaluation만 허용한 자료를 학습·fine-tuning에 쓰지 않는다. 대회가 순수 검증 데이터 제출을 면제해도 제공자의 사용 범위를 넓혀 주지는 않는다. |
| 원음 전달 제한 | JVS-MuSiC·CoreaSpeech+ 등은 학습 가능성 외에 2차 평가의 파일 전달이 문제다. 원음 대신 feature·다운로드 링크만 내면 된다고 임의 가정하지 않는다. AI-Hub는 사용 허용 답변이 있으므로 전달 절차를 별도로 확인하는 항목으로 둔다. |

### 3.3 후보별 규정 적합성 판단

아래 판단은 **이용 허락과 확보 증거를 바탕으로 한 다음 행동**이다. 개별 음원의 권리와 품질을 모두 검증한 학습 승인 목록은 아니다. `조건부 검토`는 공개 파일·명시된 이용 조건을 근거로 선별을 시작할 수 있다는 뜻이고, `편입 보류`는 적힌 사유를 해소하기 전 학습에 추가하지 않는다는 뜻이다. `배포·음원 미확인`은 규정 위반 판정과 다르다. 각 ID의 원출처는 4–7절에 연결했다.

| 대상 ID | 판단 | 남은 조건 또는 구체적 사유 |
| --- | --- | --- |
| A01, A02, A04, A06, A10, A14, A15 | 조건부 검토 | 명시된 공개 이용 조건 적용, 원본 계보·출처 표시·실제 사용할 파일 전달 범위 확인. A06은 독립 검증 목적 우선. |
| A03, A18, A38 | 편입 보류 | license 표기는 있으나 각각 실제 Drive 내용, ZIP 적용 버전, archive·manifest 대조가 남음. |
| A05, A07, A08, A12, A17, A19, A20, A24, A25, A28, A29, A36, A37 | 편입 보류 | 원음 grant·현재 공식 전달 경로·버전별 조건 중 미확인 사항이 있음. 각 행의 미확인 이유를 해결한 부분만 재판정. |
| A09 | 편입 보류 | CodecFake+의 MIT 표기만으로 생성·재구성 음원과 원자료 전체의 적용 범위를 확정하지 못함. |
| A11, A13, A34 | 편입 보류 | NC-ND의 가공·변형물 전달 범위를 실제 전처리와 2차 제출물 기준으로 확인. 학습 자체의 일괄 금지를 뜻하지 않음. |
| A16, A21, A33 | 편입 보류 | 배포 license와 별도로 원미디어 출처·이용 범위를 확인. **SCDF와 VoxPopuli의 데이터 license는 모두 CC0**이며 NC 충돌이 확인된 것은 아님. |
| A22, A23, A31 | 편입 보류 | 신청·승인·custom license 및 공개 접근성 요건 대조 필요. 학생 개인을 학술기관 자격으로 자동 간주하지 않음. |
| A26 | 편입 보류 | DSD의 AI-Hub 파생 부분은 정확한 dataset ID와 원음 재배포 권한 불명. 비AI-Hub 부분은 별도 심사 가능. |
| A27, A30 | 배포·음원 미확인 | AI4T는 미디어 링크 중심, SPADE 한국어는 공개 대기. 다른 언어의 gate는 별도 검토. |
| A32 | 추가 편입 보류 | MLAAD의 기존 권리 보류 기록과 SpeechFake의 원배포 조건을 재대조. 기존 bank에 있다는 사실은 승인 근거가 아님. |
| A35 | 학습 보강 제외 | VoiceWukong의 확인된 허용 목적은 승인된 탐지기 평가. 별도 학습 허락이 확보되기 전 training·fine-tuning에 사용하지 않음. |
| B01, B09, B22 | 편입 보류 | GTSinger/M4Singer 추가 약정, 기관 권한·면책 진술 및 파생 SVCC2025 적용 범위 확인. NC-SA 자체가 보류 사유는 아님. |
| B02, B03, B05, B06, B13, B27 | 조건부 검토 | 공개 배포 license와 원곡 조건, 실제/합성·GT 라벨, 가수·곡 계보 확인. B03/B27은 기존판 중복을 순증에서 제외. |
| B04, B10 | 편입 보류 | ND 공유 조건, 원 가창 자료의 별도 조건·전달 가능성 확인. |
| B07, B11 | 편입 보류 | custom agreement 또는 신청·배포 제한과 대회 사용·전달 요건 대조. |
| B08, B12, B15, B16, B20 | 편입 보류 | 실제 payload·원음 이용 허락·원자료별 조건 중 각 행에 적힌 미확인 사항 해소. |
| B14 | 편입 보류 | JVS-MuSiC 오디오의 재배포 제한. 태그 license로 원음 전달을 정당화할 수 없음. |
| B17, B18, B19, B21, B23, B24, B25, B26 | 배포·음원 미확인 | URL·annotation·논문·예시·다른 corpus 참조를 전체 음원 배포로 계산하지 않음. B25는 audio 라벨도 별도 확인. |
| C01, C11 | 조건부 검토 | 명시된 CC BY 4.0 적용과 출처·수정 기록 보존. C01은 오디오 비공개 모델 제외 및 battle grouping 수정이 먼저. |
| C02, C04, C06, C12 | 조건부 검토 | 실제 배포된 허용 파일만 선별하고 참조 원곡·트랙별 license 확인. URL-only 부분까지 허가된 것으로 확대하지 않음. |
| C03, C13 | 편입 보류 | HAIM은 하위집합별 license 설명 충돌, Jamendo는 로컬 TDM 미해결 기록. 상위 권리 조건부터 확인. |
| C05, C07, C08, C09, C10, C15, C16 | 편입 보류 | 각각 음원 grant, 교육용 조건, 전달 gate, 파생 원자료 권리 등 미확인. 논문·코드 license로 대체하지 않음. |
| C14, C18, C19 | 배포·음원 미확인 | metadata 또는 공개 예정 자료. 현재의 허가된 오디오 확보 수량으로 세지 않음. |
| C17 | 라벨 확인 후 검토 | CC BY 4.0 확인. MIDI·가상악기 렌더를 AI 생성 또는 사람 실연으로 자동 분류하지 않고 실제 제작 방식과 대회 정의 대조. |
| D01, D12 | 조건부 검토 | 공개 음원 조건 적용. D12는 이미 사용한 한국어 자료이므로 추가판과 원화자 차집합 확인. |
| D02, D03, D04, D05, D06, D07, D08, D09, D10 | 대회 사용 허용 확인 | AI-Hub에 대한 운영진 허용 답변 있음. 개별 승인·음원 구성·해외 반출·2차 전달 절차 확인은 남음. |
| D11 | 편입 보류 | CoreaSpeech+의 원음 재배포 금지와 대회 파일 전달 요구를 조정할 허락·절차 미확인. |

규정과 현재 확보 상태를 함께 보면, **모든 후보를 한 번에 내려받아 학습하는 방식은 권하지 않는다.** 조건이 확인된 하위집합부터 파일별 provenance·license·분할을 고정하고, 모델 선택에 사용한 검증 자료까지 별도로 기록해야 한다.

## 4. 합성 음성·부분 위조

| ID | 데이터와 원 출처 | 내용·규모·배포 상태 | 음원 이용 조건 | 보강 가치와 제한 |
| --- | --- | --- | --- | --- |
| A01 | [DFADD](https://github.com/isjwdu/DFADD) · [파일](https://huggingface.co/datasets/isjwdu/DFADD) | 영어, Grad-TTS/Matcha-TTS/NaturalSpeech 2/PFlow-TTS/StyleTTS 2. 실제 ZIP 배포 | 데이터 카드 MIT, VCTK CC BY 4.0, LJSpeech public domain | **우선 검토. 2025-04 수정 ZIP** 사용. Matcha 오디오·라벨 불일치가 수정됨. bona fide 병합 지침을 지키고 VCTK·기존 생성기 중복 확인 |
| A02 | [ODSS](https://zenodo.org/records/8370669) | 영어·독일어·스페인어 실제/TTS 쌍, 2.4GB ZIP | 자연·합성 오디오 모두 CC BY-SA 4.0 명시 | 비교적 작은 실제/합성 보강 후보. VITS/FastPitch 계열과 원문·원화자 중복 확인 |
| A03 | [MultiAPI-Spoof](https://xuepingzhang.github.io/MultiAPI-Spoof-Dataset/) | 약 230시간 fake, 30 API, Common Voice bona fide 동량. 공식 Drive 링크 | 데이터 CC BY-NC 4.0 명시. Drive 내부 파일·접근 동작 미확인 | 상용 API 다양성 후보. 익명 API ID만으로 새로운 생성기임을 확정할 수 없음 |
| A04 | [ASVspoof 5](https://zenodo.org/records/14498691) | MLS 기반 영어, 다양한 합성·공격. 전체 배포 142.3GB, train 약 37.5GB | DB ODC-By, genuine audio CC BY 4.0. 공식 README의 부분별 조건 적용 | 기존 2019/2021과 다른 공격·원음 후보. 전체 평가분까지 무차별 학습하지 않고 공식 split·codec seed 유지 |
| A05 | [PartialSpoof v1.2](https://zenodo.org/records/5766198) | 진짜 발화 일부에 합성 구간 삽입, 구간별 라벨. train/dev 각 약 2GB, eval 5.8GB | 이번 웹 추출에서 Rights 값 미표시. 원 배포 README와 ASVspoof 기반 조건 확인 필요 | 부분 위조 학습 후보. ASVspoof2019 계보와 교차 중복 확인. v1.3 구간 라벨은 별도 링크 |
| A06 | [LlamaPartialSpoof](https://github.com/hieuthi/LlamaPartialSpoof) · [파일](https://huggingface.co/datasets/HaoY0001/LlamaPartialSpoof) | LibriTTS 기반 전체·부분 위조, XTTS/GPT-SoVITS/CosyVoice/ElevenLabs 등, 28.2GB 파일 | CC BY 4.0 | **외부 검증 우선.** 원래 평가용; 후속 연구용 화자 분할도 제공. 학습에 쓴 원본·파생본을 독립 OOD로 재사용하지 않음 |
| A07 | [PartialEdit v1.1](https://zenodo.org/records/18829689) | VoiceCraft/SSR-Speech 편집, 구간 라벨·화자 split. editor별 43,358개, 전체 파일 20.2GB | 공식 README에서 명시된 라이선스 grant를 확인하지 못함 | 기술 적합성 높지만 **조건 확인 전 보류**. Audiobox E3/E4 미공개. 원본 VCTK 별도 필요 |
| A08 | [PhonemeDF](https://zenodo.org/records/19522979) | 실제 28,539 + 합성 199,773개, 약 730시간, 82.9GB. Chatterbox/XTTS/FreeVC 등 | 음원 이용 허락 미확인 | Chatterbox 등 부족한 생성기 후보. phoneme alignment를 부분 위조 정답으로 혼동하지 않음 |
| A09 | [CodecFake+](https://huggingface.co/datasets/CodecFake/CodecFake_Plus_Dataset/tree/main) | CoRS 약 100GB, **CoSG 244MB**. 2025-10 두 부분 업로드 기록·파일 확인 | 카드 MIT지만 별도 음원 권리 설명 불충분. VCTK 등 원본 조건 확인 | CoSG 생성 부분부터 검토. CoRS 재구성은 생성과 라벨 의미가 다름. 142만 text rows를 독립 음원 수로 쓰지 않음 |
| A10 | [CodecFake, Wu/Tseng](https://huggingface.co/datasets/rogertseng/CodecFake) | VCTK 영어 코덱 재구성, 표시 707,872행·102GB, 오디오 Parquet | 카드 CC BY 4.0 | 아래 Xie 자료와 다른 프로젝트. VCTK/DFADD/CodecFake+ 계보 중복 |
| A11 | [Codecfake, Xie](https://github.com/xieyuankun/Codecfake) | VCTK 영어·AISHELL3 중국어, train 740,747/dev 92,596; 코덱·ALM 테스트 파일 | CC BY-NC-ND 4.0 | 재구성·생성 구분 필요. AudioGen 일반 소리가 포함된 test를 전부 음성으로 라벨링하면 안 됨. ND 가공·제출 범위 미해결 |
| A12 | [ST-Codecfake](https://zenodo.org/records/14631091) · [추가 open set](https://zenodo.org/records/14631085) | 11 코덱, bilingual source tracing, 주 배포 39.3GB | 오디오 라이선스 값 이번 확인에서 미확보 | Xie 계보의 추적·OOD 보강. 완전히 독립한 실제 음성 출처로 계산하지 않음 |
| A13 | [AUDETER](https://huggingface.co/datasets/wqz995/AUDETER) · [설명](https://github.com/mike-qz-wang/AUDETER) | 논문 300만 clip/4,500시간 이상; HF 파일 1.08TB 표시. 11 TTS+10 vocoder, fake-only | CC BY-NC-ND 4.0 명시 | F5/Fish/Spark/Zonos 등 누락 조건만 선별. 저장소의 uploading 문구와 실제 공개 파일이 불일치. 전체 발표 규모를 완전 배포·독립 수량으로 단정하지 않음 |
| A14 | [WaveFake](https://zenodo.org/records/5270336) | LJSpeech 영어·JSUT 일본어, 생성 clip 104,885개, 23.4GB ZIP | 음원 CC BY-SA 4.0 | 오래된 vocoder 보강. 실제 원발화·화자 다양성은 제한적. 코드 MIT와 구분 |
| A15 | [WaveFake 확장](https://zenodo.org/records/10512541) | Avocodo/BigVGAN/BigVGAN-L 재구성 39,300개, 9.8GB | CC BY-SA 4.0 | 새 vocoder 출력이지 새로운 원음이 아님. 동일 LJSpeech 원문 파생본을 묶음 |
| A16 | [DECRO](https://github.com/petrichorwq/DECRO-dataset) · [파일](https://zenodo.org/records/7603208) | 영어·중국어 split 합계 118,381개, TTS/VC·Baidu/Xunfei | 공식 저장소 CC BY 4.0 표기. 원자료 일부 별도 연구 조건 확인 | 영어 bona fide가 ASVspoof2019 LA 재분할. 새 독립 real로 사용 금지. README의 일부 언어·출처 설명 불일치 |
| A17 | [CVoiceFake v2](https://zenodo.org/records/14062964) | EN/ZH/DE/FR/IT, vocoder·DiffWave, 46.3GB | 공개 파일 있으나 원 페이지의 라이선스 값을 읽지 못함 | Common Voice 계보. 언어별 출력 다양성 후보, 권리 확인 후 편입 |
| A18 | [LibriSeVoc](https://zenodo.org/records/15127251) · [저자 카드](https://huggingface.co/datasets/csun22/LibriSeVoc/blob/main/README.md) | 13,201 LibriTTS 원본에서 92,407개·243.66시간, ZIP 45GB | 저자 HF 카드 CC BY-SA 4.0. ZIP 버전 적용 범위 대조 필요 | 원발화보다 파생본이 많은 구조. LlamaPartialSpoof/CD-ADD/SONAR 등과 계보 비교 |
| A19 | [CD-ADD](https://github.com/leolya/CD-ADD) | train 300시간 이상/test 약 50시간, zero-shot TTS·codec·denoising. OpenXLab/Baidu 링크 | 음원 라이선스·접근 동작 미확인 | LibriTTS와 TEDLIUM3 기반. 조건 변화 진단용 후보이나 추가 승인·권리 검토 필요 |
| A20 | [SONAR](https://github.com/Jessegator/SONAR) | 평가용, 여러 합성 플랫폼, 공식 Drive 링크 | 복합 음원 이용 조건 미확인 | LibriTTS와 공개 데모 계보. 타 배포본의 SeedTTS 실제 prompt 오라벨 지적은 **추가 감사 단서**, 원본 검증 완료가 아님 |
| A21 | [In-the-Wild](https://huggingface.co/datasets/mueller91/In-The-Wild/blob/main/README.md) | 온라인 영어 실제/가짜, 58명 identity, 저자 ZIP | 카드 CC BY-SA 4.0 | 현실 채널·미상 생성기 진단. 영상·화자 단위 중복, AUDETER의 원음 재사용 확인. 생성기 독립이라고 단정하지 않음 |
| A22 | [DeepSpeak v2](https://huggingface.co/datasets/faridlab/deepspeak_v2) | 웹캠 real/생성 데이터, 52시간 이상, 표시 134GB, ElevenLabs/PlayHT/Speechify | 신청서+HF gate, 적격 학술 이용과 기타 이용의 조건·비용 상이 | 새로운 수집 화자 후보지만 즉시 승인 보장 없음. **audio-config로 라벨 판정**, fake video를 fake audio로 치환 금지 |
| A23 | [DeepSpeak-Agentic](https://huggingface.co/datasets/faridlab/deepspeak-agentic) | 사람·음성 에이전트 통화 37시간 이상, 43.9GB, Starfish/ElevenLabs/Cartesia | 기관·연구 신청과 custom agreement | 최신 상용 대화 도메인. 원통화·분리 stream·혼합 stream 모두 같은 session group |
| A24 | [IndieFake](https://indie-fake-dataset.netlify.app/) | 인도 영어 50명, 27.17시간 real/fake, 예시 공개 | 이름·이메일·소속·목적 신청; 상세 오디오 조건 미확인 | accent 보강 후보. 전체 파일을 즉시 받을 수 있는 자료로 분류하지 않음 |
| A25 | [TIMIT-TTS](https://zenodo.org/records/6560159) | 영어 TTS 및 time-warp, 7.2GB ZIP | 원 페이지 오디오 이용 조건 미확인 | 영상과 정렬된 합성 음성. 실제 원음이 함께 공개됐다고 가정하지 않음 |
| A26 | [DSD-Corpus](https://zenodo.org/records/13788455) | 17.4GB multipart ZIP+CSV; LibriSpeech/AIHub/VCTK, 공개 TTS·ElevenLabs | 배포 CC BY-NC 4.0. AIHub의 구체 ID·재배포 권한 미확인 | 한국어 fake 포함 여부는 확인되지 않음. AIHub 사용·한국 연구기관이라는 사실만으로 한국어 자료로 분류하지 않음 |
| A27 | [AI4T](https://github.com/davidcombei/AI4T) | 논문 196 fake+192 real source, 소셜미디어 링크 | 코드·메타데이터 CC BY 4.0과 원미디어 재배포 권리 부재를 명시 | 실제 플랫폼 변형 연구 후보. 즉시 쓸 수 있는 음원 ZIP이 아님 |
| A28 | [FoR](https://bil.eecs.yorku.ca/wp-content/uploads/2020/01/FoR-Dataset_RR_VT_final.pdf) · [원 연구실](https://bil.eecs.yorku.ca/datasets/) | 198,000개 이상이라는 원 논문, 네 가지 파생판 | 저자 학위논문 GPLv3 설명; 현행 원음 배포 경로 미확인 | 오래된 TTS와 공개 원음 계보. 네 버전을 독립 데이터로 합산하지 않음 |
| A29 | [ADD2023](https://arxiv.org/abs/2408.04967) · [Track 1.2 배포](https://zenodo.org/records/12151404) | 중국어 위조·구간 탐지·출처 추적. track별 별도 공개 | 정확한 원 배포 조건 미확인. 제3자 labels-only 패키지와 구분 | AISHELL 계보와 실제 track 공개 범위 확인 필요 |
| A30 | [SPADE](https://huggingface.co/datasets/rogertseng/spade) | 다국어 부분 위조. 공식 카드에서 한국어·영어·일본어 공개 대기 | 다른 언어는 연락처 동의 gate | **현재 한국어 보강 수량으로 계산하지 않음** |
| A31 | [SpoofCeleb](https://huggingface.co/datasets/jungjee/spoofceleb/tree/main) | 실제 환경 음성과 TTS, 소속 이메일 신청 | 저자 안내상 승인 2–3일, 세부 조건 동의 | 마감 직전 확보를 전제로 실험 일정 수립 금지 |
| A32 | [MLAAD](https://huggingface.co/datasets/mueller91/MLAAD) · [SpeechFake 수집 정의](../data_v4/sources.py) | 이미 bank에 각각 8,775/3,600개 존재. 추가 시스템·언어만 비교 대상 | MLAAD 비상업 학술 조건과 기존 권리 보류 기록 재검토. SpeechFake는 현재 source 정의의 원 배포처를 기준으로 확인 | 전체 증량보다 생성기×버전×언어의 빈 칸을 먼저 찾음. |
| A33 | [SCDF](https://zenodo.org/records/16744845) · [VoxPopuli 조건](https://github.com/facebookresearch/voxpopuli#license) | 48.1GB waveforms ZIP, 23.7만 발화 이상·500시간 이상 발표. Czech/French/English/German/Spanish, VoxPopuli 50명, XTTSv2/F5/OpenVoice v2/DDDM-VC | SCDF와 VoxPopuli **데이터 모두 CC0** 표기. VoxPopuli 모델·코드의 NC와 혼동 금지. 의회 원미디어 안내의 적용 범위 미확인 | 새 방송·의회 발화 계보 검토 가치. genuine은 화자당 25개로 fake 편중. 원미디어 조건·중복 확인 전 편입 보류 |
| A34 | [DiffSSD](https://huggingface.co/datasets/purdueviperlab/diffssd) | 70,000 영어 synthetic clip, 18.1GB TAR. ElevenLabs/PlayHT 포함 10 생성기, LJ Speech/LibriSpeech 참조 | 합성 음원 CC BY-NC-ND 4.0 명시, 생성기·원자료 조건 포함 | 상용 출력 다양성 후보. real 원음 별도 확보, ND 가공·제출 범위 검토 필요 |
| A35 | [VoiceWukong](https://github.com/VoiceWukong/VoiceWukong) | 영어 fake 265,200/중국어 148,200 발표, 다수 상용·공개 도구, 신청 필요 | 현행 조건은 학술 탐지기 **평가 목적**, 승인·재배포 제한 | **학습 보강에서 보류.** 연구용이라는 표현을 학습·fine-tuning 허가로 확대하지 않음 |
| A36 | [HABLA](https://zenodo.org/records/7370805) | 7.5GB RAR, real 22,816/fake 58,000, 라틴아메리카 스페인어 5 accent/162명, VC·Azure TTS | 상위 OpenSLR 출처는 제시되나 데이터 자체 라이선스 값 미확인 | accent·VC 보강 후보. 영어·한국어 데이터로 오인하지 않음 |
| A37 | [STOPA](https://github.com/Manasi2001/STOPA) · [공식 연결 파일](https://zenodo.org/records/15606628) | 약 70만 영어 sample 발표, VCTK, 13 synthesizer. 공식 음원 링크 있으나 이번 fetch 실패 | 저장소 MIT는 소프트웨어에 명시. 음원 라이선스 미확인 | 구조·설정별 합성 흔적 연구. 원음 다양성을 70만 개로 해석하지 않음 |
| A38 | [Diffuse or Confuse](https://github.com/AntonFirc/diffusion-deepfake-speech-dataset/) | diffusion·재보코딩 데이터. 공식 저장소에 Zenodo 배포 안내, 실제 archive 목록 이번 미확인 | README가 데이터 CC BY 4.0 명시 | 파일 확보 경로·manifest 확인 후 검토. 한국어 포함은 확인되지 않음 |

### 4.1 음성 데이터의 우선순위 해석

A01/A02는 비교적 명확한 파일·조건으로 작은 대조 실험을 설계하기 좋다. A33은 기존과 다른 원음 도메인을 검토할 가치가 크지만 배포자가 제시한 CC0만으로 상위 원음 권리 문제가 모두 해결됐다고 단정하지 않는다. A03/A09는 최신 API·코덱 생성이라는 장점이 있으나 각각 실제 파일 목록과 음원 라이선스 적용 범위가 미해결이다.

A13의 전체 TB급 수집, A14/A18의 대규모 재보코딩 증량보다, 기존 데이터에 없는 생성기 버전·화자·원발화·조건 조합을 추려 추가하는 편을 우선 제안한다. 이는 실험 비용에 관한 판단이며 성능 우월성이 검증됐다는 뜻은 아니다.

## 5. 실제·합성 가창

| ID | 데이터와 원 출처 | 내용·규모·배포 상태 | 음원 이용 조건 | 보강 가치와 제한 |
| --- | --- | --- | --- | --- |
| B01 | [GTSinger](https://github.com/AaronZ345/GTSinger) · [파일](https://huggingface.co/datasets/GTSinger/GTSinger/tree/main) · [license](https://raw.githubusercontent.com/AaronZ345/GTSinger/main/dataset_license.md) | 실제 가창 80.59시간, 20명, 9개 언어, 별도 paired speech 16.16시간. HF 54.2GB·한국어 폴더 확인 | CC BY-NC-SA 4.0 + 추가 약정 문구. 일부 문구가 M4Singer를 지칭하는 불일치 | **한국어 실제 가창 우선 후보.** KO-Soprano-1/2, KO-Tenor-1 확인. 한국어 시간은 미집계. 원곡·가수·창법 변형별 group 유지 |
| B02 | [SingMOS-Pro](https://huggingface.co/datasets/TangRain/SingMOS-Pro) | 중국어·일본어 7,981 clip, 11.15시간, 1.48GB. GT/SVS/SVC metadata | CC BY 4.0 | 실제·합성 가창 비교. MOS는 품질 점수이므로 진위 라벨로 사용하지 않음. codec/vocoder 항목은 생성 provenance 별도 판정 |
| B03 | [SingMOS](https://huggingface.co/datasets/TangRain/SingMOS) | 이전 singing MOS release, GT·SVS·SVC·vocoder·codec 유형 | CC BY 4.0 카드 | Pro와 중복 가능성이 커 합산 금지. 독립 새 데이터보다 버전·system 차집합 대상 |
| B04 | [CtrSVDD](https://github.com/SVDDChallenge/CtrSVDD_Utils) · [train/dev](https://zenodo.org/records/10467648) · [test](https://zenodo.org/records/12703261) | bona fide 47.64시간/fake 260.34시간, 164 singer ID/14 방식 발표. train/dev 7.9/3.4GB, test 9GB | CC BY-NC-ND 4.0 및 일부 원자료 추가 조건 | fake singing 핵심 후보. Oniku/Ofuton/Kiritan/JVS-MuSiC 일부 real은 별도 확보·분절 필요. metadata가 곧 모든 오디오 포함을 뜻하지 않음 |
| B05 | [ACE-Opencpop](https://huggingface.co/datasets/espnet/ace-opencpop-segments) · [논문](https://www.isca-archive.org/interspeech_2024/shi24_interspeech.pdf) | **ACE Studio 합성 가창**, 수동 tuning. 128.9시간 발표, HF 43.6GB/105,960행 | CC BY-NC 4.0 | 중국어 fake 보강. 사람 녹음으로 오인 금지. Opencpop 악곡과 다른 파생본 계보 연결 |
| B06 | [ACE-KiSing](https://huggingface.co/datasets/espnet/ace-kising-segments) | **ACE Studio 합성**, 중국어·영어, 32.5시간 발표, 10.8GB/28,960행 | CC BY-NC 4.0 | 작은 fake 가창 후보. 원 KiSing과 라벨이 다르고 B05와 생성기 계열은 같음 |
| B07 | [AT-ADD 2026 Track 2](https://huggingface.co/datasets/xieyuankun/AT-ADD-Track2/tree/main) · [공식](https://www.at-add.com/) | 노래·음악 포함 all-type. train 19.8/dev 11.1/progress 5.79/eval 29.1GB archive 목록 | custom access agreement, 비상업 학술·교육/제3자 조건/재배포 조건. 현재 HF는 동의 후 자동 접근 안내 | 유망하지만 **승인·약관 조건부**. 노래 원자료 Opencpop/M4Singer/KiSing 중복. Track 전체를 singing 전용 규모로 쓰지 않음 |
| B08 | [OpenSinger](https://multi-singer.github.io/) · [배포 설명](https://github.com/Multi-Singer/Multi-Singer.github.io) | 중국어 약 50시간, 원 문헌 66명. 공식 Drive 링크 | CC BY-NC-SA 표기. Drive 실제 payload 미열람 | 실제 가창자 다양성 후보. 후속 자료의 76 ID와 임의 통합하지 않음 |
| B09 | [M4Singer](https://github.com/M4Singer/M4Singer) · [license](https://raw.githubusercontent.com/M4Singer/M4Singer/master/dataset_license.md) | 중국어 다가수 약 30시간. 현재 README 공식 Drive 링크 | CC BY-NC-SA 4.0, 추가 약정 | real 가창 후보. 과거 수동 신청 설명만 보고 현행 전체를 email-gated로 단정하지 않음. CtrSVDD/AT-ADD와 계보 중복 |
| B10 | [Opencpop](https://wenet-e2e.github.io/opencpop/download/) · [license](https://wenet-e2e.github.io/opencpop/liscense/) | 중국어 100곡, 여성 1명, 3,756발화·약 5.2시간 | CC BY-NC-ND 4.0, 공식 접근 절차 | 새 가수 다양성은 작음. ACE-Opencpop/CtrSVDD와 원곡 중복. ND가 ML 전체 금지라는 단정은 하지 않음 |
| B11 | [PopCS](https://github.com/MoonInTheRiver/DiffSinger/blob/master/resources/apply_form.md) | 중국어 여성 1명 중심 실제 가창 | 소속·연구 목적을 기재한 공식 이메일 신청, CC BY-NC-SA 4.0, 링크 배포 제한 | 접근 시간·가수 다양성 때문에 후순위. 신청하지 않음 |
| B12 | [원 KiSing 참고 논문](https://www.isca-archive.org/interspeech_2022/shi22d_interspeech.pdf) | 원 중국어 실제 가창 corpus. 원 배포 페이지의 현재 상태 미확인 | 이용 조건 미확인 | **ACE-KiSing으로 대체하면서 real 라벨을 유지하면 오류**. 원 KiSing은 별도 식별 |
| B13 | [PJS](https://sites.google.com/site/shinnosuketakamichi/research-topics/pjs_corpus) | 일본어 짧은 노래 100개+병렬 speech 100개, 1명, 공식 Drive ZIP 0.26GB | 모든 데이터 CC BY-SA 4.0 | 작은 real 가창 보강, 다양성 효과는 제한. speech와 singing 합계 구분 |
| B14 | [JVS-MuSiC](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_music) | 일본어 100명×2곡, 원본·pitch/tempo 편집, ZIP 0.6GB | 태그는 CC BY-SA, **오디오는 학술·비상업·개인 이용 및 재배포 제한** | 화자 다양성·보정된 사람 노래의 오탐 점검. CtrSVDD 의존 자료이므로 이중 계산 금지 |
| B15 | [Kiritan](https://github.com/mmorise/kiritan_singing) · [논문](https://www.jstage.jst.go.jp/article/ast/42/3/42_E2074/_pdf) | 1명·50곡. full-song 3시간31분과 active phoneme 약57분은 다른 수치. GitHub는 label | 오디오는 원 공식 사이트 별도, 저자 설명은 일본 저작권법 30-4 범위. 단순 CC grant로 확인되지 않음 | 작은 가수 계보, CtrSVDD 의존. 현재 접근·허용 범위 확인 필요 |
| B16 | [NUS-48E](https://smcnus.comp.nus.edu.sg/) · [논문](https://smcnus.comp.nus.edu.sg/archive/pdf/2012-2013/2013_05-Pub-NUS-48E.pdf) | 12명, 48 performance/20 unique English songs, singing 115분+speech 54분. 공식 Drive 연결 | 상세 음원 라이선스 미확인 | 소규모 real 보강. 새 Zenodo mirror의 license를 원저자 권한 증명으로 사용하지 않음 |
| B17 | [SingFake](https://singfake.org/) · [저자 저장소](https://github.com/yongyizang/SingFake) | real 28.93시간/fake 29.40시간 발표, 5개 언어·40명. 주 배포는 URL annotation | 연구 조건 안내와 개별 플랫폼 원음 권리 별개 | 현실 노래 커버 사례. URL 수를 지금 내려받을 수 있는 허가된 음원 수로 계산하지 않음 |
| B18 | [WildSVDD](https://zenodo.org/records/10893604) | train/test CSV 총 657.6kB | 원음 권리·접근은 각 플랫폼에서 별도 확인 | **오디오 archive가 아님.** SingFake 계보 확대판 |
| B19 | [MIREX 2025 Song Deepfake Detection](https://music-ir.org/mirex/wiki/2025%3ASong_Deepfake_Detection) | WildSVDD·SONICS 연결 | 원 데이터 조건 적용 | 새로운 독립 음원 집합으로 합산하지 않음 |
| B20 | [ATDD](https://zenodo.org/records/14621005) · [원 논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC12856623/) | 원 segment 21,249개와 auto-tune counterpart, EN/ZH/JA, ZIP 10.9GB | 음원 라이선스 값 미확인 | **보정된 사람 노래**의 오탐 후보. 모든 auto-tune을 생성 fake로 분류하지 않음. VocalSet/MUSDB/M4Singer/Opencpop 등 광범위 중복 |
| B21 | [Descriptive Singing Voice Manipulation](https://zenodo.org/records/20210548) | 2026 부분 편집 설명·위치 annotation. Test/Training JSON만 총 222.9MB | full audio는 추후 공개 설명, 현재 확인 안 됨 | 논문이 나왔어도 **오디오 공개 완료로 판단하지 않음** |
| B22 | [SVCC2025](https://huggingface.co/datasets/lestervioleta/svcc2025) | 10.9GB, GTSinger subset·7 styles | CC BY-NC-SA 4.0 | 원본/GT release를 합성 conversion output으로 오인하지 않음. B01과 순증 중복 |
| B23 | [SingStyle111](https://shuqid.net/singstyle111) · [Zenodo](https://zenodo.org/records/10265401) | 111곡/224version/8명/12.8시간 발표, EN/ZH/IT | Zenodo는 논문 PDF만. 저자 사이트의 현재 오디오 접근·license 확인 실패 | 실제 창법 다양성 후보지만 파일·조건 보류. 논문의 CC BY를 음원에 전용하지 않음 |
| B24 | [SingNet-3k](https://singnet-dataset.github.io/) · [논문](https://arxiv.org/abs/2505.09325) | 인터넷 음악·sample pack 기반 3,000시간 발표 | 모델·논문 공개와 full raw audio/license 공개는 별개. 후자 미확인 | 현재 다운로드 가능한 3,000시간으로 계산하지 않음 |
| B25 | [Singing Head DeepFake](https://liuke3068likwix.github.io/SingingHead-DeepFake/) · [논문](https://arxiv.org/abs/2605.27944) | 2026 audiovisual singing-head benchmark | 전체 음원 archive·audio label·조건 미확인 | fake 영상이 실제 노래를 유지할 수 있음. 비디오 라벨을 오디오에 전용 금지 |
| B26 | [SingFox](https://github.com/Arth-Shah/SingFox) · [논문](https://arxiv.org/abs/2606.18985) | 113,802 clip/20언어/1,150가수 발표. 확인된 저장소는 생성 resources·예시 OGG 중심 | full licensed audio release 미확인 | 생성기·언어 다양성 연구 단서. 전체 학습 음원 확보 가능하다고 주장하지 않음 |
| B27 | [CSD](https://zenodo.org/records/4916302) · [VocalSet](https://zenodo.org/records/1193957) | 현재 bank CSD 200/VocalSet 800. 이미 수집한 real 가창 | CSD CC BY-NC-SA 4.0, VocalSet CC BY 4.0 | 같은 가수의 추가 crop·새 annotation을 새 실제 출처로 세지 않음 |

### 5.1 한국어 가창에서 달라진 판단

GTSinger는 한국어 실제 가창을 포함하므로 이전의 중국어·일본어 SingMOS-Pro 중심 제안에 추가할 가치가 있다. 다만 세 가창자만으로 실제 한국어 가요 전체의 다양성이 충분해졌다고 판단하지 않는다. 창법별 같은 가수·같은 노래의 여러 버전은 독립 음원으로 분할하면 안 된다.

GTSinger의 GitHub TODO는 한국어·일본어 정제 작업이 남았다고 표시하지만 HF 최근 commit에는 해당 언어 업데이트가 기록돼 있다. 어느 한 문구만 보고 최신 라벨 품질을 확정하지 않고, 실제 선택한 revision의 파일·메타데이터를 대조해야 한다.

별도 검색한 **“OpenCpop200”**, **독립 SVDD2026 음원 release**, **새 가수 녹음을 더한 VocalSet 확장판**은 이 조사에서 원저자 배포를 확인하지 못했다. 미발견을 존재하지 않음으로 단정하지 않는다.

## 6. 실제·합성 음악과 stem

| ID | 데이터와 원 출처 | 내용·규모·배포 상태 | 음원 이용 조건 | 보강 가치와 제한 |
| --- | --- | --- | --- | --- |
| C01 | [Music Arena](https://huggingface.co/datasets/music-arena/music-arena-dataset/blob/main/README.md) | 실제 생성 음원·battle metadata. 2026-03~07 공개 항목 1,786/1,269/859/1,514/1,959. sao/sao-small은 audio 제외 | CC BY 4.0 | **기존 928개와 ID 차집합 우선.** 월별 수는 순증 unique 곡 수가 아님. 이미 읽은 July revision을 새 release라고 재계산 금지 |
| C02 | [Echoes](https://huggingface.co/datasets/Octavian97/Echoes) | 현 카드 4,468곡·약131시간·8.6GB, 실제 MP3·CSV, TTA/ATA | CC BY-SA 4.0 | 기존 provider×TTA/ATA에서 빠진 부분만 증량. 참조 원곡 계보 유지. 과거 3,577곡/110시간 논문 수치와 버전 구분 |
| C03 | [HAIM](https://huggingface.co/datasets/mippia/HAIM) · [구성](https://github.com/Mippia/HAIM_dataset/blob/main/DATASET_DESCRIPTION.txt) | HF 카드: 오디오67,000+링크86,686. GitHub 설명: 오디오약67,000+metadata38,596. **집계 범위 차이는 아래 설명** | HF 카드 CC BY-NC 4.0과 상세 문서의 MIT/NC/NC-SA 하위집합 설명이 다름. 일괄 license 확정 불가 | 현재 폴더당150 cap. C1/C2는 MTG 원자료 조건과 SA 적용 범위 확인 전 추가 편입 보류. URL-only를 오디오로 세지 않음 |
| C04 | [SONICS](https://huggingface.co/datasets/awsaf49/sonics/tree/main) | 약49k Suno/Udio fake 음원, HF32.6GB. real은 YouTube ID/metadata | CC BY-NC 4.0, 연구·비상업 조건 | 기존 부분 수집 확대 후보. 광고된 real+fake 전체를 다운로드 가능한 음원 수로 쓰지 않음 |
| C05 | [FakeMusicCaps](https://zenodo.org/records/15063698) · [논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC12295441/) | 27,605×10초, 5생성기. 실제 ZIP12.9GB | 원 페이지에서 음원 license 값을 확인하지 못함 | 기존2,000개가 있어도 추가 편입의 권리 검토를 생략하지 않음. MusicCaps 원참조·prompt별 분할 |
| C06 | [MoM/CLAM](https://github.com/StarkVision-AI/MoM-CLAM) · [파일](https://huggingface.co/datasets/anonymous2212/MoM-CLAM-dataset/tree/main) | 파일·링크 혼재134GB. YuE/DiffRhythm/Riffusion/Udio 폴더, Suno 일부는 CSV 링크 | HF CC BY-NC 4.0, 원미디어 권리·공개 범위 확인 | **새 음악 생성기 후보.** 논문130k+와 viewer1,600을 각각 전체 usable audio 수로 해석 금지 |
| C07 | [M6](https://huggingface.co/datasets/yl7622/M6) · [논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC13000184/) | 실제 TAR30.2GB, viewer18,426행. 음악 장르·악기·언어 다양성 | dataset card/audio license 미확인. 논문 CC BY는 원음 grant가 아님 | 파일은 있어도 조건 확인 전 보류. 일부 상용 음악은 metadata만 |
| C08 | [MUSDB18-HQ](https://zenodo.org/records/3338373) | 150곡, train100/test50, vocal/drum/bass/other+mixture, ZIP22.7GB | 교육용 조건·원자료별 제한 | 실제 stem·통제 혼합 후보. 일반 MUSDB18과 HQ는 같은 곡. MedleyDB46곡 중복 |
| C09 | [MoisesDB](https://github.com/moises-ai/moises-db) · [공식 Download](https://music.ai/research/) | 240 multitrack곡, 12장르. 공식 Download 버튼 확인; HF는 코드·metadata | CC BY-NC-SA 4.0. 현행 다운로드 전달 방식·form gate 미확인 | 실제 보컬+AI 반주 구성 재료. 연구실 사이트 버튼을 실제 archive 획득으로 보고하지 않음. 가수45/47 표기는 논문·홍보 간 불일치 |
| C10 | [MedleyDB](https://medleydb.weebly.com/description.html) · [v1](https://zenodo.org/records/1649325) · [v2](https://zenodo.org/records/1715175) | 사람 연주 multitrack, 접근 요청 경로 | CC BY-NC-SA, 세부 접근 조건 | 질 좋은 실제 stem 후보지만 gate와 MUSDB 중복 확인. 즉시 확보를 전제로 일정 계획 금지 |
| C11 | [Medley-solos-DB](https://zenodo.org/records/3464194) | **21,571개 약3초**, 8악기 범주에 female singer 포함. TAR7.9GB 공개 | 음원 CC BY 4.0 명시 | **실제 악기·보컬 오탐 보강 우선.** 21k곡이 아님. 원곡 ID grouping, MedleyDB 계보 확인 |
| C12 | [FMA](https://github.com/mdeff/fma) | 106,574곡, small8,000×30초/7.2GiB, medium25k/22GiB, large106k/93GiB | 음원은 트랙별 CC. 코드 MIT·metadata CC BY와 별개 | 이미 FMA계보 존재. 허용 트랙만 선택, Echoes 참조·MusicBench/MoM 재사용과 중복 확인 |
| C13 | [MTG-Jamendo](https://github.com/MTG/mtg-jamendo-dataset) | 55k+곡/195tag, 다운로드 지침·audio_licenses.txt. instrument subset은 같은 데이터의 부분집합 | 트랙별 CC 및 연구·학술 조건. 기존 프로젝트의 TDM 조건 미해결 기록 존재 | HAIM real과 중복, blanket 승인 금지. 코드 Apache와 audio rights 구분 |
| C14 | [MusicCaps](https://huggingface.co/datasets/google/MusicCaps) | 5,521 caption·timestamp·YouTube-ID, 약2.95MB metadata | metadata CC BY-SA 4.0이 YouTube 원음 전체를 허가하지 않음 | 오디오 직접 배포가 아님. FakeMusicCaps·MusicBench 계보 식별에 유용 |
| C15 | [MusicBench](https://huggingface.co/datasets/amaai-lab/MusicBench) · [파일 revision](https://huggingface.co/datasets/amaai-lab/MusicBench/tree/cd4474aadc46ac6b1ceed7dd4d9c4ef48286062b) | 약52.8k training rows, MusicCaps 증강. audio16.8GB+FMA eval298MB | 카드 CC BY-SA 3.0, 원 YouTube/FMA 권리 검토 필요 | **52k 독립 사람 연주가 아님.** 증강을 생성 fake로 자동 라벨링 금지 |
| C16 | [Music4All](https://sites.google.com/view/contact4music4all) | 공식 설명109,269×30초/16,269artist/46가사 언어 | 현행 다운로드·음원 학습/전달 허락 미확인 | 규모는 크지만 조건부. 후속 metadata 확장 license를 원음에 전용하지 않음 |
| C17 | [Slakh2100](https://www.slakh.com/) | MIDI·virtual instrument 렌더2,100mixture, stem, 187patch/34class | CC BY 4.0 | 사람 녹음도 neural TTM도 아닌 경계 사례. **대회 정의에 맞춘 별도 라벨 정책** 필요. redux 및 MIDI 원곡 중복 확인 |
| C18 | [Instrumental-Dataset](https://github.com/evamooreshtg58-beep/Instrumental-Dataset) | README는 약7k/2GB archive 주장, 관찰한 tree는 README뿐. 7,045/7,050 수치 불일치 | 저자 CC BY-NC 주장, FMA 등 원자료 적용 불명 | **공개 완료 후보에서 제외.** 검증된 archive 링크 없이는 수집 계획에 넣지 않음 |
| C19 | [Internet Archive Music Dataset](https://adasp.telecom-paris.fr/rc/demos_companion-pages/internet_archive_music_dataset/) · [논문](https://arxiv.org/abs/2609.20870) | 2026-09-15 투고, 34,000시간 이상 발표. 공식 companion은 pipeline·tables 추후 공개 안내 | 원자료 CC 선별 설명과 실제 배포·트랙별 조건 확인 별개 | 현재 완성 배포를 확인하지 못함. 미래 공개 기대를 마감 전 사용 가능으로 계산하지 않음 |

HAIM의 수량 차이는 공개 설명의 집계 범위와 연결된다. HF 카드에는 **SONICS 실제 음악 링크 48,090개**가 추가되어 있으며 `38,596 + 48,090 = 86,686`이다. 두 값을 같은 버전·같은 범위의 수량으로 섞지 않는다. 실제 원음 수를 재집계한 결과는 아니다. [HF 구성표](https://huggingface.co/datasets/mippia/HAIM), [GitHub 설명](https://github.com/Mippia/HAIM_dataset/blob/main/DATASET_DESCRIPTION.txt)

권리 설명은 아직 단일 값으로 정리할 수 없다. HF 카드는 데이터 전체를 CC BY-NC 4.0으로 안내하지만, GitHub 상세 문서는 일부 ACE-Step 파생본에 MIT를, 요약표의 A1/C1/C2에는 CC BY-NC-SA 3.0을 표시한다. 상세 문서 내부에서도 묶음 설명과 요약표의 표기가 다르다. 적용 revision·하위집합·상위 MTG 조건을 확인해야 하며, 더 느슨한 표기만 골라 쓰지 않는다.

HAIM의 B1은 AI로 mastering한 사람 음악, B2/B3 등은 생성 음악의 후처리, C1/C2는 시간축에서 실제·AI가 섞인 사례다. “AI 도구를 거쳤다”를 하나의 fake 라벨로 묶지 말고 대회의 생성·합성 정의와 파일별 provenance로 판단한다. B6의 K-pop 가사 조건부 생성은 한국어 사람 가창을 확보했다는 근거가 아니다. [공식 하위집합 설명](https://github.com/Mippia/HAIM_dataset/blob/main/DATASET_DESCRIPTION.txt)

FoRMusic, FakeMusicNet, “DeeperForensics music”이라는 정확한 이름으로는 이번 검색에서 원저자의 신뢰할 수 있는 음원 배포를 확인하지 못했다. 이름만으로 항목·수량을 만들지 않았다.

## 7. 한국어와 AI-Hub

**AI-Hub 데이터의 대회 사용은 허용된다.** DACON.GM이 이 대회의 AI-Hub 문의에 2026-09-21 10:28 사용 가능하다고 답했다. 국내 신청·승인 절차가 있다는 이유로 이 답변을 무시하거나 AI-Hub를 전부 제외할 근거는 없다. [운영진 답변](https://dacon.io/competitions/official/236749/talkboard/417844)

AI-Hub 공통 정책은 영리·비영리 AI 연구개발 활용을 안내하지만, **국외 이용·국외 반출에 별도 합의**, 승인받지 않은 제3자에게 제공·열람시키는 행위에 제한을 둔다. FAQ도 학습 결과물 활용과 원데이터·단순 가공 데이터 재배포를 구분한다. 따라서 대회 사용 허용과 해외 H100 전송·검증기관 전달 절차를 따로 기록한다. 이미 계정에서 승인받았는지, GPU 위치가 어디인지, 어떤 제출 절차를 적용할지는 이번 조사에서 확인하지 않았다. [현행 AI-Hub 정책](https://aihub.or.kr/intrcn/guid/usagepolicy.do), [공식 FAQ](https://www.aihub.or.kr/aihubnews/faq/list.do)

| ID | 데이터와 원 출처 | 확인한 내용·규모 | 배포·권리 상태 | 현재 목적과의 관계 |
| --- | --- | --- | --- | --- |
| D01 | [Seoul Corpus / OpenSLR113](https://www.openslr.org/113/) | 한국어 자연발화, FLAC sound.tgz 약2.5GB와 TextGrid | 오디오 파일 직접 공개, CC BY-NC 2.0 | **새 한국어 실제 발화 우선 후보.** Zeroth/FLEURS 읽기음성과 다른 대화 특성. 원화자·세션 기준 분할 |
| D02 | [AI-Hub465 다화자 가창](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=465) | 18명, 실제 구축4,681개. 소개의4,500은 목표와 구분. 발라드/R&B·록/팝·트로트 | 국내 신청·승인, AI-Hub 정책 | 한국어 실제 노래 다양성에 직접 관련. 현재 S2에는 train256/dev55만 파일로 확보됨 |
| D03 | [AI-Hub473 다음색 가이드보컬](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=473) | 92가창자, 4,000곡, 157시간23분25.754초 발표. 음색·연령·성별 정보 | 국내 신청·승인, AI-Hub 정책 | 한국어 실제 가창의 큰 후보. 이번 조사에서 전체 파일 확보·학습 준비 완료를 확인하지 않음 |
| D04 | [AI-Hub71470 국악 악보 및 음원](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71470) | WAV/MIDI/JSON 각각11,069개, 총154,076초. 기존 녹음 수집·직접 제작 | AI-Hub 정책, 로컬 일부 확보 | 실제 국악·악기 도메인. MIDI 파일 수를 별도 음원으로 더하지 않음. 보컬 여부는 파일별 확인 |
| D05 | [AI-Hub71767](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71767) | WAV 원천120,000건, MIDI·JSON 각120,000건. 컨소시엄 대중음악 원자료 | AI-Hub 정책, 로컬 일부 확보 | 실제 악기·음악 구간 다양성 후보. crop·loop의 원곡 grouping. 36만 개 독립 음원으로 합산 금지 |
| D06 | [AI-Hub644](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=644) | 환경음 전체44,810 clip 중 악기3,928. 여러 환경·거리·수집 방식 | AI-Hub 정책, 로컬 악기 일부 확보 | 전체가 음악이 아님. none/none 환경음과 악기·음악 presence 라벨을 분리 |
| D07 | [AI-Hub71314 방송콘텐츠 대화체](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71314) | 구축 설명7,002.74시간, 방송 대화·배경음악·잡음 정보 | 공식 페이지는 유형/형식을 텍스트로도 표시. 실제 음원 archive 제공 범위는 추가 대조 필요 | 자연스러운 실제 말+음악 후보. **프로그램 수준 BGM 통계를 모든 clip의 MUSIC_PRESENT로 전용 금지** |
| D08 | [AI-Hub123 KsponSpeech](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=123) | 한국어 자발 대화. 공식 AI-Hub 배포 경로 확인 | 계정·승인·AI-Hub 조건. 코드·HF loader의 Apache/MIT는 원음 license가 아님 | 말더듬·간투사·대화 실제 음성 후보. 이미 로컬에 있는 부분과 원세션 중복 확인 |
| D09 | [AI-Hub542 다화자 음성합성용 데이터](https://aihub.or.kr/aihubdata/data/view.do?aihubDataSe=&currMenu=&dataSetSn=542&topMenu=) | TTS 모델 학습용 다화자 음성 자료 | 공식 검색 페이지 확인, 직접 접근 일부 실패. 실제 파일 구성·사용 승인 미확인 | “음성합성 데이터”라는 제목만으로 모든 음원이 fake라고 판단하지 않음. 실제 녹음·생성 출력 구분 |
| D10 | [AI-Hub71405 소음환경 명령어](https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=ty&currMenu=116&dataSetSn=71405&topMenu=100) | 다화자 소음2,905시간+차량소음1,116시간, 합계4,021시간 발표 | AI-Hub 승인·정책 적용 | real 채널 다양성 후보. 실제 클래스에만 잡음을 집중 추가하면 또 다른 shortcut이 생김 |
| D11 | [CoreaSpeech](https://coreaspeech.github.io/demo/) · [CoreaSpeech+](https://huggingface.co/datasets/Gong1212/CoreaSpeech-plus) | 700시간·168,790개·21,449명 발표, 기존 corpus에서 선별. 2000h 확장 구성도 설명 | HF gated, 학술·비상업, **원음 재배포 금지** 안내 | 한국어 도메인 다양성은 매력적이나 원자료 중복·대회 원음 전달 조건 미해결. “TTS용 corpus”와 fake 출력 구분 |
| D12 | [Zeroth](https://www.openslr.org/40/) · [FLEURS](https://huggingface.co/datasets/google/fleurs) | 현재 bank 각각3,000/1,500개 | 현재 수집 정의는 CC BY 4.0. 추가판의 원 배포 조건 재대조 | 이미 사용하는 한국어 실제 발화. 추가 clip보다 새로운 화자·원문·녹음 조건이 남는지 확인 |

한국어 합성 음성은 현재 MLAAD/SpeechFake를 먼저 생성기·버전·발화별로 정리하는 편이 현실적이다. DSD에 AI-Hub가 등장한다는 이유로 한국어 fake가 확인됐다고 보고할 수는 없고, SPADE의 한국어는 아직 공개 대기로 표시된다. 공개 한국어 **부분 위조 가창**을 바로 내려받을 수 있다는 근거도 이번 조사에서는 확보하지 못했다.

KSS, KEMDy 계열, 제주어 speech와 공공저작물 음악 경로도 탐색했으나 이번 범위에서 현재의 원음 배포·적용 조건을 충분히 닫지 못했다. 제3자 재배포본이나 검색 결과만으로 편입 후보의 규모·허락을 확정하지 않았으며, 위 표의 검증 수준과 구분한다.

## 8. 이미 확보한 데이터의 추가 활용

### 8.1 추가 bank의 준비 상태

다음은 이 대화에서 로컬 S2 CSV와 대응 파일을 확인한 결과다. 파일 존재는 전체 디코딩, 라벨 정확성, 최신 MERT/EAT feature 호환성을 증명하지 않는다.

| 데이터 | train / dev 행 | 대응 오디오 | feature cache train / dev | 판단 |
| --- | ---: | --- | ---: | --- |
| Music Arena | 773 / 155 | 모두 존재 | 0 / 155 | 새 다운로드 전에 split 수정과 필요한 feature 생성 범위를 정할 수 있음 |
| Jamendo | 851 / 149 | 모두 존재 | 112 / 149 | 기존 권리 검토는 TDM 조건 미해결로 추가 학습 보류 |
| AI-Hub71470 | 289 / 211 | 모두 존재 | 289 / 211 | 기술적으로 비교적 준비됐지만 정책·반출·검증 제출 경로 별도 |
| AI-Hub465 | 256 / 55 | 모두 존재 | 95 / 0 | 한국어 실제 가창 보강에 적합, 가수·원곡 grouping 확인 필요 |
| AI-Hub71767 | 244 / 136 | 모두 존재 | 0 / 0 | 실제 loop·악기 후보, 전체 구축 규모와 로컬 수량 구분 |
| AI-Hub644 | 150 / 50 | 모두 존재 | 0 / 0 | 악기 선택 부분과 전체 환경음 구분 |
| channel_fake / channel_real | 1,100 / 250 및 500 / 100 | 모두 존재 | 모두0 | 기존 MLAAD/Zeroth에 채널 처리. 새 생성기·새 실제 화자가 아님 |

근거: [S2-A train](../_workspace/s2_extra/bank_A_train.csv), [S2-A dev](../_workspace/s2_extra/bank_A_dev.csv), [S2-B train](../_workspace/s2_extra/bank_B_train.csv), [S2-B dev](../_workspace/s2_extra/bank_B_dev.csv), [권리 기록](../_workspace/lb_admission_20260924/rights_decisions.json).

### 8.2 데이터 추가보다 먼저 해결할 분할 문제

Music Arena의 **battle component 55개가 train/dev에 동시에 존재하며 관련 행은110개**다. [코드](../data_v4/build_extra.py)는 Sonauto 계열을 dev에 두면서 battle ID는 group에 기록한다. 생성기 분리와 원비교 묶음 분리가 동시에 충족되지 않는 구조다. 원음 hash 중복을 뜻하는 것은 아니지만, 이 dev를 battle·내용 계보까지 독립한 평가라고 부르면 안 된다.

가수·원곡·원발화·배틀·변형의 연결 관계를 먼저 통합한 뒤, 연결된 원자료가 양쪽에 갈라지지 않도록 분할해야 한다. group 이름에 서로 다른 dataset prefix를 붙이는 것만으로는 원음 중복이 해결되지 않는다.

이전 Music Arena 소규모 pilot의 S4 유사 설정 ADS 차이는 seed별 +0.0014/+0.0002/−0.0003, 평균 +0.0004로 기록되어 있다. 이는 큰 개선의 증거가 아니며 새 생성기 전체가 무가치하다는 증거도 아니다. **당시 pilot과 이번 S2 split은 다른 구성**이므로, 위 교차 group 발견만으로 이전 pilot에 누수가 있었다고 단정하지 않는다. [실험 기록](../research/top15/work2/diversity/critique.md)

## 9. 편입·분할·평가 계획

### 9.1 먼저 만들 편입 기록

아래는 실행 제안이며 이번 조사에서 구현하거나 실행한 결과가 아니다.

데이터별로 원 배포 URL·revision·접근일, 음원 라이선스와 원자료 조건, 실제 다운로드 목록, 원본 hash, decode 상태, 라벨 근거, train/dev/locked-eval 역할을 기록한다. clip마다 최소한 다음 정보가 필요하다.

- 원 dataset·원 파일 ID·원곡/원발화 ID·가수/화자 ID
- 생성기 이름·모델 버전·설정·생성/변환/재구성/후처리 구분
- 언어·길이·sample rate·codec·보컬/음악 presence
- voice/music 진위와 각각의 mask, 근거가 없는 항목은 unknown
- 부모 파일 목록, segment 시간, 편집·믹싱·분리·재인코딩 이력
- 권리 검토 상태와 원음 제출 가능 여부

목록에 있다는 사실, 다운로드했다는 사실, 기술 검사를 통과했다는 사실, 학습에 편입됐다는 사실을 각각 분리해 보고한다. 이 문서의 후보 표는 다운로드·편입 완료 목록이 아니다.

### 9.2 순증 다양성을 판단하는 단위

“10만 개를 더 얻었다”보다 다음의 증가분을 본다.

| 단위 | 확인할 질문 |
| --- | --- |
| 원본 계보 | 이미 가진 LibriTTS/VCTK/Common Voice/FMA/MTG 곡의 또 다른 변형인가? |
| 화자·가수 | 실제로 다른 사람인가, 변환된 target voice ID만 늘어난 것인가? |
| 생성기·버전 | 같은 모델의 wrapper·vocoder·sampling 설정을 다른 생성기로 세고 있지 않은가? |
| 악곡·가사·prompt | 같은 곡·가사를 여러 모델로 만든 경우 모두 같은 분할에 묶였는가? |
| 도메인 | 대화·읽기·가창·방송·통화·악기·배경음악 조건에서 무엇이 추가되는가? |
| 클래스 조건 | codec·길이·음량·채널이 진짜/가짜 라벨과 우연히 일치하지 않는가? |
| 평가 독립성 | 새 source를 train에도 넣고 새 OOD에서도 평가해 독립성을 과장하지 않는가? |

특히 아래 계보를 우선 연결한다.

- VCTK → DFADD, 여러 CodecFake, STOPA, PartialEdit 및 일부 ASVspoof 관련 자료.
- LibriTTS/LibriSpeech → LlamaPartialSpoof, LibriSeVoc, CD-ADD, SONAR, DiffSSD 일부.
- Common Voice → SpeechFake real, CVoiceFake, MultiAPI real, AUDETER 일부.
- Opencpop/M4Singer/KiSing/JVS/Kiritan → CtrSVDD, ACE 계열, AT-ADD, ATDD.
- GTSinger → SVCC2025.
- MusicCaps → FakeMusicCaps, MusicBench.
- FMA/MTG-Jamendo/SONICS → HAIM, Echoes 참조 자료, MoM 등에서 실제 파일 단위 교차 대조.
- MedleyDB → MUSDB 일부, Medley-solos 일부.

화자나 원곡 ID가 숨겨진 데이터는 독립성이 확인되지 않은 상태로 남긴다. hash가 다른 재인코딩·crop·stem도 동일 원본일 수 있으므로 waveform fingerprint와 메타데이터 계보가 함께 필요하다.

### 9.3 네 혼합 조합과 부분 위조

현재 적은 **실제 보컬+AI 반주**를 구성할 때, 다른 세 조합도 같은 전처리·음량·codec 조건으로 만든다. 특정 클래스만 Demucs를 거치거나, fake 반주에만 다른 sample rate·cut 길이를 쓰지 않는다.

| 보컬 | 반주 | 확인할 목적 |
| --- | --- | --- |
| 실제 | 실제 | 사람 녹음·믹싱을 fake로 오탐하지 않는가 |
| 실제 | AI | 부족한 조합의 음악 진위를 탐지하는가 |
| AI | 실제 | 실제 반주 때문에 합성 보컬을 놓치지 않는가 |
| AI | AI | 두 성분이 모두 합성일 때 안정적으로 동작하는가 |

원본을 먼저 분할한 뒤 분할 안에서만 혼합하고, 모든 부모가 같은 역할에 속하도록 제한한다. 대조군도 같은 믹싱·분리 경로를 거쳐야 한다. 만들어진 합성 혼합물이 실제 평가 도메인을 대표한다는 주장은 별도의 자연 혼합 평가로 검증한다.

부분 위조는 fake 길이·비율·위치·경계 방식별 성능을 나누어 본다. crossfade 경계만 외웠는지 확인하고, “파일에 fake가 조금이라도 있음”이라는 file label과 구간 label을 구분한다.

### 9.4 실제로 의사결정을 바꾸는 비교

| 비교 | 고정할 것 | 볼 결과 | 채택·중단 판단 |
| --- | --- | --- | --- |
| 실제 가창 추가 | 동일 인코더·head·분할·학습량 통제 | 한국어/다국어 real singing 오탐, 독립 source 전체 지표 | 특정 가수에서만 개선되거나 fake singing 탐지가 악화하면 확대 보류 |
| 생성기 추가 | 같은 real pool·전처리·sampling 정책 | unseen-generator 성능, worst-source 성능 | 추가한 생성기 자체에서만 개선되면 일반화 증거로 보지 않음 |
| 혼합·부분 위조 추가 | 네 조합의 처리 조건·부모 분할 | real voice+fake music, 짧은 fake segment 탐지, real 후처리 오탐 | 파일 길이·경계·codec shortcut이 보이면 먼저 데이터 구성 수정 |
| 고정 인코더 대 미세조정 | 같은 원자료·분할·평가 protocol | source-disjoint 성능과 seed 변동 | 같은 source val 개선만으로 full fine-tuning 선택 금지 |
| batch 확대 | 데이터·평가, 유효 batch와 optimizer step 정의 | 동일 sample exposure 또는 동일 compute에서 일반화·속도 | 처리량 상승과 정확도 상승을 구분. learning rate·warmup·step 변화도 기록 |

batch 크기를 키우는 것만으로 부족한 출처 다양성이 해결되지는 않는다. H100 여유는 여러 데이터 가설의 feature 추출과 통제 비교에 쓸 수 있지만, 이번 요청에서는 GPU 실행·추가 유료 자원 투입을 하지 않았다.

평가에는 대회 공식 산식에 맞춘 파일 수준 지표와 voice/music·source별 진단을 함께 사용한다. EER과 ADS 또는 LB 점수를 같은 척도처럼 빼거나, 과거 LB와의 차이를 모든 모델에 일정한 보정값으로 적용하지 않는다. 비공개 평가 데이터로 학습·튜닝·pseudo-labeling하지 않는다. [공식 규칙](https://dacon.io/competitions/official/236749/overview/rules)

## 10. 보류 사유와 정정 사항

| 오해하기 쉬운 주장 | 이번 조사에서의 정리 |
| --- | --- |
| 공개 논문·GitHub가 있으면 데이터도 배포됐다 | WildSVDD는 CSV, Descriptive Singing Voice Manipulation은 JSON, 일부 SingNet/SingFox 자료는 모델·코드·예시 중심이다. |
| HF에 audio라고 표시되면 전체 음원이 있다 | MoisesDB의 공식 HF 저장소는 코드·metadata다. 일부 viewer는 특정 config나 첫 부분만 인덱싱한다. |
| 논문·코드의 MIT/CC license가 음원을 덮는다 | M6, STOPA, JVS-MuSiC, MusicCaps 등의 오디오와 논문·코드·태그 조건을 구분했다. |
| ND가 붙으면 모든 ML 학습이 자동 금지다 | 그렇게 단정하지 않았다. 목적·변형·재배포·상위 source·대회 제출 조건을 확인할 때까지 해당 편입을 보류한다. |
| AI-Hub는 gate와 제3자 제공 제한 때문에 대회 사용 불가다 | **운영진이 사용 가능하다고 답했다.** 해외 반출·검증 전달 절차의 미확인은 대회 사용 불허와 구분한다. |
| AI-Hub가 학습에 허용되면 해외 GPU 업로드도 허용된다 | 현행 정책은 국외 반출과 제3자 제공에 별도 제한을 둔다. 대회 허용 답변에 구체적 해외 전송·제출 절차까지 적힌 것은 아니다. |
| NC·SA가 붙으면 상금 대회에서 무조건 실격이다 | 이 대회 운영진은 NC 모델과 NC-SA 데이터 사용을 허용했다. 개별 제한은 계속 준수해야 한다. |
| validation 파일은 어떤 경우든 제출 대상이 아니다 | 순수 검증과 모델·설정 선택에 사용한 자료를 구분해야 한다는 운영진 답변이 있다. |
| “음성합성용” 또는 SVCC 자료는 모두 fake다 | 모델 학습용 사람 녹음·GT가 포함된다. ACE-KiSing은 반대로 실제 KiSing과 달리 synthetic이다. |
| auto-tune·mastering·codec 재구성은 모두 생성 fake다 | 대회 정의와 provenance에 따라 구분한다. 처리된 real의 오탐 사례로도 중요하다. |
| 새 데이터셋 이름이면 새 출처다 | VCTK/LibriTTS/Common Voice/Opencpop/MTG/FMA 파생 관계가 광범위하다. |
| GTSinger는 한국어가 없다 | 실제로 한국어 가창자 ID와 Korean 폴더가 있다. 전체80.59시간이 한국어라는 뜻은 아니다. |
| CodecFake+는 sample만 있다 | CoRS·CoSG archive가 현재 공개돼 있다. 다만 음원 MIT 적용 범위는 아직 명확하지 않다. |
| M4Singer는 지금도 반드시 이메일 신청이다 | 현재 공식 README에는 Drive 링크가 있다. 오래된 논문의 접근 설명과 구분했다. |
| Echoes는 MIT·3,577개다 | 현재 카드 음원 조건은 CC BY-SA 4.0,4,468곡·131시간이다. 과거 논문 수치와 버전이 다르다. |
| HAIM·SONICS의 전체 발표 규모를 모두 내려받을 수 있다 | 실제 오디오와 외부 플랫폼 URL 부분이 구분되어 있다. |
| HAIM의 86,686과 38,596은 같은 범위의 링크 수다 | HF에는 SONICS real 링크48,090개가 더 있다. license도 HF의 단일 표기와 GitHub의 하위집합별 설명을 구분했다. |
| SCDF의 CC0는 VoxPopuli의 NC와 충돌한다 | VoxPopuli 데이터는 CC0, 모델·코드는 NC다. 데이터 license 충돌이 확인된 것은 아니며 원미디어 조건 검토는 별개다. |
| AI-Hub465는 4,500개다 | 소개의 목표와 달리 실적 표는4,681개다. |
| 원음 license를 못 읽었으면 금지다 | 금지가 확인된 것이 아니라 이용 허락 확인이 미완료다. 이를 사실과 구분했다. |

추가 확인이 필요한 핵심 항목은 GTSinger/M4Singer 추가 약정 적용 범위와 한국어 subset 실측, HAIM 하위집합별 적용 license, MultiAPI 실제 파일 목록·API mapping, CodecFake+ 음원 조건, SCDF 원미디어 조건, M6/FakeMusicCaps/PartialEdit/PhonemeDF/ATDD/NUS-48E 등의 정확한 음원 grant, MoisesDB 현재 전달 절차, AI-Hub 자료의 해외 처리·대회 검증 전달 경로다.

NEXDATA 구매는 위 후보로도 채워지지 않는 구체적 빈틈이 남을 때 비교할 수 있다. 그때 필요한 것은 단순 총량이 아니라 생성기·언어·가수/화자·원본 계보·real/fake 짝·부분 위조 라벨·제출 가능한 이용 조건이다. 이번 문서에서는 NEXDATA의 현행 상품·견적·납기·계약을 새로 검증하지 않았으므로 구매 가능성이나 효과를 확정하지 않는다.

## 11. 조사 방법과 한계

음성 위조, 가창, 음악·stem을 세 연구 작업으로 나누고 한국어·AI-Hub·로컬 source 검토를 병행했다. 각 분야에서 초기 탐색 뒤 파일 배포·원본 권리·파생 계보를 다시 확인했으며, 후속 탐색에서는 최신 release와 오래된 설명의 충돌을 점검했다. 검색 결과 요약보다 원저자 저장소·배포 카드·Zenodo 파일 목록·원 논문·기관 정책을 우선했다. 각 행의 링크가 해당 주장의 근거이며, 확인하지 못한 항목은 그대로 표시했다.

이 목록은 조사일에 확인한 범위의 폭넓은 검토다. 전 세계 공개 자료가 완전히 망라됐다는 주장도, 모든 약관을 법률적으로 확정했다는 주장도 아니다. 계정 뒤의 파일, 일부 Drive·Zenodo API, 동적으로 표시되는 license는 도구로 확인하지 못한 곳이 있다. 접근 실패를 미배포로 바꾸지 않았다.

이번에 수행한 로컬 계산은 기존 CSV의 행·분할·라벨 구성 재집계다. 신규 음원의 다운로드·checksum·디코딩·청취·라벨 표본 감사·원본 중복 fingerprint·학습·GPU 성능·리더보드 개선은 수행하지 않았다. 보고서의 추천 순서와 실험 설계는 판단·제안이며 측정 결과와 구분된다.

독립 corpus 수는 의도적으로 합산하지 않았다. 표에는 같은 계보의 버전·subtrack·재배포 관계와 기존 사용 source도 포함되어 있다. 자료를 실제로 수집할 때는 각 URL의 revision과 license 원문을 저장하고 본 문서의 상태를 갱신해야 한다.

### 로컬 근거 파일

- [주 bank](../results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank/bank_trainval.csv)
- [원본 test bank](../results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank/bank_test.csv)
- [수집 catalog](../data_v4/sources.py)
- [추가 bank 작성 코드](../data_v4/build_extra.py)
- [2026-09-24 권리 판단](../_workspace/lb_admission_20260924/rights_decisions.json)
- [AI-Hub 관련 기존 조사](../research/top15/work/r4/notes.md)
- [Music Arena pilot](../research/top15/work2/diversity/critique.md)

문서 변경 범위는 이 Markdown 보고서다. 조사 이전의 미커밋 코드·데이터·결과물은 수정하지 않았다.
