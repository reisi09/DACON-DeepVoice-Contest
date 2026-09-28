# 데이터셋 스카우트 R1: DACON 236749 학습 데이터 공백 보강 후보

작성: 2026-09-28 (KST) · 마감: 2026-09-29 10:00 KST · 출발점: `PUBLIC_DATASET_SURVEY_20260927.md` (이번에 근거로 쓴 항목은 배포처를 다시 열어 확인함)

대상 공백 (브리핑 기준, 검증된 상태)
1. 완성곡 생성 외의 가짜 노래 없음: SVS/SVC 가짜 가창, RVC식 AI 커버, CtrSVDD/WildSVDD/SingFake 없음
2. 한국어 가창: 반주 있는 실제 한국어 곡 0, 한국어 무반주 실제 100행(가수 1명), 한국어 가짜 노래 222행
3. 실제 보컬 + AI 반주: train 5행 (bank 전체 117행 중 held-out 107행)
4. 상업 수준으로 제작된 실제 음악 없음
5. 실제 음악·가창의 아티스트/가수 편중

## 표기와 작업 범위

- **[V] VERIFIED**: 이번에 원 배포처 페이지, Zenodo/HF/GitHub API, 논문 HTML, AI-Hub 상세 페이지를 직접 열어 확인했다.
- **[I] INFERRED**: 검색 결과 요약, 이전 조사, 파일 크기 환산, 일반 지식에서 추정했다.
- **[U] UNKNOWN**: 확인하지 못했다. 미발견이 곧 부재라는 뜻은 아니다.
- 하지 않은 일: 데이터 다운로드, 약관 동의, 계정 생성·로그인, 토큰과 `~/.aihub_key` 열람. 로컬에서는 `data/bank_*.csv` 메타데이터만 다시 집계했다.

"첫 투입" 시간은 다운로드 → 라벨 매핑·계보 group 지정 → 기존 파이프라인(HTDemucs stem, DF-Arena·AntiDeepfake·WavLM·MERT·EAT·SSLAM·PANNs feature) 추출까지를 뜻한다. P1 재학습(~15분)과 패키징·검증(~1시간)은 빠져 있다. 가정은 모두 [I]다: HF는 100 MB/s 이상, Zenodo는 10–40 MB/s, Google Drive는 할당량 오류가 날 수 있음. feature 비용은 브리핑의 "새 encoder 하나를 42k행에 15–20분(H100×4)"을 선형으로 환산해 1천 행(10–30초 길이)당 5분 안팎으로 잡았다. 긴 파일은 그만큼 늘어난다.

## 0. 결론 요약

1. 로그인 없이 바로 받을 수 있고 공백을 가장 직접 메우는 공개 데이터는 **AI-OpenBMAT, MUSDB18-HQ(+musdb-XL), CtrSVDD, GTSinger 한국어 subset** 네 가지다 [V]. 모두 비상업 이용을 허용하고 [V] 각각 1–4시간 안에 첫 투입이 가능하다고 본다 [I].
2. "실제 보컬 + AI 반주"(공백 3)를 이미 만들어 둔 공개 데이터는 **AI-OpenBMAT 하나**만 찾았다. 사람/AI 1,647쌍이 같은 음성 배치를 공유하고 음악만 다르다 [V]. 단 보컬이 노래가 아니라 TV 음성이다. 노래 버전은 **ACE-Step 1.5의 Vocal2BGM**(보컬 트랙에 반주 자동 생성, 코드·가중치 MIT)으로 직접 만드는 편이 빠르다 [V].
3. SVS/SVC 가짜 가창(공백 1)은 **CtrSVDD가 1순위**다. 14개 시스템, fake 260.34시간 [V]. 전부 중국어·일본어 무반주라 언어 shortcut 위험이 있고, bona fide 원천에 "official ACE-Studio release"가 들어 있어 라벨을 걸러야 한다 [V].
4. 공개된 한국어 가짜 가창 데이터는 사실상 없다. WildSVDD에 한국어 가수가 많지만 URL뿐이다 [V]. 한국어 SVC 가짜는 GTSinger 한국어·CSD에 zero-shot SVC(Seed-VC GPL-3.0, SoulX-Singer-SVC Apache-2.0)를 적용해 직접 만들어야 한다.
5. **반주가 있는 한국어 실제 곡(공백 2의 핵심)은 AI-Hub 71544 "음악 유사성 판별 데이터"가 거의 유일한 대량 후보**다. 원곡 10,000곡, 44.1 kHz 스테레오, 가사와 "Vocal" 악기 정보가 있다 [V]. AI-Hub 승인과 국내 처리 조건이 걸려 있어 마감 전 투입 가능성은 [U]다. AI-Hub 밖에서는 공유마당 대중음악 103건 정도가 전부였다 [V].
6. 상업 수준 실제 음악(공백 4)은 **MUSDB18-HQ 150곡(전문 믹스, stem 포함), musdb-XL(같은 곡에 상용 limiter Ozone 9 Maximizer 적용), IRMAS(상용 녹음 발췌)** 조합으로 바로 시험할 수 있다 [V]. S8 퇴행의 가설(마스터링된 상업 음악에서 오탐)을 직접 검증하는 조합이다. HAIM B3(Ozone 9 마스터링 AI 곡)·B4(전문 믹스 AI 곡)를 fake 쪽 대조군으로 함께 넣으면 "마스터링 = real" 지름길도 막을 수 있다 [I].
7. 이전 조사 이후 바뀐 점: HAIM이 2026-07-29 갱신됐고 B3·B4·B6(K-pop 가사 ACE-Step 1.5, 한국어·영어판)·B7–B9는 bank에 없다 [V]. AT-ADD Track 2는 자동 승인 gated이며 eval 메타데이터까지 공개됐다 [V]. SingFox는 음원을 공개하지 않았고 [V] TVSM은 오디오 없이 feature만 있다 [V].
8. 학습에 쓰지 말아야 할 것: WildSVDD·SingFake·HAIM B5·SONICS real·KVT(모두 YouTube 링크만 있음), HF의 비공식 K-pop 가수 음색 모델, JVS-MuSiC 원음(재배포 제한). 2차 평가 때 학습 파일을 적법하게 낼 수 없거나 권리 문제가 있다.

## 1. 종합 순위 (공백 기여 × 첫 투입까지 시간)

| 순위 | 후보 | 메우는 공백 | 접근 요건 | 규정 적합성 (공개·최소 비상업·2차 파일 제출) | 첫 투입 | 핵심 주의 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | AI-OpenBMAT | 3, 4 | 공개 Zenodo [V] | 높음. CC BY 4.0 [V]. 상위 Epidemic Sound 원음 조건은 [U] | 1–2시간 | 보컬은 유럽 TV 음성, 한국어 0 [I] |
| 2 | MUSDB18-HQ + musdb-XL | 4 (+1·3 구성 재료) | 공개 Zenodo [V] | 높음. 교육·비상업 조건 [V]. XL은 CC BY 4.0 gain 메타데이터 [V] | 2–3시간 (+XL 1시간) | 150곡뿐, 영어 위주 [I] |
| 3 | CtrSVDD | 1 | 공개 Zenodo [V]. 일본어 bona fide 4종은 별도 확보(로그인·약관) [V] | 높음. CC BY-NC-ND 4.0 원본을 그대로 제출 [V/I] | 2–4시간 | zh/ja 무반주만. ACE-Studio 계열 라벨 모호 |
| 4 | GTSinger 한국어 subset | 2, 5 (+1·3 한국어 입력) | 공개 HF, gate 없음 [V] | 높음. CC BY-NC-SA 4.0 + 추가 약정 문구 [V] | 약 1시간 | 가수 3명, 가창 8.28시간 [V] |
| 5 | 자체 생성 A: 한국어 SVC 가짜 가창과 AI 커버형 혼합 | 1, 2 | 공개 도구 (Seed-VC, SoulX-Singer-SVC, RVC) [V] | 생성 파일과 재현 정보 제출 필요 [V, 운영진 답변] | 3–6시간 | 입력·음색 prompt는 권리가 확인된 가수만 |
| 6 | 자체 생성 B: ACE-Step 1.5 Vocal2BGM | 3 (노래 버전) | 공개 도구, 가중치 MIT·gate 없음 [V] | 5와 같음 | 3–5시간 | 출력 속 보컬은 재합성본이라 반주만 떼어 원 보컬과 다시 섞을 것 |
| 7 | IRMAS | 4, 5 | 공개 Zenodo [V] | 높음. CC BY-NC-SA [V] | 1–2시간 | train 3초 클립은 학습 창(4.04초)보다 짧음 [I] |
| 8 | HAIM 미편입 subset (B3·B4·B6, 필요 시 B7–B9) | 4 (fake 대조), 2 (한국어 가짜 노래) | 공개 HF [V] | 높음. MIT / CC BY-NC 4.0 [V] | 1–2시간 | 전부 fake. ACE-Step 계열이라 생성기 다양성은 작음 |
| 9 | AI-Hub 71544 (+465·473·71767) | 2, 4, 5 | AI-Hub 신청·승인(내국인만) [V] + 국내 처리 | 대회 사용 허용 답변 있음 [V, 브리핑]. 해외 반출·제3자 제공 제한 | [U] 국내 연산 가능 여부에 달림 | EU GPU·클라우드 LLM 전송 금지 |
| 10 | AT-ADD Track 2 | 1 (SVC 가창), 음악 생성기 다양성 | HF 로그인 + 양식 + 약정 동의 → 자동 승인 [V] | 조건부. 비상업 학술·교육, 재배포는 약정 조건부 [V]. DACON 제출 해석은 [U] | 3–5시간 | 가창 원천이 CtrSVDD·ACE와 겹침 |

보조 후보 (11–20)

| 순위 | 후보 | 공백 | 접근 | 조건 | 첫 투입 | 주의 |
| --- | --- | --- | --- | --- | --- | --- |
| 11 | ACE-KiSing / ACE-Opencpop | 1 (상용 SVS) | 공개 HF [V] | CC BY-NC 4.0 [V] | 1–3시간 | 중국어(+영어). CtrSVDD A14와 같은 ACE 계열 |
| 12 | OpenSinger / M4Singer / Opencpop (실제 가창) | 5, fake의 real 짝 | Google Drive / Google Drive / Google Form→메일 [V] | NC-SA / NC-SA+약정 / NC-ND | 2–4시간 | Drive 할당량 |
| 13 | MoisesDB | 4, 5, 1·3 재료 | 연구 페이지 폼 [I] | CC BY-NC-SA 4.0 [V] | 2–4시간 (폼이 즉시 링크를 주는 경우) | 240곡·47명 |
| 14 | MTG-Jamendo / FMA-large | 5 | 공개 스크립트 / 공개 [V] | 비상업 연구, 트랙별 CC [V] | 2–4시간 (부분) | 인디 비중이 큼 |
| 15 | SVCC 2025 시스템 출력 | 1 (가창 스타일 변환) | Google Drive 링크 [V] | GTSinger 조건 [V] | [U] | 소량, 언어 [U] |
| 16 | SingMOS-Pro | 1 | 공개 HF [V] | CC BY 4.0 [V] | 약 1시간 | SingMOS 추가 실험이 +0.0002에 그침 [V, 브리핑] |
| 17 | 공유마당 대중음악 | 2, 4 (소량) | 회원 로그인 [I] | CC BY·기증저작물 자유이용 [V] | 수작업 | 103건 중 보컬곡 일부 |
| 18 | AI-Hub 71314 방송 대화체 | 4 (한국어 음성+실제 BGM) | AI-Hub | 국내 처리 | [U] | 오디오 제공 범위 [U] |
| 19 | Song Describer / SD-RR / MedleyVox | 5 | 공개 Zenodo [V] | CC BY-SA / CC BY [V] | 약 1시간 | 소량 |
| 20 | Deezer hybrid 코드 | 3 (구성 방법 참고) | GitHub [V] | 코드 | - | 코덱 재합성 stem의 라벨 해석이 애매 |

순위를 매긴 논리: 공백 3·4는 LB 퇴행 가설(S8)과 직접 이어지고 완제품 데이터가 있어 앞에 두었다. 공백 1은 CtrSVDD로 규모를 채울 수 있지만, 무반주 fake만 더하는 효과는 SingMOS 선례로 보아 작을 수 있다 [I]. 그래서 AI 커버형 혼합(5)을 붙여야 가치가 커진다. AI-Hub는 공백 2 기여가 가장 크지만 처리 장소 조건 때문에 시간 항목이 [U]라 9위에 두었다.

## 2. 공백별 정리

| 공백 | 바로 쓸 공개 데이터 | 자체 생성으로 메울 부분 | 남는 문제 |
| --- | --- | --- | --- |
| 1 가짜 가창(SVS/SVC/AI 커버) | CtrSVDD, AT-ADD T2 가창, ACE-KiSing/Opencpop, SVCC 2025 출력, SingMOS-Pro | Seed-VC/SoulX-SVC/RVC로 한국어·영어 무반주 변환, MUSDB 반주와 재혼합한 AI 커버형 | 공개 fake는 전부 zh/ja 무반주. 실제 AI 커버(WildSVDD)는 링크뿐 |
| 2 한국어 가창 | GTSinger 한국어(실제 무반주 3명), HAIM B6(한국어 가짜 완성곡) | 한국어 SVC fake, 한국어 보컬 + Vocal2BGM | 반주 있는 한국어 실제 곡은 AI-Hub 71544 말고는 공유마당 소량뿐 |
| 3 실제 보컬 + AI 반주 | AI-OpenBMAT(음성 버전) | ACE-Step 1.5 Vocal2BGM(노래 버전) | 한국어 음성 + AI BGM은 bank의 Zeroth/FLEURS에 AI 반주를 섞어 구성 가능 [I] |
| 4 상업 수준 실제 음악 | MUSDB18-HQ, musdb-XL, IRMAS, AI-OpenBMAT 사람판(제작음악) | - | MedleyDB·BAF·Music4All은 신청 승인이 필요해 마감 전 어려움 |
| 5 다양성 | MTG-Jamendo, FMA-large, OpenSinger(66명), M4Singer(20명), MoisesDB(47명) | - | 아티스트 단위 추출과 계보 group이 선행돼야 함 |

## 3. 후보 상세

### 3.1 AI-OpenBMAT (1위)

- URL: https://zenodo.org/records/20031232 · 소개 https://daveloay.github.io/AI-OpenBMAT/ [V]
- 내용 [V]: 방송 모니터링용 60초 발췌 3,294개(54.9시간), 사람/AI 1,647쌍, mono(예시 22.05 kHz), 파일마다 JSON(세그먼트 종류·길이·SNR·RMS). 사람 음악은 BAF 데이터의 Epidemic Sound 제작음악(방송·광고용 전문 제작), AI 음악은 Suno v3.5 extend로 만든 스타일 일치 이어짓기, 음성은 OpenBMAT TV 음성이다. 쌍은 같은 OpenBMAT 참조 구조를 따르고 음악만 다르다. 세그먼트 비율은 no-music 48.4%, bg-music 20.1%, similar 11.2%, music 10.7%, fg-music 5.3%, bgvl-music 4.4%.
- 언어: 예시가 스페인 다큐·독일 뉴스이므로 유럽 TV 음성 [V]. 한국어 0 [I].
- 규모: `AI-OpenBMAT.zip` 7.31 GB [V].
- 라이선스·접근: CC BY 4.0, open [V]. 상위 BAF는 "비상업 연구용, 요청 시 제공"으로 알려져 있어 [I] 파생 공개본과의 조건 차이는 [U]다.
- 규정: 공개 접근과 재배포 가능 라이선스라 2차 파일 제출에 문제가 적다고 본다 [I].
- 공백: 3 (실제 음성 + AI 음악 1,647행. 현재 train의 realV+fakeM은 5행 [V, bank 재집계]), 4 (전문 제작음악이 들어간 real 1,647행, 단 음성과 섞인 상태).
- 첫 투입: 다운로드 수 분, 54.9시간 분량 feature 추출 포함 1–2시간 [I].
- 라벨: 사람판은 FILE 0 · VOICE 0(present 1) · MUSIC 0, AI판은 FILE 1 · VOICE 0 · MUSIC 1. extend 결과에 사람 원곡 일부가 남는지는 [U]인데, 남더라도 파일 라벨은 fake다. 음악이 없는 구간이 절반 가까이라 창 단위 MIL과 잘 맞는다.
- 분할: `gen_xxxxx`와 `gen_xxxxx_ai`, 같은 OpenBMAT 참조를 한 group으로 묶는다.
- 참고: 배포처 설명에 따르면 음성에 가려진 짧은 음악 조건에서 기존 탐지기의 F1이 60% 아래로 떨어진다 [V].

### 3.2 MUSDB18-HQ + musdb-XL (2위)

- URL: MUSDB18-HQ https://zenodo.org/records/3338373 · musdb-XL https://zenodo.org/records/7041331 · musdb-XL-train https://zenodo.org/records/12194067 · 코드 https://github.com/jeonchangbin49/musdb-XL (MIT) [V]
- 내용 [V]: 150곡 전곡(train 100 / test 50), mixture와 vocals·drums·bass·other stem, 44.1 kHz 스테레오 WAV. 100곡은 DSD100에서 왔고 DSD100은 "Mixing Secrets" 멀티트랙 라이브러리에서 파생됐다. 나머지는 MedleyDB 등 [I, 이전 조사].
- 규모: `musdb18hq.zip` 22.66 GB [V]. 약 10시간 [I].
- 라이선스: "교육 목적으로만 제공, 권리자 허락 없이 상업 이용 금지"(Zenodo 표기 other-nc) [V].
- musdb-XL [V]: MUSDB18-HQ test 50곡에 iZotope Ozone 9 Maximizer(널리 쓰이는 상용 limiter)를 걸어 musdb-L/XL을 만드는 샘플 단위 gain ratio다. 파형이 아니라 복원용 메타데이터이며 9.35 GB, CC BY 4.0. musdb-XL-train은 4초 세그먼트 30만 개와 원곡 100곡용 gain 파라미터(18.23 GB, 풀면 200–210 GB)다.
- 공백: 4 (전문 믹스 실제 곡, XL은 상용 마스터링 음량), 그리고 공백 1·3 구성 재료(깨끗한 보컬 stem과 반주 stem).
- 첫 투입: 곡을 30초 조각으로 나눠 행을 확보하면 2–3시간, XL 복원 추가 1시간 [I]. XL-train 전체 복원(200 GB)은 마감 안에는 권하지 않는다 [I].
- 라벨: 원곡 mixture와 XL 모두 real. limiter는 생성이 아니며, bank가 SonicMaster 처리본(HAIM B1)을 real로 둔 정책과도 맞는다 [V, bank 재집계].
- 분할: 같은 곡의 원본·XL·조각·파생(AI 커버형, Vocal2BGM)을 모두 한 group에 둔다.
- 근거 자료: Deezer의 ISMIR 2026 연구(https://arxiv.org/abs/2607.26874)는 MUSDB18-HQ로 사람 stem과 AI stem을 섞은 곡을 모의하고, "분리 후 탐지"로는 AI stem의 흔적을 안정적으로 되찾지 못한다고 보고했다 [V]. 현재 파이프라인이 HTDemucs stem 위에서 탐지하므로, 공백 3 데이터를 같은 경로로 학습에 넣어야 하는 근거로 읽힌다 [I].

### 3.3 CtrSVDD (3위)

- URL: train/dev https://zenodo.org/records/10467648 · test https://zenodo.org/records/12703261 · 도구 https://github.com/SVDDChallenge/CtrSVDD_Utils · 논문 https://arxiv.org/abs/2406.02438 [V]
- 내용 [V]: 무반주 가창. bona fide 47.64시간(32,312 clip), deepfake 260.34시간(188,486 clip, 평균 4.97초), 14개 방식(SVS와 SVC), 가수 ID 164개. Train 12,169/72,235(A01–A08), Dev 6,547/37,078, Eval 13,596/79,173(A09–A14). bona fide 원천은 중국어 Opencpop·M4Singer·KiSing·"official ACE-Studio release", 일본어 Ofuton-P·Oniku Kurumi·Kiritan·JVS-MuSiC다. A14는 ACE Studio의 합성 엔진 ACESinger(상용)다.
- 규모 [V]: `train_set.zip` 7.911 GB, `dev_set.zip` 3.389 GB, `test_set.zip` 8.964 GB, `test.txt`. test.txt에 라벨이 있는지는 [U].
- 라이선스·접근: open, CC BY-NC-ND 4.0 [V]. 일본어 bona fide 4종은 라이선스 때문에 zip에 없어서 직접 받아 `segment.py`로 복원해야 한다 [V]. Kiritan 배포처는 로그인 페이지이고 [V] JVS-MuSiC는 재배포 제한이 있다 [I, 이전 조사].
- 규정: ND라도 비상업 목적의 원본 그대로 공유는 허용된다. 원 파일을 그대로 제출하고 전처리는 코드로 내는 방식이 안전하다 [I]. JVS-MuSiC 원음은 제출이 어려울 수 있으니 일본어 계열 행은 빼는 편이 낫다 [I].
- 공백: 1 (SVS/SVC 가짜 가창, 현재 0).
- 첫 투입: 약 20 GB 다운로드와 attack별 균형 표본 5–10k clip 추출로 2–4시간 [I].
- 라벨: VOICE_PRESENT 1, MUSIC_PRESENT 0, FILE = VOICE. bona fide 가운데 "official ACE-Studio release" 행은 ACE Studio 합성물일 수 있어 [U] 대회 정의로 real에 두면 안 된다. 메타데이터 첫 열(원 dataset 이름)로 걸러 제외한다.
- shortcut: fake가 전부 중국어·일본어 무반주다. 같은 zip에 든 중국어 bona fide를 반드시 함께 넣고, 무반주 fake를 실제 반주와 섞은 AI 커버형 파생본(3.5)도 만든다. SingMOS 추가 효과가 +0.0002였던 점을 보면 [V, 브리핑] 무반주 fake만 더해서는 LB 변화가 작을 수 있다 [I].
- 계보: Opencpop·M4Singer·KiSing 가수와 곡은 AT-ADD, ACE-*, SingMOS에도 나온다. 데이터셋을 넘어 가수·곡 기준으로 group을 묶는다.

### 3.4 GTSinger 한국어 subset (4위)

- URL: https://huggingface.co/datasets/GTSinger/GTSinger/tree/main/Korean · 약정 https://huggingface.co/datasets/GTSinger/GTSinger/blob/main/dataset_license.md · 논문 https://arxiv.org/abs/2409.13832 [V]
- 내용 [V]: 전문 스튜디오 녹음, 48 kHz 24-bit WAV. 가수는 반주를 헤드폰으로만 들었고 녹음에는 반주가 없다. 한국어 가수 3명의 가창 시간은 KO-Tenor-1 4.61시간, KO-Soprano-1 0.95시간, KO-Soprano-2 2.72시간으로 합계 8.28시간이다. 짝 음성(speech)은 1.32/0.24/0.61시간이다. 창법은 mixed voice·falsetto, breathy, pharyngeal, vibrato, glissando.
- HF 실측 [V]: Korean 폴더에 WAV 3,327개(5.66 GB), TextGrid·JSON 각 3,327개, musicxml 2,295개. 가수별 WAV는 KO-Tenor-1 1,863, KO-Soprano-2 1,087, KO-Soprano-1 377. 가수/창법/곡 조합 180개. 5.66 GB는 48 kHz 24-bit 기준 약 10.9시간으로 가창+음성 10.45시간과 맞는다 [I].
- 라이선스·접근: CC BY-NC-SA 4.0에 추가 약정 문구가 붙는다 [V]. 문구는 "M4Singer 저자"에 대한 면책(복사 오류로 보임), "고용주를 대리할 권한이 있다"는 진술, 인용 요구다. HF gate는 없다 [V].
- 규정: NC-SA 데이터 사용은 운영진이 허용했다 [V, 이전 조사의 DACON 답변]. "고용주 대리" 문구가 개인 참가자에게 어떻게 적용되는지는 사용자가 판단할 부분이다 [I].
- 공백: 2 (한국어 무반주 실제 가창이 CSD 1명에서 4명으로), 5, 그리고 3.5·3.6의 한국어 입력.
- 첫 투입: 약 1시간 [I].
- 라벨: real VOICE, MUSIC_PRESENT 0. 음성 파일은 real speech로 따로 표시하거나 뺀다. 같은 곡의 창법별 버전과 Control group은 한 group이다.
- 주의: 이것만 넣으면 "한국어 무반주 = real" 지름길이 커진다. 3.5의 한국어 SVC fake와 짝으로 넣어야 한다.

### 3.5 자체 생성 A: 한국어 SVC 가짜 가창과 AI 커버형 혼합 (5위)

- 실제 보컬 입력: GTSinger 한국어(3명), CSD 한국어(bank 안, CC BY-NC-SA), MUSDB18-HQ vocals, VocalSet(bank 안, CC BY 4.0). AI-Hub 465/473 보컬은 국내에서만 처리한다.
- 음색 prompt: 권리가 확인된 데이터의 다른 가수(VocalSet 20명, GTSinger 다른 언어 가수 17명, M4Singer, OpenSinger)만 쓴다. 실존 연예인 음색이나 HF의 비공식 K-pop 음색 모델은 쓰지 않는다.
- 도구 [V]: Seed-VC(https://github.com/Plachtaa/seed-vc, 코드와 HF 가중치 GPL-3.0, zero-shot 가창 변환), SoulX-Singer-SVC(https://github.com/Soul-AILab/SoulX-Singer, 코드·가중치 Apache-2.0, zero-shot, 언어 무관, 가사 전사 불필요), RVC WebUI(MIT, 사전학습 가중치 MIT, 대상 가수별 학습 필요), Applio(MIT), DDSP-SVC(MIT).
- 만들 행: (a) 한국어 fake 무반주(VOICE_FAKE 1). 원본 real과 같은 group의 쌍으로 둔다. (b) AI 커버형: MUSDB 곡의 vocals stem을 변환해 같은 곡 반주(drums+bass+other)에 다시 섞은 fakeV+realM. 대조군은 원곡 mixture(realV+realM)다. 실제 AI 커버처럼 HTDemucs로 분리한 보컬을 변환하는 변형도 섞는다.
- 한계: 한국어 보컬에 맞는 실제 반주가 공개 데이터에 없다(GTSinger는 반주 미배포 [V]). 한국어 AI 커버형은 AI-Hub 없이는 만들기 어렵고, 영어 MUSDB로 패턴을 익혀 언어 일반화에 기대야 한다 [I].
- 규정: 학습에 쓴 생성 파일 전부와 모델 revision·seed·설정을 제출해야 한다(운영진 2026-08-27 답변) [V, 이전 조사]. GPL-3.0 도구의 출력 오디오 제출에는 제약이 없다고 본다 [I].
- 첫 투입: 설치·가중치 30–60분, zero-shot 변환 1–2시간(수천 clip), 혼합·feature 1시간으로 3–6시간 [I].

### 3.6 자체 생성 B: ACE-Step 1.5 Vocal2BGM (6위)

- URL: https://github.com/ace-step/ACE-Step-1.5 (MIT, 2026-09-03 갱신) · 가중치 https://huggingface.co/ACE-Step/acestep-v15-base , https://huggingface.co/ACE-Step/acestep-v15-xl-base (MIT, gate 없음) [V]
- 기능 [V]: Cover, Repaint & Edit, Track Separation(Extract), Multi-Track(Lego), Vocal2BGM(보컬 트랙에 반주 자동 생성), 50개 이상 언어. extract·lego·complete 작업은 base 계열 모델이 필요하고 XL은 12 GB 이상(권장 20 GB)의 VRAM을 쓴다. 표준 생성은 A100에서 곡당 2초 미만이다. 한국어는 v1 README에 명시돼 있다 [V]. v1.5에서의 한국어 품질은 [I].
- 사용법: GTSinger 한국어·CSD·MUSDB vocals에 반주를 생성한다. 출력 mix를 그대로 쓰지 말고 반주만 분리(HTDemucs나 ACE-Step extract)해 원래의 실제 보컬 파형과 다시 섞는다. 출력 안의 보컬은 VAE를 거친 재합성본이라 real 라벨이 흔들린다 [I].
- 라벨: FILE 1, VOICE 0, MUSIC 1 (공백 3의 노래 버전). 대조군은 같은 보컬에 실제 반주(MUSDB 원 반주)를 섞은 realV+realM.
- 대안: Deezer 코드(https://github.com/deezer/ismir26-hybrid-human-ai-music-detection)는 stem을 neural codec으로 autoencode해 혼합을 모의한다 [V]. 코덱 재합성 반주를 "AI 생성 음악"으로 볼지는 대회 정의상 애매하므로 보조로만 쓴다 [I]. 상용 서비스의 반주 추가 기능은 계정·약관·업로드 권리가 필요하고 [I], AI-Hub 보컬 업로드는 해외 반출이라 금지다.
- 첫 투입: 설치·가중치 30–60분, H100 한 장에서 시간당 수백 곡 생성 [I], 분리·혼합·feature 1시간으로 3–5시간 [I].

### 3.7 IRMAS (7위)

- URL: https://zenodo.org/records/1290750 [V]
- 내용: 주 악기 주석이 달린 음악 발췌 [V]. 대상은 첼로, 클라리넷, 플루트, 어쿠스틱·일렉 기타, 오르간, 피아노, 색소폰, 트럼펫, 바이올린, 사람 노래 [V]. 상용 녹음에서 뽑은 발췌로 알려져 있다 [I]. Training 6,705개(3초), Testing 2,874개(5–20초) [I].
- 규모: TrainingData 3.18 GB와 TestingData Part1–3 합쳐 10.98 GB [V].
- 라이선스: Zenodo 표기 CC BY-NC-SA 4.0 [V].
- 공백: 4, 5.
- 첫 투입: 1–2시간 [I].
- 주의: 학습 창이 64,600 samples(16 kHz 기준 4.04초) [V, 브리핑]라 3초 train 클립은 padding이 필요하다. padding 길이가 real 쪽 단서가 될 수 있으니 Testing 부분(5–20초)부터 쓴다 [I]. 이전 조사 4순위였던 Medley-solos-DB도 클립이 약 2.97초라 같은 문제가 있다 [I].
- 라벨: real music. 노래(voi) 주석이 있는 발췌는 VOICE_PRESENT 1(real singing).

### 3.8 HAIM 미편입 subset (8위)

- URL: https://huggingface.co/datasets/mippia/HAIM · 구성 설명 `DATASET_DESCRIPTION.txt` [V]. HF 최종 수정 2026-07-29 [V].
- bank에 이미 있는 것 [V, bank 재집계]: sonicmaster 300(real로 라벨됨), fx_encoder_style_transfer 300, Lyria-3 Pro 300, crossfade 297, concat 295, acestep 400, musicgen 400.
- 아직 없는 것 [V]: B3 Ozone 9 마스터링(Greg Calbi "General Clarity" preset) AI 곡 6,000개(MIT), B4 전문 믹스 AI 곡 6,000개(MIT), B6 K-pop 가사 조건 ACE-Step 1.5 곡 2,000개(프롬프트마다 en/kr 두 버전, CC BY-NC 4.0), B7 신호 변형 2,000개, B8 의미 편집 2,000개, B9 repaint 2,000개.
- 가치: B3·B4는 "마스터링·전문 믹스 = real" 지름길을 끊는 fake 대조군이다. musdb-XL(마스터링된 real)과 짝으로 쓴다 [I]. B6 한국어판 약 1,000곡 [I]은 한국어 가짜 완성곡을 4배 이상 늘린다. 다만 자체 `generated_music`도 ACE-Step이라 [V, bank 재집계] 생성기 다양성은 크지 않다.
- 규정: 오디오가 포함된 subset은 문제없다 [V]. B5(AI 보컬 + 사람 반주, AI 커버)는 YouTube URL 2,040개뿐이라 뺀다 [V].
- 첫 투입: 폴더 단위 다운로드(곡당 평균 약 3.6 MB [I])로 1–2시간 [I].

### 3.9 AI-Hub (9위, 조건부)

공통 [V]: 상세 페이지마다 "내국인만 데이터 신청이 가능합니다"가 표시된다. 대회 사용은 운영진이 2026-09-21 허용했다 [V, 브리핑]. 국외 이용·반출과 제3자 제공에 제한이 있으므로 EU GPU와 클라우드 LLM 세션으로 보낼 수 없다 [V, 브리핑]. 첫 투입 시점은 로컬 Mac이나 국내 GPU(VESSL deneb-kr의 실제 위치와 허용 여부는 [U]) 사용 가능성에 달렸다.

| ID | URL | 확인 내용 [V] | 공백 | 주의 |
| --- | --- | --- | --- | --- |
| 71544 음악 유사성 판별 | https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71544 | 2022 구축, 2023-12 최종 개방, 20,000곡. WAV 44.1 kHz·16 bit·스테레오. 크라우드소싱 수집. 원곡(Org) Ballade 3,000·Dance 2,000·Hiphop 1,000·RnB 1,000·Rock 1,000·Trot 2,000. Similar 세트는 원곡과 Cover_Arrangement·Rhythm·Genre·Instrument·Timbre·Tempo 변형 각 300. JSON·MIDI·MusicXML 라벨, `lyricText` 필드, 악기 목록에 "Vocal" | 2, 4, 5 | 원곡마다 보컬이 있는지는 [U]. real로는 Org만 쓰고 Cover_Timbre(음색 변경) 등 제작 방법이 [U]인 변형은 뺀다 |
| 465 다화자 가창 | https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=465 | 2021, 가수 18명, 4,681개(발라드/R&B 1,560, 록/팝 1,560, 트로트 1,561), wav와 JSON·MIDI, 출처 "한국의 가요", SVS 학습용, MOS 4.52 | 2, 5 | 반주 제공 여부 [U]. 로컬에 train 256·dev 55 파일 [I, 이전 조사] |
| 473 다음색 가이드보컬 | https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=473 | 2021, 가창자 92명(연령·음색·성별·장르 16그룹, 동요·댄스 등), 4,000곡, 자체 수집, note(MIDI)+JSON | 2, 5 | 157시간 [I, 이전 조사] |
| 71767 한국 대중음악 루프 | https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71767 | 2023, loop WAV 120,000개(Vocal Melody 15,000, Drums·Piano·Guitar·Bass·Strings·Brass·Wind·Organ·Synth Lead 각 10,000, Pad 5,000, 국악 현악기·관악기 각 5,000), 44.1/48 kHz, 출처 "컨소시움 소유 20만곡 대중음악", 루프 안 가사·코드 진행 메타 | 2, 4 | 예시 길이 3,102 ms라 4.04초 창보다 짧은 루프가 많을 수 있음 [I] |
| 71314 방송콘텐츠 대화체 | https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71314 | 7,002.74시간, 공중파 TV·라디오, 환경정보(배경음악/잡음 종류와 강도) | 4 (한국어 음성+실제 BGM) | 페이지의 데이터 유형 표기가 "텍스트"라 오디오 제공 범위 [U] |

- 539 "음성 및 모션 합성 데이터"는 이름과 달리 실제 발화·모션(음성 1,017시간)이다 [V]. 합성 음성이 아니므로 이번 공백과 관계없다.
- AI-Hub 검색에서 합성·위조 음성이나 가창 탐지용 한국어 데이터셋은 찾지 못했다 [U].

### 3.10 AT-ADD Track 2 (10위)

- URL: https://huggingface.co/datasets/xieyuankun/AT-ADD-Track2 · 평가 계획 https://arxiv.org/abs/2604.08184 · 벤치마크 논문 https://arxiv.org/abs/2608.23437 [V]
- 내용 [V]: speech·sound·singing·music 네 유형의 real/fake. Train 146,781(speech 49,575 / sound 39,840 / singing 36,000 / music 21,366), Dev 91,069(49,734 / 19,929 / 16,000 / 5,406), Progress 45,875(28,813 / 5,729 / 4,872 / 6,461), Eval 229,373. 가창은 Opencpop·M4Singer·KiSing 원천의 real 대 SVC fake, 음악은 사람 작곡 대 TTM이다. eval에는 OOD real이 들어 있다. 음악 real 원천은 [U]. 메타데이터 열은 name·type·label·source·source_id·speaker_id·generation_task·generator·caption이며 eval 메타 229,373행까지 공개됐다.
- 규모: 전체 66 GB [V]. train archive만 약 20 GB [I, 이전 조사].
- 접근 [V]: HF 로그인, 짧은 양식 작성, 약정 동의 후 자동 승인. 접근 자격 증명은 공유하면 안 된다.
- 조건 [V]: 비상업 학술 연구와 교육 목적만 허용. 재배포는 같은 목적일 때 약정 사본을 함께 보내고 제3자 원 라이선스를 지키는 조건에서만 가능. 권리자 요청이 있으면 삭제해야 한다.
- 규정: DACON 2차 검증용 제출이 "학술·교육 목적 재배포"에 해당하는지 [U]다. 쓰기 전에 운영진 확인을 권한다.
- 공백: 1 (SVC 가창), 음악 생성기 다양성. CtrSVDD·ACE-*와 원천 가수·곡이 겹치므로 계보 group이 필수다.
- 첫 투입: 3–5시간 [I].

### 3.11 ACE-Opencpop / ACE-KiSing (11위)

- URL: https://huggingface.co/datasets/espnet/ace-opencpop-segments · https://huggingface.co/datasets/espnet/ace-kising-segments [V]
- 내용: ACE Studio(상용 SVS)로 합성하고 수동 튜닝한 가창이다 [V 카드, I 이전 조사]. ACE-Opencpop은 중국어, train 100,510 / val 50 / test 5,400 segment, 43.58 GB [V]. ACE-KiSing은 중국어·영어, 24,362 / 50 / 4,548 segment, 다운로드 10.78 GB [V]. 시간은 각각 128.9·32.5시간 [I, 이전 조사].
- 라이선스: CC BY-NC 4.0, gate 없음 [V].
- 공백: 1 (상용 SVS 출력). CtrSVDD의 A14(ACESinger)와 엔진이 같아 둘 중 하나로 충분할 수 있다 [I].
- 주의: 원 곡의 real 짝(Opencpop, 원 KiSing)이 없으면 언어·곡 지름길이 생긴다. "ACE-"가 붙은 쪽은 real이 아니다.
- 첫 투입: ACE-KiSing 기준 1–2시간 [I].

### 3.12 실제 가창 짝: OpenSinger / M4Singer / Opencpop (12위)

- OpenSinger https://github.com/Multi-Singer/Multi-Singer.github.io : Google Drive 공유 링크, 받으면 CC BY-NC-SA 동의로 간주 [V]. 중국어 약 50시간, 66명 [I, 이전 조사].
- M4Singer https://github.com/M4Singer/M4Singer : Google Drive "free use" 링크, 사용 시 약정 수락 [V]. 중국어 20명 약 30시간 [I, 이전 조사].
- Opencpop https://wenet-e2e.github.io/opencpop/download/ : Google Form 작성 후 메일 안내 [V]. 여성 1명 5.2시간, CC BY-NC-ND 4.0 [I, 이전 조사].
- 용도: 공백 5(가수 다양성)와 CtrSVDD·AT-ADD·ACE-*의 중국어 fake에 대한 real 짝.
- 첫 투입: Drive 할당량 오류가 없다면 2–4시간 [I].

### 3.13 MoisesDB (13위)

- URL: https://music.ai/research/ · https://github.com/moises-ai/moises-db [V]
- 내용 [V]: 이전에 발표되지 않은 240곡, 아티스트 47명, 12개 장르, 14시간 24분 46초, 4 stem을 넘는 멀티트랙. 비상업 연구에 무료, CC BY-NC-SA 4.0.
- 접근: 연구 페이지의 Download 버튼. 페이지에 HubSpot 폼 스크립트가 있어 이름·메일 입력 뒤 링크를 받는 방식으로 보인다 [I]. HF `wearemusicai/moisesdb`는 메타데이터뿐이다(0 GB) [V].
- 공백: 4, 5, 공백 1·3 구성 재료(47명 보컬 stem).
- 첫 투입: 폼이 즉시 링크를 주면 2–4시간 [I]. 수동 승인이면 마감 안에 어렵다.

### 3.14 MTG-Jamendo / FMA (14위)

- MTG-Jamendo https://github.com/MTG/mtg-jamendo-dataset [V]: 55,701곡, 320 kbps MP3 전체 508 GB(저음질 156 GB), 폴더별 tar라 일부만 받을 수 있고 스크립트가 MTG 서버와 핀란드 미러를 지원한다. 오디오는 트랙별 CC, 데이터셋은 비상업 연구·학술 전용. bank에 Jamendo 계열이 이미 있고 [V] 이전 조사에 TDM 조건 미해결 기록이 있다.
- FMA https://github.com/mdeff/fma [V]: fma_large 106,574곡×30초 93 GiB, fma_full 879 GiB, 아티스트 16,341명, 트랙별 CC.
- 용도: 공백 5. 아티스트 단위로 뽑아 real music top-10 group 집중(19%)을 낮춘다. 공백 4에는 약하다(인디 비중이 큼) [I].
- 첫 투입: tar/zip 일부만 받으면 2–4시간 [I].

### 3.15 SVCC 2025 출력 / SingMOS-Pro (15·16위)

- SVCC 2025 https://huggingface.co/datasets/lestervioleta/svcc2025 [V]: GTSinger subset(train·test·test_gt), GTSinger 조건의 CC BY-NC-SA 4.0, 10.94 GB. baseline과 일부 참가 시스템 출력은 README의 Google Drive 폴더에 있다. 과제는 가창 스타일 변환이며, AI가 만든 가창이므로 fake로 볼 수 있다 [I]. 한국어 포함 여부와 출력 규모는 [U].
- SingMOS-Pro https://huggingface.co/datasets/TangRain/SingMOS-Pro [V]: CC BY 4.0, 2.83 GB, 중국어·일본어, gate 없음. SingMOS 추가 실험이 +0.0002였으므로 [V, 브리핑] 우선순위를 낮췄다.

### 3.16 공유마당 (17위)

- URL: https://gongu.copyright.or.kr/gongu/wrt/wrtCl/listWrtSound.do?menuNo=200020&depth2ClSn=10030 [V]
- 내용 [V]: 음악 23,104건, 분류에 대중음악·성악·국악 등이 있다. 대중음악은 103건이며 "소향 보컬 버전", "윤도현 가창 음원"처럼 전문 가수의 보컬곡과 같은 곡의 MR 버전이 함께 있다. 표시 라이선스는 CC BY와 기증저작물 자유이용.
- 접근: 로그인 안내가 보인다. 다운로드에 로그인이 필요한지는 [I].
- 의미: AI-Hub가 아니라서 해외 GPU에서도 쓸 수 있는 드문 한국어 전문 보컬곡이다 [I]. 양이 적어 학습보다는 한국어 실제 곡 오탐 점검용으로 알맞다 [I].

### 3.17 기타 소량 자료 (19위)

- Song Describer https://zenodo.org/records/10072001 : Jamendo 706곡, CC BY-SA 4.0, 3.32 GB [V].
- SD-RR https://zenodo.org/records/22169646 : 2026-08-30 공개. Song Describer 기준곡 496개와 스마트폰 재녹음 쿼리 1,488개(10초), CC BY-SA 4.0, 3.41 GB [V]. 실제 음악의 공간·마이크 채널 다양성용.
- MedleyVox https://zenodo.org/records/7984549 : 다중 가수 분리 평가셋, CC BY 4.0, 0.9 GB [V].
- MedleyDB Sample https://zenodo.org/records/1438309 : 0.42 GB, CC BY-SA 4.0 [V].

## 4. 합성 도구와 라이선스

| 도구 | 용도 | 한국어 | 코드 / 가중치 라이선스 | 비고 |
| --- | --- | --- | --- | --- |
| Seed-VC (https://github.com/Plachtaa/seed-vc) | zero-shot SVC | 언어 무관 [I] | GPL-3.0 / GPL-3.0 [V] | 대상 가수 학습이 필요 없어 가장 빠름 |
| SoulX-Singer / -SVC (https://github.com/Soul-AILab/SoulX-Singer) | zero-shot SVS / SVC | SVS는 중국어·영어·광둥어만 [V]. SVC는 언어 무관 [V] | Apache-2.0 / Apache-2.0 [V] | 한국어 SVS에는 못 씀 |
| RVC WebUI (https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) | SVC, 가수별 학습 | 언어 무관 [I] | MIT / MIT (HF lj1995) [V] | 10분 분량 데이터로 학습 |
| Applio (https://github.com/IAHispano/Applio) | RVC 포크 | 언어 무관 [I] | MIT [V] | 사용 편의 |
| so-vits-svc | SVC | 언어 무관 [I] | AGPL-3.0, 2023년 보관 처리 [V] | 권하지 않음 |
| so-vits-svc-fork | SVC | 언어 무관 [I] | GitHub 표기 NOASSERTION [V] | 라이선스 확인 필요 |
| DDSP-SVC (https://github.com/yxlllc/DDSP-SVC) | SVC | 언어 무관 [I] | MIT [V] | |
| Vevo2 (Amphion) | TTS·SVS·SVC 통합 | [U] | 코드 MIT / 가중치 CC BY-NC-ND 4.0 [V] | |
| DiffSinger (openvpi) + OpenUtau | 악보 기반 SVS | 커뮤니티 한국어 보이스뱅크와 한글 가사 입력 [I] | Apache-2.0 / OpenUtau MIT [V]. 보이스뱅크별 약관 [U] | 악보(MIDI+가사) 필요. AI-Hub 465/473·GTSinger musicxml로 가능하지만 느림 |
| ACE-Step 1.5 (https://github.com/ace-step/ACE-Step-1.5) | 곡 생성, Cover, Repaint, Extract, Lego, Vocal2BGM | 50개 이상 언어 [V] | MIT / MIT [V] | 공백 3 노래 버전 |
| YuE v1 / YuE2 (https://github.com/multimodal-art-projection/YuE) | 곡 생성, YuE2는 zero-shot cover [V] | v1 한국어 지원 [I] | v1 Apache-2.0 [V] / YuE2 [U] | |
| MusicGen-melody | 멜로디 조건 음악 생성 | - | 코드 MIT / 가중치 CC BY-NC 4.0 [V] | 보컬과의 박자·화성 정합이 약함 |
| Stable Audio Open | 음악·효과음 생성 | - | HF gated(auto), 라이선스 표기 other [V] (Stability 커뮤니티 라이선스로 알려짐 [I]) | |

공통 주의: 학습에 쓴 생성물은 파일 전부와 생성·재현 정보를 2차 평가에 제출해야 한다 [V, 운영진 답변]. AI-Hub 음원을 입력으로 쓴 생성은 국내에서만 한다.

## 5. 제외·보류 목록

| 후보 | URL | 이유 |
| --- | --- | --- |
| WildSVDD | https://zenodo.org/records/10893604 | CSV(CC BY 4.0)뿐이고 오디오는 플랫폼 URL [V]. 97명, deepfake 2,007곡, bona fide 1,216곡, 한국어가 세 번째로 많음 [V]. 원음 권리와 링크 소실 문제 |
| SingFake | https://github.com/yongyizang/SingFake | URL 주석 중심 [I, 이전 조사] |
| HAIM B5 | HAIM 카드 | YouTube URL 2,040개 [V] |
| SONICS real | https://huggingface.co/datasets/awsaf49/sonics | real은 YouTube 링크(HAIM 카드 기준 48,090개) [V] |
| KVT | https://khlukekim.github.io/kvtdataset/ | K-pop 466곡 6,787 vocal segment, 오디오 미제공(링크·MFCC) [I, 검색 요약] |
| FSD | https://github.com/xieyuankun/FSD-Dataset | 신청서 제출 후 비공개 Drive, 원 권리자 보호가 이유 [I, 검색 요약] |
| Music4All | https://sites.google.com/view/contact4music4all | 이메일 요청, 이용 조건 미기재. 109,269곡·아티스트 16,269명·46개 언어 [V] |
| BAF | https://zenodo.org/records/7343030 | 요청 승인(비상업 연구). Epidemic Sound 74시간과 TV 57시간 [I, 검색 요약] |
| MedleyDB 1.0/2.0 | https://zenodo.org/records/1649325 , https://zenodo.org/records/1715175 | restricted, 요청 필요 [V] |
| DAMP-MVP 등 Smule | https://zenodo.org/records/2747436 | 로그인·신청, 재배포 금지 조항 [I, 검색 요약] |
| JVS-MuSiC 원음 | 이전 조사 참조 | 재배포 제한 [I] |
| M6 | https://huggingface.co/datasets/yl7622/M6 | HF에 30.21 GB tar만 있고 카드·라이선스 없음 [V] |
| SingFox | https://github.com/Arth-Shah/SingFox | 코드·모델만 있고 음원 미공개 [V] |
| TVSM | https://zenodo.org/records/7025971 | 오디오가 아니라 Mel·VGGish·MFCC feature [V] |
| Codec-SUPERB `*_synth` | HF Codec-SUPERB | 코덱 재합성본, fake 라벨 해석이 애매 [V 존재, I 해석] |
| HF 비공식 K-pop 음색 모델·데이터 | HF 검색 결과 | 실존 가수 음색 복제, 권리 불명 [V 존재] |
| GTZAN, Melon Playlist | - | 상용 음원 권리 불명 / mel-spectrogram만 제공 [I] |
| CoreaSpeech+ | https://huggingface.co/datasets/Gong1212/CoreaSpeech-plus | manual gate [V], 원음 재배포 금지 [I, 이전 조사], 가창 아님 |

## 6. 이전 조사(2026-09-27) 대비 달라진 점

- HAIM: 구성이 갱신됐다(B3·B4 MIT, B6 K-pop 가사, Lyria Pro 3 등). bank 편입분과 미편입분을 구분했다 [V].
- AT-ADD Track 2: gated=auto로 자동 승인되고 eval 메타데이터가 공개됐다 [V].
- MUSDB18-HQ: Zenodo에서 요청 없이 받을 수 있는 open 상태임을 확인했다 [V].
- 새 후보: AI-OpenBMAT, musdb-XL, IRMAS, SD-RR, AI-Hub 71544, 공유마당, ACE-Step 1.5 Vocal2BGM, SoulX-Singer-SVC.
- GTSinger 한국어: 가창 8.28시간, 음성 2.17시간, HF WAV 3,327개·5.66 GB로 실측했다 [V].
- CtrSVDD: bona fide 원천에 "official ACE-Studio release"가 있다는 라벨 위험을 새로 짚었다 [V].
- Medley-solos-DB·IRMAS train·AI-Hub 71767: 클립이 4.04초 학습 창보다 짧아 padding 지름길 위험이 있다 [I].
- SingFox는 음원 미공개, TVSM은 feature 전용임을 확인했다 [V].

## 7. 마감 기준 권장 투입 순서 (제안이며 측정 결과가 아님)

1. AI-Hub가 아니므로 EU 클러스터에서 바로 받을 수 있는 것부터 병렬로 받는다: AI-OpenBMAT, MUSDB18-HQ(+musdb-XL), GTSinger 한국어, CtrSVDD train/dev.
2. 기존 원칙대로 한 번에 한 종류만 추가해 비교한다. 순서는 (a) AI-OpenBMAT, (b) MUSDB+XL+IRMAS Testing, (c) CtrSVDD(중국어 bona fide 포함, 일본어·ACE 계열 bona fide 제외)와 AI 커버형 혼합, (d) GTSinger 한국어와 한국어 SVC fake 짝, Vocal2BGM 행.
3. 판단 지표는 held-out 전체 ADS보다 source별 real-music FPR(MUSDB·XL·IRMAS), held-out realV+fakeM 107행의 VoiceEER, 새 source를 뺀 재학습 비교를 우선한다.
4. GPU는 새 행만 feature를 뽑고 대기 중에는 멈춘다(크레딧 약 $70).
5. AI-Hub 71544·465·473은 국내 연산 자원이 확인될 때만 진행한다.

## 8. 한계

- 약관 해석은 법률 검토가 아니다. 특히 AT-ADD의 재배포 조항, GTSinger 추가 약정, AI-OpenBMAT 상위 원음 조건은 사용자가 원문을 확인해야 한다.
- 크기와 시간은 배포처 표기와 위의 가정에 기반한다. 다운로드·디코딩·청취 검사를 하지 않았으므로 실제로 쓸 수 있는 파일 수는 다를 수 있다.
- 작업 도중 Zenodo 검색 API가 403과 빈 응답으로 막혀 DAMP·BAF·FSD·KVT는 검색 요약에 의존했다.
- AI-Hub 상세 페이지의 용량과 파일 목록은 동적으로 로드돼 확인하지 못했다.
- 이 목록은 조사 시점 기준의 폭넓은 검토이며 전 세계 자료를 모두 망라했다고 주장하지 않는다.
