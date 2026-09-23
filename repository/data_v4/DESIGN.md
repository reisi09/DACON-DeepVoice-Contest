# Data v4 — 규정 준수 광역 데이터 수집 설계

작성 2026-09-23. **상태: 설계·코드만 완료, VESSL 업로드·다운로드·GPU 실행은 하지 않음.**
전제: v3 계획(`DATA_PLAN_236749_2026-09-19.md`)과 그 검토(`DATA_PLAN_REVIEW_236749_2026-09-19.md`)의 진단을 이어받고, 검토의 P1 지적(라벨 확정 기준, locked 구성, 지름길 검사 기준, ND 해석)을 반영한다.

| 파일 | 역할 |
|---|---|
| `sources.py` | 수집원 목록(라이선스·접근 근거·선택 규칙) → 파일 단위 plan JSON (revision 고정, upstream checksum) |
| `../candidate_data_acquire.py` | 기존 다운로더 재사용. `ACQUIRE_BANK/OUT` 환경변수, OpenSLR·asvspoof.org 호스트, gated HF 토큰, `size-only` 검증 추가 |
| `generate_speech.py` | 한국어 TTS/VC 자체 생성 (Zeroth 화자·문장과 lineage 연결) |
| `generate_music.py` | ACE-Step 한국어 노래·반주 자체 생성 (가사는 이 프로젝트에서 새로 쓴 문장) |
| `build_manifest.py` | ingest → materialize(16 kHz·crop·codec) → vocals(HTDemucs 근거) → mix |
| `split.py` | split 분석 보고서 → 사람이 쓴 규칙 → 결정적 배정 → 누수 검증 |
| `envs.sh`, `smoke_vessl.sh` | VESSL 생성기 환경, VESSL 첫 실행 점검 |

## 1. 채울 공백

현재 은행(16,042행)의 공백. 근거는 v3 §1과 검토 §1의 CSV 재계산이다.

1. **한국어 real/fake 음성 없음.** 대회 주제는 보이스피싱이다.
2. **전화 채널 없음.** 평가 데이터 설명에 "일부 전화채널"이 명시돼 있다.
3. **가짜 음성이 ASVspoof2019 한 원천뿐.** 2019년 시스템, 영어다.
4. **실제 가창·혼합 라벨 부족.** SONICS·HAIM 가사곡의 보컬 라벨이 비어 있다.
5. **음악 real/fake가 원천 단위로 갈림**(출처 = 정답).
6. **음성·음악 모두 없는(none) 대조군 없음.**

## 2. 수집원 (2026-09-23 확인)

### 2.1 공개 데이터

| 이름 | 라이선스 | 근거 | 크기 | 채우는 공백 |
|---|---|---|---:|---|
| zeroth_korean | CC BY 4.0 | HF 카드 | 2.9 GB | 한국어 real, 자체 생성의 화자·문장 원천 |
| fleurs_ko | CC BY 4.0 | HF 태그 | 1.8 GB | 한국어 real(다른 녹음 도메인) |
| **mlaad_ko** | CC BY-NC 4.0 | HF 카드, gated=auto | 4.8 GB | **한국어 fake 12계열**: Qwen3-TTS 0.6B/1.7B, VoxCPM2, Fish-S2-Pro, Higgs-Audio-V2, MOSS-TTS 1.7B/8B, OmniVoice, MiniMax speech-02, Chatterbox, XTTS-v2, Bark. 각 1,001개 |
| mlaad_other | CC BY-NC 4.0 | 위와 같음 | 계산 중 | 다국어 생성기 다양성. (언어, 시스템) 폴더당 40개 |
| asvspoof2021_la | ODC-By 1.0 | Zenodo 4837263 | 7.8 GB | **실제 전화망 전송**(a-law, G.722, PSTN, VoIP) real+fake |
| asvspoof2021_df (part00) | ODbL 1.0 | Zenodo 4835108 | 8.7 GB | mp3/m4a/ogg 압축, 보코더 100+ |
| vocalset | CC BY 4.0 | Zenodo 1193957 | 2.1 GB | 실제 a cappella 가창 |
| csd | CC BY-NC-SA 4.0 | Zenodo 4916302 | 1.9 GB | 실제 한국어 가창 |
| speechfake | CC BY 4.0 | GitHub 배지+LICENSE | 부분 선택 | 오픈 생성기 30종의 한국어 fake, Common Voice(CC0) real |
| haim (rev 371be01) | CC BY-NC 4.0 + MTG/SONICS 크레딧 | HF 카드 | 폴더당 150 | 실제 MTG, Lyria/ACE-Step/MusicGen, AI 마스터링 real(B1), 부분 편집, mixset |
| echoes | CC BY-SA 4.0 | HF 태그 | 8.6 GB | 음악 생성기 12종 원본(8초 레거시 크롭 대체) |
| sonics (shard 2개) | CC BY-NC 4.0 | HF 태그 | ≈6.5 GB | Suno/Udio 전곡(보컬 포함) |
| openmic | CC BY 4.0 | Zenodo 1432913 | 2.6 GB | 실제 음악 + voice 주석 마스크 |
| musan | CC BY 4.0 | openslr.org/17 | 11 GB | 실제 음악(보컬 Y/N 주석), 실제 음성, **noise(none 대조군)** |

- 기존 로컬 은행과 기존 16,042행(`--existing-csv`)은 라벨·마스크를 그대로 가져온다.
- 정확한 합계는 `plan/SUMMARY.json`에 기록된다.

### 2.2 규정 근거

- **규칙 2-1** ("누구나 접근 가능 + 최소 비영리 허용"): CC BY, BY-SA, BY-NC, BY-NC-SA, ODC-By, ODbL을 허용한다.
- **규칙 2-2** (출처 명시): manifest 행마다 `license`, 원천 URL, revision, sha256을 기록한다.
- **2차 평가 데이터 제출:** 운영진 답변 5~7번에 따라 원본과 재현 코드로 제출한다. 대용량 원천의 가공본을 통째로 싣지 않아도 된다.
- **ND 자료:** 운영진은 원본+코드 재현 방식을 허용했다. 다만 이번에는 새로 넣지 않는다(보수적 선택). 기존 Existing9504의 CtrSVDD(BY-NC-ND)는 이 방식으로 설명해야 한다.

### 2.3 수집하지 않는 것

| 자료 | 이유 |
|---|---|
| AI-Hub | 국적·승인 게이트. "누구나 접근 가능" 해석이 불확실하다 |
| KSS | Kaggle 로그인이 필요하다 |
| Common Voice 직접 | Mozilla Data Collective 등록이 필요하다. SpeechFake가 CC0 CV real을 포함한다 |
| SingFake, WildSVDD, SONICS real, HAIM 링크 서브셋 | 링크만 제공, 오디오 라이선스가 없다 |
| ASVspoof5 | 142 GB, 영어 |
| Codecfake | ND이고, 운영진 규칙상 코덱 복원은 REAL이라 라벨이 충돌한다 |
| RTCFake | CC BY-NC 4.0이지만 **manual 승인 게이트**. 기본 OFF다. 승인을 받으면 켜고 보고서에 명시한다 |

## 3. 자체 생성 (GPU)

MLAAD-ko가 XTTS-v2와 Chatterbox를 이미 포함하므로, 자체 생성은 **MLAAD에 없는 것**만 한다.

| backend | 라이선스 | 역할 | 수량 목표 |
|---|---|---|---:|
| cosyvoice2 | Apache-2.0 | zero-shot 한국어 TTS | Zeroth 화자당 8 → 약 800 |
| melo | MIT | 비복제 한국어 TTS(다른 계열) | 약 800 |
| seedvc | GPL-3.0 코드 | **음성 변환(VC)**. MLAAD에 없음 | 화자쌍 × 2방향 × 8 → 약 800 |
| acestep | Apache-2.0 | 한국어 가사 노래 300 + 반주 200 | 500 |
| xtts, chatterbox | CPML / MIT | 선택. MLAAD-ko와 겹친다 | 0 |

- **lineage 규칙** (component 크기 제한, `generate_speech.plan_jobs`):
  - TTS: 화자 S의 목소리로 S의 *다른* 문장을 읽힌다. 화자 S 하나에만 연결된다. 참조 발화는 다시 읽히지 않는다.
  - VC: 고정 화자쌍 A↔B 안에서만 변환한다. component = 화자 2명.
- 생성 문장은 1~5문장(약 5~30초)이다. real Zeroth와 내용·화자가 짝을 이뤄 "내용·화자로 맞히는" 지름길을 막는다.
- ACE-Step 가사는 `generate_music.py`에 들어 있는 창작 문장이다. 제3자 가사는 쓰지 않는다. 같은 prompt의 곡들은 한 component다.
- 모든 생성 행에 `generator`, `generator_repo`, 가중치 라이선스, `parent_ids`, `label_evidence=construction`을 기록한다.

## 4. 라벨 규칙

| 유형 | VP | MP | VF | MF | FILE |
|---|---|---|---|---|---|
| 실제 발화 / a cappella | 1 | 0 | 0 | 마스크 | 0 |
| AI 음성(TTS/VC) | 1 | 0 | 1 | 마스크 | 1 |
| 실제 반주 | 0 | 1 | 마스크 | 0 | 0 |
| AI 반주 | 0 | 1 | 마스크 | 1 | 1 |
| 실제 노래 | 1 | 1 | 0 | 0 | 0 |
| AI 노래 | 1 | 1 | 1 | 1 | 1 |
| 혼합(구성) | 1 | 1 | 성분대로 | 성분대로 | OR |
| 실제 + AI 후처리(B1 마스터링) | 유형대로 | | 0 | 0 | 0 — `ai_postprocessed_real=True`로 표시해 A/B 대상으로 둔다 |
| noise | 0 | 0 | 마스크 | 마스크 | 0 |

- **조건부 진위** `VOICE_FAKE_IF_PRESENT`: "있다면 가짜인가"를 존재 여부와 분리해 기록한다.
  - 실제 녹음은 보컬 존재를 몰라도 FILE=0으로 확정된다.
  - 생성 노래는 보컬이 확인되면 VF=1이 된다.
  - 존재·진위 모두 모르면 FILE=None(마스크)이다. **모르는 것을 0으로 두지 않는다.**
- **보컬 근거** (`vocals` 단계): 최종 crop의 HTDemucs 보컬/믹스 RMS 비(3초 창 최댓값)를 쓴다.
  - ≥0.35면 있음, ≤0.08이면 없음, 그 사이는 모름.
  - 메타데이터와 **일치하면 확정**, **불일치하면 마스크**한다. 라벨을 뒤집지 않는다(검토 P1).
  - 임계값은 원천별 청취 표본으로 조정한 뒤 고정한다. 조정 전 값은 기본값이다.
- **부분 생성(HAIM C1/C2):** crop에 AI 구간이 1초 이상 남아야 FAKE다. 1초 미만으로 걸치면 마스크, 실제 구간만이면 REAL이다(`crop_labels`).

## 5. 전처리 (라벨 독립)

- **디코드:** 16 kHz(평가 데이터와 동일).
- **길이:** 평가 범위 4~60초에 맞춰 4~10초 30%, 10~30초 40%, 30~60초 30%에서 뽑는다. 원본이 더 짧으면 전체를 쓴다. 이어붙이지 않는다(검토: 새 단서 생성 위험).
- **음량:** RMS −23 dBFS에 −6~+4 dB 무작위 게인, 피크 0.97.
- **채널:** 스테레오는 50%를 다운믹스하고, 모노는 20%를 2채널로 복제한다.
- **코덱 표 (파일당 1개):**
  - WAV 30%, FLAC 15%, MP3 64/128/192k 25%, Opus 32k 5%, AAC 96k 5%
  - **전화 20%:** 300–3400 Hz 대역 + 8 kHz + μ-law / A-law / GSM / Opus 8k를 거친 뒤 16 kHz로 복원
  - 없는 인코더는 실행 시 자동으로 빼고 확률을 재정규화한다(`usable_codecs`). 로컬 ffmpeg 7.1에는 libgsm이 없었다.
- **무작위성:** 모든 추첨은 `ID`만의 함수다. 라벨·원천과 독립이다. 실제 체인은 `codec_chain`, `crop_*`, `gain_db`, `channels`로 행에 남는다.

## 6. 혼합 합성

- **재료:** 실제·가짜 음성 단독 행 × 실제·가짜 음악 단독 행(보컬 없음이 확인된 것).
- **방식:** 동시 60%(SNR −5~+15 dB, 음성 위치 무작위), 순차 40%(음성 먼저 또는 음악 먼저).
- **component 1:1 짝짓기:** 음성 component 하나는 음악 component 하나와만 섞는다. 혼합이 component들을 사슬처럼 이어 거대 component를 만들지 않게 한다.
- **코덱:** 혼합물도 같은 코덱 표를 거친다. 혼합물만 FLAC이 되는 지름길을 막는다.

## 6.5 존재(presence) 판정: PANNs 교체 대신 학습형 head (`presence.py`)

로컬 실측 (실제 발화 50 / 반주 50 / 가사곡 50, 10 s, 깨끗함·전화 μ-law):

| 방식 | voice AUC 깨끗/전화 | music AUC |
|---|---|---|
| PANNs CNN14 (현재) | 0.824 / 0.838 | 0.996 |
| AST-AudioSet | 0.807 / 0.805 | 0.997 / 0.990 |
| CED-base | 0.806 / 0.811 | 1.000 / 0.999 |
| 보컬 stem에 PANNs | 0.825 / 0.798 | — |
| 보컬 stem에 CED | 0.801 / 0.785 | — |
| 보컬 stem 에너지 비율 | **0.864** / 0.815 | — |

깨끗한 음질에서는 stem 에너지가, 전화 음질에서는 원음 PANNs가 낫다. 두 신호가 서로 보완하므로 둘을 함께 쓰는 head의 근거가 된다.

AudioSet 분류기 교체는 차이가 없다(노래 속 보컬 누락이 공통 원인). 그래서 **고정 모델 피처 위의 작은 head**를 학습한다.
- 피처: PANNs 클래스 그룹(voice/music/speech/singing/Music)의 창별 max·mean·q90 + HTDemucs 보컬·반주 stem 에너지 비율(3 s 창 max·mean·q90·비율) + 믹스 RMS. 제출 추론에서 이미 돌아가는 두 모델이므로 추가 비용이 거의 없다.
- 모델: 표준화 + 로지스틱 회귀, JSON 저장(numpy만으로 추론, pickle 없음). VOICE/MUSIC 각각.
- 학습: data_v4 manifest의 존재 라벨(마스크 제외), split 규칙의 train. val/test에서 raw PANNs 대비 AUC를 보고한다. 추출은 (원천, 존재 패턴) 셀당 400행으로 GPU 시간을 제한한다.
- 연결: head 출력이 `VOICE/MUSIC_PRESENT_PROB`와 FILE 결합식의 Pv·Pm을 대체한다. 채택은 리더보드 1요인 A/B로 결정한다.

## 7. Split 방법론 — seed 무작위 배정이 아닌 분석 기반

사용자 지시(09-23): train/val/test는 seed로 나누지 않고 분석해서 정한다.

1. **`split.py analyze` 보고서:**
   - component(화자, 트랙, 문장, prompt, 부모 공유 키의 연결 요소) 크기 분포와 최대 component의 구성
   - 셀(유형 × FILE × 원천 × 생성기)별 파일 수와 component 수
   - 유형·라벨별 길이 분위수, 코덱 분포
   - **메타데이터 프로브:** 길이·채널·게인·코덱만으로 FILE을 맞히는 grouped-CV AUC. fold별 값을 보고하고, 통과할 때까지 재추첨하지 않는다(검토 P1).
2. **보고서를 읽고 `split_rules.json`을 직접 작성한다.**
   - `test_holdout`: 학습에서 완전히 빼는 생성기·원천. 예: MLAAD-ko의 일부 계열, Echoes의 일부 생성기, ASVspoof2021 LA의 특정 전송 조건.
   - DF-Arena 사전학습 노출(ASVspoof2019, MLAAD)은 "사전학습 노출 미확인"으로 따로 표시한다. 미학습 생성기라고 주장하지 않는다.
   - `val_fraction`, `cell_keys`, 셀별 최소 component 수를 정한다.
3. **`apply`:** 결정적 greedy 배정이다.
   - 희귀 셀부터, 작은 component부터, 동률이면 component ID 순으로 채운다.
   - component 수가 부족한 셀은 train 전용으로 두고 보고서에 남긴다.
4. **`verify`:** component 교차 누수 0, split·유형별 단일 클래스 여부를 확인한다. 누수가 있으면 exit 1.
5. **기준:** 기존 9,504개의 val/test처럼 과거에 노출된 자료는 새 test에 넣지 않는다. `prior_split`과 `prior_seen`은 행에 보존한다.

## 8. VESSL 실행 계획 (미실행)

### 8.1 환경

과거 workspace 기록(`_workspace/*_workspace_private.json`) 기준이다.

| 항목 | 값 |
|---|---|
| workspace | `wsp-9683432r72m7` (AGENTS.md: 재사용, 새로 만들지 않음) |
| GPU | H100 SXM×1 (capella-emea), 15 vCPU, RAM 200 GB, ephemeral 200 GB. 대안 A100 SXM×1 (betelgeuse-na), 11 vCPU, RAM 128 GB |
| image | `quay.io/vessl-ai/torch:2.9.1-cuda13.0.1-py3.13-slim` |
| mount | `/root`(cluster volume), `/shared`, object volume `dacon-v6-training`(`objvol-a5gxdg7er70i`) |
| 경로 | 코드 `/root/dacon_v4`, 원본 아카이브 `/root/data_v4/archives`, 전개 `/root/data_v4/raw/<source>`, 산출 `/root/data_v4/out`, venv `/root/envs/<backend>` |

### 8.2 예산

사용자 승인: GPU 8시간, 저장 200 GB.

- **저장:**
  - 아카이브 약 70–80 GB. 전개 후 sha256을 기록하고 아카이브를 지운다. 다시 받을 수 있다: URL과 checksum이 plan에 고정돼 있다.
  - 전개 약 80 GB, 산출 약 30 GB.
  - 여유가 부족하면 mlaad_other → sonics → asvspoof2021_df 순으로 줄인다.
- **GPU:**
  - 생성: CosyVoice2 약 1시간, Melo 약 0.3시간, Seed-VC 약 0.7시간, ACE-Step 약 1시간
  - 보컬 근거: crop 약 15k × 평균 25초 → HTDemucs 약 0.7시간
  - 환경 설치·smoke 약 1시간
  - 다운로드와 materialize는 CPU·네트워크 작업이라 생성과 **병렬**로 돌린다.
  - 모든 생성기는 `--deadline-epoch`로 절대 마감 시각에 스스로 멈춘다.

### 8.3 순서

1. **로컬:** `vesslctl auth login`, `hf auth login`, MLAAD 페이지에서 동의.
2. **로컬:** `python data_v4/sources.py build --out data_v4/plan`. SUMMARY로 크기를 확인한다.
3. **저장소 먼저:**
   - object volume 목록을 조회해 기존 은행 오디오가 올라가 있는지 확인한다.
   - 코드와 plan 묶음(sha256 manifest 포함)을 업로드한다.
   - 다음 실행 bootstrap을 먼저 올려 옛 실험이 재시작되지 않게 한다(AGENTS.md).
4. **workspace 시작** → `bash data_v4/envs.sh data` → `bash data_v4/smoke_vessl.sh`. 실패하면 멈춘다.
5. **백그라운드 다운로드:** `ACQUIRE_BANK=/root/data_v4/archives ACQUIRE_OUT=/root/data_v4/receipts python candidate_data_acquire.py data_v4/plan/<src>.json`을 원천별로 돌린다.
6. **생성:** `envs.sh cosyvoice2 melo seedvc acestep` → backend마다 샘플 1개로 smoke → 본 생성.
7. **전개 → 파이프라인:**
   - 아카이브 전개
   - `build_manifest.py ingest`(`--existing-csv` 포함)
   - `materialize --workers 14` → `vocals` → `mix --mixes 1500`
8. **분석:** `split.py analyze` → 보고서를 로컬로 내려받는다. **여기서 멈추고 규칙을 작성한다.**
9. **감시와 정지:** 실행 중 5분마다 확인하고, 변화가 없으면 조용히 기다린다. 결과를 내려받아 검증한 뒤 workspace를 정지한다.

## 9. 검증 현황

| 검사 | 환경 | 결과 |
|---|---|---|
| `sources.py check` (항목 캡·sidecar 묶음·include/exclude) | 로컬 Mac | 통과 |
| `generate_speech --self-check` (TTS 화자 단일 연결, 참조 비재사용, VC 쌍 고정, 재현성) | 로컬 | 통과 |
| `generate_music --self-check` | 로컬 | 통과 |
| `build_manifest --self-check` (조건부 진위, 마스크, 보컬 근거 불일치 시 마스크, crop 부분 라벨, 혼합 1:1) | 로컬 | 통과 |
| `split --self-check` (holdout → test, 부모 공유 → 같은 split, 결정성, 누수 탐지) | 로컬 | 통과 |
| 합성 오디오 E2E (materialize → mix → split, 16 kHz 확인) | 로컬 + imageio-ffmpeg 7.1 | 통과. libgsm 없음 → 자동 제외 |
| HAIM adapter를 실제 로컬 2,000 파일에 적용 | 로컬 | 2,000행. B1 real 500, C1/C2 fake 1,000, A2 가사곡 393 VP=1 |
| `candidate_data_acquire.py --self-check` | 로컬 | 통과 |
| plan 생성(API) | 로컬 | zeroth, fleurs, mlaad_ko, asvspoof LA/DF, vocalset, csd 완료. 나머지 진행 중 |
| **VESSL smoke / 생성기 / 실제 다운로드** | VESSL | **미실행** |

## 10. 남은 위험

- **생성기 API:** CosyVoice2, Seed-VC, ACE-Step, MeloTTS의 Python API와 CLI는 버전마다 바뀐다. `generate_*` adapter는 VESSL에서 샘플 1개 smoke로 먼저 확인한다. 실패한 backend는 빼고 MLAAD-ko에 의존한다.
- **스키마 추정:** ingest adapter 가운데 FLEURS tsv 열 순서, MLAAD `meta.csv` 열, SONICS csv 열, OpenMIC 경로, MUSAN ANNOTATIONS 열은 공개 문서 기준이다. 실제 파일로 검증하지 않았다. ingest 직후 원천별 행 수·라벨 분포를 확인하고 0행이 나온 원천은 고친다.
- **SpeechFake:** ModelScope 서브셋 이름을 모른다. 다운로드 후 구조를 보고 adapter를 붙인다(현재 미포함).
- **보컬 임계값(0.35/0.08):** 청취로 조정하기 전의 기본값이다.
- **전화 채널:** 시뮬레이션이다. 실제 전송은 ASVspoof2021 LA뿐이다.
- **제출 마감:** 09-29 10:00까지 학습·패키징 시간을 고려하면, 수집은 09-25 안에 끝내야 한다.
