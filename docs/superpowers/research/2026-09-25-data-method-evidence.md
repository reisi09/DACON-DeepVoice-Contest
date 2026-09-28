# PANNs 후속 데이터·방법론 조사

확인일: 2026-09-25. 공식 배포처·저자 논문과 현재 저장소 문서를 대조했다. 이 문서는 Opus 토론에 제공할 근거와 후속 실험 제안이다. 아래 후보를 다운로드하거나 학습·평가하지 않았으며, 확정된 v1 설계와 D0를 변경하지 않는다. 접근 신청, 외부 연락, 새 토큰 발급, 유료 자원 실행도 하지 않았다.

## 기존 자료와 구분

- 현재 D0는 42,939개이며, 신규 1,477개를 제외한 구조 비교 자료다. 1,477개는 LibriSpeech 449개·MusicArena 882개·MLAAD 146개로 이미 확보·적격 처리됐다. 새 다운로드보다 구조 선택 후 같은 총 step으로 D0와 D+를 비교하는 순서가 앞선다. [확정 설계](../specs/2026-09-25-panns-cnn-expanded-design.md), [MLAAD 편입 기록](../../MLAAD_DECISION_2026-09-24.md).
- VocalSet, CSD, MUSAN, OpenMIC, HAIM, Echoes, FLEURS, Zeroth는 기존 수집 계획에 있다. NSynth도 이전 평가에 사용됐다. 이들을 신규 독립 자료로 다시 세지 않는다. [기존 데이터 설계](../../../repository/data_v4/DESIGN.md), [이전 평가](../../RESULTS_V2.md).
- Jamendo 신규 1,000곡은 확보했지만 별도 TDM/ML 약관 때문에 보류됐다. 기존 자료의 이용 판단을 신규 곡 전체에 확대하지 않는다. MusicArena의 미확정 음성 존재 라벨도 그대로 마스킹한다. [수집 기록](../../ADDITIONAL_DATA_COLLECTION_2026-09-24.md), [이용 판단](../../MLAAD_DECISION_2026-09-24.md).

## HF 외 추가 원천 후보

우선순위는 현재 PANNs의 음성·음악 존재 판별을 기준으로 제안했다. 가짜 탐지 성능에 직접 도움이 된다는 증거는 아직 없다. 아래 여섯 원천은 확인한 저장소의 수집 계획·결과에서 편입 기록을 찾지 못했다. 전 장치의 모든 파일을 조사한 것은 아니므로 실제 수집 전 manifest·파일/PCM 해시·원본 계보를 다시 대조한다.

| 후보 | 보완할 오류와 역할 제안 | 접근·이용 조건 | 용량과 Mac 24 GB 처리 판단 |
|---|---|---|---|
| **ESC-50** | MUSAN 한 원천에 치우친 둘 다 없음 진단을 넓힌다. 사이렌·새소리·기계음을 음성/음악으로 오인하는지 확인. 우선 독립 진단용으로 잠근다. | 공식 GitHub 공개, 토큰 요구 없음. 전체 CC BY-NC, ESC-10은 CC BY. 클립별 출처 보존. 환경음 class만으로 두 presence를 0으로 확정하지 않고 배경음·사람 소리를 확인한다. | 공식 안내 약 600 MB, 2,000개×5초. CPU 순차 디코딩·청취 큐에 적합. 원음이 같은 crop은 공식 fold와 parent group을 함께 보존. [공식 자료](https://github.com/karolpiczak/ESC-50). |
| **BabySlakh** | 합성 악기 음색을 사람 목소리로 오인하는지 확인하는 작은 대조군. MIDI·stem으로 어느 악기가 오류를 만드는지 좁힌다. | Zenodo 공개, CC BY 4.0, 토큰 요구 없음. 샘플 기반 MIDI 렌더링이며 최신 생성 AI 음악의 대표 자료가 아니다. `MUSIC_FAKE` 정답을 임의 부여하지 않는다. | 20곡, 16 kHz WAV, archive **882,818,115 bytes**. CPU 분석이 가능하지만 원본 곡 20개이므로 crop 수를 늘려도 독립 표본이 늘지 않는다. [공식 자료](https://zenodo.org/records/4603870). |
| **MUSDB18-HQ의 권리 확인 subset** | 실제 노래의 보컬·반주·mix 대응쌍으로 가창과 약한 반주의 영향을 분리한다. 부모 곡을 먼저 나눈 뒤 train 또는 잠긴 평가 역할 하나만 배정. | HQ record는 공개 다운로드. 전체는 교육용·비상업 조건이며, 46곡은 MedleyDB CC BY-NC-SA 4.0, 2곡은 Easton Ellises CC BY-NC-SA 3.0, 나머지는 별도 원천 조건이다. 48곡도 자동 적격으로 선언하지 않고 곡별 대응·대회 검증자 전달 조건을 기록한다. | 전체 ZIP **22,656,664,047 bytes**. 현재 로컬 여유 공간을 확인하기 전 전량 수집하지 않는다. 적격 곡·필요 stem만 처리할 방안을 먼저 확인. [공식 HQ 배포](https://zenodo.org/records/3338373). |
| **MoisesDB** | 여러 장르의 실제 보컬/악기 대응쌍. artist·genre·activity·bleeding 메타데이터로 같은 곡의 보컬 있음/없음과 누음을 구분하는 데 유용할 가능성. | 공식 GitHub는 CC BY-NC-SA 4.0, 배포자 안내는 비상업 연구. 공식 연구 사이트를 통한 다운로드 경로다. 현재 접근 폼의 동의·인증 조건과 정확한 archive 크기는 **미확인**, 대기. | 원본 용량 미확인. 전량 상주 로딩 대신 곡별 처리 후보. CPU로 신호·계보를 검사할 수 있다는 판단이며 처리량은 미측정. [공식 코드·배포 안내](https://github.com/moises-ai/moises-db), [배포자 설명](https://music.ai/blog/press/introducing-moisesdb-the-ultimate-multitrack-dataset-for-source-separation-beyond-4-stems/). |
| **MedleyDB** | 실제 stem과 상세 악기 주석을 이용한 가창·관악기·현악기 혼동 분석. MUSDB18에 포함된 곡과 중복을 제거해야 한다. | Zenodo restricted: 계정과 접근 승인 필요. 비상업 연구·CC BY-NC-SA 4.0. 배포자는 전체/부분 재게시 전에 동의를 요청하므로 대회 검증 전달과 공개 재배포를 구분해 확인한다. **접근 대기**. | 공개 record에서 파일 목록·크기를 확인할 수 없어 **미확인**. 접근 승인 전 다운로드량을 확정하지 않는다. [공식 record](https://zenodo.org/records/1649325), [공식 조건](https://medleydb.weebly.com/downloads.html). |
| **FSD50K** | 여러 환경음·인간 비언어 소리·악기 소리의 추가 진단. 약한 라벨을 없는 라벨로 오해하지 않는 검토 자료. | Zenodo 공개지만 공식 페이지에 **ML 대회 사용 시 저자에게 연락**하도록 명시돼 있어 편입 보류. dataset CC BY 외에 클립별 CC0/BY/BY-NC/Sampling+가 섞여 있다. 새 연락·동의는 하지 않았다. | 전체 약 24.7 GB, eval audio 두 파일 합 **6,258,901,239 bytes**. dev 주석은 불완전할 수 있고 eval도 지정 vocabulary 안에서 완전하다. 무주석을 대회 presence 부재로 바로 바꾸지 않는다. [공식 자료·조건](https://zenodo.org/records/4060432). |

MUSDB/MedleyDB 같은 stem 자료는 고정 HTDemucs의 사전학습·평가 이력과도 대조해야 한다. [Demucs 공식 설명](https://github.com/facebookresearch/demucs)은 `htdemucs`가 MUSDB와 추가 800곡으로 학습됐다고 밝힌다. 새로 수집했다는 사실만으로 전체 시스템에 독립적인 holdout이 되지는 않는다. PANNs AudioSet 사전학습 노출까지 배제했다는 주장도 하지 않는다. 기존에 제안됐던 In-The-Wild·Deepfake-Eval-2024는 음성 fake 분기의 별도 후보로 유지하되, 이번 PANNs 구조 비교에 대량 추가하는 우선순위는 낮춘다.

## 더 정밀하게 비교할 방법

아래 방법은 문헌과 현 구조를 보고 세운 가설이다. 현 데이터에서 개선을 확인한 결과와 구분한다.

| 제안 | 먼저 볼 관측 | 최소 비교와 해석 한계 |
|---|---|---|
| **오류 라벨 검토를 먼저 정밀화** | 같은 parent를 제외한 예측에서 정답과 지속적으로 충돌하는 파일, 모델 간 불일치, 드문 source, 무작위 대조 표본 | 이미 있는 미확정 1,889개를 모두 0으로 채우지 않는다. 점수는 청취 우선순위에만 쓰고 판정자·근거·미확정을 기록한다. Confident Learning은 클래스 조건부 label-noise 가정에서 출발하므로 누락된 다중 라벨과 source shift를 자동 해결한다고 볼 수 없다. [저자 논문](https://jair.org/index.php/jair/article/view/12125). |
| **한 부모의 조건만 바꾼 대응 실험** | 동일 음성·반주에서 SNR, 음성 점유 시간, codec만 바꿀 때 presence·FILE 점수가 어떻게 바뀌는지 | 허용된 train 부모로 작은 학습 후보를 만들고, 별도 역할의 부모로 진단 패널을 만든다. 두 역할은 섞지 않는다. SNR/offset/시간 점유율/codec/부모 해시를 기록하고 clip·비청취 성분은 제외 또는 마스킹한다. 기존 mixer를 재사용하며 Scaper 방식의 사건·조건 기록만 참고한다. [저자 도구·주석 방식](https://github.com/justinsalamon/scaper/blob/master/docs/tutorial.rst). |
| **시간 pooling 대안을 구분해서 검토** | R5 attention의 peak·entropy, 긴 파일에서 짧게 나온 음성 누락, 한 구간 오탐이 파일 점수를 지배하는 정도 | linear-softmax·auto-pool은 조건부 v2 후보다. 문헌은 frame별 event probability를 모으는 경우가 많아 현재 feature-level temporal pooling과 같지 않다. 무작정 R5를 대체하지 않고 같은 입력·동결 경로에서 한 변인만 비교한다. [5개 MIL pooling 비교](https://arxiv.org/abs/1810.09050), [auto-pool](https://arxiv.org/abs/1804.10070). |
| **보정 문제와 순위 문제 분리** | AUC/EER은 비슷한데 Brier/log loss 또는 FILE 결합만 나빠지는지 | 구조 비교의 CDF는 고정한다. 별도 calibration 역할에서 temperature 또는 제한된 Platt 보정을 비교할 수 있다. 양의 단조 보정은 단일 점수 순위·AUC를 고치지 못하지만 여러 점수의 FILE 결합에는 영향을 줄 수 있다. calibration 결과로 구조를 반복 선택하지 않는다. [temperature scaling 원 논문](https://proceedings.mlr.press/v70/guo17a.html). |
| **source가 정답의 대리 변수가 되는지 진단** | duration/codec/source만으로 라벨이 쉽게 구분되는지, 두 class가 있는 source 내부와 전체 AUC가 다르게 움직이는지 | 기존 잠긴 역할을 재배치하지 않는다. 사전에 고른 개발 자료에서 source·parent별 오차와 알려진 mask를 보고한다. leave-one-source-out은 새 개발 실험이며 원래 holdout의 대체물이 아니다. 소수 source 최악값만 최적화하는 Group-DRO는 작은 그룹에 과적합할 수 있어 바로 도입하지 않는다. [독립 split 재학습 논문](https://arxiv.org/abs/2204.09583). |

가창·약한 음악·비언어 사람 소리의 판정 기준을 먼저 고정해야 한다. 대회 목표가 음성인지 모든 인간 발성인지, 아카펠라를 음악 성분으로도 보는지에 따라 같은 자료의 라벨이 달라진다. 기존 대회 라벨 계약과 청취 근거를 따르며 새 dataset class 이름만으로 이를 바꾸지 않는다.

## AI 생성 자료와 Mac 사용

가장 먼저 만들 후보는 **이미 허용된 train 원본의 혼합·codec 대응쌍**이다. 새 생성 모델 다운로드 없이 CPU에서 수행할 수 있고, 원천 한쪽에만 codec를 걸어 fake/real shortcut을 만들지 않도록 양쪽에 같은 변환을 적용한다. 32 kHz mono PCM16의 15초 클립 400개는 음원 payload만 약 384 MB다. 이는 산술상 저장량이며 작업 시간·activation cache 크기의 실측이 아니다.

새 생성 모델은 확인된 빈칸을 채우는 소규모 후보로 둔다. [MeloTTS 공식 저장소](https://github.com/myshell-ai/MeloTTS)는 한국어와 CPU 추론을 지원하고 library MIT를 명시한다. 그러나 M5 Pro에서의 속도·메모리, 선택할 checkpoint 조건과 생성물 이용 범위를 이번에 검증하지 않았다. 짧은/긴 발화와 낮은 음량 음성 후보를 만들 때 기존 승인 모델·가중치의 재사용부터 확인한다. 생성기가 이미 train에 있으면 새 생성기 독립 평가로 세지 않는다.

[ACE-Step 1.5 공식 저장소](https://github.com/ace-step/ACE-Step-1.5)는 MPS/CPU 및 Apple Silicon용 MLX 실행 경로를 안내한다. 코드 MIT 표기만으로 모든 checkpoint·참조 음원·생성물의 조건을 확정하지 않는다. 24 GB에서 대량 가창을 만들 수 있다고 아직 보증할 수 없으며, 채택한다면 가장 작은 허용 모델로 단일 작업의 메모리·실시간 배율부터 측정한다. `instrumental` prompt가 실제 음성 부재를 보장하지 않으므로 생성물은 청취·성분 검사 전까지 후보 상태다. 음성 복제나 외부 유료 생성 API는 이번 제안에 필요하지 않다.

Mac에는 metadata·권리·계보 확인, CPU 순차 디코딩, hash/PCM 중복 탐지, 청취 큐, 대응 혼합, cached score 분석을 맡기는 편이 적절하다. 새 생성 모델과 PANNs 검증을 동시에 실행하지 않고 기존 18 GiB 작업 상한을 지킨다. 클라우드 GPU는 확정된 입력과 환경에서 v1 비교를 끝내는 데 우선 사용한다.

## 진행 순서 제안

1. 현재 설계 이탈 수정과 D0 기준 비교를 완료한다. 추가 자료 때문에 이 경로를 다시 흔들지 않는다.
2. 기존 자료의 판정 모호성·source별 오류를 정리하고, ESC-50 또는 작은 권리 확인 stem 패널을 **모델 선택 전에 역할 고정**해 확보한다. 모든 라벨이 없다면 presence 진단에만 쓰고 G 채택 판단은 하지 않는다.
3. 구조를 고정한 뒤 D0 대 D+ 1,477개를 같은 총 optimizer step으로 비교한다. 그 다음에만 대응 혼합을 별도 추가해 효과를 분리한다.
4. 남은 오류가 시간 집계에 집중되면 pooling, 보정에 집중되면 calibration 한 가지를 새 버전으로 비교한다. 대량 데이터·새 생성기는 앞 단계에서 드러난 빈칸에 맞춰 제한한다.

원천 페이지 응답·API 파일 크기·조건 확인 영수증은 로컬 `_workspace/panns_cnn_v1/data_method_research/primary_receipts.json`에 보존했다. 내려받은 것은 공개 설명·메타데이터뿐이며 audio payload·가중치 수집과 접근 승인 완료를 뜻하지 않는다.
