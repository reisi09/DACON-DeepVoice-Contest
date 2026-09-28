# 신규 방법 후보 감사 (2026-09-18)

현재 운영 기준선은 **공식 DACON 제출 점수 0.7479394074의 MERT12**이며, live 검증된 production artifact다. 별도로 E6에서 얻은 **로컬 음악 test EER 8.995%**는 해당 실험과 split에 조건부인 진단 수치이며 공식 점수와 같은 지표가 아니다.

## 범위와 판정 기준

기존 이력에는 이미 AST_LAST_BLOCK(약 7M trainable), V4 P2 AST 전체 backbone fine-tuning, S0/S1/S2 EAT full fine-tuning이 있다. 따라서 “새 spectrogram ViT를 end-to-end로 fine-tune한다” 자체는 독립 방법이 아니다. 실질적인 신규성은 backbone 이름이 아니라 **patch/localization supervision, source-disjoint controls, 그리고 file pooling objective**를 바꾸는 데 있다. 아래 후보는 그 감독 신호와 시간 집계의 결함을 겨냥한다. 음성 anti-spoofing 결과는 음악 일반화의 증거로 사용하지 않는다.

## 순위

| 후보 | 방법 | 새로 검증하는 것 | 공개 사전학습/구현 | 판단 |
|---|---|---|---|---|
| A | **SSAST patch 기반 spectrogram encoder** | masked patch SSL 목표가 기존 AST/EAT fine-tuning과 다른가 | 공식 PyTorch 코드와 checkpoint 공개 | 표현 사전학습 대조군 |
| B | **BEATs encoder** | acoustic-tokenizer 목표가 다른 forensic detail을 보존하는가 | Microsoft 공식 코드와 checkpoint 공개 | 일반 오디오 사전학습 대조군 |
| C | **RawNet2/AASIST raw waveform branch** | mel/MERT 밖의 파형 단서를 source-disjoint 음악에서도 포착하는가 | 공식 코드/checkpoint 공개 | speech 중심, 음악 검증 필요 |
| D | **codec-aware contrastive MERT12** | codec identity를 지우면서 fake/real identity를 보존하는가 | CLAD 공식 구현은 음성용 | E3 consistency의 후속 가설 |
| E | **wavelet/STFT forensic residual head** | 주파수 잔차가 generator/source shortcut 없이 남는가 | 공식 보조 구현 공개 | speech 중심, 음악 증거 없음 |
| F | **temporal coherence / localized multi-task** | 파일 평균 대신 시간적 불연속과 짧은 구간을 학습하는가 | 1차 논문·모델카드 근거 부족 | 낮은 신뢰도 탐색 |
| G | **patch/localization supervision** | 기존 AST/EAT backbone에 window/patch 감독과 다른 pooling objective를 붙이면 개선되는가 | 기존 checkpoint 재사용 | backbone 변경이 아닌 감독·집계 변경 |
| H | **multi-resolution spectrogram CNN control** | Transformer 문맥보다 local spectral inductive bias가 필요한가 | 추가 artifact 없이 소형 CNN | architecture control |

## 근거와 한계

### 1. SSAST

SSAST는 AST 구조를 재사용하지만, patch/frame masking에 대해 생성·판별 SSL을 수행한다. 공식 저장소는 patch 기반 Base가 89M이고, AudioSet+LibriSpeech 사전학습 checkpoint와 fine-tuning 경로를 제공한다. 저자 문서도 patch 기반은 audio, frame 기반은 speech에 권장한다고 명시한다. 따라서 현재 AST와 비교할 핵심은 “Transformer 대 CNN”이 아니라 **사전학습 목표와 patch granularity**다. 공개 성능은 AudioSet/음성 등 downstream 평균이며 DACON 음악 위조 성능을 보장하지 않는다.

코드: [SSAST pinned repository](https://github.com/YuanGongND/ssast/tree/a1a3eecb94731e226308a6812f2fbf268d789caf), [paper](https://arxiv.org/abs/2110.09784).

### 2. BEATs

BEATs는 acoustic tokenizer와 audio Transformer를 반복적으로 서로 distill하고, tokenizer가 만든 이산 acoustic label을 예측하도록 학습한다. 이는 MERT의 음악·오디오 표현을 단순히 다른 layer에서 읽는 것과 다른 pretraining objective다. 공식 Microsoft 저장소는 Iter1–3 및 AS20K 변형의 tokenizer/model checkpoint와 MIT 라이선스를 명시한다. 다만 AudioSet 중심 일반 오디오 분류 성능은 deepfake forensic sensitivity와 동일하지 않으며, 고수준 의미를 강조하면서 미세 생성 artefact를 버릴 가능성도 있다.

코드: [BEATs pinned README](https://github.com/microsoft/unilm/blob/50224e387211f15ac6a3b2685730b9a0c850f145/beats/README.md), [primary paper](https://arxiv.org/abs/2212.09058).

### 3. RawNet2/AASIST

RawNet2는 raw waveform을 직접 입력해 자동 특징을 학습하고, AASIST는 sinc front-end/RawNet2 계열 encoder 뒤에 spectro-temporal graph attention을 둔다. 공식 논문과 구현은 ASVspoof logical-access 음성 공격을 대상으로 한다. 따라서 “voice EER가 낮다”는 이유로 음악에 채택할 수 없다. 음악에서는 stereo 처리, 16 kHz 재표본화 손실, 긴 파일 windowing을 별도로 고정해야 한다.

코드: [RawNet2 pinned implementation](https://github.com/eurecom-asp/rawnet2-antispoofing/tree/e259c539afd0252028b2d7d273db567f7f2d9a0f), [RawNet2 paper](https://arxiv.org/abs/2011.01108), [AASIST pinned implementation](https://github.com/clovaai/aasist/tree/a04c9863f63d44471dde8a6abcb3b082b07cd1d1).

### 4. Codec-aware contrastive/domain-invariant training

최근 robustness 연구는 압축·변형, 특히 neural codec에서 detector가 취약하다고 보고한다. CLAD는 음성 deepfake detector에 contrastive learning을 적용해 manipulation robustness를 높이는 공식 구현을 제공한다. 이 프로젝트의 E3 consistency는 codec pair를 넣었지만 seed 평균 개선이 작고 ensemble 방향이 반대였다. 따라서 다음 단계는 “consistency loss를 더 세게”가 아니라, 동일 source의 REAL/FAKE와 codec 변형을 함께 묶어 **fake/real identity는 유지하고 codec identity는 제거하는** 소규모 contrastive pilot이어야 한다. 이 가설은 speech 논문에서 옮겨온 것이므로 음악에서 검증 전에는 주장할 수 없다.

코드: [CLAD pinned implementation](https://github.com/CLAD23/CLAD/tree/518efcb80c8dafa602cd74f52628337373fbc3be), [robustness study](https://arxiv.org/abs/2503.17577), [CLAD paper](https://doi.org/10.1016/j.ins.2025.121928).

### 5. Wavelet/STFT forensic residual

`gan-police/audiodeepfake-detection`은 wavelet-packet과 STFT로 생성기별 안정적 주파수 패턴을 찾고, unseen generator 일반화를 겨냥한 경량 detector를 공개했다. 이는 현재 spectral residual branch와 방향은 비슷하지만, MERT semantic feature를 보조하는 고정 wavelet expert로 두면 비교가 더 명확하다. 해당 결과는 speech vocoder/gan 계열 중심이며, 음악 generator와 source-disjoint한 성능을 입증하지 않는다.

코드: [pinned implementation](https://github.com/gan-police/audiodeepfake-detection/tree/eaed30201667c33bc7e15cfb3a3af95d13feaec6), [paper repository description](https://github.com/gan-police/audiodeepfake-detection).

### 6. Temporal coherence / localized multi-task

파일 단위 평균은 짧은 편집 구간을 희석한다. 2026년 공개 저장소는 시간적 일관성 전문가를 표방하지만, 이 감사에서는 이를 뒷받침하는 1차 논문·모델카드와 독립 benchmark를 확인하지 못했다. lower-confidence exploratory work로 두고, E5와 구분되는 temporal multi-task head로만 검증한다.

코드: [2026 implementation pinned repository](https://github.com/stbiadmin/audiodeepfake_public/tree/3e9908915338f6e8a367ff1cc35eb91fbc4cdbc2). 저장소가 제공하는 pretrained classifier의 speech/music 적용 범위는 모델 파일별로 다르므로 그대로 음악 품질 근거로 쓰지 않는다.

### G–H. Patch/localization supervision과 CNN control

기존 AST/EAT backbone을 다시 end-to-end fine-tune하는 것만으로는 새 제안이 아니다. 새 요소는 log-mel/PCEN patch token에 window BCE·patch consistency·경계 인접 window objective를 추가하고, file pooling을 사전등록하는 것이다. window label이 파일 label에서 유도됐다면 fake boundary 검출 주장은 진단 수준이다. 같은 입력에서 multi-resolution STFT/mel 소형 CNN을 control로 두면 Transformer 장거리 문맥과 local spectral inductive bias를 분리할 수 있다.

## 권장 3단계 controlled pilot

모든 후보는 같은 source/group-disjoint train/calibration/dev/test, 같은 4초 window와 파일 aggregation, 같은 seed 41/42/43, 같은 MERT12 baseline으로 비교한다. 현재 test를 선택에 재사용하지 않는다.

1. **Representation/architecture pilot:** MERT12 frozen head, SSAST, BEATs, end-to-end patch model, multi-resolution CNN을 비교한다. 동일 split·window·trainable head budget·pooling 후보 수를 고정하고 frozen/last-block fine-tuning 상태만 사전등록한다.
2. **Forensic complement pilot:** 1위 representation에 RawNet2/AASIST raw branch 또는 wavelet/STFT residual 중 하나만 붙인다. branch 단독, MERT12 단독, gated fusion 세 arm만 둔다. 성능 외에 generator/source별 paired difference와 codec slice를 기록한다.
3. **Robustness/localization pilot:** 1위 조합에 codec-invariant contrastive loss 또는 temporal multi-task 중 하나만 추가한다. codec 변형은 train에만 노출하고, 완전히 보지 않은 codec/source를 holdout으로 둔다. 파일 EER와 window recall을 함께 보고, window recall만 올리고 file EER가 악화되면 채택하지 않는다.

### 자원·배포 경계

GPU 시간은 공개 checkpoint와 입력 길이, batch, gradient policy가 달라 **측정 전에는 알 수 없다**. 현재 MERT12의 22.4 GiB/약 60분 offline L4 목표와 호환되는지 각 후보를 먼저 20–50개 파일 smoke benchmark로 측정한다. checkpoint 다운로드·라이선스·가중치 재배포 조건은 실제 artifact의 LICENSE/model card를 확인한 뒤 기록하며, 논문이 공개됐다는 사실만으로 상업적 재배포 권한을 추정하지 않는다.

Open questions: SSAST/BEATs checkpoint의 실제 16 kHz·stereo/긴 음악 입력 비용, 각 artifact의 최신 redistribution terms, source-disjoint DACON holdout에서의 성능은 아직 측정하지 않았다.
