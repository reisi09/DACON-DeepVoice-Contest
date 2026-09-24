# PANNs Cnn14 v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` for inline implementation, or `superpowers:subagent-driven-development` if that execution method is explicitly selected. Steps use checkbox syntax. This document does not start implementation or provision compute.

**Goal:** 확정된 PANNs v1을 구현하고, 실제 음원·동일 조건 비교로 presence와 FILE 경로의 개선 여부를 판단한다.

**Architecture:** 보존 Cnn14에 identity-init SE를 넣고 block6 convolution만 미세조정하는 R3가 주 실험 모델이다. R0/R1/R2 대조군과 시간 가중 pooling R5를 비교하며, 기존 fake 경로·presence logistic·CDF는 고정한다. 데이터 검증·모델 연결·품질 비교·배포 검증을 순서대로 통과한다.

**Tech Stack:** 기존 Python 3.11 venv, 설치된 PyTorch·NumPy·panns_inference, stdlib JSON/hashlib/tarfile/zipfile, 기존 pytest. 신규 의존성 설치는 기본 작업에 포함하지 않는다.

**Spec:** [확정 설계 v1.0](../specs/2026-09-25-panns-cnn-expanded-design.md).

## Global Constraints

- 현재 완료된 것은 설계·workflow 문서다. 아래 체크박스는 모두 실제 실행 전 상태다.
- 코드 저장소: `/Users/choiwuseck/Documents/ChatGPT/데이콘/_workspace/lb_cpu_20260924`, 작업 branch: `cpu/lb-improvement-20260924`.
- 원본 자료: `/Users/choiwuseck/Documents/ChatGPT/데이콘`. 원본 저장소의 다른 미커밋 변경을 건드리지 않는다.
- 실행 Python: `/Users/choiwuseck/Documents/ChatGPT/데이콘/.venv/bin/python`. 코드 실행 cwd는 위 코드 저장소의 `repository`다.
- W0~W2는 CPU로 시작한다. W3은 명시적 로컬 CPU/MPS 비교이며 유료 cloud GPU를 기동하지 않는다.
- AdamW: conv LR=`1e-5`, SE LR=`1e-3`, pooling LR=`1e-4`, weight decay=`1e-4`, effective batch=4 files, parameter group별 gradient norm clip=1.0, pos_weight=1.
- 파일럿: 조건당 seed 41, 최대 200 optimizer step 또는 60분. 메모리 상한: RSS+MPS driver allocation 18 GiB. 본 학습 step은 파일럿 처리량으로 산정하고 품질 지표를 보기 전에 고정한다.
- 전체 BN·dropout·SpecAugment는 eval 상태. 미확정 라벨은 mask하며 split·parent·원음·기존 checkpoint를 보존한다.
- 원음·activation·checkpoint·원본 예측 bulk는 Git 제외 작업 폴더에 둔다. 코드·검토 문서·작은 수치/해시 영수증은 의미 있는 완료 단위로 검증 후 commit·push한다.
- 이미 수행한 Opus 검토를 반영한다. 아래 구현 결과에 대한 검토와 실제 수치 검증은 아직 하지 않았다.

## Review Focus

| 실제 실패 위험 | 담당 단계와 필요한 관측 |
|---|---|
| 짧은 음원의 반복 또는 forced tail 중복이 사라져 학습/추론이 달라짐 | W2: 실제 길이별 파일에서 baseline과 segment 시작점·sample 내용·통계 비교 |
| parent가 train/evaluation을 넘어 중복되거나 오염 cache를 재사용 | W1/W2: 실제 join 결과와 hash 불일치 자료를 거부·격리하는 경로 실행 |
| 동결 block/fc에 `no_grad()`를 적용해 SE gradient가 끊김 | W2/W3: 최종 file loss에서 SE·block6 실제 업데이트와 동결 tensor 불변 확인 |
| 미확정 라벨, dtype 승격, 긴 파일 재계산이 다른 손실을 만듦 | W2/W3: mask 불변성·기준 dtype·직접 autograd 대비 loss/gradient/update 측정 |
| 패키저가 새 PANNs를 누락하거나 runtime이 구형 경로로 조용히 되돌아감 | W7: checkpoint 필수 로드, 실제 adapter 변화의 예측 영향, 누락·변조 거부 |

## 단계별 진행 순서

`W0 기준 고정 → W1 데이터 검증 → W2 구현·캐시 → W3 Mac 파일럿 → W4 v1 비교 → W5 개발 평가·선택 → W6 조건부 확장 → W7 최종 독립 평가·패키지 검증`

W1의 원음 복구와 W2의 순수 모델 구현은 기준 identity를 고정한 뒤 독립적으로 진행할 수 있다. 동일 파일을 여러 작업자가 동시에 수정하지 않는다. W3 이전 수치 확인에는 W1에서 실제로 검증된 소규모 음원만 쓴다. W6이 불필요하면 W5에서 W7로 간다. 독립 평가가 없으면 W5의 개발 보고까지 진행하고 공식 품질 채택은 유보한다.

## 파일별 책임과 구현 예정 interface

아래 `Create` 파일·함수는 **아직 존재하지 않는 구현 계약**이다. 명령도 해당 단계 구현 후에만 실행한다. 기존 스크립트를 이미 새 CNN trainer처럼 실행할 수 있다고 보지 않는다.

| 파일 | 책임 |
|---|---|
| Create `repository/panns_adaptation.py` | SE/시간 pooling, 원본 Cnn14의 prefix/suffix 연결, trainable 범위, Torch presence readout·masked loss |
| Create `repository/panns_adaptation_run.py` | audit/cache/pilot/train/evaluate 명령. 기존 해시·역할·평가 기능을 재사용하고 실행 manifest·resume·file 예측 관리 |
| Create `repository/test_panns_adaptation.py` | 실제 checkpoint·음원의 초기 parity, gradient/update, 동결 상태, mask/cache/role·resume 거부, runtime 연결 검증 |
| Modify `repository/full_bank_submission.py` | 명시된 adapted PANNs가 있으면 strict loading하고 기존 두 presence·FILE 경로에 연결 |
| Modify `repository/package_datav4_submission.py` | 보존 DataV4 기준의 새 PANNs checkpoint·runtime·설정·hash 포함. 기존 fake 모델·CDF·presence head 유지 |

모델 interface는 `AdaptedCnn14(base, channel_se=False, temporal_pool=False, train_blocks=(6,))`로 고정한다. `prefix(waveforms_32k)`는 eval/no_grad block1~5 출력, `tags_from_block5(features)`는 A→block6→C→기존 fc의 527 확률을 반환한다. `presence_logits(tags, stem_features, heads, labels, groups)`는 파일의 모든 구간을 집계한 CPU의 두 float64 logit을 반환한다. `masked_file_loss(logits, targets, known, known_fraction)`는 effective file batch에 대한 scalar를 반환한다. 배치 축과 segment 축을 혼용하지 않는다.

driver는 `audit(config)`, `cache(config)`, `pilot(config)`, `train(config)`, `evaluate(config)`를 제공하고 공통 `--run PATH`의 고정 JSON manifest를 받는다. 경로·hash·허용 arm/seed·device·step·output을 manifest로 전달한다. CLI의 `--arm`·`--seed`는 그 목록의 항목을 선택하며 고정 값을 임의로 덮어쓰지 못한다. `audit`만 아직 고정되지 않은 입출력 경로를 읽고 검증 결과를 새 manifest에 쓴다. 나머지는 그 manifest·spec hash가 맞지 않으면 실행을 거부한다. 범용 scheduler나 plugin registry는 만들지 않는다.

## W0. 기준 모델·입력·환경 고정

**Files:** 보존 `submissions/DataV4_DurationBalanced/submit.zip`와 `.json` 영수증, `repository/local_head_experiments.py::load_reference`, `repository/package_datav4_submission.py`, 확정 spec. 산출물은 원본 프로젝트 `_workspace/panns_cnn_v1/reference_lock.json`이다.

**Consumes:** 기존 DataV4 패키지·영수증과 현재 source commit. **Produces:** archive/member hash, 6개 fake head, PANNs base, presence head, CDF, label 순서·groups, decoder/runtime 버전의 identity.

- [ ] 패키지 SHA-256과 member hash를 영수증에 대조하고 원본을 덮어쓰지 않는 검증 출력 경로를 만든다. 패키지에 없거나 정체성이 다른 파일은 원본 복구 전까지 대체하지 않는다.
- [ ] 현재 패키지의 실제 전처리·label·head/CDF·requirements와 로컬 버전을 기록한다. NumPy 2.2.6/2.4.6 차이를 숨기지 않고 W2의 기준 runtime을 고정한다.
- [ ] 과거 실제 CPU parity 10개 음원과 저장 예측의 identity를 확인해 R0 재현 입력으로 지정한다. 알려진 노출 이력을 적고 독립 평가라고 부르지 않는다.
- [ ] 사용자 최고점 제출 ZIP과의 동일성이 확인되지 않은 상태를 유지한다. R0은 검증된 DataV4 reference라는 이름으로 고정한다.

최소 해시 확인은 stdlib로 수행한다. 관련 identity가 없으면 이후 명령을 실행하지 않는다.

```python
with archive_path.open('rb') as stream:
    actual_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
if actual_sha256 != receipt['sha256']:
    raise ValueError('Reference archive identity mismatch')
```

**완료 기준:** R0에 필요한 실제 파일·설정·출처가 하나의 identity로 묶인다. 새 모델 성능을 검증한 단계가 아니다.

## W1. presence용 데이터 복구·검증

**Files:** Create `panns_adaptation_run.py`의 `audit`, `test_panns_adaptation.py`의 데이터 거부 검사. Read `data_v4/prepare_roles.py`, `data_v4/build_manifest.py`, `data_v4/stage_gpu_inputs.py`, `data_v4/verify_acquired_audio.py`.

**Consumes:** 원본 `_workspace/lb_mlaad_admission_20260924/roles-v6/{manifest.jsonl,roles.csv,splits.csv}`, `_workspace/lb_a100_preparation_20260924/audio_direct_readback/portable-audio.tar`, W0 identity. **Produces:** `data_audit.json`, `presence_rows.jsonl`, D0 train/development/selection/calibration ID 목록·누락 사유·parent component·known fraction.

- [ ] archive 전체와 실제 사용할 각 음원의 manifest hash·bytes·path·decode를 대조한다. 예전 헤더 경로 일치 기록을 현재 payload 검증으로 재사용하지 않는다.
- [ ] MUSAN 900행의 보존 원음을 허용된 프로젝트 경로·기존 저장소에서 복구한다. 접근·라이선스·새 token이 필요한 자료는 사유를 남겨 보류하고, 가능한 자료의 코드 작업은 계속한다. 새 CPU/GPU 인스턴스를 자동 생성하지 않는다.
- [ ] 안전한 member allowlist, 중복·절대경로·`..`·symlink 거부, 임시 파일 해시 확인 후 설치를 적용한다. `stage_gpu_inputs.stage`는 `r:gz`와 `audio/additional-v2/*.wav` 전용이므로 전체 tar/FLAC에 그대로 호출하거나 기존 검증을 느슨하게 바꾸지 않는다. 필요한 안전 검증 패턴만 새 reader에 적용한다.
- [ ] 추가 1,477행은 D0에서 제외한다. 모든 미확정 target을 mask하고 source/parent를 닫아 역할 충돌을 격리한다. 행이 쓰는 hash가 processed audio인지 original인지 확인하고 필드 이름만 보고 대체하지 않는다.
- [ ] negative/positive·4가지 presence 조합·source별 실제 확보 수와 제외 수를 적는다. 둘 다 없음의 자료가 없는 경우 전체 presence 품질 평가를 제한한다.
- [ ] 알려진 hash 하나를 바꾼 복사 manifest가 실제 거부되고, 역할 충돌 parent가 train으로 들어가지 않는지 확인한다. 원본 파일·라벨은 바꾸지 않는다.

**완료 기준:** 동일 D0를 모든 arm이 읽으며 결손이 기록된다. audit가 통과하지 않은 행은 학습용 목록에 없다. D0가 변경되면 관련 arm 전체를 새 버전으로 다시 실행한다.

## W2. 모델·파일 손실·block5 캐시 구현

**Files:** Create `panns_adaptation.py`; extend driver `cache`; extend `test_panns_adaptation.py`. Read `decision_heads.py`와 baseline의 `get_segment_starts`, `extract_segment`, `make_panns_segments`.

**Consumes:** W0 base checkpoint, W1 실제 음원·역할. **Produces:** A/C 모델, differentiable file loss, 검증된 block5 cache index, 초기 parity·동결/연결성 측정.

- [ ] 32개 실음원 패널의 waveform/segment→원본 tags→NumPy readout 출력을 먼저 저장한다. 부족한 유형은 미검증으로 남긴다. fixture가 없는 경우 실제 검사를 skip하되 단계 통과로 처리하지 않는다.
- [ ] 아래 두 모듈을 기존 Cnn14에 연결하고 원본 state를 strict loading한다. 처음부터 B나 새 2출력 head를 넣지 않는다.

```python
class ChannelSE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.down = torch.nn.Linear(1024, 64)
        self.up = torch.nn.Linear(64, 1024)
        torch.nn.init.zeros_(self.up.weight)
        torch.nn.init.zeros_(self.up.bias)

    def forward(self, x):
        g = 2 * torch.sigmoid(self.up(torch.relu(self.down(x.mean((2, 3))))))
        return x * g[:, :, None, None]

class TemporalPool(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2048))

    def forward(self, z):
        attention = torch.softmax((z * self.weight[None, :, None]).sum(1), dim=1)
        return z.max(2).values + (z * attention[:, None, :]).sum(2)
```

- [ ] block1~5 prefix만 `no_grad()`로 처리하고, A·block6·C·동결 fc를 통한 gradient는 유지한다. 전체 BN affine/buffer/counter는 고정한다. R2도 동결 block6를 gradient 경로 안에서 실행한다.
- [ ] `decision_heads.py`의 tag max/mean/q90·특징 순서·scale·logistic을 기준 dtype으로 재현한다. stem 값은 identity가 확인된 행만 재사용하며 다른 행은 고정 HTDemucs CPU 추출 비용을 별도로 기록한다.
- [ ] 미확정 target을 안전하게 치환한 뒤 mask된 loss를 계산한다. `NaN * 0`으로 미확정을 처리하지 않는다.

```python
def masked_file_loss(logits, targets, known, known_fraction):
    if (known_fraction <= 0).any():
        raise ValueError('Target has no known training labels')
    safe_targets = torch.where(known, targets, torch.zeros_like(targets))
    per_target = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, safe_targets, reduction='none')
    return (per_target * known / known_fraction).sum() / (2 * logits.size(0))
```

- [ ] 원래 전처리·prefix weight/code·dtype·음원 hash·segment 위치로 cache identity를 만들고 오염 cache는 재계산 전에 거부한다. 모든 segment를 보존하고 전량 캐시 용량을 실제 segment 수로 계산한다.
- [ ] 명세의 초기 출력·readout 허용 오차를 검사하고, 실제 file loss의 3~5회 update에서 SE·block6의 변화와 동결 tensor 불변을 확인한다. unknown 값을 바꿔도 손실/gradient가 같고 알려진 target이 실제 update에 참여해야 한다.

향후 검증 명령은 `DACON_LOCAL_DATA_ROOT=/Users/choiwuseck/Documents/ChatGPT/데이콘 /Users/choiwuseck/Documents/ChatGPT/데이콘/.venv/bin/python -m pytest test_panns_adaptation.py -k 'identity or connectivity or masked or cache' -q`다. 이 시점에는 실음원 fixture와 해당 검사가 구현돼 있어야 한다.

**완료 기준:** 초기 값만 맞는 것이 아니라 intended optimizer 연결·frozen 상태·file loss·cache 거부 경로가 실제 자료에서 확인된다.

## W3. Mac CPU/MPS 파일럿과 본 비교 예산 고정

**Files:** driver `pilot`, tests의 device/replay/resume 검사. **Consumes:** W2 모델·cache. **Produces:** `pilot.json`, resource trace, 고정 `run_manifest.json`.

- [ ] CPU 기준을 먼저 측정한 뒤 Cnn14 suffix 자체를 MPS로 실행한다. 기존 `AudioTagging(device='mps')`의 CPU 동작을 MPS 검증으로 보고하지 않는다.
- [ ] 같은 파일에서 microbatch 1→2→4, forward/backward/optimizer 시간, RSS/driver allocation·swap 압박을 기록한다. prefix 추출·stem 비용과 suffix 학습 비용을 분리한다.
- [ ] 파일 segment 수 분포와 직접 autograd 메모리를 확인한다. 실제 긴 파일에서 필요한 경우에만 spec의 두-pass replay를 구현한다. 두 pass의 확률·선택 index가 달라지면 gradient를 버리고 optimizer step을 하지 않는다.
- [ ] 같은 real file의 direct/replay loss·gradient·update, CPU/MPS 오차를 명세 기준과 비교한다. 허용치를 넘는 MPS 경로는 CPU와 동등하다고 부르지 않는다.
- [ ] 실행 중단→checkpoint 재개를 실제로 한 번 수행하고 계속 실행한 경우와 sampler 순서·optimizer state·예측을 비교한다. 다른 data/spec hash의 재개는 거부한다.
- [ ] 처리량으로 본 비교 step·예상 시간·disk와 장치를 산정해 manifest에 고정한다. step은 모든 arm에 동일하게 적용하고 candidate 품질 수치를 보고 늘리거나 줄이지 않는다. 이 예산이 없으면 W4를 시작하지 않는다.
- [ ] Mac 처리량이 정한 실행 예산을 충족하지 못해 A100 등 cloud 학습이 필요하면, 같은 코드·data/cache/checkpoint identity를 대상으로 업로드·원격 hash·기존 workspace 상태·실행 비용과 종료 조건을 먼저 확인한다. 현재 workflow 작성은 workspace 기동이나 새 유료 자원 생성을 수행하지 않는다.

**완료 기준:** 선택 장치에서 실음원 gradient/update·재개가 작동하고 메모리 상한 안에서 진행할 본 비교 예산이 있다. CPU 완료는 MPS 완료를 뜻하지 않는다. 200 step 파일럿을 성능 학습 완료라고 보고하지 않는다.

## W4. 확정 v1 비교 실행

**Files:** driver `train`, `evaluate`; W3 run manifest. **Produces:** arm/seed별 checkpoint, 완료 step·학습 파일 수·실제 시간, 파일별 5열 예측·coverage.

기본 비교는 R0 추론과 **4개 학습 조건 × 3 seeds = 12회 학습**이다. R0을 seed별로 중복 학습하지 않는다. 후보별 총 비용은 W3에서 측정한 cache·stem 준비 시간과 각 arm의 step 시간으로 합산하며, 과거 작은 fake head 학습 시간을 그대로 적용하지 않는다.

- [ ] R0을 고정 패널에서 재현한다. R1(block6), R2(SE), R3(SE+block6)를 각 seed 41·42·43으로 같은 D0·순서·step에서 학습한다.
- [ ] R5(시간 가중 pooling+block6)를 같은 조건으로 비교한다. R3 결과가 좋다는 이유로 R5에 더 많은 step이나 다른 데이터 비중을 주지 않는다.
- [ ] fixed final-step checkpoint만 사용한다. 중단된 arm은 동일 step까지 재개하거나 미완료로 남기고 최종 비교에서 빼되 사유를 보고한다.
- [ ] 매 arm에서 frozen tensor hash와 보존 fake 확률을 대조한다. 결과는 독립 출력 경로에 저장하고 이전 모델을 덮어쓰지 않는다.

구현 후 실행 형태는 아래와 같다. `RUN_MANIFEST`는 W3에서 생성·검토한 실제 JSON 경로를 전달한다. 이 명령을 현재 존재하는 명령으로 보고 실행하지 않는다.

```bash
/Users/choiwuseck/Documents/ChatGPT/데이콘/.venv/bin/python panns_adaptation_run.py train --run "$RUN_MANIFEST" --arm R3 --seed 41
/Users/choiwuseck/Documents/ChatGPT/데이콘/.venv/bin/python panns_adaptation_run.py evaluate --run "$RUN_MANIFEST" --arm R3 --seed 41
```

**완료 기준:** R0/R1/R2/R3/R5의 비교 가능한 결과가 있고, R3−R1의 구조 추가 효과와 R3−R0의 전체 효과를 구분할 수 있다.

## W5. 개발 선택과 독립 평가 준비

**Files:** driver `evaluate`, 기존 `graph_scoring.py`; 새 품질 보고서. **Consumes:** W4 per-file 예측, 잠긴 역할·parent·label mask. **Produces:** paired 비교, 개발 후보 lock 또는 유지 결정.

- [ ] 알려진 target별 presence AUC/Brier/log loss와 고정 CDF FILE EER을 같은 ID·mask에서 비교한다. 현재 fake EER이 그대로인지 확인한다.
- [ ] 현 평가 코드에서 관련 정답이 모두 알려진 공통 패널을 예측 확인 전에 고정한다. G용 패널 제외 수와 나머지 파일의 알려진 presence coverage를 같이 보고한다.
- [ ] parent component paired bootstrap 2,000회(seed 20260925)를 실행한다. 같은 재추출을 arm·seed에 적용하고, class가 사라져 계산되지 않은 비율을 기록한다. 과거 노출 selection/calibration은 개발 진단이다.
- [ ] source·presence 조합·실제 annotation이 있는 오류군별 퇴행을 확인한다. 개발상 G>0, 두 AUC 평균 차이≥0 조건과 spec의 불확실성을 보고한다.
- [ ] 독립 평가 자료의 출처·역할·라벨 적격성을 확인하되, 후보 예측과 품질 지표는 W6까지 모든 개발 선택을 끝낸 후 W7에서 한 번만 연다. 독립 자료가 없으면 개발 결과까지만 보고한다.

계산식은 코드의 성분 metric을 재사용한다. `voice_presence_auc`를 `voice_eer` 또는 fake AUC로 혼동하지 않는다.

```python
gain = (0.45 * (base['file_eer'] - new['file_eer'])
        + 0.05 * (new['voice_presence_auc'] - base['voice_presence_auc'])
        + 0.05 * (new['music_presence_auc'] - base['music_presence_auc']))
```

**완료 기준:** 개발 후보와 조건부 확장 필요 여부가 정해진다. 독립 평가를 확보하지 못해도 개발 보고는 완료할 수 있지만 leaderboard 향상·최종 품질 채택을 선언하지 않는다.

## W6. 결과가 요구할 때만 후속 확장

**Files:** 해당 조건을 실제 시작할 때 모델/driver/spec 버전을 함께 수정한다. 미리 placeholder 모듈을 추가하지 않는다.

- [ ] **R6:** 단독 이득을 보인 모듈이 둘 이상일 때 spec의 G 순위로 상위 두 개 조합 한 번만 비교한다. A+C 조합도 각각의 단독 결과와 비교한다.
- [ ] **R4:** 개발 오류와 block4 실제 비용이 추가 비교를 뒷받침할 때 시간·주파수 gate를 구현한다. block4 캐시를 새로 만들고 동결 block5에도 gradient를 통과시킨다.
- [ ] **D1/D2:** block6 비교 뒤 표현 수정 필요와 자원 여유가 확인되면 block5+6로 확대한다. 구조 없는 깊이 대조군을 같이 둔다.
- [ ] **D+:** 적격 판정된 신규 1,477행을 같은 구조의 D0 대조군과 비교한다. MusicArena 음성 미확정 mask와 기존 roles를 유지한다. 접근 token·이용 조건 보류 행은 학습에서 제외한다.
- [ ] **합성:** 검증된 train 부모끼리의 혼합부터 별도 데이터 실험으로 다룬다. 저음량 presence 라벨의 청취 적격성과 parent 폐쇄를 확인한다. 생성 AI 대량 수집은 실제 부족 유형과 이용 조건·비용을 확인한 뒤 별도 범위로 잡는다.

**완료 기준:** 각 후속 변경의 근거와 대조군이 있다. 결과는 W5의 개발 평가만 다시 적용하고 최종 holdout을 보며 확장하지 않는다. 구조+데이터+CDF를 동시에 바꾼 결과를 단일 구조의 효과라고 보고하지 않는다.

## W7. 최종 독립 평가·실제 제출 runtime·패키지 검증

**Files:** Modify `full_bank_submission.py`, `package_datav4_submission.py`, tests의 integration 검사. **Consumes:** 후보 lock·W0의 보존 DataV4 identity. **Produces:** 새 ZIP, member hash·실제 CLI parity·자원 보고.

- [ ] W6까지 마친 한 architecture와 seed 41을 잠근 뒤 새 독립 평가를 한 번 수행한다. G의 95% 하한>0, 두 presence AUC 차이의 95% 하한≥0을 충족해야 개선 근거로 채택한다. 평가 부족·넓은 구간은 기존 모델 유지와 판단 유보다. 평가 결과를 보고 다시 모델을 바꾸면 그 패널은 개발 노출로 기록하고 새 독립 자료 없이는 재확증하지 않는다.
- [ ] `Engine`에 선언된 adaptation을 strict loading하는 경로를 연결한다. 설정에 adaptation이 있는데 파일이 없거나 hash가 다르면 실패시킨다. 기존 PANNs로 조용히 fallback하지 않는다.
- [ ] `package_datav4_submission.py::replacements/package`의 trained member 목록과 runtime 목록에 adaptation checkpoint·설정·`panns_adaptation.py`를 포함한다. 원래 6개 fake head, presence head, CDF를 W0 identity 그대로 보존한다.
- [ ] 기존 패키저가 목록에 없는 `model/trained/` 파일을 버리는 동작 때문에 새 PANNs가 빠지지 않는지 실제 ZIP을 검사한다. 7,144행 고정 조건을 가진 별도 `package_full_bank_submission.py`를 이 작업의 대체 패키저로 쓰지 않는다.
- [ ] 실제 음원으로 연구 reference와 새 패키지 CLI를 비교한다. 순서·부분집합 변경 시 파일별 결과 불변, fake 확률·`max_nopresence` 불변, presence 변화의 `max/noisy_or` 전달, adapter 누락·변조 거부를 확인한다.
- [ ] 실제 runtime requirements·archive 크기·member hash·원치 않는 자료 포함 여부를 확인한다. CPU/MPS 통과를 공식 L4 환경 검증으로 보고하지 않는다. 필요 cloud 검증은 실행 전 저장소·업로드를 확인하고 허용된 기존 workspace만 재사용한다.
- [ ] 결과 ZIP·hash·변경점·검증 범위를 보고하고 공식 제출은 별도 사용자 지시가 있을 때만 한다. 개발 후보의 QA용 패키지 생성과 공식 품질 채택을 구분한다.

**완료 기준:** 선택 모델이 실제 실행 경로에서 동작하고, 보존 모델로 되돌릴 수 있는 별도 패키지·identity가 남는다. 파일 존재나 ZIP 생성 성공만으로 완료하지 않는다.

## 진행 기록·검토·Git

각 단계 종료 시 `실제로 끝낸 작업 / 측정 근거 / 남은 제한 / 다음 단계 진입 여부`를 보고한다. 새로운 코드의 meaningful check가 통과한 완료점에서 해당 파일만 stage하고 전체 staged diff를 검토해 commit·push한다. 원본 데이터·checkpoint와 무관한 변경은 포함하지 않는다. 실패한 검사를 삭제하거나 형식 검사로 바꾸지 않는다.

계획 자체는 spec 각 항목을 W0~W7에 연결하고, 현재 존재하는 경로와 앞으로 만들 경로를 구분해 자체 검토한다. 설계 해시와 workflow 해시는 [v1 고정 기록](../../evidence/panns_cnn_review_20260925/design_v1_lock.json)에 저장한다. 이 문서 작성으로 어떤 체크박스도 자동 완료되지 않는다.
