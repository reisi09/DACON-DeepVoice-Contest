# PANNs Cnn14 내부 구조·미세조정 검토

상태: **코드·데이터 조사와 Claude 토의 완료. 아래 구조는 제안이며 아직 구현·학습하지 않았다.** 사용자는 Cnn14 내부 CNN 구조와 미세조정을 포함하고 Claude도 활용하도록 요청했다. 이번 요청은 CNN 구조와 학습 경로를 바꾸는 설계 작업으로 분류한다. 모델 성능 개선이 입증됐다는 뜻은 아니다.

후속 요청에 따른 [확장 설계](superpowers/specs/2026-09-25-panns-cnn-expanded-design.md)가 추가됐다. 채널·시간·주파수 후보와 미세조정 깊이, 데이터·캐시·평가 조건을 명세했다. 아래 내용은 최초 토의 기록으로 보존하며, class weight 등 변경된 결정은 확장 설계를 따른다.

## 실제 Claude 사용

기존 검증된 Claude Code 2.1.281 실행 파일로 `claude-opus-5-5`, `--effort high`를 지정해 두 번 검토했다. 두 응답 모두 실제 `modelUsage`가 `claude-opus-5-5`이고 `is_error=false`였다. 프로세스 시간은 253.988초, 59.899초다. 내장 도구·MCP·사용자 정의 훅을 비활성화한 읽기 전용 요청에 필요한 코드 발췌와 실험 수치만 제공했다. 원음·가중치·인증값을 전송하지 않았다.

첫 실행은 API 호출 전 `Not logged in`으로 종료됐다. 최소 환경에서 제외한 `USER`를 기존 값 그대로 전달하자 `auth status`와 실제 요청이 성공했다. `LOGNAME`이나 `__CF_USER_TEXT_ENCODING`만 추가한 경우에는 실패했다. 인증값을 읽거나 재발급하지 않았고 전역 설정도 바꾸지 않았다. 실패·성공 기록은 원본 프로젝트의 `_workspace/panns_architecture_review_20260925/`에 보존했다.

## 확인한 현재 경로

`원음 → 기존 정규화·분할 → 32 kHz → Cnn14의 6개 conv block → 주파수 평균·시간 max+mean → fc1 → 527개 sigmoid 태그 → 24개 PANNs/stem 통계 → 고정 logistic presence 2개 → 고정 CDF FILE 결합`

현재 `panns_inference.AudioTagging`은 CUDA 외 장치를 CPU로 선택한다. 앞서 전체 파이프라인을 MPS로 실행했어도 PANNs 자체는 CPU였다. **Cnn14의 MPS forward/backward는 아직 확인하지 않았다.** 기존 `v5_experts.PresenceExpert`는 새 2출력 헤드를 쓰고 다수 CNN 파라미터를 학습하므로 이번 비교에 그대로 사용하면 변경 범위가 섞인다. 해당 로더의 `weights_only=False`도 재사용하지 않는다.

PANNs의 AudioSet 사전학습 가중치를 재사용하는 근거는 [PANNs 논문](https://arxiv.org/abs/1912.10211)에 있다. 제안하는 SE는 채널별 반응을 입력에 따라 조절하는 구조다. [SE 논문](https://arxiv.org/abs/1709.01507)은 그 일반적인 구조 근거이며, 이 대회에서의 성능 향상을 입증하지는 않는다.

PANNs 변경은 presence AUC/CPS와 FILE 경로에 영향을 줄 수 있다. 현재 평가 코드에서 Voice/Music EER은 정답 presence로 마스킹하므로, fake 출력이 고정된 이번 범위에서 이 두 EER은 직접 달라지지 않는다.

## 데이터 조사에서 확인한 문제

- 로컬 `portable-audio.tar` 헤더의 정규 파일은 43,517개이며, 현재 bank의 음원 43,516개 경로가 모두 일치한다. 이번 조사에서는 archive 내부 음원 내용 해시를 재검사하지 않았다. 추출 시 개별 manifest 해시 검증이 필요하다.
- 그 bank의 presence 패턴은 음성 없음·음악 있음 5,345개, 음성 있음·음악 없음 28,664개, 둘 다 있음 7,618개, 음성 미확정·음악 있음 1,889개다. 둘 다 없는 사례는 0개다.
- 전체 manifest 44,424행에는 **MUSAN 잡음 900개가 둘 다 없음으로 기록**돼 있다. 이들은 fake 판별용 bank와 portable 음원 묶음에서 빠져 있다. 현재 역할은 train 527, selection 49, calibration 48, development 276개다. 기록된 원본 경로는 원격 `/data/...`이며, 확인한 로컬 경로에서는 원음을 찾지 못했다.
- 따라서 해당 잡음 자료의 원음·라벨을 확인하고 presence 학습·평가 대상에 포함해야 한다. 확보 전 실행한 검사는 비대상 잡음에 대한 부재 판정을 검증한 것으로 보고하지 않는다. 900개가 모두 한 원천이라는 편향도 별도로 기록한다.
- 추가 1,477개는 현재 전부 train 역할이다. 595개는 음성 있음·음악 없음, MusicArena 882개는 음악 있음·음성 미확정이다. 미확정을 0으로 바꾸지 않는다. 고정된 train 역할을 임의로 바꿔 독립 holdout이라고 부르지 않는다.
- 기존 presence 특징 8,748행은 파일당 통계 24개만 담고 있다. CNN 미세조정용 block5 활성값은 없으므로 원음에서 새로 추출해야 한다. frozen prefix의 가중치·코드·정규화·분할·리샘플러·음원 해시를 캐시 정체성에 포함한다.

## 제안하는 최소 구조와 비교

**block5의 pooling 출력과 block6 사이 한 곳에 SE를 추가한다.** 1,024채널을 64채널로 줄였다 복원하며, 게이트는 `2 * sigmoid(W2(ReLU(W1(mean(x)))))`다. bias를 포함한 마지막 선형층만 0으로 초기화해 시작 게이트가 1이 되도록 한다. 첫 선형층은 기본 초기화를 사용한다. 이 계산의 신규 파라미터 수는 132,160개다. 초기 출력 재현은 구현 후 실제 가중치와 음원에서 검증해야 한다.

| 조건 | 학습 대상 | 비교 목적 |
|---|---|---|
| R0 | 없음 | 기존 모델 재현 |
| R1 | block6의 convolution 가중치 | 구조 변경 없는 미세조정 |
| R2 | SE만 | SE 단독 효과 |
| R3 | SE + block6의 convolution 가중치 | **R1 대비 구조 추가 효과** |

block6의 두 convolution 가중치는 코드상 56,623,104개다. 이는 산술 계산이며 Mac 학습 메모리·속도를 실측한 값이 아니다.

모든 조건에서 frontend, block1~5, 모든 BN 파라미터·running buffer, fc1, 527출력층, 기존 presence logistic, stem 특징과 FILE 결합기를 고정한다. backbone은 `eval()` 상태에서 gradient가 필요한 파라미터만 켜 dropout·SpecAugment·BN 상태 변화가 섞이지 않도록 한다. 같은 원음·역할·seed·학습 step을 사용한다. R1과 R3의 공통 convolution 학습률도 같게 둔다. 증류, 새 2출력 헤드, 시간 attention pooling은 첫 비교에 추가하지 않는다.

파일의 실제 구간을 집계해 기존 presence 통계와 logistic을 미분 가능한 Torch 계산으로 재현하고, 최종 presence **logit**에 알려진 라벨만 사용하는 BCEWithLogits를 적용한다. 양성 파일의 모든 구간을 양성으로 복제하지 않는다. 클래스 가중치는 train에서만 계산한다. 마지막 checkpoint를 사용하며 평가 지표를 보고 epoch를 고르지 않는다. 학습률과 전체 step은 아직 실행 계획에서 확정하지 않았다.

## Opus 의견 중 채택·수정한 부분

1. block4·5 두 위치를 한꺼번에 바꾸자는 제안은 block5 한 곳으로 줄인다. 동결된 block5 출력 캐시를 재사용할 수 있다.
2. Opus가 처음 제안한 static channel scale 대조군은 후속 진단으로 미룬다. 첫 질문은 일반 미세조정에 SE를 더했을 때 실용적 이득이 있는지다. SE의 입력 의존성이 순수한 원인이라고 주장하지 않는다.
3. Opus는 정답 presence 대입을 성능 상한이라 부른 주장을 철회했다. 고정 결합기에서 변수를 바꿔 보는 진단일 뿐, 달성 가능한 개선 폭이나 수학적 상한이 아니다.
4. stem 특징과 연관된 오류를 CNN이 보정할 수 없다는 단정도 철회했다. 실제 tag-level gradient와 고정 분류기의 민감도를 측정해야 한다.
5. 표본 재추출은 공유 부모가 닫힌 component를 기본 단위로 삼고 source별 분석을 병행한다. seed 변동이나 임의의 0.5 threshold 오류 수만으로 채택하지 않는다.
6. Opus는 고정 CDF를 쓰는 FILE EER을 선택 지표에서 제외하자고 했다. 이 부분은 그대로 채택하지 않는다. 고정된 실제 파이프라인에서 FILE 퇴행 여부는 사용자 목표에 필요한 평가다. 다만 이는 고정 결합기와의 호환성을 포함한 결과이며 PANNs 표현력만의 평가가 아니다. CDF 재적합은 별도 실험으로 분리한다.

## 구현 후 필요한 검증과 중단 조건

- 실제 음원 32개를 먼저 사용해 CPU 초기 출력 재현, Torch/NumPy presence 계산 일치, 알려진·미확정 라벨 마스크를 검사한다. 단일 구간·forced tail·혼합 음원·잡음을 포함하되 자료가 없으면 그 범위를 미검증으로 표시한다.
- 실제 오류 파일에서 최종 logit과 527 tag logit까지의 gradient를 측정한다. 손실만 감소하고 의도한 CNN까지 gradient가 전달되지 않는다면 전체 학습으로 확대하지 않는다.
- SE 첫 선형층은 초기 마지막 층이 0이라 첫 step gradient가 0일 수 있다. 3~5회 실제 갱신에서 누적 경로가 열리는지 확인한다. 첫 step의 값만 보고 연결이 끊겼다고 판정하지 않는다.
- 학습 후 모든 동결 파라미터와 BN buffer 해시를 대조한다. 변하면 해당 실험을 무효로 처리한다.
- CPU frontend와 MPS suffix의 실제 forward/backward를 확인한다. batch 1→2→4 순서로 측정하고 MPS 할당·driver 메모리·프로세스 RSS를 각각 기록한다. MPS fallback을 자동 활성화하지 않는다. CPU float64 readout으로 이동해도 gradient가 연결되는지 확인한다.
- 소규모 단계는 한 seed·최대 200 optimizer step으로 제한한다. 전체 비교 예산은 실제 처리량·메모리와 자료 검증 결과를 보고 정한다. 새 클라우드 GPU를 만들지 않는다.
- 채택은 component별 paired 비교, 음성·음악 presence AUC, Brier·보정, source별 변화와 FILE EER을 함께 보고 결정한다. 독립 평가가 확보되지 않으면 개발 진단으로만 남긴다. 문서·모델 간 합의·초기 출력 재현은 성능 증명이 아니다.

이 문서는 요청한 Claude 토의와 조사의 기록이다. 제안의 구현 승인이나 학습 완료 기록이 아니다. 모델 코드·가중치·공식 제출물은 변경하지 않았다.
