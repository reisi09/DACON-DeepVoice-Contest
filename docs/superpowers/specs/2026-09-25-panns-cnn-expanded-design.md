# PANNs Cnn14 구조·미세조정 확장 설계

상태: **설계 v1.0 확정 — 2026-09-25. 새 모델 구현·학습·성능 검증은 아직 하지 않았다.** 사용자의 “설계를 확정해라” 요청에 따라 아래 구현 기준과 실험 순서를 고정한다. MERT, DF-Arena, HTDemucs, fake 분류기를 바꾸거나 새 GPU를 시작하는 작업은 포함하지 않는다. 이전 [최소 구조 검토](../../PANNS_CNN_OPUS_REVIEW_2026-09-25.md)를 세 가지 구조 후보와 학습·데이터·평가 계약으로 확장한다.

## 확정 결정과 v1 범위

| 항목 | 확정 내용 |
|---|---|
| 주 실험 모델 | **R3: Cnn14 block5 뒤 identity-init SE + block6 convolution 미세조정**. 학습 대상은 56,755,264개이며 성능 우승 모델로 미리 채택하는 것은 아니다. |
| 필수 비교 | R0(보존 기준), R1(block6만), R2(SE만), R3(SE+block6)를 먼저 비교하고, 같은 기준에서 R5(시간 가중 pooling+block6)를 비교한다. |
| v1 구현 모듈 | A(SE)와 C(시간 가중 pooling), 파일별 masked loss, block5 캐시, 학습·평가·실제 runtime 연결. |
| 조건부 후속 | R4(block4 gate), R6(검증된 두 모듈 조합), D1/D2(block5+6 미세조정), D+(추가 데이터). 최초 구현에 미리 넣지 않는다. |
| 동결 대상 | block1~5, 전체 BN·frontend·fc1·527출력층·기존 presence logistic·stem·CDF·다른 모델. R2에서는 block6도 동결한다. |
| 기준 artifact | 보존 DataV4 패키지와 그 영수증을 해시로 고정한다. 사용자 최고점 제출물과의 정확한 동일성은 미확인이다. |
| 품질 기준 | 같은 패널의 presence AUC와 고정 CDF의 FILE EER, component 단위 paired 비교. 독립 평가 없이는 개발 후보까지만 선정한다. |
| 기본 배포 형태 | seed 41의 단일 PANNs. seed 42·43은 변동 확인용이며 가장 좋은 seed를 사후 선택하지 않는다. |
| 로컬 실행 | CPU 기준 검증 후 Cnn14 자체 MPS forward/backward를 확인한다. 조건당 최대 200 step/60분 파일럿과 메모리 합계 18 GiB 상한을 유지한다. |

설계 확정과 실제 학습 입력·실행 예산 확정은 구분한다. D0 적격 파일·해시, runtime 버전과 최대 200에포크·early stopping 정책은 [실행 workflow](../plans/2026-09-25-panns-cnn-v1-workflow.md)에 따라 학습 전에 run manifest에 고정한다. 파일럿 처리량은 소요 시간 추정에 사용하며 에포크 상한을 줄이는 데 사용하지 않는다. 실제 종료 에포크와 step 수는 조건마다 다를 수 있다. 이를 측정하지 않고 전체 소요 시간이나 데이터 준비 완료를 선언하지 않는다. 품질 결과를 본 뒤 구조·손실·split·채택 기준을 바꾸려면 새 설계 버전과 비교를 만든다.

## 1. 목표와 변경 범위

목표는 음성·음악 존재 여부를 더 잘 판별하고, 기존 FILE 결합기에 연결했을 때의 퇴행을 막는 것이다. 가창, 음성+배경음악, 짧은 음성, 약한 음악, 둘 다 없는 잡음에서 어떤 변화가 생기는지 구분한다. 특정 오류의 원인이 구조라고 확정한 상태는 아니다.

기존 경로와 변경 후보의 위치는 다음과 같다.

```text
원음 → 기존 gain 처리·분할·32 kHz 리샘플링 → log-mel/BN0
 → block1…4 → [B: 시간·주파수 gate]
 → block5 → [A: 채널 SE gate]
 → block6 → 주파수 평균 → 시간 max + [C: 시간 가중 평균]
 → 기존 fc1 → 기존 AudioSet 527 sigmoid
 → 기존 PANNs 통계 15개 + 고정 stem 통계 9개
 → 기존 voice/music logistic → 기존 CDF FILE 결합
```

대괄호는 비교할 위치이며, 처음부터 세 모듈을 함께 넣는다는 뜻이 아니다. **첫 후보는 A+block6 미세조정이다.** 확장은 기존 block5 캐시를 쓰는 C를 먼저 비교하고, B는 개발 오류와 block4 비용 측정에 근거해 추가 가치가 있을 때 실행한다. B와 C는 같은 block6 미세조정 대조군에 각각 비교하고, 단독 이득이 확인된 모듈만 조합한다.

고정 대상은 frontend, 명시적으로 해제하지 않은 convolution, 모든 BN의 affine 파라미터와 running buffer·counter, fc1, fc_audioset, 기존 presence logistic의 계수·표준화값·특징 순서, stem 특징, FILE 결합기, 나머지 인코더와 fake 출력이다. 동결된 계층도 학습 모듈 뒤에 있으면 역전파를 통과시켜야 한다. `requires_grad=False`와 `no_grad()`를 혼동하지 않는다.

입력부터 최초 학습 모듈 **앞까지**의 동결 prefix만 `no_grad()` 또는 검증된 캐시로 처리한다. 최초 학습 모듈 이후에는 동결된 block·fc·logistic도 입력에 대한 gradient가 필요하므로 autograd 경로를 유지한다.

PANNs만 바꾸면 현재 평가 코드의 Voice/Music fake EER은 직접 개선되지 않는다. 두 EER은 정답 presence로 대상이 정해지고 fake 출력은 고정되기 때문이다. 평가 대상은 presence AUC·보정과 FILE EER이다. 공식 ADS/CPS와 로컬 지표의 명칭·범위는 구분한다.

## 2. 실제 해상도와 세 가지 구조

기존 구간은 16 kHz의 64,600 sample, 약 4.0375초다. 짧은 파일은 기존 코드대로 반복하고, 긴 파일은 마지막 시작점을 추가하므로 끝 구간이 겹칠 수 있다. 32 kHz에서는 129,200 sample이다. n_fft=1024, hop=320, center=True일 때 log-mel 시간축은 404칸이다. 다음 값은 확인한 코드로 계산한 크기이며 이번 작업에서 실제 forward로 측정한 값은 아니다.

| pooling 뒤 위치 | 채널 | 시간×주파수 | FP32 segment 캐시 |
|---|---:|---:|---:|
| block3 | 256 | 50×8 | 400 KiB |
| block4 | 512 | 25×4 | 200 KiB |
| block5 | 1,024 | 12×2 | 96 KiB |
| block6 | 2,048 | 12×2 | 192 KiB |

각 축은 넓은 수용영역을 가진 학습 특징이다. 시간 칸을 정확한 사건 경계, 주파수 칸을 특정 Hz 대역으로 해석하지 않는다. block5의 두 주파수 칸에 복잡한 주파수 attention을 넣는 대신 B는 block4에서 비교한다. block4도 충분한 해상도라는 보장은 없다.

### A. block5 채널 SE — v1 주 실험 구조

입력 `X[C=1024,T,F]`를 시간·주파수 평균한 뒤 `1024→64→1024`의 두 선형층과 ReLU를 통과시킨다.

`Y = X * 2*sigmoid(W2(ReLU(W1(mean_TF(X)))))`

두 선형층에 bias를 사용한다. W1은 기본 초기화, W2와 두 번째 bias는 0으로 초기화한다. 초기 gate는 1이며 신규 파라미터는 **132,160개**다. 첫 step에서 W1 gradient가 0일 수 있으므로 여러 실제 갱신에 걸쳐 연결성을 확인한다. 마지막 층과 별도 residual 계수를 동시에 0으로 만들어 경로를 막지 않는다.

가설은 사전학습 채널의 상대적 기여를 대회 presence 자료에 맞추면 block6만 학습하는 것보다 도움이 될 수 있다는 것이다. 가창·혼합음 개선을 미리 보장하지 않는다. 채널별 보정의 근거는 [SE](https://arxiv.org/abs/1709.01507)이며, 여기서 쓰는 identity 초기화와 삽입 위치는 프로젝트 설계다.

### B. block4 시간·주파수 gate — 조건부 후속 구조

입력 `X[C=512,T,F]`에서 채널 평균과 채널 최댓값을 구해 `[2,T,F]`로 쌓는다. 이를 `Conv2d(2,1,kernel=3×3,padding=1,bias=True)`에 넣고 `Y=X*2*sigmoid(conv(descriptor))`를 적용한다. convolution의 가중치·bias를 0으로 시작해 초기 gate를 1로 만든다. 신규 파라미터는 **19개**다.

이 후보는 채널마다 다른 보정 대신 위치별 보정을 모든 채널에 공유한다. [CBAM](https://arxiv.org/abs/1807.06521)의 채널 집계·공간 attention을 참고한 작은 변형이며, 논문 구조를 그대로 재현한 것이 아니다. 주파수가 4칸뿐인 조건에서 유효한지는 실험으로 판단한다.

B의 `3×1`·`1×3` 축별 파생 실험은 기본 행렬에서 제외한다. 주된 비용인 block5 역전파·block4 캐시는 그대로여서 커널을 더 줄이는 것만으로 실행 비용 문제가 해결되지 않는다. B의 결과는 시간·주파수 보정을 함께 추가한 실용적 효과이며 어느 축이 원인인지 확정하지 않는다.

block4 캐시를 쓰며 block5는 동결하되 gradient를 통과시킨다. **B 뒤의 block5를 `no_grad()`로 실행하거나 기존 block5 캐시로 대체하면 B가 학습되지 않으므로 금지한다.** A와 비교해 메모리·연산량이 더 들 수 있다.

### C. 시간 가중 평균 — v1 필수 비교 구조

block6의 주파수 평균 후 특징을 `Z[C=2048,T]`라고 한다. `a=softmax(wᵀZ, time)`를 구하고 기존 `max_time(Z)+mean_time(Z)`를 `max_time(Z)+sum_time(a*Z)`로 바꾼다. bias 없는 `w[2048]`를 0으로 초기화한다. 처음에는 균등 가중치라 원래 평균과 수학적으로 같고 신규 파라미터는 **2,048개**다. 부동소수점 연산 순서 차이는 실제 수치 검증 대상이다.

max 경로는 유지한다. 별도 2출력 head나 Transformer를 추가하지 않는다. 가설은 평균에 희석되는 일부 구간에 다른 가중치를 주는 것이 도움이 될 수 있다는 것이다. 시간축 12칸의 clip 내부 보정이며, 파일 간 문맥이나 정밀한 사건 위치 추정은 제공하지 않는다. attention entropy와 가중치 집중을 진단하되 높은 집중 자체를 오류나 성공으로 판정하지 않는다.

## 3. 미세조정 깊이와 비교 순서

| 단계 | 조건 | 학습 대상 | 핵심 비교 |
|---|---|---|---|
| 기준 | R0 | 없음 | 원본 재현 |
| 첫 비교 | R1 | block6 conv | 미세조정 자체: R1−R0 |
| 첫 비교 | R2 | A만 | 구조 단독: R2−R0 |
| 첫 비교 | R3 | A + block6 conv | **우선 비교: R3−R1** |
| v1 후속 필수 비교 | R5 | C + block6 conv | R5−R1 |
| 조건부 확장 | R4 | B + block6 conv | R4−R1 |
| 조건부 | R6 | 단독 효과를 보인 두 모듈 + block6 conv | 각 단독 후보와 비교 |
| 조건부 | D1/D2 | block5+6 conv / 선택한 단일 모듈 + block5+6 conv | 깊이 효과와 구조 효과 분리 |

R6은 개발 평가에서 순위를 고정한 상위 두 모듈의 조합 한 개만 비교한다. 단독 이득이 둘 미만이면 생략한다. 깊이 확장은 오류 분석·처리량에 근거해 필요할 때만 실행한다. 전체 조합 탐색, block1부터 전체 학습, dilation·stride 변경, 다중 backbone은 이번 설계에서 제외한다. 표의 ID는 조건 식별자이며 숫자 순서가 실행 순서는 아니다.

block6 conv 파라미터는 **56,623,104개**, block5는 **14,155,776개**, 합계는 **70,778,880개**다. BN은 포함하지 않는다. A의 경량성은 추가 모듈만을 뜻하며, block6 미세조정 전체가 13만 파라미터 학습이라는 뜻이 아니다. FP32 AdamW의 파라미터·gradient·1차/2차 상태 네 묶음만 계산해도 block6 약 **864 MiB**, block5+6 약 **1,080 MiB**다. 전체 backbone·활성값·임시 버퍼·OS 메모리는 별도다.

같은 비교 안에서는 데이터 ID·순서·seed·파일 수·학습률과 최대 에포크·early stopping 정책을 맞춘다. 실제 종료 에포크와 optimizer step 수는 함께 보고한다. R1과 R3의 convolution 학습률은 동일하다. 각 후보를 원래 pretrained 가중치에서 새로 시작하며, 다른 후보의 학습 가중치를 이어받지 않는다. R0은 seed별 중복 학습을 하지 않는다. 깊이 비교에서는 같은 데이터로 새 block6 대조군도 함께 두며, 이전 단계에서 평가로 고른 checkpoint를 그대로 유리한 출발점으로 쓰지 않는다.

기본 배포 후보는 **사전에 정한 seed 41의 단일 PANNs**다. seed 42·43은 학습 변동을 확인하며, 가장 좋은 seed를 사후 선택하거나 세 모델 평균을 조용히 배포하지 않는다. 따라서 seed 평균 지표와 실제 배포 후보의 지표를 따로 보고한다. ensemble은 추론 비용과 결합 방식이 바뀌므로 이번 기본 후보에 포함하지 않는다.

## 4. 학습 경로와 실제 손실

모델은 `eval()`을 유지해 dropout·SpecAugment·BN 갱신을 끈다. 학습할 파라미터만 명시적으로 `requires_grad=True`로 설정한다. 증강은 후속 데이터 비교에서만 다룬다.

527 sigmoid → 클래스 그룹 max → 파일 max/mean/q90 → 기존 표준화·logistic을 Torch로 재현한다. NumPy `quantile(.9)`의 기본 linear 보간과 동일한 값을 사용한다. q90은 연속적인 soft pooling으로 바꾸지 않는다. max·정렬의 tie에서는 derivative가 유일하지 않으므로 결정된 Torch 규칙을 기록하고 작은 입력 변화 진단과 실제 gradient 측정을 병행한다.

전부 float64로 승격한 뒤 통계를 계산하면 기존 NumPy 경로와 달라질 수 있다. 로컬 NumPy 2.4.6에서 float32 입력의 max·mean·q90 결과 dtype이 float32인 것을 확인했고, mean을 먼저 float64로 승격했을 때 값이 달라지는 작은 수치 사례도 확인했다. 따라서 기준 runtime의 NumPy 버전·입력 및 통계 dtype을 먼저 고정하고 그대로 재현한다. logistic의 표준화·내적은 기존 코드처럼 float64로 처리한다. 과거 보존 requirements의 NumPy 2.2.6과 로컬 2.4.6을 같은 환경으로 간주하지 않는다. 패키지 runtime에서의 전체 parity는 여전히 별도 검증 대상이다.

2026-09-26 원인 재현에서 확인한 3개 파일·18개 구간은 Mac과 Linux의 16 kHz 정규화 음원·구간 시작점이 같았지만, 32 kHz 리샘플링 결과는 최대 6.705522537231445e-8 달랐다. 같은 CPU 모델에서 Linux 입력으로만 바꾸어도 기존 tag 오차가 재현됐고, 과거 CUDA 출력과의 잔여 오차는 8.643e-7 미만이었다. 이 결과를 전체 파일의 검증 통과로 확대하지 않는다.

A100 실행 전에는 같은 Linux runtime에서 같은 43개 파일의 CPU 기준 cache를 생성한다. 각 장치에서 보존된 원본 Cnn14 forward와 adapted prefix/suffix·readout의 일치 검사를 유지하고, 그 뒤 CPU와 CUDA를 tag·presence 최대 절대 오차 1e-5 및 gradient 상대 L2 오차 1e-3으로 비교한다. 두 adapted 경로의 일치만으로 통과시키지 않는다. 기존 Mac gold는 그대로 보존하고 플랫폼 간 수치 차이를 별도 진단으로 기록한다. Mac과 Linux의 구간 시작점·구간 수·tensor shape가 다르면 미세한 수치 차이로 취급하지 않고 학습을 차단한다. 원본 전처리나 허용 오차를 바꾸지 않으며, 전체 검증을 통과하기 전에는 본학습을 시작하지 않는다.

위 장치 비교에서 frozen stem 입력은 실제 학습·선택·평가에 쓰는 CUDA cache 값으로 고정한다. 2026-09-26 저장된 43개 파일·137개 구간을 재생한 결과, 독립 CPU/CUDA stem을 쓸 때 presence 최대 오차는 9.821374400675875e-5였지만 동일한 CUDA stem을 쓰면 9.665041580259048e-8이었다. tag 최대 오차는 7.748603820800781e-7이었다. 독립 Demucs 추출 차이는 `frozen_frontend_diagnostic.json`에 보존하며 PANNs 학습 차단 기준과 분리한다. 이 검사는 동결 stem을 입력받는 PANNs 경로를 검증하며 전체 Engine이나 독립 CPU/CUDA Demucs 출력의 일치를 뜻하지 않는다. 기존 tag·presence·gradient 허용 오차와 원본 forward 검사는 유지한다.

파일별 두 presence logit에 **masked BCEWithLogits**를 적용한다. 음성·음악은 독립 target이며 서로 배타적인 softmax를 사용하지 않는다. file 양성을 모든 segment 양성으로 복제하지 않는다. 미확정 라벨은 0이 아니며 손실에서 제외한다.

두 target 모두 미확정인 파일은 supervised sampler에서 제외한다. 나머지 train 파일을 균등하게 섞어 순회한다. target k의 알려진 비율 `r_k=N_known_k/N_train`을 해당 고정 train 자료에서 계산하고, 실제 effective batch 파일 수를 B라고 할 때 `L=(1/(2B))*sum_k(sum_i(mask_ik*BCE(logit_ik,y_ik))/r_k)`를 사용한다. mask 자체가 달라져도 batch마다 알려진 파일 수로 나누지 않는다. 특정 파일의 가중치가 함께 뽑힌 미확정 파일 수에 따라 달라지는 문제를 줄이기 위한 정규화다. 한 batch에서 알려진 target이 없으면 해당 항은 0이며, train에 알려진 양·음성이 없으면 해당 target 학습을 시작하지 않는다.

**첫 구조 비교의 pos_weight는 1이다.** 이전 최소안의 class weight 제안을 변경한다. 고정 logistic·CDF에 연결할 확률을 학습하면서 class weighting까지 한꺼번에 바꾸지 않기 위해서다. D+에서도 pos_weight=1을 유지하고 `r_k`는 그 자료의 알려진 비율로 기록한다. segment 수가 많은 파일에 더 큰 파일 가중치를 주지 않는다. label-known 비율로 정규화해도 missing label이 무작위라는 보장은 없으며, 미확정 집단의 성능을 대변한다고 해석하지 않는다.

수치 파일럿의 시작 설정은 AdamW, conv LR=`1e-5`, gate LR=`1e-3`, C의 w LR=`1e-4`, weight decay=`1e-4`, effective batch=4 files이다. block5, block6, 각 adapter를 분리한 parameter group별 gradient norm clip=1.0을 사용하고 clip 발생률·실제 norm을 기록한다. 새 gate의 gradient가 global clipping을 통해 기존 convolution의 step까지 축소하는 일을 분리하기 위한 선택이다. 상수 학습률로 시작하며 평가 결과를 보고 후보별로 다르게 조정하지 않는다. 이는 검증된 최적값이 아니라 비교 가능한 시작값이다. 수치 발산으로 공통 설정을 고쳐야 하면 해당 비교 전체를 새 실험으로 다시 시작한다.

첫 비교에는 증류나 tag auxiliary loss를 넣지 않는다. 최종 logit이 미분 가능하더라도 중간 527 sigmoid가 포화하거나 그룹 max가 극소수 tag에만 gradient를 줄 수 있다. 실제 오류 파일에서 tag logit → block6 → gate별 gradient와 optimizer update를 확인한다. 경로가 사실상 작동하지 않으면 긴 학습으로 넘어가지 않는다. 고정 readout이 병목이라는 근거가 생겼을 때만 별도 readout 실험을 다시 설계한다.

### 긴 파일의 메모리 계약

먼저 실제 파일의 segment 수 분포를 집계하고, 한 파일의 모든 segment를 이용하는 직접 autograd의 메모리를 측정한다. 임의로 앞부분만 쓰거나 segment를 잘라 기존 파일 집계를 바꾸지 않는다. 메모리 측정상 긴 파일이 들어가지 않을 때에만 다음 두 번의 forward를 구현한다.

1. 가중치를 고정한 채 segment microbatch를 순회해 모든 tag 확률을 수집한다. 학습 모델의 activation graph는 저장하지 않는다.
2. CPU의 원래 dtype을 유지한 tag 확률 복사본을 leaf로 삼아 동일한 파일 집계·logistic·masked loss를 계산하고 `dL/dp`를 구한다. 통계는 기준 dtype, logistic은 float64를 사용한다.
3. 같은 가중치·같은 segment·같은 microbatch 분할로 다시 forward한 뒤, 각 microbatch의 확률에 저장된 `dL/dp`를 주어 backward한다. leaf gradient를 model dtype으로 바꾸는 수치 오차를 측정한다. 두 번째 pass의 확률도 모아 비교한다.
4. optimizer step 전에 두 pass의 확률과 파일 통계에서 활성화된 class-group max·file max·q90의 index/tie 집합이 일치하는지 검사한다. CPU에서는 동일 출력의 bitwise 반복을 기준으로 삼고, MPS에서는 확률 max absolute difference ≤1e-6과 동일 index/tie 집합을 요구한다. 실패하면 누적 gradient를 버리고 step을 수행하지 않는다. GPU의 eval 모드만으로 결정성을 가정하지 않는다.
5. effective batch의 모든 파일에서 검사가 끝난 후 parameter group별 clip·optimizer step을 한 번 실행한다. 두 pass 사이와 한 파일의 중간에는 optimizer를 갱신하지 않는다.

이는 두 pass가 같은 함수를 계산할 때 chain rule을 사용하는 메모리 절약 방식이다. tolerance 안의 MPS 차이는 수치 근사이며 bitwise 동일한 연산이라고 부르지 않는다. 직접 autograd와 손실·gradient·업데이트를 실제 다구간 파일로 비교해야 하며, 비교 전에는 정확한 구현이라고 보고하지 않는다. batch의 target별 분모는 replay 시작 전에 고정한다. CPU readout은 float64를 지원하지 않는 MPS 경로와 분리하지만 중간에서 gradient를 버리지 않는다. AMP, dropout, 확률적 증강은 최초 parity 비교에 넣지 않는다.

## 5. 데이터와 오염 방지

### D0: 구조 비교에 사용할 고정 자료

추가 1,477개를 제외한 기존 manifest 계열을 시작점으로 한다. 현재 roles-v6 역할을 보존하고 적격·해시 검증을 통과한 행만 사용한다. fake-task bank를 presence 데이터 목록으로 그대로 사용하지 않는다. D0에서 확인할 제외·누락 목록을 먼저 고정하고, arm마다 다른 가용 파일을 쓰지 않는다.

MUSAN 복구 등으로 D0가 바뀌면 기존 arm과 이어 비교하지 않고 새 데이터 버전으로 모든 관련 arm을 다시 실행한다. 평가 중간에 확보된 유리한 파일만 추가하지 않는다.

portable archive에는 bank 음원 43,516개 경로가 있지만 음성·음악이 모두 없는 MUSAN 900개는 빠져 있다. 900개는 train 527, development 276, calibration 48, selection 49개로 기록돼 있다. 보존 원본이나 적법한 동일 자료에서 복구하고 manifest 개별 해시·crop·codec 경로를 확인한다. 새로 가공해야 하면 새 파생 ID·해시를 만들고 원래 파일인 것처럼 대체하지 않는다. 역할은 부모 원본에서 상속한다.

900개 모두가 반드시 복구돼야 연결성 파일럿을 할 수 있는 것은 아니다. 다만 각 행의 복구·제외 사유와 역할별 실제 수를 남겨야 한다. train과 평가에 검증된 ‘둘 다 없음’ 자료가 전혀 없으면 그 유형의 품질 개선을 판정하거나 전체 presence 채택 평가로 넘어가지 않는다. MUSAN 한 원천의 결과만으로 다양한 비음성 잡음에 일반화한다고 말하지 않는다.

### D+와 합성: 구조 선택 이후 데이터 효과를 분리

신규 LibriSpeech 449개와 MLAAD 146개는 음성 있음·음악 없음, MusicArena 882개는 음악 있음·음성 미확정이다. D0에 이를 추가한 D+는 **같은 구조·같은 최대 에포크와 early stopping 정책의 D0 대조군**과 비교한다. 자료량이 달라지므로 실제 optimizer step 수와 학습 파일 노출 수도 함께 보고한다. train 역할은 학습 편입 완료를 뜻하지 않으며, 법적·대회 이용 적격 상태가 보류된 행은 그대로 제외한다. 조건을 새로 승인했다고 간주하지 않는다.

현재 bank의 미확정 음성 라벨 1,889개 전체에 같은 mask 규칙을 적용한다. MusicArena만 예외 처리하지 않는다. 청취 판정은 음원·부모·source별로 추적하고, 모델 예측은 판정 대상 우선순위를 정하는 보조 수단으로만 쓴다. 특정 개수 청취만으로 전체 라벨 정확성을 보증하지 않는다.

추가 합성은 허용된 **train 부모끼리** 음성·음악·잡음을 섞는 방식부터 검토한다. source와 parent가 확인된 음원만 쓰고 gain, offset, SNR, mixing seed와 부모 전체를 남긴다. 한 parent라도 held-out 역할이면 train 파생물을 만들지 않는다. 합성물을 부모와 다른 역할에 배치하지 않는다. 저음량 성분은 물리적으로 존재해도 청취상 라벨이 모호할 수 있으므로 단순 OR로 presence를 확정하지 않는다. 청취·규칙상 불확실한 target은 마스킹하거나 합성 후보에서 제외한다. presence 파생자료의 fake label은 새로 추정하지 않는다.

새 AI 음원 생성 모델·TTS 대량 생성은 기본 실행에 넣지 않는다. 기존 자료와 단순 혼합에서 확인된 빈칸이 남을 때에만 라이선스·생성 원천 편향·실제 청취 품질·CPU/MPS 비용을 검토한다. 생성 모델의 판정이나 다른 모델의 동의만으로 라벨 적격성을 인정하지 않는다.

### 역할과 독립성

train은 gradient와 알려진 라벨 비율 계산에 사용한다. development는 오류 분석, selection은 early stopping과 사전에 정한 arm 비교, calibration은 별도 보정 실험에서만 사용한다. selection을 여러 에포크의 모델 선택에 반복해서 쓰므로 독립 평가로 해석하지 않는다. 현재 패널의 과거 학습·선택 노출을 조사해 기록한다. 역할 이름이 holdout을 뜻하지 않는다. 이미 본 패널은 개발 진단으로 취급한다.

공유 원음, crop, remix, 파생 음원, hash 중복을 연결한 parent component를 역할 경계 검사와 paired bootstrap 단위로 사용한다. 한 component가 여러 역할에 걸치면 해당 비교에서 격리하고, 고정 역할을 몰래 재배치하지 않는다. 새 source 평가를 만들더라도 AudioSet 등 사전학습 노출까지 배제했다고 표현하지 않는다. 독립 평가 자료가 준비되지 않으면 개발 후보까지만 선정한다.

## 6. Mac 24 GB 실행·캐시 계약

현재 `AudioTagging` wrapper는 CUDA 외 장치를 CPU로 선택한다. 앞선 전체 Engine MPS 실행은 **Cnn14 MPS 학습 검증이 아니다.** 직접 Cnn14 모듈을 인스턴스화하고 CPU frontend + 명시적 CPU/MPS suffix를 시험한다. 설치된 패키지 파일을 수정하지 않으며 silent MPS fallback을 켜지 않는다. 미지원 연산은 좁은 CPU 경로에서 연결성을 확인하거나 해당 MPS 실험을 중단한다.

32개의 실제 파일로 단일·다중 구간, forced tail, 짧은 반복, 혼합음, 잡음을 확인한다. 자료가 없는 항목은 미검증으로 남긴다. microbatch는 1→2→4 segment 순서로 올리고 effective file batch는 gradient accumulation으로 유지한다. CPU thread는 4에서 시작한다. 같은 파일·같은 경계·동일 모델의 CPU/MPS 시간, forward/backward, optimizer step, 파일 길이별 처리량을 구분해 기록한다.

초기 연결성 파일럿은 조건당 seed 41, 최대 200 optimizer step 또는 60분 중 먼저 도달한 시점까지다. 이는 품질 학습 예산이 아니라 장치·수치·메모리 확인 상한이다. 메모리 이상, 비유한 손실·gradient, cache 불일치, 동결 상태 변화가 있으면 더 일찍 중단한다. 파일럿 처리량과 checkpoint 비용은 예상 시간·저장소 산정에 사용하며, 시간에 맞추려고 본학습 에포크를 줄이지 않는다.

2026-09-26 사용자의 고에포크·early stopping 요청에 따라 본학습은 최대 200에포크, patience 15, min_delta 0.0001로 시작한다. 학습 전 epoch 0과 각 전체 에포크가 끝난 뒤 selection의 파일 단위 presence BCE를 계산한다. 음성·음악 target별로 알려진 라벨의 평균 손실을 구한 뒤 두 평균을 동일 비중으로 합친다. development·calibration·test는 조기 종료 기준에 넣지 않는다. 이 설정이 최적이라는 실험 근거는 아직 없다.

실제 최저 검증 손실의 가중치를 보존하고, patience 기준 손실보다 0.0001 이상 낮아지면 대기 횟수를 초기화한다. 그만큼의 개선 없이 15에포크가 지나거나 최대 에포크에 도달하면 학습을 끝내고 최적 가중치로 평가한다. 모든 arm·seed에 같은 규칙을 적용하되 실제 종료 에포크는 달라도 된다. 각 조건의 학습 에포크, 최적 에포크, 종료 사유를 함께 보고한다. 검증 손실 감소만으로 ADS 개선을 주장하지 않는다.

현재 모델·optimizer·sampler·RNG와 최적 모델·patience·마지막 검증 에포크를 하나의 checkpoint에 저장한다. 재개는 현재 학습 상태에서 이어가고 평가는 최적 가중치를 사용한다. 에포크 마지막 update 뒤 검증 전에 중단됐다면, 재개 후 그 검증부터 수행한다. 기존 4시간 학습 상한은 없애지만 승인된 클라우드 세션의 절대 종료 시각은 유지한다. 시간 만료·시그널 중단은 수렴이나 early stopping으로 기록하지 않으며, 미완료 조건을 완료 조건과 품질 비교하지 않는다.

프로세스 RSS와 MPS driver allocation을 각각 기록하고, **합계 18 GiB**를 초기 중단 상한으로 둔다. 통합 메모리에서 중복 집계될 수 있으므로 실제 사용량 추정치가 아니라 보수적인 작업 상한이다. CPU에서는 MPS 항을 0으로 둔다. 24 GB에서 실행 가능하다는 보증은 아니며, 메모리 압박과 swap 증가가 지속되면 상한에 닿기 전에도 안전 지점에서 checkpoint 후 중단한다. 다음 실행에서 batch를 줄이거나 CPU로 전환하고 동일 장치 비교를 새로 확인한다. 측정·재개 상태에는 model, optimizer, RNG, sampler 위치, 데이터·역할 해시, 완료 파일과 segment 수를 포함한다.

캐시는 float32·구간 단위로 시작한다. A/R1은 block5, B와 block5+6 미세조정은 block4 캐시를 사용한다. block6를 학습하는 C는 block5 캐시가 필요하다. 동결 구간 뒤의 activation만 저장하고, 상위 구조나 augmentation을 바꾸면 관련 캐시를 무효화한다. block4 캐시를 block5 캐시처럼 이름만 바꿔 쓰지 않는다.

키에는 음원 내용 해시, decoder·gain·분할·resampler 설정과 버전, 시작점·반복 규칙, prefix 코드·checkpoint 해시, dtype, 캐시 위치를 넣는다. role은 별도 manifest에 고정해 같은 음원이 다른 역할로 섞이는지 검사한다. 파일당 구간 수가 달라 총 용량을 파일 수만으로 확정하지 않는다. 100,000 segment 기준 block5 약 9.16 GiB, block4 약 19.07 GiB에 index·metadata가 추가된다. 전량 생성 전에 실제 segment 수·여유 디스크를 확인하고 필요한 역할부터 만들며, block4·5 전량 중복 생성은 피한다.

기존 8,748행의 24개 presence 통계는 CNN 미세조정 activation을 대신하지 못한다. 그 안의 stem 특징은 원음 해시·gain·분리기 버전·ID가 정확히 맞는 행만 재사용한다. 나머지는 고정 HTDemucs로 다시 추출해야 한다. 가용 fake 확률도 동일 checkpoint·전처리·ID임을 확인한 뒤 FILE 평가에 재사용한다.

## 7. 품질 판단과 실험 종료

파일 단위 예측을 보존하고 같은 ID·label mask에서 paired 비교한다. 각 target의 지표는 그 정답이 알려진 파일에서만 계산하고, arm별로 ID·coverage가 같은지 확인한다. 미확정은 음성 없음으로 포함하지 않는다. presence AUC, Brier, log loss, 음성만/음악만/둘 다/둘 다 없음, 실제 annotation이 있는 가창·혼합음·저음량 집단을 보고한다. 메타데이터가 없는 집단은 새로 확인하기 전까지 ‘가창’ 등으로 단정하지 않는다. source에 두 class가 없으면 AUC는 미정의로 표시하고 Brier·손실·표본 수를 사용한다. MUSAN 둘 다 없음의 selection은 manifest상 49개뿐이며 실제 확보 수는 더 적을 수 있어 넓은 불확실성을 예상해야 한다.

FILE은 저장된 **고정 CDF**에서 평가한다. CDF 재적합은 calibration 역할의 적격 자료로만 하는 별도 실험이며 최초 구조 비교에 섞지 않는다. 정답 presence 대입은 민감도 진단일 뿐 달성 가능한 성능 상한이 아니다.

현재 `graph_scoring.py`의 가중치에 따르면 다른 fake 출력과 평가 집합이 고정된 경우 로컬 총점 차이는 다음과 같다.

`G = 0.45*(FILE_EER_base − FILE_EER_new) + 0.05*Δvoice_AUC + 0.05*Δmusic_AUC`

이 식은 현 코드의 비교용 지표다. 라벨이 불완전하거나 같은 지표의 평가 집합이 arm 사이에 다르면 G를 공식 점수처럼 계산하지 않고 성분별 결과·coverage를 보고한다. leaderboard ADS/CPS 예상치로 변환하지 않는다.

G를 비교할 때는 예측을 보기 전에 정한 **전체 정답이 있는 공통 파일 패널**에서 현 평가 코드가 정의하는 각 mask를 동일하게 적용한다. 전체 패널에서 빠진 미확정 라벨 파일 수·source도 공개하고, 이 파일들의 알려진 presence 평가는 별도로 유지한다. G용 적격 패널이나 양·음성 표본이 없으면 구조 후보 순위는 presence 진단으로만 기록하고 종합 채택은 유보한다.

개발 선택은 seed 41·42·43을 각각 비교하고, 공유 parent component 단위 paired bootstrap 2,000회(seed 20260925)의 95% 구간을 함께 제시한다. 재추출에서 class가 사라져 지표가 계산되지 않는 비율도 기록한다. 세 seed를 독립 음원 표본처럼 합치지 않는다. bootstrap의 한 번의 재추출 안에서는 같은 component 선택을 모든 seed·arm에 적용하고 seed 평균 차이를 계산한다. 이는 고정 seed 집합에서의 표본 불확실성이며 더 넓은 최적화 변동을 보증하지 않는다.

eval 모드이므로 주된 seed 차이는 sampler 순서와 A의 첫 선형층 초기값이다. 세 seed가 dropout·다양한 데이터 분할·모든 초기화 조건을 포괄한다고 보고하지 않는다.

여러 후보를 본 개발 구간의 CI를 확증적 유의성으로 제시하지 않는다. 기본 채택 원칙은 G의 양의 개선과 두 presence AUC의 비퇴행이다. 최종 한 후보를 고정한 뒤, 새 독립 평가에서 G의 95% 하한이 0보다 크고 각 AUC 차이의 95% 하한이 0 이상일 때 개선 근거로 채택한다. 이는 퇴행 허용 폭을 임의로 두지 않은 보수적 정책이며, 구간이 넓으면 실패로 단정하지 않고 **판단 유보·기존 모델 유지**로 결론낸다. 불완전 라벨이면 전체 채택 판단을 유보한다. source별 퇴행도 별도 확인하고 전체 평균으로 숨기지 않는다.

R6에 진입할 개발 후보는 seed 평균 G>0이면서 두 presence AUC의 평균 차이가 각각 0 이상인 단독 모듈로 한정하고 G 순으로 정한다. 점수는 반올림 전 값으로 비교하며 같으면 신규 파라미터가 적은 후보를 우선한다. 이는 개발용 선택 규칙이지 독립 검증 통과가 아니다. 최종 holdout의 CI와 판정은 **실제로 배포할 seed 41 모델 대 R0**에 적용하고, 구조 기여를 설명할 때는 같은 seed의 R1과의 차이도 제시한다.

불확실한 후보를 통과시키려고 평가를 반복하거나 margin·패널·seed를 사후 변경하지 않는다. 기존 모델과 후보를 모두 보존하고, 실제 제출은 별도 사용자 지시가 있어야 한다.

## 8. 구현 시 검증해야 할 실패 모드

아래는 **향후 검증 명세**이며 이번에 통과했다는 기록이 아니다.

| 방지할 결함 | 의미 있는 관측·통과 조건 |
|---|---|
| 구조 삽입만으로 기준 출력 변화 | 실제 checkpoint·32파일에서 초기 527 tags와 두 presence 확률, FILE 출력 비교. 같은 CPU FP32 경로에서 max absolute error ≤1e-6을 목표로 사전 고정; 초과하면 원인을 규명하고 기준을 사후 완화하지 않는다. |
| Torch readout 계산 변경 | 먼저 같은 24개 특징을 넣은 CPU float64 logistic에서 logit 차이 ≤1e-9, 확률 차이 ≤1e-10을 확인한다. tag 통계 자체는 기준 dtype·tie·한 구간·q90 보간을 검사하고, [0,1] 범위 통계의 절대 오차를 해당 dtype의 machine epsilon 두 배 이내로 둔다. 전체 logit 오차는 그 통계 허용 오차에 각 `abs(coef/scale)`를 곱해 합한 값+1e-9 이내인지 확인한다. 이는 float32 통계의 반올림과 float64 내적 오류를 분리하기 위한 기준이다. |
| adapter 연결 끊김·optimizer 누락 | 실제 알려진 오류 파일의 loss에서 의도한 모듈로 finite gradient가 흐르고 해당 optimizer가 가중치를 바꾸는지 확인. A의 W1은 첫 step만으로 판정하지 않고 3~5 step 누적을 본다. 임의 최소 gradient 값을 성능 기준으로 쓰지 않는다. |
| 알 수 없는 라벨이 음성 없음으로 학습 | 미확정 target 값을 0/1로 바꿔도 loss·gradient가 같고, 같은 파일의 알려진 target은 loss와 의도한 파라미터 업데이트에 기여해야 한다. |
| 동결 모델이 몰래 바뀜 | update 전후 동결 tensor·BN buffer 전체 해시 일치. fake 출력과 FILE의 `max_nopresence` 입력은 같은 파일의 기존 값과 동일. presence 변경이 `max`·`noisy_or`로 전달되는 실제 통합 사례를 확인한다. CDF plateau 때문에 최종 FILE이 항상 변해야 하는 것은 아니다. |
| 긴 파일 replay가 다른 목적함수 학습 | 실제 다구간 파일에서 직접 autograd와 replay의 loss·global gradient 및 한 optimizer update 비교. gradient 차이의 L2 비율 ≤1e-5, 상대 오차가 불안정한 near-zero 값은 절대 오차를 병기하고 지배적인 오차 원인을 분석한다. |
| CPU/MPS가 다른 계산 실행 | 같은 입력·가중치에서 forward와 gradient 비교. FP32 probability max diff ≤1e-5, gradient L2 상대차 ≤1e-3을 파일럿 계약으로 두고 CPU 기준 repeatability부터 확인. 실패 시 MPS 동등성으로 보고하지 않고 CPU 사용 가능성을 분리한다. |
| 캐시·역할 오염 | 음원·prefix·전처리 해시 불일치 캐시의 실제 거부, parent가 역할을 넘는 파생 자료의 실제 격리, 재개 시 데이터·sampler 불일치 거부. |
| 패키지에서 구형 모델로 우회 | adapted checkpoint strict loading, 실제 adapter weight 변화에 따른 예측 변화, 저장·재로드 전후 예측 재현, 파일 순서·부분집합 변경에도 각 파일 예측 불변. 후보 선택 후 실제 제출 실행 경로까지 검증. |

검증 파일 개수나 pytest 성공 수를 품질 근거로 쓰지 않는다. 특히 구조 초기 재현, gradient 연결, 다른 모델의 동의는 일반화 성능 증명이 아니다. 수치 허용치는 leaderboard 채택 기준과 구분한다.

## 9. 연결할 기존 코드와 산출물

새 production 파일은 실제 구현 단계에서 필요한 만큼만 만든다. 설계상 추가 핵심은 Cnn14 adapter와 학습 driver이며, 임의 plugin registry나 범용 실험 프레임워크는 만들지 않는다.

- [`baseline_original/script.py`](../../../repository/baseline_original/script.py): 분할·짧은 반복·리샘플링 규칙 재사용. home으로 label을 복사하는 부작용은 새 연구 loader에 가져오지 않는다.
- [`decision_heads.py`](../../../repository/decision_heads.py): 기준 tag·stem·logistic의 유일한 의미 기준. Torch 학습 mirror는 이 경로와 실제 parity를 유지한다.
- [`data_v4/presence.py`](../../../repository/data_v4/presence.py): 고정 stem 특징 재사용·추출 시 현재 전처리와 ID 계약 확인.
- [`data_v4/prepare_roles.py`](../../../repository/data_v4/prepare_roles.py): 기존 역할·부모 관계 확인. 잠긴 역할을 재생성해 바꾸지 않는다.
- [`graph_scoring.py`](../../../repository/graph_scoring.py): 공식 형식·라벨 mask·부분 metric 처리 재사용.
- [`full_bank_submission.py`](../../../repository/full_bank_submission.py): 선택된 PANNs만 교체하는 최종 통합 위치. 나머지 fake 경로·FILE 결합을 고정한다.

체크포인트는 기존 안전한 `weights_only=True` 로딩 경로를 사용하고 state_dict key를 엄격하게 검증한다. 원본 checkpoint는 보존한다. 기존 `v5_experts.PresenceExpert`의 새 2출력 head·광범위 해제·`weights_only=False` 경로를 그대로 재사용하지 않는다.

필수 결과물은 입력·역할·cache 해시가 있는 실행 manifest, arm별 trainable/frozen 명세, loss·메모리·처리량 로그, 파일별 예측과 coverage, paired 비교 보고서, checkpoint·재개 상태, 선택 후보의 package parity다. 원음·대용량 activation·가중치를 Git에 넣지 않는다. 모델 자체의 일반 근거는 [PANNs 논문](https://arxiv.org/abs/1912.10211)이며 여기의 확장 후보 성능을 입증하는 자료는 아니다.

## 10. 실제 Opus 검토와 반영

기존 Claude Code 2.1.281로 확장 초안을 `claude-opus-5-5`, `--effort high`에 전달했다. 실제 `modelUsage`는 `claude-opus-5-5`, `is_error=false`, API 기록 109,570ms, 프로세스 시간 111.109초였다. 도구·MCP·사용자 정의 훅을 차단한 요청에 이 설계와 필요한 코드 사실만 전달했으며 원음·가중치·인증값은 포함하지 않았다. 이전 두 차례 최소안 검토와 별개인 추가 검토다.

반영한 수정은 다음과 같다.

1. 첫 비교의 class weighting 제거, 미확정 라벨 비율을 이용한 고정 분모, parameter group별 clipping으로 학습 비교를 명확히 했다.
2. replay의 두 pass 출력·max/q90 선택 집합 확인과 step 취소 조건을 추가했다. eval 상태만으로 MPS 결정성이 보장된다는 표현을 없앴다.
3. 불완전 라벨 문제를 피하려고 예측 전에 정한 공통 완전 라벨 패널을 G용으로 지정하고, 미확정 파일의 알려진 지표·coverage는 별도 유지한다. R6 선택 조건과 실제 배포 seed 41을 명시했다.
4. 메모리 합계를 보수적 작업 상한으로 쓰고, B보다 C를 먼저 평가하며 B의 축별 파생 실험을 제외했다. D0 변경 시 관련 arm을 모두 다시 실행하도록 했다.
5. 최종 step checkpoint와 미완료 arm 처리, 실제 FILE 입력의 변동·불변 항목을 확인하는 통합 검증을 명시했다.

별도 자체 검토에서는 NumPy의 실제 통계 dtype을 확인해 float64 일괄 승격을 피하도록 수정했다. 이는 작은 수치 probe이며 Cnn14 추론·역전파·품질 검증은 아니다.

그대로 채택하지 않은 의견도 있다. Opus는 AUC 항이 fake AUC일 가능성을 제기했지만 실제 `graph_scoring.py`에서 두 항은 presence AUC임을 확인했다. “static scale을 convolution이 흡수하므로 R3−R1은 입력 의존성만의 효과”라는 해석도 채택하지 않았다. 파라미터화·최적화 경로·정규화가 함께 달라져 그 효과만 분리한 실험은 아니다. CDF 재적합을 모든 arm의 필수 부가 평가로 넣자는 제안은 이번 최소 비교에서 제외하고 후속 보정 실험으로 유지한다. 현재 평가가 고정 결합기와의 호환성을 포함한다는 한계는 명시한다.

검토 영수증과 코드·산술 확인 범위는 [expanded_design_review.json](../../evidence/panns_cnn_review_20260925/expanded_design_review.json)에 남긴다. 이 기록의 `final_spec_sha256`은 확장안이 저장된 commit `2e58d9330809b92e28058cf77c80c218c6ff7b1c`의 문서에 해당하며, v1 확정 후 해시로 덮어쓰지 않는다. Opus가 검토한 초안과 수정 후 문서를 구분하며, 확정 요청에서 Opus를 다시 호출하지 않았다. 현재 v1 문서·workflow 해시는 [design_v1_lock.json](../../evidence/panns_cnn_review_20260925/design_v1_lock.json)에 보존한다. 사람·모델의 검토는 runtime·학습 품질 검증을 대신하지 않는다.

## 11. 이번 설계 작업의 완료 경계

확장된 것은 구조 후보, 학습 깊이, 미분 가능한 파일 손실, 긴 파일 처리, 데이터·캐시·장치 조건, 채택·중단 기준이다. 새로운 실험 성능이나 MPS 학습 성공을 보고하지 않는다. 독립 평가 자료, MUSAN 복구, 실제 CNN gradient·속도·메모리 측정은 아직 남아 있다. 이 문서를 구현 완료나 새 GPU 사용 승인으로 해석하지 않는다.
