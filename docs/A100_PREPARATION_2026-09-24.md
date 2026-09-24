# A100 실험 준비 — 2026-09-24

E00/E04 실행 코드에 A100 지원을 추가하고 별도 실행 묶음을 VESSL에 준비했다. GPU는 시작하지 않았다. 실제 A100 모델 로딩·속도·VRAM 검증은 남아 있으며, 현재 VESSL에서는 음원 이동과 실행 가능한 A100 workspace 확보가 필요하다.

## 코드 변경

`data_v4/gpu_experiments.py`는 기본 동작이 CPU preflight이고 GPU 단계의 기본 선택은 A100이다. `--gpu-name A100`을 특징 추출·E00·E04의 실제 실행 명령까지 전달한다. H100은 `--gpu-name H100`으로 선택한다. 기존 직접 호출용 `full_bank_features.py`와 `full_bank_train.py`의 기본 선택은 H100으로 보존했다.

선택한 GPU 종류가 다르거나 CUDA·native BF16을 사용할 수 없으면 모델 로딩 전에 중단한다. A100은 BF16을 지원하므로 기존 학습 정밀도는 유지한다. 지원 여부는 실행 시 PyTorch로 확인하며 실제 A100 환경 통과를 미리 선언하지 않는다. [NVIDIA A100 자료](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/a100-80gb-datasheet-update-a4-nvidia-1485612-r12-web.pdf), [PyTorch BF16 확인 API](https://docs.pytorch.org/docs/main/generated/torch.cuda.is_bf16_supported.html).

배치 후보는 1·2·4·8·16·32·64·128·256이다. MERT 배치는 매 실행에서 다시 측정해 다른 GPU의 측정값을 그대로 쓰지 않는다. 학습 배치도 실제 forward/backward 측정과 VRAM 80% 기준으로 고른다. 배치 측정 후 가중치와 난수 상태를 복원한다. 실행 GPU 이름·메모리·compute capability·PyTorch/CUDA 버전은 `device.json`으로 남긴다.

seed 41, 최대 50 epoch, patience 5, E00/E04 데이터 역할, frozen encoder, F0 비교 방식은 유지한다. 합집합 43,516개에서 특징을 한 번 추출하고 E00/E04가 함께 사용한다. 코드 해시가 바뀌므로 새 cache/output 경로를 사용하며, 이전 캐시의 provenance 검사를 느슨하게 바꾸지 않는다.

## 확인한 범위

CPU에서 실제 하위 CLI가 A100 선택을 받는지, GPU가 없을 때 CPU로 학습을 진행하지 않는지 확인했다. 메모리 제한은 모의 조건으로 주고 실제 CPU forward/backward/AdamW를 실행해 배치 1 선택과 가중치 복원을 확인했다. GPU 조회 모의 검사는 선택 조건만 검증하며 A100에서의 실행·메모리 검증을 대신하지 않는다. 기존 역할 분리, 캐시 무결성, microbatch 수치 동등성, 고정 F0 검증도 통과했다.

실행 묶음은 기존의 명시된 파일 목록 31개만 포함한다. 압축을 풀어 모든 멤버 해시와 CLI import를 확인했고, 입력표·설정 7개 파일은 앞서 CPU 검증한 바이트와 동일하다. 기존 Capella 볼륨의 43,516개 음원과 25개 자산에 대한 검증 기록은 유효하지만, A100 클러스터로의 전달 완료를 뜻하지 않는다.

## VESSL 상태와 남은 배치

현재 조회에서 A100 SXM 80GB 1개 자원 `resourcespec-a100x1`은 사용 가능으로 표시되며 시간당 1.48 credit이다. CPU 11개, RAM 128 GB이고 `cluster-betelgeuse`에 있다. 이는 조회 시점의 표시이며 이후 예약 가능 여부를 보장하지 않는다.

기존 A100 workspace `wsp-5i3knd3gxeox`는 state=`terminating`, phase=`terminated`다. 기존 H100 workspace `wsp-9683432r72m7`는 stopped/standby이며 데이터는 `cluster-capella`의 `clustervol-qgze68bffy6l`에 있다. 서로 다른 클러스터이므로 H100의 `/root`를 A100에 그대로 마운트할 수 있다고 가정하지 않는다. 프로젝트 AGENTS.md의 기존 workspace 재사용 규칙에 따라 새 A100 workspace는 만들지 않았다.

음원 내보내기는 검증된 manifest의 43,516개, 12,329,271,166 bytes만 읽어 새 tar로 묶는다. 복사할 바이트 자체의 SHA-256을 검사하고 기존 원본·기존 export를 덮어쓰지 않는다. 결과는 기존 object volume `objvol-a5gxdg7er70i`의 `lb-improvement-20260924/a100-preparation-v1/audio/`에 저장하고 전체 재읽기 해시를 확인한다. 가중치 등 고정 자산 25개는 이미 같은 object volume에 있으므로 다시 업로드하지 않는다.

이 작업을 위한 CPU 1개 요청안을 준비했다. CPU 3개·RAM 13 GiB, 시간당 0.30 credit, 명령 제한 1,800초와 종료 유예 30초다. 기동·종료 시간은 추가될 수 있으며 0.15 credit는 총비용 상한이 아니다. 새 유료 CPU 인스턴스는 아직 만들지 않았다. 음원 내보내기 완료 후 A100 측에 새 경로로 배치하고, 준비 코드·음원·자산 preflight와 실행 환경을 확인해야 한다.

## 실행 방법

object volume의 `lb-improvement-20260924/a100-preparation-v1/README.md`에 묶음 해시와 전체 명령을 기록했다. 입력 배치와 GPU 실행 예산을 승인받은 환경에서 새 묶음을 사용해야 한다. 이전 `prepared-v1`의 H100 전용 코드를 그대로 실행하면 A100 지원 변경이 적용되지 않는다.

[패키지](evidence/lb_a100_preparation_20260924/package_receipt.json) · [검증 범위](evidence/lb_a100_preparation_20260924/verification.json) · [클라우드 상태](evidence/lb_a100_preparation_20260924/readiness.json) · [CPU 내보내기 요청안](evidence/lb_a100_preparation_20260924/cpu_export_job_request.json)
