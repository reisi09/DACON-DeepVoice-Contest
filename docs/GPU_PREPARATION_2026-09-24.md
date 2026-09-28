# GPU 학습 사전준비 — 2026-09-24

GPU 없이 수행할 데이터·코드 배치와 CPU 입력 사전검증을 완료했다. 재검증에서 전체 bank 음원 43,516개와 고정 자산 25개의 해시가 일치했고, CPU 작업은 성공 상태로 종료됐다. GPU는 중지 상태이며 실제 모델 로딩·GPU 실행과 성능 검증은 아직 수행하지 않았다.

## 준비한 실험

E00은 수정된 기존 데이터 기준선, E04는 여기에 신규 1,477개를 추가하는 비교다. 같은 seed 41, 최대 50 epoch, patience 5를 사용한다. MERT·DF-Arena·PANNs·HTDemucs는 고정하고 기존 music WindowMIL과 voice linear head만 학습한다.

| 입력 | 행 수 | 용도 |
|---|---:|---|
| E00 | 32,197 | 기존 train·selection·calibration |
| E04 | 33,674 | E00에 신규 train 1,477개 추가 |
| development | 9,842 | 두 실험의 동일 비교 패널 |
| 특징 추출 합집합 | 43,516 | E04와 development를 한 번만 추출 |

라벨이 부족해 bank에 내보내지 않는 행이 있어 전체 roles-v6 manifest 44,424행과 다르다. E00의 기존 행은 E04에서 전체 필드가 일치하며 selection·calibration·development는 추가하거나 재배정하지 않았다.

`--feature-manifest`를 명시한 경우에만 합집합 캐시를 재사용한다. 원본 행, 역할, 라벨, 오디오, 모델 자산, 전처리, 구현과 runtime 계약이 일치해야 한다. 과거 MERT 일부 캐시는 이 조건을 만족하지 않아 사용하지 않는다. 실제 새 특징 추출은 GPU 실행 단계다.

E00이 끝난 뒤 E04를 실행한다. FILE 보정 F0는 E00 calibration만으로 적합하고 두 모델의 development 예측에 그대로 적용한다. E04나 development 정답으로 다시 적합하지 않는다.

## 검증과 전달

학습 준비 코드 commit은 `b4d1009902bc4cd04c79f850eee43bc10f0e338d`, MLAAD 이용 조건·146개 편입 기록은 `c951608`이다. CPU에서 실제 NPZ 값·마스크 보존, 잘못된 캐시·행·가중치·checkpoint 거부, F0 적합 자료 분리, 시간 제한 후 작업 프로세스 종료, 경로 이탈과 기존 파일 덮어쓰기 방지를 확인했다. 테스트 통과를 모델 품질 근거로 사용하지 않는다.

두 실제 데이터 archive의 1,477개 음원, 709,024,988 bytes를 임시 디렉터리에 배치해 전부 해시를 대조했다. 기존 파일 표본을 보존했고 검증용 임시 디렉터리는 정리했다. GPU 실행에 필요한 프로젝트 Python 모듈 21개와 입력표·설정을 묶어 별도 디렉터리에서 실제 CLI import도 확인했다. 기존 cloud root에서도 같은 배치 코드를 실행해 신규 음원 1,477개를 설치했고 모든 신규 음원 해시를 대조했다. 배치 단계에 진입하기 전에 고정된 가중치 파일 25개의 해시 검사도 통과했다. 이후 재검증에서 전체 bank 음원 해시 검사를 통과했고 완료 영수증도 회수해 대조했다.

VESSL `qwdkw / objvol-a5gxdg7er70i`의 `lb-improvement-20260924/gpu-preparation-v1/`에 실행 묶음, CPU 준비 스크립트, CPU job 요청안을 올렸다. 재다운로드 SHA-256이 일치했다. 실행 archive는 12,942,393 bytes이며 SHA-256은 `11ff29888fb29041d4789f6386efa3d99b0e0731ebfcf871cbaf57a7de4e21e1`이다.

기존 시작 스크립트의 조건부 phase A 자동 실행을 제거했다. 원본은 같은 volume의 별도 경로에 백업하고 해시를 확인했으며 현재 스크립트는 안내만 출력한다. 마지막 조회에서 workspace `wsp-9683432r72m7`는 stopped/standby다.

## CPU 실행 결과

사용자가 CPU 인스턴스 1개 생성을 승인해 `job-aj1ufwbahfpk`를 실행했다. 자원은 CPU 3개·RAM 13 GiB, 시간당 0.30 credit이고 GPU는 없다. 명령에는 600초 제한과 종료 유예 30초를 적용했다. 기동·종료 시간이 추가될 수 있어 0.05 credit를 실제 청구액이나 총비용 상한으로 제시하지 않는다.

2026-09-24 UTC 기준 스크립트는 10:11:53에 시작했다. 10:20:53에 신규 음원 1,477개, 709,024,988 bytes의 해시 확인과 설치를 마쳤다. 작업은 10:21:59에 `failed`로 종료됐다. 시작 후 약 606초여서 시간 제한 종료로 추정하지만, CLI가 종료 코드와 사유를 제공하지 않아 확정하지 않는다. Object volume에서 완료 영수증은 회수되지 않았다. GPU workspace는 이후 조회에서도 stopped/standby다.

추가 승인을 받은 검증 전용 작업 `job-lregdqqbkv9q`는 기존 파일을 다시 복사하지 않고 검사했다. 준비 코드·입력표 30개 파일, 음원 43,516개(12,329,271,166 bytes), 고정 자산 25개의 해시가 모두 일치했다. 검사 대상은 기존 bank 음원 42,039개와 신규 1,477개이며, 원래 manifest의 기존 42,947행 전체를 검사한 것은 아니다. E00·E04의 역할별 행 수, seed 41과 설정도 고정한 입력과 일치했다.

재검증은 317.25초 걸렸고 10:40:07 UTC에 `succeeded`로 종료됐다. 새 실행 디렉터리의 `preflight.json`, `environment.json`, `readiness.json`을 object volume에서 회수했다. 완료 영수증의 해시를 cloud 로그 및 별도 readiness 영수증과 대조하고 입력표·자산 해시와 역할별 행 수를 확인했다. 기존 Python 3.11.13 환경을 재사용했으며 의존성 설치나 GPU 실행은 하지 않았다. 파일시스템 여유 공간도 검사 전후 모두 4 GiB 기준을 충족했다. 보고된 여유 공간은 파일시스템 값이며 이 프로젝트에 예약된 용량을 뜻하지 않는다.

검증 전용 스크립트는 변경된 코드·추가 코드·변경된 해시 목록을 실제 실행 전에 거부하는 검사를 통과했다. 기존 결과를 성공으로 재사용하지 않도록 새 실행 디렉터리를 요구한다. 첫 작업과 재검증 요청안은 실행 전 기록으로 보존하고, 실제 실행 결과는 각 실행 영수증으로 구분한다. 현재 추가 CPU 실행을 기다리는 항목은 없다.

GPU 특징 캐시는 아직 생성하지 않았다. 실행 안내에는 캐시·출력 파일시스템에 최소 4 GiB가 남아 있는지 확인하는 명령을 추가했다. 이는 운영상 여유 공간 기준이며 최종 산출물 크기를 보장하지 않는다. 학습 launcher 자체에는 디스크 여유 공간 자동 점검 기능이 없다.

GPU 명령은 기본값이 CPU preflight이며 실행 단계와 `--max-seconds`를 명시해야 한다. 시간 제한은 작업 프로세스에 적용되고 유료 workspace 자체를 정지하지는 않는다. 실제 GPU 실행 시 승인된 시간에 workspace를 중지하는 운영 절차가 별도로 필요하다. 정확한 명령과 경로는 object volume의 `README.txt` 및 [첫 CPU 요청안](evidence/lb_gpu_preparation_20260924/cpu_job_request.json)과 [검증 전용 재요청안](evidence/lb_gpu_preparation_20260924/cpu_verify_job_request.json)에 있다. 공식 제출은 이번 준비에 포함하지 않는다.

[입력표 해시](evidence/lb_gpu_preparation_20260924/manifests.json) · [검증 범위](evidence/lb_gpu_preparation_20260924/verification.json) · [실제 배치 검증](evidence/lb_gpu_preparation_20260924/staging_verification.json) · [전송](evidence/lb_gpu_preparation_20260924/transfers.json) · [시작 스크립트 변경](evidence/lb_gpu_preparation_20260924/startup_change.json) · [CPU 실행 기록](evidence/lb_gpu_preparation_20260924/cpu_job_execution.json) · [재검증 성공 기록](evidence/lb_gpu_preparation_20260924/cpu_retry_execution.json) · [준비 상태](evidence/lb_gpu_preparation_20260924/readiness.json)
