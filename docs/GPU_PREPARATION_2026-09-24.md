# GPU 학습 사전준비 — 2026-09-24

코드·실험 입력·신규 데이터·가중치 전달과 CPU 검증을 마쳤다. 기존 음원이 중지된 VESSL 영구 디스크에 실제로 보존돼 있는지 확인하는 단계가 남아 있어, 전체 사전준비 완료나 즉시 학습 가능으로 판정하지 않는다. GPU는 켜지 않았다.

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

두 실제 데이터 archive의 1,477개 음원, 709,024,988 bytes를 임시 디렉터리에 배치해 전부 해시를 대조했다. 기존 파일 표본을 보존했고 검증용 임시 디렉터리는 정리했다. GPU 실행에 필요한 프로젝트 Python 모듈 21개와 입력표·설정을 묶어 별도 디렉터리에서 실제 CLI import도 확인했다. 기존 학습 환경과 전체 음원이 있는 cloud root에서의 통합 검증은 아직 하지 않았다.

VESSL `qwdkw / objvol-a5gxdg7er70i`의 `lb-improvement-20260924/gpu-preparation-v1/`에 실행 묶음, CPU 준비 스크립트, CPU job 요청안을 올렸다. 재다운로드 SHA-256이 일치했다. 실행 archive는 12,942,393 bytes이며 SHA-256은 `11ff29888fb29041d4789f6386efa3d99b0e0731ebfcf871cbaf57a7de4e21e1`이다.

기존 시작 스크립트의 조건부 phase A 자동 실행을 제거했다. 원본은 같은 volume의 별도 경로에 백업하고 해시를 확인했으며 현재 스크립트는 안내만 출력한다. 마지막 조회에서 workspace `wsp-9683432r72m7`는 stopped/standby다.

## 마지막 확인 단계

기존 음원의 기대 경로는 `/root/data_v4/out/audio/`다. `/root`는 `clustervol-qgze68bffy6l`에 보존되지만 CLI는 중지된 cluster volume 내부 파일 조회를 지원하지 않는다. 현재 이 볼륨에 접근할 수 있는 실행 중 CPU job도 없다. Object storage의 원천 archive나 이전 결과 ZIP은 이 음원의 현재 보존 증거를 대신하지 못한다.

GPU 없이 마무리할 수 있도록 CPU 3개·RAM 13 GiB 인스턴스에서 수행할 요청안을 준비했다. 기존 볼륨을 마운트하고 신규 데이터·가중치를 기존 파일을 덮어쓰지 않고 배치한 뒤 전체 bank 43,516행과 가중치 해시를 확인한다. 기존 환경을 재사용하며 의존성을 자동 설치하지 않는다. 비용은 시간당 0.30 credit, 검증 명령은 600초 제한과 종료 유예 30초다. 기동·종료 시간이 추가될 수 있어 총비용 0.05 credit 상한을 보장하지 않는다. CLI에는 read-only mount와 scheduler TTL 옵션이 없다. 새 유료 CPU 인스턴스는 아직 만들지 않았다.

GPU 명령은 기본값이 CPU preflight이며 실행 단계와 `--max-seconds`를 명시해야 한다. 시간 제한은 작업 프로세스에 적용되고 유료 workspace 자체를 정지하지는 않는다. 실제 GPU 실행 시 승인된 시간에 workspace를 중지하는 운영 절차가 별도로 필요하다. 정확한 명령과 경로는 object volume의 `README.txt` 및 [CPU 요청안](evidence/lb_gpu_preparation_20260924/cpu_job_request.json)에 있다. 공식 제출은 이번 준비에 포함하지 않는다.

[입력표 해시](evidence/lb_gpu_preparation_20260924/manifests.json) · [검증 범위](evidence/lb_gpu_preparation_20260924/verification.json) · [실제 배치 검증](evidence/lb_gpu_preparation_20260924/staging_verification.json) · [전송](evidence/lb_gpu_preparation_20260924/transfers.json) · [시작 스크립트 변경](evidence/lb_gpu_preparation_20260924/startup_change.json) · [남은 확인](evidence/lb_gpu_preparation_20260924/readiness.json)
