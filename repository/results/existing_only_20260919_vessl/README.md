# A100 데이터 배치 — 2026-09-19

확정한 9,504행을 기존 A100 워크스페이스에 연결된 오브젝트 볼륨에 배치했다. GPU를 시작하거나 시작 스크립트를 바꾸지 않았다.

- 워크스페이스: `wsp-5i3knd3gxeox` (`jolly-routine`, A100 SXM ×1, stopped)
- 볼륨: `objvol-a5gxdg7er70i`, A100 마운트 `/root-1`
- 배치 경로: `/root-1/existing-only-9504-20260919/`
- 실행용 목록: `dataset/manifest.csv` — train 6,652 / val 1,426 / test 1,426
- 각 분할의 real/fake는 1:1이다. 원본 라벨·마스크·그룹·분할을 모두 보존했다.

## 저장된 데이터

기존 업로드의 6,885개를 재사용하고 2,619개를 19개 ZIP으로 추가 업로드했다. 선택된 원본 음원 합계는 **14,939,622,208바이트**다. 연결된 75개 오디오 ZIP은 기존 뱅크의 미선택 음원도 담고 있으므로 합계가 **20,039,338,953바이트**다. 실행 시 선택 목록에 있는 멤버만 꺼내야 한다.

새 오디오 ZIP은 **4,984,907,429바이트**, 동일한 사전학습 파일 27개를 담은 `assets.zip`은 **6,265,308,398바이트**다. 음원 원본·이전 분할·이전 제출 파일은 보존했다.

`audio_upload.json`은 정확히 9,504개 ID의 오디오 경로와 SHA-256, 원격 ZIP의 키·크기·SHA-256·ETag를 기록한다. 오디오 ZIP의 `key`는 `/root-1` 기준 경로이며, 현재 배치 폴더 기준이 아니다. `deployment.json`은 목록·모델·코드의 원격 파일 기록을 묶는다.

실행용 목록 SHA-256:
`96ba67dd2dfdbcfbf041465e883e0e003e6489e14f82e93054df4b42075a9ae3`

## 검증 범위

원격 목록을 다시 읽어 로컬 파일과 해시를 대조했다. 9,504행의 라벨·마스크·출처·그룹·split 보존, 오디오 업로드 목록과의 일치, 세 분할 사이의 알려진 그룹 중복 부재를 검사했다. `assets.zip` 내부 27개 파일의 SHA-256과 CRC도 기존 모델 기록과 일치한다. 새 업로드는 multipart별 체크섬과 크기를 확인했고, 재사용 객체는 기존 기록의 크기·ETag·SHA-256 메타데이터를 대조했다. 원격 대형 ZIP 전체를 새로 다운로드한 검증은 아니다. 노트북 실행 시 원격 ZIP과 추출 음원의 SHA-256을 다시 확인한다.

```sh
.venv/bin/python existing9504_upload.py
.venv/bin/python verify_existing9504_upload.py
```

첫 명령은 기존 기록을 재사용해 업로드를 재개하며, GPU를 시작하는 기능이 없다. 두 번째 명령은 실제 클라우드 목록과 원본 분할을 대조한다.

## 노트북

Claude **Opus 5 엑스트라(xhigh)**가 제작한 v3를 Codex가 CPU에서 재검수했다. 작성 자료와 설정 근거는 `author_request.json`, `AUTHOR_REQUEST.txt`, `claude_model_setting.txt`에 있다. 검수 원본과 재현 코드는 `claude_v3/`에 보관했다.

v3는 변경된 val/test 예측값·지표, 다른 모델에 연결된 parity 기록을 완료 ZIP 검사에서 거부한다. 검사 직후 파일을 바꾸는 경우도 실제 ZIP 내부 바이트를 다시 대조해 차단했다. 생성된 노트북의 마지막 셀을 실행한 뒤 로컬 Jupyter 서버에서 ZIP을 받아 HTTP 200과 SHA-256 일치를 확인했다. 기존 링크는 같은 조건에서 HTTP 403이었다. 같은 폴더의 기존 사본, 하드링크, 파일시스템이 다를 때의 검증 복사도 확인했다. 이 검사는 실제 9,504개 ID와 합성 결과·체크포인트를 사용한 CPU 검사이며, 실제 모델 학습이나 VESSL UI 검사가 아니다.

검수한 `DACON_Existing9504_A100.ipynb`, `author_code.zip`, `README_REVIEWER.md`, `codex_review.json`을 위 배치 경로에 업로드하고 원격 파일을 다시 받아 해시를 대조했다. `notebook_review_pending=false`, `notebook_ready_for_manual_run=true`다. 실제 A100 환경·배치·처리량 확인은 아직 남아 있으므로 `training_ready=false`, `runtime_preflight_pending=true`를 유지한다. 워크스페이스와 시작 스크립트는 변경하지 않았다.

노트북을 열고 Run All 하면 준비부터 결과 ZIP 생성까지 순서대로 진행하도록 구성했다. Jupyter에서 `/root-1`을 볼 수 없다면 노트북 파일만 `/root` 아래에 업로드해 열면 된다. 실행 코드는 `/root-1/existing-only-9504-20260919/author_code.zip`을 사용한다. 마지막 셀은 노트북 폴더에 검증된 ZIP을 제공하고, 오브젝트 볼륨에도 사본을 남긴다. 결과 파일은 `DACON_Existing9504_A100_Handoff.zip`이며, 이후 제출 패키지를 만들 때 이 파일 하나를 전달하면 된다. 이 파일 자체가 DACON 제출용 `submit.zip`은 아니다.

요청한 실행 흐름은 준비·검증 → A100 배치/처리량 측정 → 전체 특징 추출 → train 학습·val 선택 → 체크포인트 고정 후 test 평가 → 단일 결과 ZIP이다. 이번에는 GPU 실행을 하지 않으므로 A100 처리 시간·VRAM·성능과 전체 실행 성공을 검증했다고 주장하지 않는다.

이 분할의 test는 독립 평가셋이 아니다. test 1,426행 중 1,001행에 과거 개발 노출이 있고, 가창은 train에만 있다. 전체 데이터 감사와 한계는 `../existing_only_20260919_final/README.md`를 따른다.
