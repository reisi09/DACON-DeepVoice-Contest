# DACON 재설계 실험 노트북 증거

## 결과물과 고정 해시

- 실행 노트북: `DACON_Redesign_H100_Experiments.ipynb`
  - SHA-256: `781d89c868773c12e539f935dca31194cae37c47d94a291a87f77854ec501fab`
  - 22 cells, 저장 출력 0개, `nbformat.validate` 통과
- 내장 소스 payload
  - SHA-256: `285fd682e08f352381a8d20765df2bad5f17fc957d7d2eb55e9235ac8291e058`
  - 로컬 import closure 29개, ZIP CRC 정상, 모든 member가 최종 workspace source와 byte 단위로 일치
- 빌더와 실행 도구: `build_redesign_notebook.py`, `redesign_notebook_runtime.py`, `redesign_notebook_integration.py`, `redesign_notebook_results.py`, `test_redesign_notebook.py`
- 기계 판독 receipt: `.omo/evidence/redesign-notebook-20260917/notebook-verification.json`

## 검증한 시나리오

### 안전한 기본 Run All

- 실행: `.venv/bin/python test_redesign_notebook.py default .omo/evidence/redesign-notebook-20260917/default-notebook-final .omo/evidence/redesign-notebook-20260917/DACON_Redesign_H100_Experiments_Default_Executed.ipynb`
- 관찰값: exit 0. CPU provenance·manifest hash·저장소 계획만 기록됐고, 특징 추출·학습·File fitting·비교/잠금·sealed·export·CPU fixture는 모두 `not_run`이었다. 실제 fixture로 조용히 전환되지 않았다.
- 증거: `default-notebook-final.log`, `default-notebook-final/results/progress.json`, 실행 노트북 SHA-256 `97896ba7b38fb3e9375d3574f2c8fc2fc1d8a5d596acd0280bdf27a8fb18ef06`

### 2시드 실제 CPU 알고리즘 통합

- 실행: `.venv/bin/python test_redesign_notebook.py cpu .omo/evidence/redesign-notebook-20260917/cpu-notebook-final .omo/evidence/redesign-notebook-20260917/DACON_Redesign_H100_Experiments_CPU_Executed.ipynb`
- 관찰값: exit 0. 결정적 labeled feature fixture에서 A/B/C/D, `D_raw_equals_stem`, voice residual, 시드 41·42, 2 folds, OOF File relation, grouped comparison, checkpoint 재로딩과 ZIP readback이 실제 구현을 거쳐 실행됐다.
- 무결성: OOF member 2개가 각각 music·voice proof를 보존했고, 정렬된 voice weight hash 2개가 identity에 포함됐다. 저장 artifact와 재로딩 artifact의 5개 확률 최대 차이는 `0.0`, 전체 upstream identity도 일치했다. 학습된 voice 경로의 최대 확률 변화는 `2.3305416107177734e-05`였다.
- 판정 경계: fixture 후보는 `rejected`, 잠금 시도도 거부됐다. receipt는 `algorithm_qa_only`이며 실제 데이터 성능·H100·L4·sealed·제출을 모두 `false`로 남겼다. 표에는 seed별 값, seed 평균·SD, probability ensemble 값을 따로 표시했고 계산 불가능한 공식 `score`는 `null`이었다.
- 증거: `cpu-notebook-final.log`, `cpu-notebook-final/cpu_integration/integration.json`, `cpu-notebook-final/cpu_integration/algorithm_qa_results.zip`, 실행 노트북 SHA-256 `bea26bb8d858614f03c961afa9a95dc6c35bd2b68b5dfb982756727c8cedb105`, ZIP SHA-256 `fa6003708f3fc4b242a9f73e7f2b5cbe746590300ad9c108a8992678b6b57da4`

### 만료된 GPU 예산과 부분 결과 보존

- 실행: `.venv/bin/python test_redesign_notebook.py expired .omo/evidence/redesign-notebook-20260917/expired-notebook-final .omo/evidence/redesign-notebook-20260917/DACON_Redesign_H100_Experiments_Expired_Executed.ipynb`
- 관찰값: 이미 만료된 영속 budget receipt에서 GPU 특징 추출을 호출하기 전에 `extract_features=interrupted`가 기록됐다. 기존 OUTPUT 5개 member를 부분 결과 ZIP으로 내보낸 뒤 CRC와 member hash를 다시 확인했고, 원래 예외를 재발생시켰다.
- 증거: `expired-notebook-final.log`, `expired-notebook-final/results/progress.json`, `expired-notebook-final/exports/interrupted_readback.json`, `expired-notebook-final/exports/interrupted_results.zip`; ZIP SHA-256 `ee9cb98ee632cd7dad6b879ed49cb39d08ff2336dbf9a911028069e0e8f8dfd7`

### 소스·패키징 경계

- 실행: `.venv/bin/python build_redesign_notebook.py`와 `.venv/bin/python -m py_compile build_redesign_notebook.py redesign_notebook_runtime.py redesign_notebook_integration.py redesign_notebook_results.py test_redesign_notebook.py`
- 관찰값: builder가 최종 notebook을 재생성했고 모든 Python 파일이 compile됐다. programming no-excuse 검사도 5개 파일에서 위반 0건이었다. `git diff --check`도 통과했다.
- payload는 명시한 로컬 source closure만 담는다. `.venv`, site-packages, audio, weight, token, secret은 포함하지 않는다. 기존 versioned source가 payload와 다르거나 이전 kernel import가 남아 있으면 덮어쓰지 않고 재시작을 요구한다.

## 실제 H100 실행 순서

1. `sources.csv`는 notebook이 만든 header-only template에 실제 파일, SHA-256, 라이선스와 출처를 채운다. baseline ZIP 경로와 고정 SHA-256도 설정한다.
2. 새 VESSL notebook 복사본에서 환경 변수 `DACON_REDESIGN_ROOT`, `DACON_INPUT_MANIFEST`, `DACON_BASELINE_ZIP`, `DACON_ASSETS`, `DACON_DATA`, `DACON_CACHE`, `DACON_OUTPUT`, `DACON_DEVICE=cuda`를 지정한다.
3. 의존성이 없는 새 image라면 `INSTALL_DEPENDENCIES=True`인 dependency cell만 한 번 실행한 뒤 kernel을 재시작하고 다시 `False`로 둔다. torch·torchaudio·torchvision 2.7.1/2.7.1/0.22.1 cu128 묶음과 검증된 backbone pin을 사용한다.
4. copied notebook의 `FLAGS`에서 `prepare_data`, `prepare_assets`, `extract_features`, `train_experiments`, `fit_file_relation`, `compare_and_lock`, `export_results`를 필요한 단계만 `True`로 바꾼 뒤 위에서 아래로 실행한다. 첫 GPU 단계가 영속 4시간 예산을 시작하며 마지막 15분은 export용으로 남긴다.
5. File 진단은 max·noisy-OR·OOF relation을 모두 기록한다. 승격 판단은 고정 `A_ensemble` control과 최종 후보를 비교하며 A→B, A→C, B→D, C→D causal report도 함께 남긴다. A artifact가 없으면 잠그지 않는다.
6. 후보가 실제 라벨의 통계·개수 gate를 통과한 뒤에만 lock이 생긴다. sealed 단계는 정렬된 music/voice checkpoint hash, 잠긴 validation identity, `audit['sealed_panel_identity']`를 특징 추출 전에 검사한다. 개봉 기록은 OUTPUT이 아닌 `DATA/.sealed_access_registry`에 남고, sealed report에는 공식 metric이 들어간다.
7. 결과 ZIP과 readback SHA-256을 확인한 뒤 VESSL workspace는 사용자가 직접 종료한다. notebook은 외부 종료 명령을 자동 호출하지 않는다.

Headless 실행도 같은 notebook 복사본의 config cell에서 환경 변수와 `FLAGS`를 고정한 뒤 `nbclient` 또는 `jupyter nbconvert --execute`로 실행하면 된다. 실제 run에서는 설정이 반영된 notebook 복사본, 실행된 notebook, `progress.json`, `gpu_budget.json`, 결과 ZIP과 readback JSON을 한 묶음으로 보존해야 한다.

## 아직 검증하지 않은 항목

- 실제 출처 데이터의 성능 선택, H100 전체 학습, L4 50분/메모리 package gate, sealed 1회 평가, DACON 제출 ZIP은 실행하지 않았다.
- 9-state joint output, log-mel 추가 경로, partial/full fine-tuning은 이번 primary A/B/C/D notebook 범위 밖의 후속 gate다.
- CPU fixture의 0 EER은 wiring 확인용으로만 쓰며 성능 근거가 아니다.
- 첫 통합 시 OOF와 full-fit identity가 달라 relation 적용이 실패한 기록은 `cpu-integration-first-failure.log`에 보존했다. 이후 family identity와 다중 시드 proof를 분리해 최종 2시드 실행에서 해결했다.
