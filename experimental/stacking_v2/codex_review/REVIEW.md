# v2 독립 검토 — 2026-09-21

판정: 이전에 지적한 두 결함은 수정됐으며 로컬 재검증을 통과했다. 신규 데이터 실험을 준비하는 단계로 진행할 수 있다. 실제 탐지 성능 향상이나 GPU 배포 적합성을 승인한 것은 아니다. 반환 코드는 수정하지 않았다.

## 확인 결과

- 반환 inventory 110개 파일의 SHA-256이 모두 일치했다.
- preserve_gain=True, manifest/cache 정책 불일치, 정책 누락·변경을 거부한다. 같은 정책에서 파일마다 다른 peak/gain 측정값은 허용한다. 추출 전 검사부터 dev, candidate lock, OOD, 배포 진입점까지 정책 전달을 확인했다.
- XGBoost가 없어도 LR/ExtraTrees CV 결과와 후보별 NOT_RUN 상태가 보고서에 남는다. 이전처럼 serialization_results.json 저장 전에 전체 작업을 중단하지 않는다.
- 실제 head/cache 재생, 부모 그룹 누수·노출 데이터 거부, 피처 순서·캐시 해시, 시간 구간 정렬, 기존 네 출력 보존을 포함한 전체 CPU 검증이 통과했다. 실행 결과는 35 passed이며, native encoder 경계는 일부 test double을 사용하므로 실제 음원/GPU parity 증거가 아니다.
- synthetic F0/F1 × LR/ExtraTrees/XGBoost의 여섯 CV 경로가 모두 OK였다. 세 모델의 실제 저장·재로드도 성공했다. LR/ExtraTrees는 skops, XGBoost는 JSON을 사용했고 최대 예측 차이는 각각 0.0이었다. 합성 데이터 결과이므로 탐지 성능 근거로 사용하지 않는다.

## 환경 차이와 재현

기존 .venv에는 XGBoost와 skops가 없었다. 처음 실행한 선택 검증은 29 passed, 2 skipped, 2 deselected, 2 failed였다. 실패한 두 테스트는 XGBoost의 save/load 단계에 도달한다고 가정하지만, 미설치 환경에서는 fit 단계에서 먼저 NOT_RUN이 발생한다. 실제 미설치 동작과 보고서 보존을 검사하는 테스트 및 별도 재현은 통과했다. 이 테스트 환경 제약을 runtime 결함으로 분류하지 않았다.

기존 .venv를 변경하지 않고 `uv pip install --python .venv/bin/python --target _workspace/stacking_review_deps --no-deps xgboost==3.1.3 skops==0.13.0`으로 임시 경로에 설치했다. 처음 pip 명령은 .venv에 pip가 없어 실패했으며 로그를 보존했다. 설치 후 전체 suite와 smoke를 실행해 위 결과를 확인했다.

프로젝트 루트에서 다음과 같이 재현한다. smoke 출력은 새 디렉터리를 지정한다.

```bash
export PYTHONPATH="$PWD/_workspace/stacking_return_v2/src:$PWD/_workspace/stacking_review_deps"
export STACKING_INPUT_ROOT="$PWD/_workspace/stacking_return_review_input"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python -m pytest _workspace/stacking_return_v2/tests -q
.venv/bin/python -m dacon_stacking smoke --out _workspace/stacking_v2_smoke_recheck
```

## 남은 작업

신규 미노출 meta-dev·locked OOD 확보, 실제 GPU 추출과 기존 추론 대비 parity, 일반화 성능 비교, L4 시간·VRAM, 최종 제출 패키지 검증이 남아 있다. 이번에는 GPU/VESSL 실행이나 공식 제출을 하지 않았다. v1의 정책 없는 production bundle/lock은 재사용하지 않고 v2 계약으로 다시 만들어야 한다.
