# DACON Stacking Return v2 — 두 검토 결함 수정

수정일: 2026-09-21 (Asia/Seoul). **전처리 계약과 optional-dependency smoke 처리만 수정했다.** 새로운 설계, 후보 모델, 피처, scorer, encoder/head 재학습은 추가하지 않았다. 새 데이터 성능 개선이나 배포 준비 완료를 주장하지 않는다.

## 입력과 검토 자료

- 기존 입력: `DACON_Chat_Stacking_Input.zip`, SHA-256 `7adfb7b426c45b0efbb23de581ca1a5a073079d2110f4f4f84cdcf0fca2fc75a`.
- 수정 기준: `DACON_Stacking_Return.zip`, SHA-256 `1f35252e0d005be18cb456736c74f62eeaf8e10bd17a437adb91e78decac4d32`.
- 현재 snapshot/input manifest 133개와 v1 반환 inventory 74개를 다시 검증했다. 기존 ZIP 자체는 변경하지 않았다.
- 별도 Codex 검토 첨부와 `reproduce_preprocessing_gap.py`는 이 대화의 파일 목록 및 런타임에 없었다. 읽거나 실행했다고 주장하지 않는다. 사용자 메시지의 재현 조건으로 **`tests/reproduce_review_findings.py`를 새로 작성**하여 v1/v2에서 각각 실행했다. 신규 수집 데이터나 원래 첨부 reproducer가 아니다.

## 1. 전처리 계약 수정

production은 현재 배포 정책을 그대로 유지한다: **16 kHz, mono, peak cap 0.6, preserve_gain=False**, 동일 snapshot의 Demucs 설정. `preserve_gain=True`와 비불리언 값, 정책 누락/변경은 거부한다. 새로운 진단용 예외 플래그를 만들지 않았다.

공통 정책 키는 `sr`, `mono`, `default_peak_cap`, `preserve_gain`, `demucs`다. `source_peak`, `normalization_gain`, 선택적 `normalized_rms`는 **파일별 측정값**이며 행 간 동일성을 요구하지 않는다. 측정값은 각 파일의 캐시 identity에는 남지만 dev/lock 정책에는 복제하지 않는다. 허용 정책하에서 peak 0.3/gain 1.0과 peak 0.7/gain 약 0.85714가 함께 읽히는 것을 확인했고, CPU encoder test double의 0.3/1.0 및 1.2/0.5 측정값도 실제 adapter→캐시 쓰기→읽기로 검사했다.

| 실제 호출 경로 | 변경 |
|---|---|
| manifest 검증 / `extract_audio` | 모든 행 정책을 native engine 로딩 전에 검사한다. 뒤쪽 행의 True도 선제 거부한다. |
| `record_from_engine` | False 정책을 사용하고 실제 측정 peak/gain과 함께 기록한다. True를 직접 호출해도 거부한다. |
| `workflow.load_records` / dev | sidecar hash뿐 아니라 manifest/cache 정책을 검사하고 정책만 동일성 키에 넣는다. |
| 최종 fit / bundle / lock | dev report와 cache 정책을 확인하고 production bundle·lock에 공통 정책을 저장한다. |
| OOD / cache inference | OOD 캐시 및 production 모델·lock 정책을 비교한다. 실패한 OOD 열람 marker는 보존한다. |
| `load_deployment` / `predict_engine_file` | 명시적인 production 정책 없이는 native 실행에 진입하지 않는다. 기존 False 추론과 File-only 교체를 유지한다. |

정책이 없는 v1 dev report/production bundle/lock에 기본값을 덧붙여 통과시키지 않는다. 검증된 manifest/cache로 dev·fit·lock을 다시 만들어야 한다. 완전한 False 정책을 가진 기존 extended cache는 CPU에서 읽을 수 있지만, 기존 코드/체크포인트 identity 검사는 유지하므로 v1 캐시를 v2 GPU 추출로 자동 재서명하지 않는다. 실제 배포와 추출 코드 identity가 맞아야 한다.

## 2. optional dependency smoke 수정

`demo.run_smoke`에서 `fit_final_dev`를 후보별 try 안으로 옮겼다. `fit`, `save`, `load`, `roundtrip` 단계를 기록하고 `NotRunError`를 해당 후보의 `NOT_RUN`/`stage`/`reason`으로 남긴다. 각 후보 종료 때 `serialization_results.json`을 저장한다. 다른 모델 대체, pickle/joblib 우회, skops trusted type 자동 확대는 없다.

XGBoost import를 차단한 작은 재현에서 v1은 예외로 중단되어 serialization report가 없었다. v2는 F0/F1의 LR·ExtraTrees 실제 sklearn 학습/CV 결과를 남겼고, XGBoost는 `NOT_RUN`(fit), 최종 보고서는 세 후보를 모두 포함했다. fit/save/load 각 단계의 NotRunError 검사도 수행했다.

## 실제 실행한 검증

| 검증 | 이번 실행 결과 | 로그/자료 |
|---|---|---|
| 수정 전 기존 suite | 14 passed. 이후 추가한 결함 검사는 수정 전 16 failed/1 passed로 누락을 재현 | `logs/v2/01_*`, `03_*` |
| 최종 전체 CPU suite | **35 passed**, 오류/실패 없음; parameterized cases 포함 | `logs/v2/09_final_pytest.txt`, `artifacts/v2_tests.xml` |
| v1/v2 직접 재현 | v1의 정책 누락·미설치 중단 확인, v2의 정책 거부·보고서 보존 확인 | `logs/v2/08_*`, `11_*` 및 `artifacts/v2_review_reproduction*/` |
| 동일 정책/다른 측정값 | 허용. False/True 혼합, manifest False/cache True, 전부 True 및 누락/다른 정책은 거부 | 위 재현 + `tests/test_preprocessing_contract.py` |
| dev→fit→bundle→lock→OOD | 실제 모듈 호출로 통과. **합성 fixture에 대한 연결 검증**이며 real OOD 평가가 아님 | 최종 pytest / 해당 workflow tests |
| 실제 기존 head/cache | 6 head + 10 NPZ CPU 재계산; 제공된 과거 진단값과 최대 차이 0.0012562871, 허용치 0.002 이내 | `logs/v2/13_input_audit.txt`, `artifacts/v2_input_audit/` |
| 과거 diagnostic baseline | 2,852행 max 수식 재현 오차 0.0, 불완전 라벨 총점은 null | 위 input audit |
| 현재 환경 기본 smoke | F0/F1 × LR/ExtraTrees/XGBoost 6개 시스템 실행 | `logs/v2/12_final_smoke.txt` |
| XGBoost 표준 JSON 저장·재로드 | PASS, 최대 예측 차이 **0.0** | `artifacts/v2_synthetic_wiring/serialization_results.json` |
| LR/ExtraTrees skops 저장·재로드 | **NOT_RUN, save 단계**. skops 미설치; 두 pip 설치와 공식 wheel 다운로드 실패 | `logs/v2/02_*`, `04_*`, `07_*` |
| 변경 범위 / 원본 해시 | 수정 src는 6개 모듈. 후보 grids/allowlist/feature 계산/scorer/기존 Engine 패치는 v1과 동일 | `logs/v2/15_scope_and_v1_integrity.txt` |

설치 실패의 원인은 로그에 관측된 `No matching distribution found`이며, 공식 PyPI의 0.13.0 릴리스가 없다는 뜻으로 해석하지 않는다. 버전은 바꾸지 않았다. 이 환경은 Python 3.13.5, numpy 2.3.5, sklearn 1.8.0, XGBoost 3.1.3, torch 2.10.0+cpu다. 상세 설치 버전과 명령은 `VERIFICATION.json`에 있다.

일괄 검증 runner 한 번은 도구의 120초 제한으로 종료됐다. 그 전에 완료된 pytest/compile/reproducer 결과만 인정했고, 미완료 smoke는 별도 명령으로 다시 실행해 성공을 확인했다. 제한 종료도 `logs/v2/12a_aggregate_timeout.txt`와 검증 receipt에 남겼다.

## 미실행 범위

**신규 음원 수집 0개**, 신규 독립 meta-dev/real OOD 성능 평가, 실제 production 후보 선정·배포, native GPU extraction/parity, 전체 실제 모델 자산 identity 검증, F2/F2R 신규 추출, L4 시간/VRAM, 공식 점수는 모두 미실행이다. GPU 시작·VESSL 작업·GitHub 변경·공식 제출을 하지 않았다. CPU 테스트의 native encoder/자산 경계는 명시적 test double이므로 GPU parity라고 표현하지 않는다. LR/ET 안전한 skops save/load도 미실행이며, 정상 설치 후 같은 smoke를 다시 실행해야 한다.

v1의 기존 로그와 결과는 삭제하지 않고 보관했다. **이번 검증은 `logs/v2/`, `artifacts/v2_*`, 현재 `VERIFICATION.json`으로만 식별**한다. `history/v1_*`와 이전 `logs/01_...`~`21_...`는 보관 증거다. 원래 실행을 이번 실행처럼 합산하지 않는다.

## 적용 파일과 Codex 검사 순서

실제 변경한 모듈은 `common.py`, `provenance.py`, `workflow.py`, `gpu.py`, `models.py`, `demo.py`이며 모두 `src/dacon_stacking/` 아래에 완전한 내용으로 제공한다. `CHANGES.json`에는 파일별 원본/반환 해시, 적용 경로, 변경 이유가 있다. `patches/v2_review_fixes.patch`는 v1 반환 루트에 적용하는 구현·테스트·관련 문서 diff다. 생성된 결과/해시 inventory는 패치로 재현하지 않고 현재 완전한 파일을 제공한다. 기존 `patches/full_bank_submission.py`와 `.patch`는 v1 그대로다.

```bash
export RETURN_ROOT=/path/to/DACON_Stacking_Return_v2
export STACKING_INPUT_ROOT=/path/to/DACON_Chat_Stacking_Input
cd "$RETURN_ROOT"
export PYTHONPATH="$RETURN_ROOT/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m pytest -q
python tests/reproduce_review_findings.py --expect fixed --out local_review_v2
python -m dacon_stacking smoke --out local_smoke_v2
```

`--out`에는 새 디렉터리를 사용한다. 전체 실행·미실행 명령은 `RUNBOOK.md`, `VERIFICATION.json`, 최종 파일별 해시는 `OUTPUT_SHA256.json`에서 확인한다. 이 패키지를 그대로 검사하는 것이 기본이며, v1 복사본에 diff만 적용할 때는 아래를 사용할 수 있다.

```bash
# v1 반환 루트에서 실행. 기존 제출 프로젝트 루트에 바로 적용하는 패치가 아니다.
patch --dry-run -p1 < /path/to/DACON_Stacking_Return_v2/patches/v2_review_fixes.patch
patch -p1 < /path/to/DACON_Stacking_Return_v2/patches/v2_review_fixes.patch
```
