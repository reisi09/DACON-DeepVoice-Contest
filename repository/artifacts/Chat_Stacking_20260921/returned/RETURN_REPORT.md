# DACON Stacking 반환 보고서

**상태: 구현 파일과 CPU 검증 산출물 제공. 새 데이터 기반 학습 완료/성능 개선/배포 준비 완료를 뜻하지 않는다.**

## 완료한 작업

현재 제출 snapshot과 학습 소스를 읽고 실제 수치 흐름·feature availability·timebase를 감사했다. 입력 SHA-256은 `7adfb7b426c45b0efbb23de581ca1a5a073079d2110f4f4f84cdcf0fca2fc75a`다. 133개 manifest 항목의 해시/크기가 일치했다. 기준 제출 파일은 `current/submission_source/model/runtime/full_bank_submission.py`이며 과거 reference/legacy는 실행 경로에 연결하지 않았다.

현재 6개 head와 NPZ를 재사용하는 CPU adapter, F0 4개/F1 32개/F2 48개/F2R 52개/F1S 38개 피처 계약, 실제 시간 겹침 정렬, 금지 피처/누락/order 검사, 새 데이터 exposure·연결 parent 검증, grouped nested meta-CV, LR/ExtraTrees/XGBoost의 작은 grid, File-only inference, 모델·캐시 provenance, 별도 candidate lock/OOD 명령을 구현했다. CPU/GPU 경로는 분리했다.

기존 파일 변경은 **현재 full_bank_submission.py 한 개**다. 완전한 수정본과 unified diff를 `patches/`에 제공한다. telemetry는 opt-in이며 기본 실행은 기존 max다. 명시적 production lock이 있는 경우에만 File을 교체한다. 나머지 네 출력은 기존 수치를 그대로 보존한다. 소스 복사본에 패치/manifest staging을 실제 실행했지만, 큰 자산이 누락된 상태이므로 deployment_ready=False다.

데이터 목표는 약 3,200개(dev 2,560 / locked OOD 640)로 계획하되 원본/출처/생성기 부족 시 축소한다. 실제 신규 수집은 0개다. 독립 parent/화자/곡/생성기 count와 파일 count를 분리했고, 기존 노출·upstream unknown·라이선스/재배포 조건을 문서화했다. 기존 base 학습 데이터를 재사용하려면 전체 stack의 nested/cross-fitting이 필요한 대안을 별도로 설명했다.

## 실제 실행한 검증

| 검증 | 이번 관측 결과 | 증거 |
|---|---|---|
| 입력 무결성 | 133개 모두 일치 | artifacts/audit/input_audit.json |
| 실제 cache + 현재 head 재계산 | 10 NPZ / 6개 head, CPU | artifacts/audit/fixture_replay.json |
| 제공된 과거 예측과 비교 | 최대 절대 차이 0.0012562871, 사전 테스트 허용 0.002 이내 | logs/18_verified_input_audit.txt |
| 기존 max baseline | val 1,426 + test 1,426행, 최대 수식 차이 0 | artifacts/audit/input_audit.json |
| 원래 scorer + mask | 기존 val/test 총점은 null; 누락/클래스 사유 유지 | 같은 audit의 diagnostic |
| 의미 있는 unit/integration 검사 | **14 passed**, 최종 실행 exit 0 | logs/15_verified_pytest.txt |
| 현재 numerical AST/scorer byte 보존 | 검사 통과 | tests/test_integration.py |
| synthetic F0/F1×세 모델 | 같은 outer split/행에서 6개 시스템 실행 | logs/20_verified_smoke.txt |
| XGBoost 표준 JSON save/load | 최대 예측 차이 0.0 | artifacts/final_synthetic_wiring/serialization_results.json |
| 파일 순서·부분집합·단독 입력 | XGBoost 결과 동일, 금지 feature order 거부 | tests/test_guards.py |
| 원본 vs opt-in runtime wiring | stub encoder + 실제 head에서 baseline/네 출력 보존 | tests/test_integration.py |
| 정적 컴파일 / CLI | exit 0 | logs/16_verified_compileall.txt, logs/17_verified_cli_help.txt |
| 로컬 source copy staging | 파일 hash 검증·patch 적용, 자산 누락 명시 | logs/19_verified_stage_runtime.txt |

실행 환경은 Python 3.13.5/Linux, torch 2.10.0+cpu, numpy 2.3.5, pandas 2.2.3, scipy 1.17.0, sklearn 1.8.0, XGBoost 3.1.3이다. 정확한 명령·환경·exit code·검증 당시 source hash는 VERIFICATION.json에 있다. 테스트 최초 실패 로그도 보존했으며 마지막 suite에서는 실패가 없다.

**CPU cache 재생은 GPU parity가 아니다.** GPU에서 계산된 encoder 특징/과거 bf16 head 결과와 이번 CPU fp32 head 계산의 비교다. stub encoder 테스트도 실제 음원/큰 encoder를 통과한 결과가 아니다. synthetic 성능 수치는 wiring만 확인하며 일반화·개선·추천 모델 선정의 근거로 사용하지 않는다.

## 아직 실행하지 않은 작업

LR/ExtraTrees의 안전한 skops 저장·재로드는 **NOT_RUN**이다. skops 설치 및 wheel 다운로드 시도가 실패했으며 `logs/02_install_skops.txt`에 pip 오류가 있다. 해당 두 모델의 학습·예측·CV는 실행했다. 구현은 skops를 사용하고, pickle 또는 다른 모델로 조용히 대체하지 않는다. 설치/버전 호환성은 후속 환경에서 확인해야 한다.

신규 meta-dev/OOD 원본 수집, 실제 메타 모델 학습·선택, production lock부터 새 OOD까지의 전체 실행, GPU 추출/F2/F2R, native GPU parity, L4 시간/VRAM, 공식 점수·업로드·제출은 실행하지 않았다. 필요한 정확한 package member 경로/예상 hash 및 신규 오디오·라벨·cache 요구는 MISSING_ASSETS.json에 있다. 기존 full train caches 전체와 큰 encoder assets는 입력에 없었다.

공식 데이터 후보 페이지/라이선스 원문은 일부 웹 확인했지만 오디오를 수집하지 않았다. 개별 음원/플랫폼 권리와 upstream 노출은 미해결 항목을 그대로 남겼다. GitHub·VESSL 접속, Work 생성, 원격 업로드, GPU 인스턴스 생성은 하지 않았다.

## Codex가 검사할 위치

먼저 `RUNBOOK.md`의 CPU 명령과 `tests/README.md`의 결함별 체크를 실행한다. 현재 원본 SHA와 `configs/runtime_patch.json`, `configs/snapshot_symbols.json`, `CHANGES.json`을 확인한다. 이어 `src/dacon_stacking/features.py`의 head 재생/시간 정렬, `provenance.py`의 exposure/두 부모 검사, `models.py`의 fold-local fit/저장, `evaluation.py`의 nested tuning/정확한 scorer, `workflow.py`의 lock/OOD/단일 파일 경로, `gpu.py` 및 완전한 runtime patch를 검토한다.

실제 사용 전 skops round-trip과 GPU parity를 별도로 실행해야 한다. 새로운 데이터가 확보된 후에만 production dev/lock/OOD 절차를 진행한다. `artifacts/final_synthetic_wiring`의 bundle은 배포 후보가 아니며 production loader가 거부한다. 독립 reviewer를 이 세션에서 별도로 실행한 것은 아니다.

## 파일 안내

DESIGN.md는 수치 코드 근거·비교 실험·누수 없는 평가 설계, FEATURE_SPEC.csv는 개별 피처 명세, DATA_PLAN.md/DATA_CANDIDATES.csv는 목표·실제 확보·권리/출처 상태, RUNBOOK.md는 로컬 실행 순서다. VERIFICATION.json/logs/는 실제 실행 증거, OUTPUT_SHA256.json은 반환 파일별 해시이며 자기 자신은 제외한다. ZIP 최상위에서 이 보고서를 바로 찾을 수 있다.
