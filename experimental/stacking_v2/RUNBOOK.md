# 실행 안내 — v2 한정 수정 / 로컬 CPU / 추후 로컬 GPU

이 문서의 신규 데이터·GPU·production lock·OOD·배포 명령은 **실행 안내**다. v2에서 실제 실행한 명령은 VERIFICATION.json과 logs/v2/로 한정한다. history/와 기존 logs/의 01~21 로그는 v1 보관 자료이며 재실행 주장에 사용하지 않는다. GitHub/VESSL 접속·인스턴스 생성·업로드·공식 제출을 수행하지 않았다. Notebook은 만들지 않았다. 아래 CLI가 동일 핵심 모듈을 호출한다.

## 1. CPU 재검사

반환 ZIP과 입력 ZIP을 각각 풀고 경로를 지정한다. `INPUT_ROOT` 바로 아래에 00_START_HERE.md와 fixtures/가 있어야 한다.

```bash
export RETURN_ROOT="$PWD/DACON_Stacking_Return_v2"
export INPUT_ROOT="$PWD/DACON_Chat_Stacking_Input"
cd "$RETURN_ROOT"
export PYTHONPATH="$RETURN_ROOT/src"
export STACKING_INPUT_ROOT="$INPUT_ROOT"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python -m dacon_stacking --help
python -m dacon_stacking audit --input-root "$INPUT_ROOT" --out local_audit
pytest -q
python -m dacon_stacking smoke --out local_synthetic_wiring
```

기대: 133개 입력 manifest 파일 검증, 실제 캐시 10개 재생, 원래 diagnostic baseline max 오차 0, CPU vs 제공된 과거 예측의 최대 차이 <=0.002. 같은 CPU 수식/저장 후 XGBoost 재로드 비교는 별도이며 현재 측정 round-trip 차이는 0이다. 과거 GPU가 생성한 캐시로 CPU head를 계산한 것이므로 0.002 허용은 **새 GPU parity 허용을 승인한 것이 아니다**.

smoke는 synthetic 72행/36그룹으로 6개 시스템의 train/eval wiring을 실행한다. 이름을 바꿔도 이 데이터/모델은 독립 실험 또는 배포 후보가 아니다. 이번 실제 결과는 `artifacts/v2_synthetic_wiring/`에 있다.

## 2. 의존성

현재 컨테이너에서는 Python 3.13.5/Linux CPU와 requirements-cpu-verified.txt의 설치된 버전을 사용했다. XGBoost 3.1.3이 설치돼 있었다. scratch 설치 및 macOS/Windows 호환성은 검증하지 않았다. 필요한 환경에서 설치하는 명령은 다음과 같다.

```bash
python -m pip install -r requirements-cpu-verified.txt
python -m pip install -r requirements-xgb.txt
python -m pip install -r requirements-serialization.txt
```

v2에서 skops==0.13.0의 두 pip 설치 시도는 `No matching distribution found`로 실패했고, 공식 wheel 직접 다운로드 도구도 실패했다. 설치된 배포본/모듈은 없으므로 skops 저장·재로드는 NOT_RUN이다. 이 오류를 해당 PyPI 버전이 존재하지 않는다는 뜻으로 해석하지 않는다. LR/ExtraTrees fit/predict/CV는 가능했지만 안전한 save/load는 NOT_RUN이다. 설치 후 동일 라이브러리 조합으로 smoke를 재실행해야 한다. skops의 untrusted type이 나오면 자동 trusted 확대나 pickle로 우회하지 말고 원인을 검토한다. XGBoost가 없는 환경에서는 해당 후보가 NOT_RUN이며 다른 모델로 대체하지 않는다.

GPU deployment의 기존 torch/torchaudio/transformers 환경을 CPU 검증 버전으로 덮어쓰지 않는다. requirements-gpu-overlay.txt는 **미검증 overlay**다. 최종 meta 학습과 배포의 numpy/scipy/sklearn/XGBoost 또는 skops 버전을 맞춘다. loader는 버전 차이를 거부한다. 현재 synthetic CPU bundle을 L4에서 바로 쓰는 경로는 지원/검증하지 않는다.

## 3. 원본 수집·split·manifest

`DATA_PLAN.md`, `DATA_CANDIDATES.csv`, `configs/data_contract.json`을 따른다. CSV old val/test를 JSONL로 바꾸는 것은 신규 데이터 수집이 아니다.

```bash
python -m dacon_stacking split-parents \
  --parents new_data/original_parents.json \
  --reservations new_data/approved_ood_reservations.json \
  --out new_data/parent_assignments.json
python -m dacon_stacking validate-data \
  --manifest new_data/all_manifest.jsonl \
  --input-root "$INPUT_ROOT" --out new_data/data_audit.json
```

`original_parents.json`은 parent dict 목록이며 `approved_ood_reservations.json`은 reserved_sources/reserved_generators 목록이다. 이 파일들은 신규 자산 확보 뒤 작성해야 한다. 예시 reservation의 REPLACE_WITH 값은 실제 승인 목록이 아니다. 원본들을 나눈 뒤 `provenance.derivative_split(parent_ids, assignments)`로 파생 파일의 부모 split을 확인한다. mix/partial/codec 렌더링 후 hash와 실제 의미 라벨을 다시 기록한다. 일반 공개 음악에 보컬이 있을 수 있으므로 source만 보고 Music-only로 라벨링하지 않는다.

새로운 JSONL 각 행에는 ID, split(meta_dev/locked_ood), category, audio_path/audio_sha256, parents, 다섯 label/mask, base_exposure가 필요하다. 부모별 source/generator_family는 canonical 값으로 검토해야 한다. registry와 충돌하거나 head_train/head_selection/historical_development가 yes/unknown이면 거부한다. upstream unknown은 기록하되 downstream 신규성 증거와 혼동하지 않는다.

## 4. GPU 추출은 별도 경로

`SOURCE_ROOT`는 전체 현재 제출 source 패키지 복사본이다. 입력 ZIP의 소형 submission_source만으로는 큰 weights/ffmpeg/head 파일이 누락되어 실행할 수 없다. 여섯 head는 fixture로 확인했지만 실배포 root에는 원래 trained 파일 위치가 필요하다.

```bash
export SOURCE_ROOT="$PWD/local_full_submission_source"
python -m dacon_stacking stage-runtime \
  --source-root "$SOURCE_ROOT" --return-root "$RETURN_ROOT"
python -m dacon_stacking extract \
  --manifest new_data/meta_dev.jsonl --input-root "$INPUT_ROOT" \
  --source-root "$SOURCE_ROOT" --out new_cache/meta_dev
```

stage는 원래 runtime SHA를 확인하고 반환 수정본 및 package를 로컬 copy에 적용하며 manifest를 갱신한다. 기존 package/다른 runtime을 덮어쓰지 않는다. stage receipt는 deployment_ready=False다. 큰 자산을 다운로드하거나 인스턴스를 만들지 않는다. extract는 CUDA가 있어야 하며 현재 모델·코드·가중치 manifest를 검사한다. production gain 정책은 False(peak cap 적용)로 고정한다. `preserve_gain=True`와 문자열/정수/None은 manifest 검증 및 extract 사전검사에서 거부한다. extract는 모든 행을 검사한 뒤에만 native engine을 로드한다. `record_from_engine`도 True를 거부한다. 기존 Engine의 진단용 메서드와 역사적 캐시 replay를 바꾸지는 않지만, gain-preserving 결과를 production 후보로 가져오는 예외는 제공하지 않는다.

기본 extract는 F1/F2 및 shortcut 원자료를 기존 호출에서 기록한다. F2R는 다음처럼 raw 비교를 켠다. 별도 디렉터리를 사용하며 캐시를 덮어쓰지 않는다.

```bash
python -m dacon_stacking extract \
  --manifest new_data/meta_dev.jsonl --input-root "$INPUT_ROOT" \
  --source-root "$SOURCE_ROOT" --out new_cache/meta_dev_f2r --raw-comparison
```

locked OOD 추출은 데이터 담당자가 같은 코드/가중치로 별도로 수행하고 JSONL과 index의 SHA-256만 개발자에게 제공하는 것을 권장한다. 추출은 평가/튜닝이 아니며 labels를 feature로 사용하지 않는다. OOD 오디오/점수/라벨을 모델 선택에 열람하지 않는다. 원본/확장 cache 파일의 provenance를 서로 바꿔 끼우면 hash/identity 검사에서 실패한다.

이미 구성된 VESSL 또는 다른 로컬 CUDA 환경에서도 같은 filesystem 명령을 실행하는 구조다. 이 문서는 VESSL API/워크스페이스/인스턴스 생성 방법을 호출하거나 계정에 접근하지 않는다. L4 시간·VRAM 적합성은 실제 profiling/parity 후 결정해야 한다.

## 5. 새 meta-dev 비교

```bash
python -m dacon_stacking dev \
  --manifest new_data/meta_dev.jsonl --index new_cache/meta_dev/index.json \
  --config configs/experiment.json --input-root "$INPUT_ROOT" --out experiments/f0_f1
```

같은 행과 outer fold에서 F0/F1×LR/ExtraTrees/XGBoost, max/noisy-OR, 다섯 출력 지표/분모/unknown, File paired bootstrap, 조건별 분석과 train/eval group trace를 저장한다. 파일별 CSV는 정확히 ID+5열이다. F2 비교는 configs/experiment_f2.json, shortcut 비교는 experiment_shortcut.json, 표본 가중치 민감도는 experiment_file_equal.json을 별도로 사용한다. 선택 기준/추가 실험 계획은 OOD를 열기 전에 고정한다.

## 6. 최종 후보 lock — OOD는 파일 경로가 아니라 해시만 전달

다음 예시의 LR/F1은 명령 사용법이지 성능상 추천/선정 결과가 아니다. 실제 dev 비교와 사전 결정 기준으로 후보를 하나 고른다. OOD digest는 데이터 담당자가 확인한 값을 환경변수로 제공한다.

```bash
export OOD_MANIFEST_SHA256="$(sha256sum new_data/locked_ood.jsonl | cut -d ' ' -f1)"
export OOD_INDEX_SHA256="$(sha256sum new_cache/locked_ood/index.json | cut -d ' ' -f1)"
python -m dacon_stacking lock \
  --manifest new_data/meta_dev.jsonl --index new_cache/meta_dev/index.json \
  --config configs/experiment.json --input-root "$INPUT_ROOT" \
  --dev-report experiments/f0_f1/dev_report.json --kind lr --features F1 \
  --ood-manifest-sha256 "$OOD_MANIFEST_SHA256" \
  --ood-index-sha256 "$OOD_INDEX_SHA256" --out candidate
```

최종 fit의 hyperparameter 선택도 dev 내부에서만 이루어진다. synthetic/exposed rows, 일치하지 않는 dev report/config, 평가되지 않은 모델, 기존 candidate 디렉터리는 거부된다. skops가 없으면 LR/ET lock은 NOT_RUN이며 XGBoost로 몰래 변경하지 않는다.

## 7. 별도 locked OOD 평가

```bash
export LOCK_SHA256="$(sha256sum candidate/candidate_lock.json | cut -d ' ' -f1)"
python -m dacon_stacking ood \
  --manifest new_data/locked_ood.jsonl --index new_cache/locked_ood/index.json \
  --input-root "$INPUT_ROOT" --lock candidate/candidate_lock.json \
  --lock-sha256 "$LOCK_SHA256" --out final_ood_evaluation
```

manifest/index/후보 hash, dev-parent/speaker/track/hash/source/family 겹침, extraction identity를 검사한다. OOD_OPENED.json은 exclusive creation이며 실패에도 남긴다. 이를 지우고 튜닝·재평가하여 같은 OOD를 미노출로 부르지 않는다. 공식 총점이 null이면 원래 분모/unknown/invalid 원인을 유지한다.

## 8. 파일별 추론과 배포 이전 점검

```bash
python -m dacon_stacking infer-audio \
  --audio local_audio/example.wav --id example --source-root "$SOURCE_ROOT" \
  --lock candidate/candidate_lock.json --lock-sha256 "$LOCK_SHA256" \
  --out single_file_prediction.csv
```

한 파일의 피처만 사용하고 File만 교체한다. 기존 `run` 진입점을 사용할 때는 다음 두 환경변수를 모두 설정한다. 설정하지 않으면 기존 max 경로다.

```bash
export DACON_STACKING_LOCK="$PWD/candidate/candidate_lock.json"
export DACON_STACKING_LOCK_SHA256="$LOCK_SHA256"
```

최종 오프라인 제출 ZIP에 포함할 때 candidate_lock.json, bundle/와 해당 SHA를 함께 관리하고 원래 packager의 manifest/검증·offline dependency 설치를 다시 수행해야 한다. 이 단계는 실행하지 않았다. 입력이 제공하지 않은 큰 자산을 이 반환 ZIP에 임의로 복제하지 않는다.

배포 전 Codex 검사는 **실제 동일 음원**의 기본 Engine vs patched-disabled vs patched-enabled를 비교해야 한다. 네 출력은 동일 numeric mode에서 일치해야 하며, single/reordered/subset/chunk 실행도 동일해야 한다. default gain/short audio/silence/tail/MP3 decode를 포함하고, production의 preserve_gain=True는 GPU 추론 전에 거부되는지 확인한다. native GPU vs cached CPU 특징 차이, peak memory, L4 시간제한, 전체 6열 CSV 검증을 끝내기 전에는 제출 적합성을 주장하지 않는다.


## v2 재검사 및 전처리 계약

이번 수정은 전처리 계약과 optional-dependency smoke 복구뿐이다. 모델 후보, feature allowlist, scorer, base heads/encoder, 원래 제출 패치의 수치 연산은 바꾸지 않았다.

```bash
# --out에는 기존 결과가 없는 새 경로를 쓴다.
python -m pytest -q
python tests/reproduce_review_findings.py --expect fixed --out local_v2_review_reproduction
python -m dacon_stacking smoke --out local_v2_smoke
```

reproducer는 제공된 검토 설명을 바탕으로 작성한 별도 파일이다. `reproduce_preprocessing_gap.py`와 별도 검토 첨부는 이 실행에서 확보되지 않았다. v1 코드에서의 재현은 `PYTHONPATH=/path/to/v1/src python tests/reproduce_review_findings.py --expect gaps --out local_v1_gap_reproduction`으로 실행한다. `--expect gaps`의 PASS는 결함이 관측됐다는 뜻이며 v1 수정 성공이 아니다.

정책 키는 `sr`, `mono`, `default_peak_cap`, `preserve_gain`, `demucs`다. `source_peak`, `normalization_gain`, 선택적 `normalized_rms`는 파일별 측정값이다. 측정값은 각 sidecar identity에 계속 포함하되 행 간/lock 간 정책 비교에는 넣지 않는다. 유효한 서로 다른 peak/gain은 허용한다. 정책 누락·지원하지 않는 정책·manifest/cache 불일치는 실패한다.

`dev_report.json.preprocessing_policy`, bundle의 `provenance.preprocessing_policy`, `candidate_lock.json.preprocessing_policy`는 공통 정책만 저장한다. dev/lock/OOD/`infer-cache`의 production bundle/`load_deployment`/`predict_engine_file`에서 검사한다. 구 v1 dev report·production bundle·lock처럼 필드가 없는 파일은 자동 보정하지 않는다. 검증된 입력으로 dev와 lock을 다시 만들고, policy를 수동 삽입하여 과거 결과를 v2 검증 결과처럼 만들지 않는다.

정책이 완전하고 False인 구 extended cache는 CPU에서 읽을 수 있다. 그러나 전체 코드 SHA identity도 유지하므로 v1 추출 코드와 v2 deployment 코드가 다른 캐시를 자동 재서명하지 않는다. 실제 배포용 캐시는 배포와 동일한 검증된 extraction identity가 필요하며 해당 GPU 추출/parity는 이번에 실행하지 않았다.

smoke는 각 후보의 `fit` → `save` → `load` → `roundtrip` 단계를 별도로 추적한다. `NotRunError`는 후보별 `NOT_RUN`과 `stage`/`reason`으로 남기고 다음 후보로 진행한다. `serialization_results.json`은 각 후보가 끝날 때 갱신하여 이미 끝난 후보의 상태를 보존한다. XGBoost가 없다고 LR/ET로 바꾸거나, skops가 없다고 pickle/joblib로 우회하지 않는다.

CPU 회귀 테스트에는 실제 NPZ/JSON I/O 및 합성 점수 기반 dev→fit→JSON bundle→lock→OOD 경로가 포함된다. native loader/encoder 경계만 test double로 검사한 extraction/deployment 연결은 실제 오디오·GPU parity가 아니다. 신규 음원·F2 GPU·L4·공식 점수는 여전히 미실행이다.
