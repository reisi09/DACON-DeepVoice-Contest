# DACON 236749 · existing-only 9504 · A100 Run All — reviewer README

작성: Claude (author). 검수: Codex. **GPU 실행·VESSL 시작·다운로드·학습은 하지 않았습니다.** 이 문서는 무엇을 만들었고, 무엇을 CPU에서 실제로 검증했으며, 무엇이 A100에서만 확인 가능한지 구분합니다.

## 0. v3 변경 (Codex v2 검수: handoff 결함 2건)

아키텍처·데이터·분할·학습·평가 코드는 v2와 같고, handoff 게이트와 마지막 다운로드 셀만 고쳤습니다.

| # | v2 결함 | v3 처리 | 확인 (CPU) |
|---|---|---|---|
| 1 | `coverage_report`가 예측 **ID**만 보고 해시를 lock·receipt 증거와 대조하지 않음(같은 ID로 값만 바꾼 test 예측, 0으로 채운 parity lock 해시가 통과) | 완료 ZIP 게이트가 handoff 멤버 바이트만으로 판정: val 예측·val 지표·results·run_identity·초기 fc5·feature index·manifest = `candidate_lock` 기록 해시, test 예측·test 지표 = `test_receipt` 해시, test_receipt·test 지표·parity receipt의 `lock_sha256` = 현재 lock 파일 해시(val/test 지표 안의 예측 해시도 대조). 체크포인트 6개·정확 커버리지 검사는 유지. 같은 게이트를 **패키징될 파일**에 한 번, **ZIP 안에 실제로 들어간 바이트**에 다시 한 번 돌려 두 결과가 같아야 게시 | 실제 `build(..., "complete")` 게이트로: 같은 ID에서 test FILE_FAKE_PROB 1개 변경, parity/test receipt의 lock 해시 0으로 변경, val 예측 1개 값 변경, val/test 지표 변경 → 모두 거부. 게이트 통과 직후 파일을 바꾸면 ZIP 재검사에서 거부되고 partial ZIP이 남지 않음. 완성 ZIP을 다시 읽으면 기록된 coverage와 동일, ZIP 안 test 예측 한 값만 바꾼 사본은 거부 |
| 2 | 마지막 셀 `FileLink(os.path.relpath(/root-1/...))`가 `/root` 범위 Jupyter에서 `../root-1/...`가 되어 받을 수 없음 | 오브젝트 볼륨 사본은 그대로 두고, 노트북 디렉터리(커널 cwd)에 SHA-256을 확인한 사본을 노출: 같은 파일시스템이면 job 로컬 검증본의 **하드링크**(추가 디스크 없음), 아니면 해시 검증 복사(여유 공간 부족 시 명확한 오류). 심볼릭 링크는 Jupyter가 루트 밖 대상을 거부하므로 쓰지 않음. 링크는 `..` 없는 상대 경로라 Jupyter `/files/` 핸들러가 스트리밍. base64 삽입 없음 | `tests/test_download_link.py`: 실제 생성 노트북의 부트스트랩+마지막 셀을 커널에서 실행하고, 루트를 제한한 로컬 `jupyter_server`로 링크를 받아 sidecar SHA-256과 바이트 일치 확인. A: 노트북은 루트 아래, deployment·job은 루트 밖 → 새 링크 200·해시 일치·하드링크, v2식 링크는 실패. B: 노트북이 deployment 디렉터리(루트 안)에 있을 때 게시본을 그대로 제공. EXDEV(파일시스템 다름)·같은 이름의 오래된 파일 → 검증 복사로 교체 |

## 0-1. v2 변경 (Codex v1 검수 결함 수정)

아키텍처·데이터·분할은 그대로이고, 지적된 결함만 고쳤습니다. 각 항목 옆은 그 결함을 CPU에서 재현·확인한 검사입니다.

| # | v1 결함 | v2 처리 | 확인 |
|---|---|---|---|
| 1 | `train_route_seed` 재개 시 RNG 미복원, 재개 때 벤치마크 재측정 | resume 파일에 Python/NumPy/Torch CPU/CUDA RNG 상태, 측정된 microbatch·window limit 저장. 재개 시 벤치마크를 다시 하지 않고(그룹이 바뀌면 부동소수 누적 순서가 바뀜) 모델·옵티마이저·조기종료 상태 뒤에 RNG 복원. 벤치마크 전후 RNG 보존. resume은 `weights_only=True`로 읽음 | 실제 WindowMIL로 5 epoch 무중단 vs 3 epoch 첫 step에서 중단 후 재개: 체크포인트·최종 가중치·val 예측·이력 bitwise 동일, 매 step 세 RNG 추출값 동일. RNG를 소비하는 벤치마크 대역도 추출값 불변 |
| 2 | `compare_arrays`가 NaN 대 수치를 0 차이로 봄, parity mismatch여도 handoff complete | NaN 위치 불일치·inf·비유한 차이는 mismatch(양쪽 NaN만 일치 허용). 예측도 같은 규칙. mismatch면 receipt(NaN은 문자열로) 저장 후 raise → FAILED ZIP. 완료 handoff는 parity가 bitwise/within_tolerance일 때만 | `[NaN,1]` vs `[0,1]`, `[1]` vs `[NaN]`, inf → mismatch. test 예측 CSV를 바꿔 실제 parity를 돌리면 raise, receipt `mismatch`, 완료 handoff 거부 |
| 3 | `_gpu_idle`·`StageLock`이 살아 있는 같은 job 프로세스를 종료, 소유 확인 전 config/state 기록 | job 단위 controller lock(flock)을 **어떤 기록보다 먼저** 획득. 다른 살아 있는 controller는 파일·프로세스를 건드리지 않고 거부. 같은 커널에서 부트스트랩 셀 재실행 시 소유권 이전, 이전 객체는 실행 거부. 자식 프로세스는 pid+커널 시작시각+controller로 기록하고, controller가 죽은(또는 유휴인 자기 자신) 기록된 고아만 회수. 기록되지 않은 stage 프로세스가 있으면 명확히 거부. StageLock은 누구도 죽이지 않음 | `tests/test_controller.py`(두 번째 controller: 파일 스냅샷 불변·첫 controller와 자식 생존, StageLock 보유자 생존, 미기록 프로세스 → 거부·미종료, 죽은 controller의 고아만 회수, 같은 프로세스 재실행 소유권 이전·다른 프로세스 거부) |
| 4 | pilot receipt 무조건 재사용, `verify_lock`이 run_identity·설정·val 증거를 안 봄, 자산 marker+파일 누락 시 해시 단계에서 크래시 | pilot·lock에 manifest·자산·코드·환경·설정 fingerprint 저장, 재사용 전(커널과 자식 양쪽)·test 특징을 열기 전에 재계산해 비교. lock은 run_identity(해시·모듈·설정·장치·torch), results/val 예측/val 지표/초기 fc5 해시까지 확인. 로컬 자산 누락·변조는 assets.zip에서 해당 파일만 다시 풀어 복구 | 다른 설정의 pilot → full 추출 거부, val_metrics 1바이트·patience·환경 lock 해시 변경 → test 전에 거부, 복구 후 재검증 통과, 삭제한 자산 파일 복구 |
| + | handoff 요건 | 노트북·`environment/bin/ffmpeg`·run_identity·test_receipt·pilot/feature receipt 필수. 완료 ZIP은 manifest 9504행(런타임 SHA `96ba67dd…a75a9ae3` 고정), 특징 9504행 정확, val/test 예측이 각 1426행을 manifest 순서대로 정확히 덮음, lock의 체크포인트 6개 바이트 일치, test receipt가 이 lock에 묶임, parity 통과를 모두 확인하고 결과를 `handoff_manifest.json`의 `coverage`에 기록 | 예측 1행 누락 → coverage 거부, 잘못된 manifest SHA 고정값 → staging 거부, 완료 ZIP에 노트북·ffmpeg·coverage 기록 확인 |

## 1. 전달물

| 파일 | 역할 |
|---|---|
| `DACON_Existing9504_A100.ipynb` | Run All 노트북. deployment root(`/root-1/existing-only-9504-20260919/`)에 둡니다(Jupyter 루트가 `/root`라서 `/root` 아래에 올려 연다면, `author_code.zip`은 deployment root에 두고 그 노트북을 엽니다). 셀 편집 불필요. 마지막 셀이 노트북 옆의 검증된 ZIP 다운로드 링크를 보여 줍니다. |
| `author_code.zip` | 노트북 옆에 둡니다. 노트북에 이 ZIP의 SHA-256이 박혀 있어 다른 바이트면 첫 셀에서 멈춥니다. |
| `README_REVIEWER.md` | 이 문서 (ZIP 안에도 포함). |

ZIP 구성: `dacon9504a100/`(실행 코드 + `requirements.in`/`requirements.lock.txt`), `reference/`(코드가 기준으로 삼은 검수본 dataset README·audit·provenance·policy·verification·schema, asset_hashes.json, 참조 requirements — handoff의 `dataset/reference/`로도 들어감), `tests/`(CPU fixture·통합 검사), `tools/`(결정적 ZIP 재빌드, 노트북 재생성). 코드를 고치면 `python tools/build_author_zip.py --root . --out author_code.zip` 후 `python tools/make_notebook.py --author-zip author_code.zip --out DACON_Existing9504_A100.ipynb`로 해시를 다시 묶어야 합니다.

작업 디렉터리: 기본 `/root/dacon_existing9504_a100_20260919`(A100 워크스페이스 로컬 디스크에 새로 만듦; H100 시절 `/root` 파일에 의존하지 않음). 오디오·자산 해제, venv, 특징 캐시, 로컬 handoff 사본으로 약 55–60 GB 여유가 필요하며 preflight가 부족하면 즉시 멈춥니다. 결과물(handoff ZIP, 진행 상태 `a100_run_status.json`, lock 증거)은 deployment root(`/root-1/...`)에 게시됩니다. 필요할 때만 환경변수 `DACON_JOB_ROOT`로 바꿀 수 있고, 셀 편집은 필요 없습니다.

## 2. Run All 단계

| 단계 | 실행 위치 | 내용 |
|---|---|---|
| preflight | 커널(표준 라이브러리) | `nvidia-smi` GPU0=A100, `/root-1` 마운트·`deployment.json`, 다른 GPU 프로세스 없음, 디스크 예산, PyPI/GitHub 도달성 |
| stage_inputs | 커널 | deployment 파일 전부 bytes+SHA-256, 런타임 manifest SHA, 9504/6652·1426·1426, split별 FILE_FAKE 3326/713/713, route 수, 노출 4176/997/1001, 가창 2034 train 전용, 라벨·마스크 일치, route 정답 known, provenance+content(그룹·component·parent·pair·audio/PCM 해시) 연결 성분의 split 교차 거부. split CSV가 배포돼 있으면 provenance 해시와 보존 열 대조. assets.zip은 정확히 27개+asset_hashes.json(검수된 해시와 동일), source.zip은 검수된 33개 모듈 해시. 오디오는 **선택된 멤버만**, 사용한 shard마다 해시 검증 후, 파일마다 SHA-256 검증 |
| environment | 커널 → 격리 venv | uv 0.8.17(휠 SHA 고정) → CPython 3.11.13 → 해시 고정 lock 설치 → freeze 정확 일치 → venv 안에서 A100·import·torchaudio→torchcodec 디코드 probe |
| pilot | venv 자식 프로세스 | 저장된 receipt는 fingerprint가 같을 때만 재사용. train 행만으로 MERT 배치 선택(원래 `select_batch`: VRAM 80%·처리량 90%), 대표 행 실측, 9504행 전체 시간 투영(16 h 대비 보고만, 중단 없음) |
| features | venv | 9504행 전 구간 특징, 행 단위 원자 저장·재개, 정확한 커버리지 |
| train | venv | 6개 헤드(epoch 경계 재개 시 RNG·배치 설정 복원), val 전용 선택, 불변 `candidate_lock.json`(fingerprint 포함), 테스트 특징 파일 미개봉. lock이 있으면 전부 재검증만 |
| test_eval | venv | lock 검증 후 test 채점, 마스크 적용 지표 |
| parity | venv | 새 프로세스에서 원본 오디오 재추출 → 캐시·예측과 대조(NaN 위치·inf 엄격). mismatch면 증거 저장 후 실패 |
| handoff | 커널 | 정확한 coverage(9504 특징, val/test 각 1426 예측, 체크포인트 6개, parity) 확인 → 단일 ZIP 작성·CRC/SHA 일괄 검증·게시 |

모든 예외는 `DACON_Existing9504_A100_Handoff_FAILED.zip`(status `failed`, traceback, 로그, receipt, coverage 진단)을 남기고 다시 raise합니다. 다시 Run All 하면 끝난 단계는 fingerprint·해시 검증 후에만 재사용하고, 특징·학습은 이어서 진행합니다. 커널이 죽어 남은 자식 프로세스는 다음 controller가 기록과 커널 시작시각으로 증명되는 고아만 종료합니다. 한 job에는 살아 있는 controller가 하나뿐입니다.

## 3. 기존 알고리즘 재사용(원본 수정 없음)과 파생 변경

원본 모듈은 `source.zip`에서 해시 검증 후 **import로 그대로** 씁니다: `full_bank_features.row_identity/validate_cache/window_truth`, `full_bank_train.forward/step/predict/benchmark`, `full_bank_batches.Item/epoch_order/loss_weights`, `mert12_submission.window_layout/windows/pin_torchaudio_loader`, `localization_features.select_batch`, `localization_models.WindowMIL`, `graph_scoring.official_metrics/_roc_metrics/validate_predictions`, `v6_runtime.baseline_module`, `eight_metrics.eer`. 특징 계산 본문(게인 정책, 임시 FLOAT wav → HTDemucs, MERT layer 12 평균, DF-Arena fc5 입력 hook, PANNs 최대값, 무음 판정)은 `full_bank_features.prepare` 루프를 그대로 옮겼습니다.

AUTHOR_REQUEST의 위험 항목별 처리:

| 위험 | 처리 |
|---|---|
| `load_rows` train/val만 허용, H100 가드 | 파생 코드에서 3분할 계약 검증, `A100` 이름 확인. GPU 이름 monkeypatch 없음. CLI는 `device=cuda`·`A100` 외 설정을 거부 |
| train만 학습·val 선택·evaluate val 전용 | 학습 프로세스는 train/val 특징만 열고 개봉 수를 lock에 기록. lock 뒤 별도 프로세스가 test 채점. test를 val로 바꾸는 경로 없음 |
| 전체 manifest 캐시 identity | pilot 캐시는 별도 디렉터리·별도 identity(`purpose=pilot`), full은 전체 manifest SHA에 묶임. pilot↔full 특징을 parity 단계에서 대조 |
| legacy/constructed 게인 보존, 원본 peak 0.6 | 원래 식 그대로 |
| 고정 candidate 라벨, 7144 하드코딩 | 원본 제출 스크립트는 쓰지 않음. submit.zip은 만들지 않음(아래 7) |
| 정규화 통계 train만 | 원래 코드 그대로(train route 파일 평균의 mean/std) |
| 노출·독립성·가창 | 지표와 README에 노출 997/1001, 가창 평가 없음, 공식 점수 미산출(unknown 라벨 때문에 `score=None`) 명시 |
| 16 h | pilot 투영과 단계별 실측(재시도 포함 누적)만 보고. 벽시계 중단 없음 |
| CUDA 모델 정리 | 커널은 torch를 import하지 않음. GPU 단계 전 우리 고아 프로세스 종료·다른 compute 프로세스 있으면 중단 |
| 사전학습 가중치 재작성 금지 | 추출 파일 read-only, 해시로 재확인 |
| 안전한 압축 해제 | 절대경로·`..`·백슬래시·심볼릭 링크·중복·암호화 멤버 거부, 멤버별 크기·해시 확인 |

## 4. 환경

`requirements.lock.txt`(해시 포함)는 보충 ZIP의 `reference_requirements.txt` 고정값(numpy 2.2.6, scipy 1.15.3, scikit-learn 1.7.1, librosa 0.11.0, soundfile 0.13.1, matplotlib 3.10.3, transformers 4.57.6, demucs 4.0.1, einops 0.8.1, torchcodec 0.8.1)을 그대로 쓰고, torchcodec 0.8에 맞는 torch/torchaudio 2.9.1(PyPI 리눅스 휠 = CUDA 12.8, sm_80 포함)과 baseline이 실제로 import하는 `panns-inference 0.1.1`/`torchlibrosa 0.1.0`/`tqdm`, 정적 ffmpeg용 `imageio-ffmpeg 0.6.0`을 더했습니다. 노트북 전용 패키지(nbformat 등)는 커널이 아니라 venv에 필요 없어 뺐습니다. **fairseq는 넣지 않았습니다**: 실제 `baseline/script.py`와 DF-Arena `backbone.py`는 `transformers.Wav2Vec2Model`을 씁니다.

`torchaudio.load`(2.9)는 torchcodec으로 디코드하므로 FFmpeg 공유 라이브러리가 필요합니다. 이미지에 없으면 apt(root) → conda-forge FFmpeg 7.1(micromamba 2.9.0-0, 공개 체크섬 확인) 순으로 채우고, 그래도 안 되면 특징 추출 전에 멈춥니다.

## 5. 이 환경에서 실제로 검증한 것 (CPU)

- lock은 `uv pip compile`로 해소했고, 같은 lock을 Python 3.11 venv에 `--require-hashes`로 설치해 모든 import가 됩니다. `low_cpu_mem_usage=True`는 accelerate 없이 동작했습니다.
- 커널 측 모듈(orchestrator·staging·dataset·handoff·environment)은 Python 3.11/3.12/3.13에서 표준 라이브러리만으로 import되고 torch를 올리지 않습니다.
- `tests/test_cpu_integration.py`: **실제 baseline/DF-Arena/MERT 코드 + 축소 설정의 무작위 가중치**로 만든 합성 deployment(26행, shard 2개, mp3/flac/wav)에서 staging → pilot → 중단 후 재개한 full 추출 → 학습 → lock → test → parity → handoff를 끝까지 실행했습니다. 확인한 결함 유형:
  - 변조된 shard, `../`·symlink 멤버, split 교차 group/parent, 라벨·마스크 불일치, route 정답 unknown → 모두 거부
  - 추출 강제 중단 후 재개 시 정확한 커버리지와 캐시 검증 통과
  - test 특징 파일을 모두 치운 상태에서도 학습·lock 완료, lock의 test 개봉 수 0
  - 학습된 voice 헤드가 fc5 초기값과 다름(그래디언트 연결)
  - lock 뒤 체크포인트 1바이트 변경 시 test 채점 거부
  - FILE = max(voice·VP, music·MP) 그래프 유지, unknown 라벨 때문에 공식 총점 미산출
  - 새 프로세스 원본 재추출 특징·예측이 CPU에서 bitwise 일치
  - handoff ZIP 멤버 전부 SHA256SUMS·CRC 일치, 체크포인트 6개·lock·run_config·manifest·audit/provenance 포함, raw audio와 캐시는 parity 샘플만
  - rename을 거부하는 볼륨을 흉내 내면 게시가 직접 쓰기+재해시로 전환되고 임시 파일이 남지 않음
  - 예외 시 FAILED ZIP 생성 후 재raise
  - 네 CLI 모두 CPU 설정 거부
- 노트북 자체(`tests/test_notebook_bootstrap.py`): 전달한 `.ipynb`를 **Python 3.13 Jupyter 커널**(nbclient, 베이스 이미지와 같은 3.13)에서 Run All 했습니다. 부트스트랩 셀이 해시 대조 → `author_code.zip` 안전 해제 → `Pipeline` 생성까지 완료하고, `/root-1`이 없는 호스트라 preflight 셀에서 멈추며 `failed` 상태 FAILED ZIP(같은 author_code.zip 바이트와 노트북 포함)과 `a100_run_status.json`을 deployment root에 남기는 것을 확인했습니다. 1바이트 바뀐 author_code.zip은 첫 셀에서 거부되고 해제되지 않습니다.

재현(CPU, GPU 불필요; 원본 `source/`와 보충 `baseline_src/`가 필요):

```bash
# lock으로 만든 Python 3.11 venv 안에서
python tests/build_fixture.py --out /tmp/fx --baseline-src <보충ZIP>/baseline_src --source-dir <컨텍스트>/source
python tests/test_cpu_integration.py --fixture /tmp/fx --work /tmp/run --source-dir <컨텍스트>/source
# 표준 라이브러리만(아무 Python 3.11–3.13): 두 controller·고아 회수 검사
python tests/test_controller.py --work /tmp/ctl
# nbclient+ipykernel이 있는 커널 Python(예: 3.13)에서
python tests/test_notebook_bootstrap.py --notebook DACON_Existing9504_A100.ipynb --author-zip author_code.zip \
    --deploy /tmp/fx/objvol/deploy --work /tmp/nb
# 같은 인터프리터 + jupyter_server: 루트 제한 서버에서 마지막 셀 다운로드 링크
python tests/test_download_link.py --notebook DACON_Existing9504_A100.ipynb --author-zip author_code.zip --work /tmp/dl
```

## 6. A100에서만 확인되는 것 (미검증)

- 실제 가중치 로드, VRAM 배치 선택, 실측 처리량과 16 h 대비 결과
- A100 이미지의 FFmpeg 공유 라이브러리 유무와 provisioning 경로(apt/conda는 여기서 실행 불가)
- `uv python install 3.11.13`(GitHub 도달 필요; 실패하면 시스템 python3.11로 대체, 없으면 중단)
- deployment.json·audio_upload.json 실제 구조: 계약(`files`의 key/bytes/sha256, `runtime_manifest_sha256`, shard key는 `/root-1` 기준, 멤버 이름은 manifest `path` 또는 유일한 basename)과 다르면 staging에서 명확한 메시지로 멈춥니다.
- GPU 수치 재현성: parity 허용오차(특징 1e-3, 예측 1e-3)를 넘거나 NaN 위치·inf가 다르면 receipt에 `mismatch`로 저장하고 parity 단계가 실패합니다(v2). 그때는 FAILED ZIP에 증거가 담기고 완료 handoff는 만들어지지 않습니다.
- 재개 RNG의 CUDA 부분(`torch.cuda.get/set_rng_state_all`)과 실제 `full_bank_train.benchmark`(CUDA 전용)는 CPU에서 실행되지 않습니다. CPU 검사는 Python/NumPy/Torch CPU 스트림과, RNG를 소비하는 벤치마크 대역으로 보존 로직을 확인했습니다. 원래 학습 루프 자체는 RNG를 쓰지 않으므로(순서는 `seed+epoch` 고정 생성기), 재개 동일성의 핵심은 같은 측정 배치 설정 재사용입니다.
- 실제 VESSL Jupyter의 루트 디렉터리·UI(JupyterLab/Notebook)와 `/root`와 job root가 같은 파일시스템인지(하드링크 여부): 로컬 `jupyter_server`에서 `/files/` 경로로만 확인했습니다. 파일시스템이 다르면 검증 복사가 되며 노트북 디렉터리에 ZIP 크기만큼 여유가 필요합니다.
- 실제 오브젝트 볼륨(`/root-1`)의 rename·fsync 동작: rename 실패 경로는 CPU에서 흉내 내 검증했지만 실제 FUSE 동작은 미확인입니다. 어느 경로였는지 sidecar의 `publish_mode`에 기록됩니다.

## 7. 의도적으로 하지 않은 것

- **submit.zip 미포함**: 기존 제출 요구사항은 torch 2.7.1(공식 환경)인데 이번 학습 환경은 참조 성공환경의 torch 2.9.1·torchcodec입니다. 공식 환경에서의 오프라인 패키지 동등성은 여기서 증명할 수 없으므로 만들지 않았습니다. 대신 handoff의 `parity/`(원본 오디오, 캐시 특징, 재계산 특징, 기대 예측)와 `assets/assets.zip`, `environment/bin/ffmpeg`, 런타임 코드로 이후 패키징 동등성을 검증할 수 있습니다.
- 데이터 재수집, 분할 변경, val/test로 학습, test 기반 선택, 서비스 시작·중지, API 키, DACON 제출.

## 8. 원래 누락됐던 baseline 소스

첫 컨텍스트 ZIP에는 `baseline/script.py`, DF-Arena 코드·설정(`backbone.py`, `conformer.py`, `modeling_antispoofing.py`, `configuration_antispoofing.py`, `feature_extraction_antispoofing.py`, `pipeline_antispoofing.py`, `config.json`, `preprocessor_config.json`, `facebook/wav2vec2-xls-r-1b/config.json`), MERT 코드·설정(`modeling_MERT.py`, `configuration_MERT.py`, `config.json`, `preprocessor_config.json`)이 없었습니다. 보충 ZIP으로 모두 받았고 `asset_hashes.json`과 해시가 일치합니다. 가중치 4개(`df_arena_1b/pytorch_model.bin`, `htdemucs/955717e8-8726e21a.th`, `panns/Cnn14_mAP=0.431.pth`, `mert/pytorch_model.bin`)는 소스가 아니며 assets.zip에서만 해시로 검증합니다. 현재 누락된 스크립트 소스는 없습니다.
