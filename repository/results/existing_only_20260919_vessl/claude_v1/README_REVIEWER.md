# DACON 236749 · existing-only 9504 · A100 Run All — reviewer README

작성: Claude (author). 검수: Codex. **GPU 실행·VESSL 시작·다운로드·학습은 하지 않았습니다.** 이 문서는 무엇을 만들었고, 무엇을 CPU에서 실제로 검증했으며, 무엇이 A100에서만 확인 가능한지 구분합니다.

## 1. 전달물

| 파일 | 역할 |
|---|---|
| `DACON_Existing9504_A100.ipynb` | Run All 노트북. deployment root(`/root-1/existing-only-9504-20260919/`)에 둡니다. 셀 편집 불필요. |
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
| pilot | venv 자식 프로세스 | train 행만으로 MERT 배치 선택(원래 `select_batch`: VRAM 80%·처리량 90%), 대표 행 실측, 9504행 전체 시간 투영(16 h 대비 보고만, 중단 없음) |
| features | venv | 9504행 전 구간 특징, 행 단위 원자 저장·재개, 정확한 커버리지 |
| train | venv | 6개 헤드, val 전용 선택, 불변 `candidate_lock.json`, 테스트 특징 파일 미개봉 |
| test_eval | venv | lock 검증 후 test 채점, 마스크 적용 지표 |
| parity | venv | 새 프로세스에서 원본 오디오 재추출 → 캐시·예측과 대조 |
| handoff | 커널 | 단일 ZIP 작성·CRC/SHA 일괄 검증·게시 |

모든 예외는 `DACON_Existing9504_A100_Handoff_FAILED.zip`(status `failed`, traceback, 로그, receipt)을 남기고 다시 raise합니다. 다시 Run All 하면 끝난 단계는 검증 후 재사용하고, 특징·학습은 이어서 진행합니다. 커널이 죽어 남은 자식 프로세스는 다음 실행에서 이 파이프라인의 것만 골라 종료합니다.

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
# nbclient+ipykernel이 있는 커널 Python(예: 3.13)에서
python tests/test_notebook_bootstrap.py --notebook DACON_Existing9504_A100.ipynb --author-zip author_code.zip \
    --deploy /tmp/fx/objvol/deploy --work /tmp/nb
```

## 6. A100에서만 확인되는 것 (미검증)

- 실제 가중치 로드, VRAM 배치 선택, 실측 처리량과 16 h 대비 결과
- A100 이미지의 FFmpeg 공유 라이브러리 유무와 provisioning 경로(apt/conda는 여기서 실행 불가)
- `uv python install 3.11.13`(GitHub 도달 필요; 실패하면 시스템 python3.11로 대체, 없으면 중단)
- deployment.json·audio_upload.json 실제 구조: 계약(`files`의 key/bytes/sha256, `runtime_manifest_sha256`, shard key는 `/root-1` 기준, 멤버 이름은 manifest `path` 또는 유일한 basename)과 다르면 staging에서 명확한 메시지로 멈춥니다.
- GPU 수치 재현성: parity 허용오차(특징 1e-3, 예측 1e-3)를 넘으면 status `mismatch`로 기록하고 결과는 보존합니다.
- 실제 오브젝트 볼륨(`/root-1`)의 rename·fsync 동작: rename 실패 경로는 CPU에서 흉내 내 검증했지만 실제 FUSE 동작은 미확인입니다. 어느 경로였는지 sidecar의 `publish_mode`에 기록됩니다.

## 7. 의도적으로 하지 않은 것

- **submit.zip 미포함**: 기존 제출 요구사항은 torch 2.7.1(공식 환경)인데 이번 학습 환경은 참조 성공환경의 torch 2.9.1·torchcodec입니다. 공식 환경에서의 오프라인 패키지 동등성은 여기서 증명할 수 없으므로 만들지 않았습니다. 대신 handoff의 `parity/`(원본 오디오, 캐시 특징, 재계산 특징, 기대 예측)와 `assets/assets.zip`, `environment/bin/ffmpeg`, 런타임 코드로 이후 패키징 동등성을 검증할 수 있습니다.
- 데이터 재수집, 분할 변경, val/test로 학습, test 기반 선택, 서비스 시작·중지, API 키, DACON 제출.

## 8. 원래 누락됐던 baseline 소스

첫 컨텍스트 ZIP에는 `baseline/script.py`, DF-Arena 코드·설정(`backbone.py`, `conformer.py`, `modeling_antispoofing.py`, `configuration_antispoofing.py`, `feature_extraction_antispoofing.py`, `pipeline_antispoofing.py`, `config.json`, `preprocessor_config.json`, `facebook/wav2vec2-xls-r-1b/config.json`), MERT 코드·설정(`modeling_MERT.py`, `configuration_MERT.py`, `config.json`, `preprocessor_config.json`)이 없었습니다. 보충 ZIP으로 모두 받았고 `asset_hashes.json`과 해시가 일치합니다. 가중치 4개(`df_arena_1b/pytorch_model.bin`, `htdemucs/955717e8-8726e21a.th`, `panns/Cnn14_mAP=0.431.pth`, `mert/pytorch_model.bin`)는 소스가 아니며 assets.zip에서만 해시로 검증합니다. 현재 누락된 스크립트 소스는 없습니다.
