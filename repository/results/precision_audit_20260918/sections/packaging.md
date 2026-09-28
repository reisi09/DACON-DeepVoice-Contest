# 제출 패키지·실행 계약 감사

감사일은 2026-09-18이다. 현재 MERT12 제출 ZIP, ZIP 내부의 작은 파일 41개, 3개 head checkpoint, 저장된 실제 추론 CSV와 공식 제출 목록을 대조했다. 새 GPU 실행이나 재제출은 하지 않았다. 이 범위에서 현재 MERT12의 패키지 손상, 잘못된 checkpoint 포함, CSV 순서 오류는 발견하지 못했다.

## 현재 제출본과 공식 결과의 연결

`submissions/MERT12_full1322_submit.zip`의 전체 바이트를 이번 감사에서 다시 읽어 계산한 SHA-256은 다음과 같다.

```text
e7bebbded6fef06517db0d42421a575a0440849248de99856796ffc3e5eb5ee5
```

파일 크기는 6,268,222,843 bytes이며, 기존 패키지 receipt와 92294 제출 메모의 해시가 모두 같다. 2026-09-18 03:02:26 UTC에 상위 감사 작업이 인증된 공식 페이지에서 확인한 제출 92294는 총점 `0.7479394074`, ADS `0.7211111111`, CPS `0.9893940741`, 소요 시간 `19분 12초`다. 공식 페이지의 해시는 사용자가 적은 제출 메모이며 서버가 재계산해 표시한 해시는 아니다. 파일명·접수 기록·메모·로컬 receipt를 함께 대조한 연결이다. 해당 제출의 공식 실행 완료는 점수와 실행 시간으로 확인된다. A100에서 일부 파일만 처리한 기록으로 L4 전체 실행을 추정할 필요가 없다. 다만 공식 페이지가 서버의 메모리 peak, 패키지 설치 소요 시간, 전체 CSV를 공개한 것은 아니다.

근거: [이번 전체 해시 계산](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/precision_audit_20260918/evidence/packaging_fresh_archive_sha256.json), [이번 공식 제출 목록](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/precision_audit_20260918/live_submissions_20260918.txt), [최종 A100 QA와 ZIP 해시 연결](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/mert12_a100_20260918/downloaded/evidence/output/final_qa_receipt.json:2).

## 이번에 직접 확인한 패키지 내용

| 확인 대상 | 결과 |
|---|---|
| ZIP 구성 | 중복 없는 45개 멤버, 절대경로·상위경로 이동 멤버 없음 |
| ZIP/해제 크기 | 6,268,222,843 / 6,268,214,325 bytes; 저장된 대회 계약의 10 GB / 32 GB 한도 이내 |
| manifest와 외부 receipt | 모든 멤버 이름이 정확히 일치 |
| 작은 멤버 41개 | 이번에 직접 읽고 SHA-256 계산; 내부 manifest와 외부 receipt 모두 일치 |
| 대형 사전학습 가중치 4개 | 이번 개별 해시는 재계산하지 않음; 전체 ZIP의 새 해시가 기존 검증본과 일치 |
| 실제 실행 코드 | ZIP의 runtime 6개와 baseline script가 현재 로컬 소스와 byte 단위로 같음 |
| head 3개 | seed 41·42·43, 각 500 steps, batch 32, `source_generator_parent+coverage_prefix`, `mlp` |
| 각 head | 파라미터 131,329개; 길이 1,024의 유한한 mean/scale, 모든 scale 양수 |
| 학습 연결 | 고유 train ID 1,322개, training manifest와 receipt의 ID 순서 일치, manifest 해시 일치 |
| GPU 학습 receipt 연결 | 3개 checkpoint 해시와 GPU receipt의 checkpoint 증거 일치; 기록된 최소 노출 수 각각 1 |

ZIP의 최상위는 `model/`, `script.py`, `requirements.txt`로 구성된다. root `script.py`가 자기 위치를 기준으로 `model/runtime/mert12_submission.py`의 `run(ROOT)`를 호출하므로 로컬 작업 폴더의 다른 구현을 우연히 가져오는 구조가 아니다. 예측 캐시나 과거 제출 CSV를 ZIP에 넣어 출력하는 경로도 확인되지 않았다. 학습 manifest는 provenance 자료이며 추론 정답표로 읽히지 않는다.

최종 head SHA-256:

```text
seed41 91f5bd17f786d5c9afccffb7516484cba6b695ba0156dce01c04a11aaca746d0
seed42 a3dbec68027d686f6465d9144166741a330c80309a0b3538e78300d3d2990d6a
seed43 a877808bd54326d51e0dedfc2331402a6f204f1534c8cc163db8036b5e895c9c
```

근거: [ZIP 직접 검사 결과](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/precision_audit_20260918/evidence/packaging_archive_inspection.json), [패키지 생성·검증](/Users/choiwuseck/Documents/ChatGPT/데이콘/package_mert12_submission.py:83), [실행 전 멤버 해시 확인](/Users/choiwuseck/Documents/ChatGPT/데이콘/mert12_submission.py:123), [head 로드](/Users/choiwuseck/Documents/ChatGPT/데이콘/mert12_submission.py:67).

## 출력 CSV와 파일 독립성

현재 경로는 sample CSV를 읽고, 실제 오디오 ID와 정확히 대조한 다음 sample의 순서로 파일을 재배열한다. 추론 후에는 `ID`, `FILE_FAKE_PROB`, `VOICE_FAKE_PROB`, `MUSIC_FAKE_PROB`, `VOICE_PRESENT_PROB`, `MUSIC_PRESENT_PROB`의 6개 열과 유한한 `[0,1]` 확률을 검사한다. 임시 CSV를 저장한 뒤 최종 경로로 교체한다.

이번 CPU 점검은 실제 baseline 함수와 validator를 실행했다. 파일명 정렬과 sample 순서가 다를 때 sample 순서를 유지했고, 중복 sample ID, 같은 stem의 오디오, 누락·추가 오디오 ID, 뒤집힌 예측 순서, NaN, 범위 밖 확률, 추가 출력 열을 거부했다. 이 검사는 CSV/ID 계약을 확인한 것이며 빈 임시 파일의 음질이나 모델 성능을 확인한 것이 아니다.

저장된 최종 A100 CSV도 이번에 다시 읽었다. 원본 4개, 역순 4개, 부분집합 2개의 공통 ID에서 다섯 확률의 최대 차이는 `0.0`이었다. 4초 무음 WAV, 60초 스테레오 FLAC, 6초 MP3를 포함한 보충 CSV도 6개 열·고유 ID·유한 확률 조건을 통과했다. 각 CSV의 새 SHA-256은 runtime receipt와 같고, receipt의 manifest 해시는 현재 ZIP 내부 manifest와 일치한다.

코드에서 PANNs 존재 확률은 파일마다 따로 계산한다. 음악 head의 mean/scale은 checkpoint에 저장된 학습 통계이며 테스트 파일 전체에서 다시 계산하지 않는다. DF-Arena·MERT·head는 평가 상태에서 실행한다. 코드 검토와 저장된 순서/부분집합 비교에서 파일 간 예측 보정이나 테스트 시 학습 경로를 발견하지 못했다. 이는 검사한 경로의 근거이며 모든 가능한 입력 조합에 대한 수학적 증명은 아니다.

근거: [실제 오디오 ID 대조](/Users/choiwuseck/Documents/ChatGPT/데이콘/baseline_original/script.py:80), [sample 순서 반영](/Users/choiwuseck/Documents/ChatGPT/데이콘/baseline_original/script.py:129), [실행과 최종 CSV 저장](/Users/choiwuseck/Documents/ChatGPT/데이콘/mert12_submission.py:166), [확률·ID 검증](/Users/choiwuseck/Documents/ChatGPT/데이콘/graph_scoring.py:48), [이번 계약 점검과 CSV 재계산](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/precision_audit_20260918/evidence/packaging_output_contract.json).

## 로컬 가중치와 설치 환경

| 구성 요소 | 실제 로드 경로와 판단 |
|---|---|
| MERT | ZIP 안 `model/assets/mert`, `local_files_only=True`; `trust_remote_code=True`는 포함된 로컬 Python 구현을 실행한다. 원격 모델 다운로드 허용을 뜻하지 않는다. 누락·불일치 loading info가 있으면 중단한다. |
| DF-Arena | ZIP 안 경로에 `local_files_only=True`로 로드한다. 내부 backbone의 `facebook/wav2vec2-xls-r-1b` 문자열은 모델 폴더로 cwd를 바꾼 상태에서 같은 상대경로의 포함된 config로 해석된다. 이 중첩 config도 실제 ZIP에 있다. |
| HTDemucs | `get_model("htdemucs", repo=HTDEMUCS_DIR)`에 로컬 repo를 준다. 포함된 YAML은 `955717e8` 모델을 가리킨다. 로컬 설치된 demucs 구현은 repo가 주어지면 `LocalRepo`를 사용한다. |
| PANNs | 포함된 327,428,481-byte checkpoint 경로를 명시한다. label CSV를 홈의 `panns_data`로 복사한 다음 라이브러리를 import한다. |
| head | 포함된 seed별 파일을 `torch.load(..., weights_only=True)`로 읽고 state dict를 strict하게 로드한다. |

PANNs 라이브러리 자체에는 label CSV가 없거나 checkpoint가 300 MB보다 작을 때 `wget`을 실행하는 fallback이 있다. 현재 정상 ZIP은 checkpoint 크기 조건을 충족하고, 코드가 label을 먼저 복사하며, 모든 asset 해시를 모델 로드 전에 검증한다. 따라서 이를 현재의 외부 다운로드 결함으로 분류하지 않았다. 파일이 깨지거나 없어지면 앞의 해시 검사에서 실패한다. HTDemucs의 `weights_only=False`는 포함된 기존 직렬화 포맷을 읽는 호환 처리다. 사용자 제공 오디오를 pickle로 읽는 경로는 아니다.

최종 `requirements.txt`는 PyTorch CUDA index와 `torch==2.7.1+cu128`, `torchaudio==2.7.1+cu128`, `torchvision==0.22.1+cu128`, `diffq==0.2.4`, `einops==0.8.1`만 지정한다. `librosa`, `transformers`, `demucs`, `panns-inference`, `numpy` 등은 대회 기본 환경에 의존한다. 따라서 이 ZIP은 임의의 빈 Python 환경에서 곧바로 재현되는 독립 배포판은 아니다. 저장된 대회 환경 계약과 공식 채점 성공을 기준으로 실행 가능성을 판단해야 한다. 이번에는 설치를 다시 수행하지 않았다.

최종 A100 실행 노트북은 Python 3.11, PyTorch 2.7.1+cu128, CUDA 12.8을 assertion으로 확인한다. 다만 subprocess에 별도 socket 차단기나 비어 있는 HF cache를 설정한 코드는 없다. `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`과 로컬 asset 로드는 확인했지만, 이 노트북만으로 네트워크를 물리적으로 차단했다고 쓰면 근거를 넘는다. 공식 서버의 정상 채점은 실제 제출 환경에서의 실행 근거로 별도 취급한다.

근거: [MERT 로드](/Users/choiwuseck/Documents/ChatGPT/데이콘/mert12_submission.py:173), [PANNs 로드](/Users/choiwuseck/Documents/ChatGPT/데이콘/baseline_original/script.py:182), [HTDemucs 로드](/Users/choiwuseck/Documents/ChatGPT/데이콘/baseline_original/script.py:248), [DF-Arena 로드](/Users/choiwuseck/Documents/ChatGPT/데이콘/baseline_original/script.py:315), [ZIP에서 꺼낸 중첩 backbone](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/precision_audit_20260918/evidence/packaging_model__assets__baseline__model__df_arena_1b__backbone.py:12), [requirements 생성](/Users/choiwuseck/Documents/ChatGPT/데이콘/package_mert12_submission.py:19), [저장된 대회 환경 계약](/Users/choiwuseck/Documents/ChatGPT/데이콘/COMPETITION_CONTRACT.md:41), [실행된 최종 QA 노트북](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/mert12_a100_20260918/downloaded/evidence/output/DACON_MERT12_Final_QA_Executed.ipynb).

## 수리된 과거 오류와 현재 남은 한계

| 분류 | 확인 내용 | 영향과 최소 후속 조치 |
|---|---|---|
| 현재 기록 불일치 | `official_submission.json`은 아직 `grading`, 점수 null이고 REPORT는 새 공식 점수가 없다고 쓴다. 현재 공식 페이지는 92294 채점 완료다. | 최신 모델과 성과를 잘못 판단할 수 있다. 현재 감사의 공식 목록과 해시를 근거로 상태 요약을 갱신하면 된다. 모델 재실행은 필요 없다. |
| 생성 시점 metadata | MERT12 package receipt의 `gpu_end_to_end_verified:false`는 생성 당시 값이다. 같은 해시의 최종 QA receipt는 complete다. | false만 읽으면 후속 검증을 누락한다. 시점별 receipt를 함께 제시한다. 과거 파일의 시점 정보를 임의로 덮지 않는다. |
| 수리 완료 | 초기 cached/raw 음악 확률 차이 `0.00019127128483675815`가 `1e-5`를 넘었다. 최종 ZIP에서 다시 계산한 저장 CSV 차이는 `3.971516315814938e-08`이다. | 현재 MERT12 미해결 오류로 다시 열지 않는다. 최종 ZIP의 torchaudio Demucs 입력 경로를 유지하고 decoder 변경 시 실제 오디오 parity를 재확인한다. |
| 과거 의존성 실패 | 공식 89372는 `libtorchaudio.so` 로드 오류, 89989는 NumPy dtype ABI 오류로 실행 실패했다. 이후 수정 제출과 현재 MERT12는 채점됐다. | 과거 실패를 현재 MERT12 실행 실패로 합산하지 않는다. 환경을 바꿀 때 torch/torchaudio 짝과 NumPy ABI를 검사할 실질 근거로 남긴다. |
| M330 과거 미해결 측정 | 기존 M330 receipt에 CPU/A100 음악 확률 차이 `0.006940650919647129`와 cross-environment parity false가 남아 있다. 같은 장치의 head/feature 연결은 별도로 통과했다. | M330의 장치 독립 수치 재현성은 미확인이다. 공식 점수가 무효라는 뜻도, 같은 원인이 최종 MERT12에 남았다는 뜻도 아니다. 원인이 필요하면 환경·decoder를 고정해 동일 실제 오디오 하나부터 비교한다. |
| 재현성 범위 | 현재 ZIP의 기본 라이브러리 전체 lock과 공식 실행 메모리 log는 ZIP에 없다. | 임의 환경 재배포나 서버 peak 메모리를 단정하지 않는다. 필요 시 공식 이미지/설치 버전 기록을 보존하고 별도 자원 측정을 한다. 이미 완료된 제출을 확인하려고 GPU를 다시 켤 필요는 없다. |

근거: [채점 중으로 남은 로컬 기록](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/mert12_a100_20260918/official_submission.json:5), [과거 실패 로그](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/mert12_a100_20260918/downloaded/unpacked/output/recovery_notebook.log:13), [decoder 고정](/Users/choiwuseck/Documents/ChatGPT/데이콘/mert12_submission.py:109), [M330 기존 차이 기록](/Users/choiwuseck/Documents/ChatGPT/데이콘/submissions/M330_MLP_full1206/package_verification.json:26).

## 이전 후보와 공식 제출의 연결 강도

| 후보 | 보존된 로컬 SHA-256 | 공식 제출 연결 |
|---|---|---|
| MERT12 full1322 | `e7bebbded6fef06517db0d42421a575a0440849248de99856796ffc3e5eb5ee5` | 이번 전체 해시 재계산 + 92294의 동일 파일명·메모 해시, `0.7479394074` |
| M330 full1206 | `1f6cfd6c9feacead01d4ebcfd217d10c1796969b1f863ba4b0283d46e7432c7e` | 90735의 고유 파일명, `0.7396465503`; 이번에 이전 ZIP 전체를 다시 해시하지 않음 |
| C08 | `0c3c328947c2f6d83a07763ac482a945004f2bad0c3df6d6b16bb160b18f9ead` | 90692의 고유 파일명, `0.7314822646`; 현재 해당 제출 디렉터리에는 ZIP이 없고 receipt가 남음 |
| S2 EAT | `19c7a66a4a82d8cb117c7a7ae6c71ed370016c1528fa2e96801105c5a5b0897c` | 91238의 `submit.zip`, `0.6378679788`; S2 식별은 과거 사용자 확인이며 공식 파일명만으로 입증되지 않음 |
| B full1696 | `1fda295b44925d5b255cc1235f9bee57141ace7469370ae3245af113572bfb00` | H100 패키지 QA receipt는 있으나 제출 91610과의 연결은 미확보 |

91610은 공식 점수 `0.6837822646`, ADS `0.6498253968`, CPS `0.9893940741`, `21분 10초`의 `submit.zip`이다. 이름과 제출 순서만 보고 B full1696으로 단정하면 안 된다. 해당 제출의 업로드 원본 해시나 사용자의 명시적 식별이 최소 추가 근거다. C08·S2·B의 ZIP이 해당 디렉터리에 없다는 관찰은 원격 보관본까지 없다는 뜻은 아니다.

근거: [M330 receipt](/Users/choiwuseck/Documents/ChatGPT/데이콘/submissions/M330_MLP_full1206/package_verification.json:2), [C08 receipt](/Users/choiwuseck/Documents/ChatGPT/데이콘/submissions/C08/verification.json:2), [S2 receipt](/Users/choiwuseck/Documents/ChatGPT/데이콘/submissions/S2_EAT/package_verification.json:2), [B receipt](/Users/choiwuseck/Documents/ChatGPT/데이콘/submissions/B_full1696/submit.json), [S2 식별의 기존 근거 수준](/Users/choiwuseck/Documents/ChatGPT/데이콘/EXPERIMENTS_ALL_2026-09-17.md:53).

<!-- HUMANIZE-SUMMARY: Korean final editing pass applied with meaning, evidence boundaries, numbers, paths, identifiers and technical terms preserved. Checks 6/6. No new claims introduced by editing. -->
