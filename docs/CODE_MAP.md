# 코드 위치와 재현 조건

## 현재 실험에서 먼저 읽을 파일

아래 경로는 ZIP을 푼 최상위 폴더 기준입니다. `repository/` 아래에서는 원래 프로젝트의 상대 경로를 유지했습니다.

| 역할 | 경로 |
|---|---|
| H100 노트북 | `repository/results/existing9504_h100_20260920/DACON_Existing9504_H100.ipynb` |
| 단계별 실행 제어 | `repository/results/existing9504_h100_20260920/code/dacon9504a100/orchestrator.py` |
| 데이터 로딩·검사 | 같은 디렉터리의 `dataset.py`, `staging.py` |
| 특징 추출 | 같은 디렉터리의 `features.py` |
| 분류기 학습 | 같은 디렉터리의 `training.py` |
| val/test 평가·선택 고정 | 같은 디렉터리의 `evaluation.py` |
| 원본 재추출 일치 검증 | 같은 디렉터리의 `parity.py` |
| 결과 압축·검증 | 같은 디렉터리의 `handoff.py` |
| 학습 당시 공통 소스 | `repository/results/existing9504_h100_20260920/training_snapshot/source/` |
| 최종 제출 진입점 | `repository/submissions/Existing9504_MIL/source/script.py` |
| 최종 결합 추론 | 그 아래 `model/runtime/full_bank_submission.py` |
| MERT 모델 정의 | 그 아래 `model/assets/mert/modeling_MERT.py` |
| MERT 설정 클래스 | 그 아래 `model/assets/mert/configuration_MERT.py` |
| DF-Arena 모델 정의 | 그 아래 `model/assets/baseline/model/df_arena_1b/` |
| HTDemucs·PANNs 연결 | 그 아래 `model/assets/baseline/script.py` |
| WindowMIL | 그 아래 `model/runtime/localization_models.py` |
| MLP | 그 아래 `model/runtime/eight_heads.py` |
| 제출 패키징 | `repository/package_existing9504_submission.py` |
| 실제 제출 추론 검증 | `repository/verify_existing9504_submission.py` |
| 최종 데이터 재구성 | `repository/rebuild_existing_bank.py` |
| 분할 정책 검사 | `repository/verify_existing_bank.py`, `verify_reconstructed_bank.py` |
| 추가 데이터 확보·목록화 | `repository/candidate_data_acquire.py`, `candidate_data_catalog.py`, `reconstruct_candidate_bank.py` |

학습 당시 소스는 완료 Handoff의 `code/source.zip`, `code/author_code.zip`에서 가져왔습니다. 최종 추론 소스는 제출 ZIP에서 가져왔습니다. 루트의 과거 소스와 같은 이름이어도 이 두 고정본이 현재 결과의 기준입니다. 원본 source ZIP 두 개도 소형 코드 묶음으로 함께 보존했습니다.

기존 통합 소스에 있던 프로젝트 루트 스크립트·노트북 및 버전 관리된 과거 실험 코드를 모두 `repository/`에 넣었습니다. 내부 에이전트 임시 기록, 가상환경, 과거 실행 디렉터리의 추적되지 않은 중복 사본은 포함하지 않았습니다. 원본 노트북의 셀 코드는 유지하고 출력과 실행 번호만 제거했습니다.

## 라이브러리와 포함 범위

MERT와 DF-Arena의 커스텀 Python 구현은 포함했습니다. HTDemucs·PANNs·Transformers·PyTorch의 설치 패키지 전체를 복제한 것은 아닙니다. 이들 의존성은 학습 환경의 `requirements.in`, `requirements.lock.txt` 및 제출 코드의 `requirements.txt`에 기록돼 있습니다. 모델 설정·기존 라이선스 문서는 보존했습니다.

학습 환경과 제출 환경은 구분해야 합니다. 제출 `requirements.txt`는 대회에 사전 설치된 패키지를 전제로 한 추가 설치 목록이므로, 빈 로컬 가상환경의 전체 설치 목록으로 사용하면 안 됩니다.

## 다른 환경에서 필요한 것

소스를 읽거나 데이터 구성을 분석할 때는 이 ZIP이면 됩니다. 모델을 실행하려면 별도 파일이 필요합니다.

- 추론: 기존 `submissions/Existing9504_MIL/submit.zip`의 모델 가중치와 실행 환경. 이 패키지의 소스 전용 디렉터리에는 가중치가 없습니다.
- 학습: `data/selected_list.csv`에 대응하는 실제 오디오, 고정 모델 자산, GPU·디스크, 노트북에서 지정한 저장소 경로.
- 제출 재생성·원본 추론 검증: 완료 Handoff인 `DACON_Existing9504_H100_Handoff.zip`. 원본 검증 오디오도 그 Handoff에 있으며 이번 패키지에는 넣지 않았습니다.
- 데이터 분할 재생성: 원래 재구성 코드가 참조하는 이전 검증 스냅샷과 실제 오디오. 최종 목록은 보존했지만 수집 디렉터리 전체를 복제하지 않았으므로, 스크립트 실행 전에 입력 경로와 해시를 확인해야 합니다.

다음 명령은 `repository/`를 작업 디렉터리로 삼고 기존 프로젝트의 `.venv`, 완료 Handoff 등 필요한 파일을 복원했을 때의 명령입니다. 패키지를 여는 것만으로 실행되지 않습니다.

```sh
.venv/bin/python package_existing9504_submission.py \
  --handoff results/existing9504_h100_20260920/completed/DACON_Existing9504_H100_Handoff.zip \
  --output submissions/Existing9504_MIL/submit.zip
.venv/bin/python verify_existing9504_submission.py
```

학습 노트북은 기존 VESSL 마운트와 H100 식별자를 확인하도록 작성돼 있습니다. 다른 장비에서 그대로 실행된다고 가정하지 말고 데이터 경로·자산 해시·장비 조건을 맞춰야 합니다. 이번 인수인계에서는 이 조건을 바꾸거나 GPU를 재실행하지 않았습니다.
