# 실험별 코드 안내

실제 실행 결과와 완료·중단·미실행 상태는 [9월 17일 실험 기록](docs/EXPERIMENTS_ALL_2026-09-17.md)을 함께 확인합니다. 한 노트북에 포함된 모든 후보가 완료됐다는 뜻은 아닙니다. 아래 경로는 `repository/` 기준입니다.

| 실험 계열 | 노트북·실행 진입점 | 모델·학습 구현 |
|---|---|---|
| G0–G4 결합 수식 | `DACON_236749_five_graphs.ipynb` | `graph_scoring.py` |
| AST 및 epoch 연장·전체 재학습 | `DACON_236749_v2_ast.ipynb`, `DACON_236749_epoch_extension.ipynb`, `DACON_236749_full_refit.ipynb` | `ast_multitask.py`, `run_v2.py`, `full_refit.py` |
| Recovery / SONICS / V4 | `DACON_236749_recovery_v3.ipynb`, `DACON_236749_trainable_v4.ipynb` | `recovery_sonics.py`, `train_v4.py`, `v4_model.py` |
| V5 전문가·joint 결합 | `DACON_236749_compare_v5.ipynb` | `v5_experts.py`, `v5_joint.py`, `v5_fusion_training.py` |
| V6 / V7 및 데이터 ablation | `V6_H100_Experiment.ipynb`, `V7_A100_DataAblation.ipynb` | `v6_runtime.py`, `v6_train.py`, `v7_notebook_runner.py` |
| Temporal Conv1D·해상도·증강 | `Temporal_Conv1D_Tuning.ipynb`, `Temporal_H100_Resolution_Softmax.ipynb`, `Temporal_H100_Augmentation.ipynb` | `temporal_backbone.py`, `temporal_experiment.py`, `temporal_resolution.py`, `temporal_augmentation_experiment.py` |
| C08 / MERT330 / AF-Whisper / MLP | `C08_MERT330_H100.ipynb`, `C08_Methods_A100.ipynb`, `C08_AFWhisper_A100.ipynb`, `DACON_MERT12_Full_A100.ipynb` | `eight_heads.py`, `mert12_full_refit.py`, `mert12_submission.py` |
| MERT/EAT full fine-tuning, raw/stem | `MERT_EAT_FullFT_H100x1_7Experiments.ipynb`, `MERT_EAT_HTDemucs_H100x1_8Experiments.ipynb` | 노트북 내 실험 구현과 `redesign_train.py` 등 공통 소스 |
| Redesign / B full-data / 결합 | `DACON_Redesign_H100_Experiments.ipynb` | `redesign_train.py`, `redesign_fit.py`, `redesign_file.py`, `redesign_voice.py`, `redesign_submission.py` |
| FullBank / MIL / 확장 실험 | `FullBank_MERT330M_MIL_H100_8020.ipynb`, `DACON_Eight_H100.ipynb`, `DACON_Expanded_H100_Experiments.ipynb` | `full_bank_train.py`, `full_bank_features.py`, `localization_train.py` |
| 최신 Existing9504 완료 실험 | `results/existing9504_h100_20260920/DACON_Existing9504_H100.ipynb` | `results/existing9504_h100_20260920/code/dacon9504a100/`, `training_snapshot/source/` |
| 최신 제출 추론 | `submissions/Existing9504_MIL/source/script.py` | 같은 디렉터리의 `model/runtime/`, `model/assets/` |

[소스 열기](repository/) · [최신 코드 상세 지도](docs/CODE_MAP.md) · [현재 계획](plan.md)

## 다음 실험 준비 코드

[Stacking v2](experimental/stacking_v2/)는 기존 네 성분 출력을 유지하면서 File 결합만 LR·ExtraTrees·XGBoost로 비교하기 위한 코드입니다. [독립 검토 기록](experimental/stacking_v2/codex_review/REVIEW.md)에 CPU·합성 검증 결과가 있습니다. 신규 데이터에서의 학습·일반화 성능과 GPU/L4 검증은 아직 완료되지 않았습니다. 포함된 실행 안내의 로컬 경로와 외부 fixture는 원래 검토 환경 기준이며, 이 저장소만으로 검증 입력까지 복원되지는 않습니다.
