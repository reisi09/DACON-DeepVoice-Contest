# DACON Existing9504 — consolidated source archive

This archive contains source code, not the model-weight submission. No GPU job is launched by creating or extracting it.

## Entry points

- `results/existing9504_h100_20260920/DACON_Existing9504_H100.ipynb`: notebook from the completed H100 handoff.
- `results/existing9504_h100_20260920/code/dacon9504a100/`: exact training controller, data loader, features, training, evaluation and handoff implementation.
- `results/existing9504_h100_20260920/training_snapshot/source/`: exact shared runtime sources from the H100 handoff. Prefer these over historical repository-root versions when inspecting that run.
- `submissions/Existing9504_MIL/source/script.py`: exact final submission entry point.
- `submissions/Existing9504_MIL/source/model/assets/mert/modeling_MERT.py`: MERT architecture; `configuration_MERT.py` contains its configuration class.
- `submissions/Existing9504_MIL/source/model/runtime/full_bank_submission.py`: actual packaged inference, including frozen MERT330M layer12 and WindowMIL heads.
- `package_existing9504_submission.py`, `verify_existing9504_submission.py`: submission packaging and real offline inference verification.
- Repository-root Python scripts and notebooks include data collection, reconstruction, splitting, auditing, earlier experiments and packaging helpers. Historical experiments are included for reference; they are not the current run entry point.
- `results/existing_only_20260919_final/`: selected train/val/test manifests and provenance. Audio files themselves are not included.

## Scope and execution

The latest run used 9,504 selected rows: 6,652 train, 1,426 validation, 1,426 test. MERT is frozen; downstream heads are trained. The test is historically exposed, not independent.

Included: repository tracked code outside internal agent evidence folders, current root scripts/notebooks, the completed run's nested source archives and extracted sources, final submission Python/config/license files, requirements, split manifests, package QA report. Notebook outputs and execution counters are removed; cell source is preserved.

Excluded: original audio, feature caches, learned weights, FFmpeg executable, virtual environments, installed third-party packages, Git history, credentials/session logs, and obsolete untracked temporary run copies. HTDemucs/PANNs/Transformers library implementations are dependencies documented in the requirements, not vendored packages. Existing third-party license notices are preserved.

This is a source handoff, not a standalone runnable model ZIP. Inference also needs weights from the separate `submissions/Existing9504_MIL/submit.zip`. Training needs the audio/assets paths and storage configured by the notebook; environment and workspace paths must be adapted on another machine. Packaging/parity scripts require the completed Handoff ZIP at their documented path. No notebook, training, or GPU job was executed for this source export.

`SOURCE_MANIFEST.json` records origins, sizes, and SHA-256 for every exported file except itself. `archive_receipt.json` beside the ZIP records archive integrity verification. Archive validation does not revalidate historical experiments or establish L4 compatibility.
