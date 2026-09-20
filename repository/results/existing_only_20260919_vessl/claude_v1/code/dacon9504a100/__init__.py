"""DACON 236749 existing-only 9504-row A100 pipeline (author code; Codex reviews).

Kernel-side (standard library): constants, common, dataset, staging, environment,
orchestrator, handoff.  Child-side (isolated Python 3.11 venv): gpu, features,
training, evaluation, parity — these import the unchanged original modules from
source.zip (full_bank_*, mert12_submission, localization_*, graph_scoring, ...).
"""
__version__ = "2026.09.19-a100-9504"
