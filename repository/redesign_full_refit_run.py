# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: use the verified CUDA environment: python redesign_full_refit_run.py CONFIG.json
"""Bounded full-data B refit, real packaged inference, and small verified export."""
from __future__ import annotations

import csv
import gc
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

from redesign_h100_run import atomic_json, sha256
from redesign_notebook_runtime import export_results, safe_extract_zip


class RefitError(RuntimeError):
    """The full-data training or submission verification contract failed."""


def execute(root: Path, previous: Path, deadline: float) -> None:
    """Train the predeclared three heads and exercise their packaged audio runtime."""
    import torch
    from redesign_data_validation import read_rows
    from redesign_features import extract_features, load_records
    from redesign_fit import fit_music
    from package_redesign_submission import package

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RefitError("CUDA is required; no CPU fallback")
    if shutil.disk_usage(root).free < 24_000_000_000:
        raise RefitError("Full refit requires 24 GB free for packaging and verification")
    output = root / "output"
    rows = read_rows(root / "source/full_training.csv")
    if (len(rows) != 1696 or len({row['ID'] for row in rows}) != 1696
            or len({row['audio_sha256'] for row in rows}) != 1696):
        raise RefitError("Full-data manifest must contain all 1696 distinct source files")
    eligible = [row for row in rows if row['MUSIC_PRESENT'] == '1' and row['MUSIC_FAKE'] in {'0', '1'}]
    if len(eligible) != 1216 or any(row['split'] != 'full_train' for row in rows):
        raise RefitError("Music label eligibility or full-training assignment changed")
    speech_unknown = [row for row in rows if row['kind'] == 'speech'
                      and row['MUSIC_PRESENT'] == '0' and row['MUSIC_FAKE'] == '']
    if len(speech_unknown) != 480 or any(
        sum(row['original_split'] == name for row in rows) != expected
        for name, expected in (('train', 1469), ('validation', 129), ('sealed', 98))
    ):
        raise RefitError("Authorized speech cohort or former split membership changed")
    atomic_json(output / "progress.json", {"stage": "features", "time": time.time()})
    extraction = extract_features(rows, previous / "assets", previous / "cache", deadline=deadline)
    atomic_json(output / "extraction.json", extraction)
    records = load_records(rows, previous / "cache")
    checkpoints = []
    receipts = []
    for seed, epochs in ((41, 17), (42, 17), (43, 8)):
        atomic_json(output / "progress.json", {"stage": "training", "seed": seed, "epochs": epochs, "time": time.time()})
        trained = fit_music(rows, records, [], output / "models" / f"B_seed{seed}",
                            arm="B", seed=seed, epochs=epochs, patience=1, batch_size=64,
                            device="cuda", deadline=deadline, fixed_epochs=True)
        checkpoints.append(trained.checkpoint)
        atomic_json(trained.checkpoint.parent / "training.json", trained.report)
        receipts.append({"seed": seed, "epochs": epochs, "checkpoint_sha256": sha256(trained.checkpoint)})
    receipt = {"training_mode": "full_data_refit", "arm": "B", "seeds": [41,42,43],
               "epochs": [17,17,8], "checkpoints": receipts, "train_ids": sorted(row['ID'] for row in rows),
               "label_eligible_music": len(eligible), "independent_validation": False, "file_fusion": "noisy_or",
               "former_split_counts": {name: sum(row['original_split'] == name for row in rows) for name in ('train','validation','sealed')},
               "former_sealed_is_now_training": True, "voice_and_presence": "frozen_baseline",
               "source_manifest_sha256": sha256(root / "source/full_training.csv"),
               "feature_receipt_sha256": extraction['receipt_sha256'],
               "feature_receipt": json.loads((previous / 'cache/receipts' / f"{extraction['receipt_sha256']}.json").read_text()),
               "audible_music_label_eligible": sum(bool((records[row['ID']]['valid'] & records[row['ID']]['stem_audible']).any()) for row in eligible),
               "official_score": None}
    atomic_json(output / "fullrefit_receipt.json", receipt)
    shutil.copy2(root / "source/full_training.csv", output / "full_training.csv")
    atomic_json(output / "progress.json", {"stage": "packaging", "time": time.time()})
    archive = root / "B_full1696_submit.zip"
    packaged = package(root / "source", archive, assets=previous / "assets",
                       checkpoints=checkpoints, fullrefit_receipt=output / "fullrefit_receipt.json")
    atomic_json(output / "package_report.json", packaged)
    with zipfile.ZipFile(archive) as packed:
        (output / "submission_manifest.json").write_bytes(packed.read('model/manifest.json'))
    smoke = root / "smoke_package"
    safe_extract_zip(archive, smoke)
    atomic_json(output / "progress.json", {"stage": "offline_packaged_audio_verification", "time": time.time()})
    del trained
    gc.collect()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    torch.cuda.empty_cache()
    atomic_json(output / 'pre_smoke_gpu_memory.json', {
        'allocated_bytes': allocated, 'reserved_before_bytes': reserved,
        'reserved_after_bytes': torch.cuda.memory_reserved(),
        'reason': 'release training allocator cache before independent inference process'})
    smoke_script = root / "source/redesign_submission_smoke.py"
    env = os.environ | {'PYTHONPATH': str(root / 'source')}
    with (output / 'smoke.log').open('w') as log:
        subprocess.run([sys.executable, str(smoke_script), str(root), str(previous)], check=True,
                       env=env, stdout=log, stderr=subprocess.STDOUT, timeout=max(1, deadline-time.time()))
    atomic_json(output / "progress.json", {"stage": "complete", "time": time.time()})


def main() -> int:
    """Export partial evidence on failure; the local watchdog owns compute shutdown."""
    cfg = json.loads(Path(sys.argv[1]).read_text())
    root, previous, shared = (Path(cfg[key]) for key in ('root','previous','shared'))
    started, hard = float(cfg['started_epoch']), float(cfg['pause_deadline_epoch'])
    if not 0 < hard - started <= 4 * 3600:
        raise RefitError("Invalid absolute GPU budget")
    output = root / 'output'
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'failure.txt').exists():
        (output / 'failure.txt').replace(output / 'previous_attempt_failure.txt')
    shared.mkdir(parents=True, exist_ok=True)
    state = {'status': 'running', 'archive_ready': False, 'started_unix': started,
             'hard_deadline_unix': hard, 'updated_unix': time.time(), 'error': None}
    atomic_json(shared / 'run_state.json', state)
    status = 'failed'
    try:
        if sha256(root / 'source/full_training.csv') != cfg['source_manifest_sha256']:
            raise RefitError('Training manifest differs from preflight pin')
        execute(root, previous, hard - 600)
        status = 'exported'
    except Exception as error:  # noqa: BROAD_EXCEPT_OK -- top-level boundary preserves evidence then fails.
        state['error'] = f'{type(error).__name__}: {error}'
        (output / 'failure.txt').write_text(traceback.format_exc())
    finally:
        if (root / 'run.log').exists():
            shutil.copy2(root / 'run.log', output / 'run.log')
        result = export_results((output,), shared / 'results.zip')
        receipt = {'status': 'exported' if status == 'exported' else 'partial_exported',
                   'archive_ready': True, 'archive_verified': result['verified'],
                   'archive_sha256': result['archive_sha256'], 'archive_bytes': result['bytes'],
                   'export_member_count': result['member_count'], 'exported_unix': time.time()}
        atomic_json(shared / 'export_receipt.json', receipt)
        (shared / 'results.zip.sha256').write_text(f"{result['archive_sha256']}  results.zip\n")
        state.update(receipt, status=status, updated_unix=time.time())
        atomic_json(shared / 'run_state.json', state)
    return 0 if status == 'exported' else 1


if __name__ == '__main__':
    raise SystemExit(main())
