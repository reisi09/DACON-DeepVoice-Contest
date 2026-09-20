"""Review actual manifests, NaN parity rejection, lock exclusion and CPU resume.

Run from repository root with .venv/bin/python. Uses synthetic frozen features
only for optimizer-resume verification; no encoders, GPU or cloud mutations.
"""
import importlib.util
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
CODE = HERE / "code"
sys.path[:0] = [str(CODE), str(ROOT)]

import numpy as np
import torch

from dacon9504a100.common import ControllerLock, canonical_sha, sha256_file
from dacon9504a100.constants import DATASET_EXPECTATIONS, RUNTIME_MANIFEST_SHA256, SOURCE_HASHES
from dacon9504a100.dataset import read_manifest, validate_manifest
from dacon9504a100.features import compare_arrays, identity_key
from dacon9504a100.parity import FEATURE_TOLERANCE, status_for
from dacon9504a100.handoff import coverage_report
from full_bank_features import row_identity
from mert12_submission import WINDOW, window_layout


def main():
    torch.set_num_threads(2)
    manifest = HERE.parent / "manifest.csv"
    assert sha256_file(manifest) == RUNTIME_MANIFEST_SHA256
    report = {"device": "cpu", "torch": str(torch.__version__), "gpu_executed": False}
    audit = validate_manifest(read_manifest(manifest), DATASET_EXPECTATIONS)
    report["manifest"] = {k: audit[k] for k in (
        "rows", "splits", "cross_split_components", "file_fake_by_split", "historical_exposure_by_split")}
    assert all(sha256_file(ROOT / name) == digest for name, digest in SOURCE_HASHES.items())
    report["unchanged_original_modules"] = len(SOURCE_HASHES)
    cases = [([np.nan, 1], [0, 1], "mismatch"), ([1], [np.nan], "mismatch"),
             ([np.nan], [np.nan], "bitwise"), ([np.inf], [np.inf], "mismatch")]
    report["nan_parity"] = []
    for a, b, expected in cases:
        actual = status_for(compare_arrays({"window_truth": np.asarray(a, dtype=float)},
                                          {"window_truth": np.asarray(b, dtype=float)}), FEATURE_TOLERANCE)
        assert actual == expected, (a, b, actual)
        report["nan_parity"].append(actual)

    with tempfile.TemporaryDirectory(prefix="existing9504-v2-review-") as scratch:
        work = Path(scratch)
        lock = ControllerLock(work / "locked").acquire()
        before = lock.path.read_bytes()
        child = subprocess.run([sys.executable, "-c",
            "from pathlib import Path; from dacon9504a100.common import ControllerLock; "
            "ControllerLock(Path(__import__('sys').argv[1])).acquire()", str(lock.job)],
            env={**os.environ, "PYTHONPATH": str(CODE)}, text=True, capture_output=True)
        assert child.returncode != 0 and "Another live controller" in child.stderr
        assert lock.path.read_bytes() == before and lock.owns()
        replacement = ControllerLock(lock.job).acquire()
        assert replacement.owns() and not lock.owns()
        replacement.release()
        report["controller"] = "cross-process rejection preserves holder; same-process ownership transfers"
        report["controller_limit"] = "Linux /proc orphan recovery not exercised on macOS"

        job = work / "job"
        cache = job / "cache"
        cache.mkdir(parents=True)
        identity = {"review_fixture": "synthetic-frozen-features"}
        index = {"identity": identity, "items": {}}
        rows = []
        rng = np.random.default_rng(11)
        for i in range(12):
            row = {"ID": f"fixture_{i}", "split": "train" if i < 8 else "val", "route": "music",
                   "MUSIC_FAKE": str(i % 2), "source": "fixture", "generator": "real" if i % 2 == 0 else "fake"}
            rows.append(row)
            samples = WINDOW + (i % 3) * 16000
            starts = window_layout(samples)
            path = cache / f"fixture_{i}.npz"
            np.savez(path, identity=row_identity(row, identity_key(identity)),
                     mert_windows=rng.normal(size=(len(starts), 1024)).astype(np.float32),
                     voice_embeddings=np.zeros((1, 1280), np.float32), presence=np.asarray([0., 1.]),
                     starts=starts, valid_samples=np.minimum(WINDOW, samples-starts), samples=samples,
                     window_truth=np.full(len(starts), np.nan, np.float32), voice_silent=False)
            index["items"][row["ID"]] = {"path": path.name, "sha256": sha256_file(path)}
        (cache / "index.json").write_text(json.dumps(index))
        torch.save({"fake_label_index": 1, "module": "fixture_fc5", "state_dict": torch.nn.Linear(1280, 2).state_dict()},
                   cache / "voice_classifier_initial.pt")
        spec = importlib.util.spec_from_file_location("author_cpu_checks", CODE / "tests/test_cpu_integration.py")
        checks = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checks)
        checks.check_training_resume({"train_benchmark_override": {"microbatch": 256, "window_limit": 10**6}},
                                     rows, job, work)
        report["optimizer_resume"] = "actual WindowMIL loop: uninterrupted/resumed weights, predictions, history and RNG equal"
        report["optimizer_fixture"] = "8 synthetic train bags and 4 synthetic val bags, 5 epochs, seed 41, CPU"

        # Packaging must bind the bytes it packs to the evidence already approved.
        gate = work / "gate"
        config = {"manifest": str(manifest), "runtime_manifest_sha256": RUNTIME_MANIFEST_SHA256,
                  "dataset_expectations": DATASET_EXPECTATIONS}
        for kind in ("cache", "features", "train", "eval", "parity"):
            directory = gate / kind
            directory.mkdir(parents=True)
            config[kind + "_dir"] = str(directory)
        real_rows = read_manifest(manifest)
        (gate / "cache/index.json").write_text(json.dumps({"items": {r["ID"]: {} for r in real_rows}}))
        (gate / "features/features_complete.json").write_text(json.dumps({"rows": 9504,
            "index_sha256": sha256_file(gate / "cache/index.json")}))
        checkpoint_hashes = {}
        for route in ("music", "voice"):
            for seed in (41, 42, 43):
                name = f"{route}_{seed}.pt"
                (gate / "train" / name).write_bytes(b"synthetic lock byte fixture")
                checkpoint_hashes[name] = sha256_file(gate / "train" / name)
        for split in ("val", "test"):
            with (gate / "eval" / f"{split}_predictions.csv").open("w") as stream:
                writer = csv.writer(stream)
                writer.writerow(["ID", "FILE_FAKE_PROB"])
                writer.writerows((r["ID"], "0.1") for r in real_rows if r["split"] == split)
            (gate / "eval" / f"{split}_metrics.json").write_text('{"fixture": true}')
        lock_body = {"checkpoints": checkpoint_hashes,
                     "validation_predictions_sha256": sha256_file(gate / "eval/val_predictions.csv"),
                     "validation_metrics_sha256": sha256_file(gate / "eval/val_metrics.json")}
        (gate / "train/candidate_lock.json").write_text(json.dumps({**lock_body, "lock_body_sha256": canonical_sha(lock_body)}))
        lock_sha = sha256_file(gate / "train/candidate_lock.json")
        (gate / "eval/test_receipt.json").write_text(json.dumps({"lock_sha256": lock_sha,
            "predictions_sha256": sha256_file(gate / "eval/test_predictions.csv"),
            "metrics_sha256": sha256_file(gate / "eval/test_metrics.json")}))
        (gate / "parity/parity_receipt.json").write_text(json.dumps({"status": "bitwise", "lock_sha256": lock_sha}))
        assert not coverage_report(config)["problems"]
        prediction = gate / "eval/test_predictions.csv"
        prediction.write_text(prediction.read_text().replace(",0.1", ",0.9", 1))
        after_prediction_edit = coverage_report(config)["problems"]
        (gate / "parity/parity_receipt.json").write_text(json.dumps({"status": "bitwise", "lock_sha256": "0"*64}))
        after_wrong_parity_lock = coverage_report(config)["problems"]
        assert not after_prediction_edit and not after_wrong_parity_lock
        report["handoff_evidence_binding"] = {"status": "defect_reproduced",
            "changed_test_probability_without_changing_ids": "accepted despite test_receipt hash mismatch",
            "parity_receipt_from_wrong_lock": "accepted",
            "scope": "Actual coverage_report using real 9504 IDs and synthetic evidence; no real results modified"}

    notebook = json.loads((HERE / "DACON_Existing9504_A100.ipynb").read_text())
    final_source = "".join(notebook["cells"][-1]["source"])
    assert 'FileLink(os.path.relpath(_final, Path.cwd()))' in final_source
    report["download_link"] = {"status": "requires_fix_for_root_scoped_jupyter",
        "notebook_cwd": "/root", "published_zip": "/root-1/existing-only-9504-20260919/DACON_Existing9504_A100_Handoff.zip",
        "generated_relative_path": os.path.relpath("/root-1/existing-only-9504-20260919/DACON_Existing9504_A100_Handoff.zip", "/root"),
        "boundary": "Outside a Jupyter server rooted at /root; live VESSL download not tested"}
    (HERE / "codex_cpu_results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
