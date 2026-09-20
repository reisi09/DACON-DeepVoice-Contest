"""Verify downloaded H100 archive integrity, split/label invariants and checkpoint bindings.

Run from the repository: .venv/bin/python results/existing9504_h100_20260920/verify_handoff.py
"""
import csv
import hashlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "author_code.zip"))
from dacon9504a100 import constants as C
from dacon9504a100.dataset import validate_manifest
from dacon9504a100.handoff import coverage_report, verify_zip, zip_reader

state = json.loads((ROOT / "completed/a100_run_status.json").read_text())
archive_path = ROOT / "completed" / Path(state["handoff"]["file"]).name
with archive_path.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
assert digest == state["handoff"]["sha256"], "Downloaded archive differs from remote completion receipt"
with zipfile.ZipFile(archive_path) as archive:
    sums = archive.read("SHA256SUMS.txt")
    expected = dict((name, sha) for sha, name in (line.split("  ", 1) for line in sums.decode().splitlines()))
    expected["SHA256SUMS.txt"] = hashlib.sha256(sums).hexdigest()
verify_zip(archive_path, expected)
with zipfile.ZipFile(archive_path) as archive:
    config = json.loads(archive.read("config/run_config.json"))
    assert config["dataset_expectations"] == C.DATASET_EXPECTATIONS
    assert config["runtime_manifest_sha256"] == C.RUNTIME_MANIFEST_SHA256
    assert hashlib.sha256(archive.read("code/author_code.zip")).hexdigest() == config["code_zip_sha256"]
    rows = list(csv.DictReader(io.StringIO(archive.read("dataset/manifest.csv").decode())))
    dataset = validate_manifest(rows, C.DATASET_EXPECTATIONS)
    coverage = coverage_report(config, zip_reader(archive))
    assert not coverage["problems"], coverage["problems"]
    receipts = {name: json.loads(archive.read(name)) for name in (
        "metrics/val_metrics.json", "metrics/test_metrics.json", "parity/parity_receipt.json",
        "measurements/pilot_receipt.json", "model/results.json", "measurements/features_features_complete.json")}
    for name, value in receipts.items():
        target = ROOT / "completed/review" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value, indent=2) + "\n")
    (ROOT / "completed/HANDOFF_README.md").write_bytes(archive.read("HANDOFF_README.md"))
report = {"archive_sha256": digest, "members_crc_and_sha256_verified": len(expected),
          "dataset": dataset, "coverage": coverage, "verified_epoch": time.time(),
          "scope": "archive integrity, data and evidence bindings; this does not run packaged inference"}
(ROOT / "completed/local_verification.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"rows": len(rows), "splits": dataset["splits"], "features": coverage["features"],
                  "checkpoints": len(coverage["checkpoints"]["files"]), "recorded_raw_audio_parity": coverage["parity_status"],
                  "verified_members": len(expected)}, indent=2))
