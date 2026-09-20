"""Build and hash-verify the offline MERT12 full-data submission ZIP."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Final

from mert12_full_refit import EXPECTED_ROWS
from mert12_submission import DEMUCS_INPUT_DECODER, SEEDS, load_heads
from package_redesign_submission import ASSET_NAMES, _write_member
from redesign_assets import validate_assets
from v6_data import sha256

RUNTIME: Final = ("mert12_submission.py", "eight_heads.py", "v6_runtime.py", "v6_core.py",
                  "v6_data.py", "graph_scoring.py")
REQUIREMENTS: Final = """--extra-index-url https://download.pytorch.org/whl/cu128
torch==2.7.1+cu128
torchaudio==2.7.1+cu128
torchvision==0.22.1+cu128
diffq==0.2.4
einops==0.8.1
"""
ENTRYPOINT: Final = """import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'model/runtime'))
from mert12_submission import run
if __name__ == '__main__':
    run(ROOT)
"""


class PackageError(ValueError):
    """The refit receipt, assets, or completed archive violates its contract."""


def validate_run(run: Path, assets: dict[str, str]) -> tuple[dict, list[Path]]:
    """Bind all heads to full coverage, frozen assets, and actual CUDA update evidence."""
    receipt_path = run / "full_refit_receipt.json"
    training_manifest = run / "training_manifest.json"
    gpu_receipt = run / "gpu_training_receipt.json"
    audit_receipt = run / "joined_data_audit.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    lineage = json.loads(training_manifest.read_text(encoding="utf-8"))
    gpu = json.loads(gpu_receipt.read_text(encoding="utf-8"))
    audit = json.loads(audit_receipt.read_text(encoding="utf-8"))
    benchmark = receipt.get("batch_benchmark")
    ids = receipt["train_ids"]
    if (receipt["training_mode"] != "full_data_refit" or receipt["candidate"] != "E6_MERT_layer12"
            or receipt["seeds"] != list(SEEDS) or receipt["steps"] != 500 or receipt["batch_size"] != 32
            or receipt["architecture"] != "1024->128 GELU->1" or receipt["independent_validation"] is not False
            or not receipt["all_former_splits_are_training"] or receipt["eligible_rows"] != EXPECTED_ROWS
            or len(ids) != EXPECTED_ROWS
            or len(ids) != len(set(ids)) or not ids
            or receipt["training_manifest_sha256"] != sha256(training_manifest)
            or receipt["audit_receipt_sha256"] != sha256(audit_receipt)
            or audit.get("eligible_ids") != ids
            or audit.get("manifest_sha256") != lineage.get("source_manifest_sha256")
            or not isinstance(benchmark, dict) or benchmark.get("final_recipe_batch_size") != 32
            or [item.get("batch_size") for item in benchmark.get("results", [])] != [32, 128, 256]
            or receipt["frozen_asset_sha256"] != assets or lineage["frozen_asset_sha256"] != assets
            or [row["ID"] for row in lineage["rows"]] != ids
            or gpu["checkpoints"] != receipt["checkpoints"]):
        raise PackageError("Full-refit receipt, lineage, GPU evidence, or frozen assets disagree")
    checkpoints = [run / f"head_seed{seed}.pt" for seed in SEEDS]
    load_heads(checkpoints, "cpu")
    for path, evidence in zip(checkpoints, receipt["checkpoints"], strict=True):
        counts = evidence["after_counts_by_id"]
        if (evidence["checkpoint_sha256"] != sha256(path) or set(counts) != set(ids)
                or any(int(value) < 1 for value in counts.values()) or evidence["after_zero_exposure_ids"]
                or evidence["parameter_update_l2"] <= 0
                or not evidence["first_gradient_l1"]
                or any(value <= 0 for value in evidence["first_gradient_l1"].values())
                or evidence["device"] != "cuda"):
            raise PackageError(f"Incomplete training or coverage evidence: {path.name}")
    return receipt, checkpoints


def package(root: Path, assets_path: Path, run: Path, output: Path) -> dict:  # noqa: DICT_OK
    """Stream a deterministic ZIP and verify every member hash, CRC, and size limit."""
    assets = validate_assets(assets_path)
    if set(assets) != ASSET_NAMES:
        raise PackageError("Expected exactly the frozen baseline and MERT asset set")
    receipt, checkpoints = validate_run(run, assets)
    checkpoint_names = [f"model/trained/head_seed{seed}.pt" for seed in SEEDS]
    members = {f"model/runtime/{name}": root / name for name in RUNTIME}
    members.update({f"model/assets/{name}": assets_path / name for name in sorted(assets)})
    members["model/assets/asset_hashes.json"] = assets_path / "asset_hashes.json"
    members.update(zip(checkpoint_names, checkpoints, strict=True))
    for name in ("full_refit_receipt.json", "training_manifest.json", "gpu_training_receipt.json",
                 "joined_data_audit.json"):
        members[f"model/trained/{name}"] = run / name
    if (run / "batch_benchmark.json").is_file():
        members["model/trained/batch_benchmark.json"] = run / "batch_benchmark.json"
    expected = {f"model/assets/{name}": digest for name, digest in assets.items()}
    expected.update({name: item["checkpoint_sha256"]
                     for name, item in zip(checkpoint_names, receipt["checkpoints"], strict=True)})
    expected.update({name: sha256(path) for name, path in members.items() if name not in expected})
    choice = {"candidate": "E6_MERT_layer12_full_data", "seeds": list(SEEDS),
              "checkpoints": checkpoint_names, "file_fusion": "max",
              "demucs_input_decoder": DEMUCS_INPUT_DECODER,
              "window_pooling": "all_valid_mean_no_audible_mask",
              "training_rows": receipt["eligible_rows"],
              "training_manifest_sha256": receipt["training_manifest_sha256"]}
    content = {"script.py": ENTRYPOINT.encode(), "requirements.txt": REQUIREMENTS.encode()}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial")
    written: dict[str, str] = {}
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name, path in sorted(members.items()):
            written[name] = _write_member(archive, name, path)
            if written[name] != expected[name]:
                raise PackageError(f"Input changed while packaging: {name}")
        for name, value in sorted(content.items()):
            archive.writestr(zipfile.ZipInfo(name), value)
            written[name] = hashlib.sha256(value).hexdigest()
        manifest = json.dumps({"schema_version": 1, "choice": choice, "sha256": written},
                              indent=2, sort_keys=True).encode() + b"\n"
        archive.writestr(zipfile.ZipInfo("model/manifest.json"), manifest)
    written["model/manifest.json"] = hashlib.sha256(manifest).hexdigest()
    with zipfile.ZipFile(temporary) as archive:
        expanded = sum(member.file_size for member in archive.infolist())
        if temporary.stat().st_size > 10_000_000_000 or expanded > 32_000_000_000:
            raise PackageError("Archive exceeds competition size limits")
        if len(archive.namelist()) != len(written) or set(archive.namelist()) != set(written):
            raise PackageError("Archive contains missing or duplicate members")
        for name, digest in written.items():
            with archive.open(name) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                    raise PackageError(f"Archived member hash mismatch: {name}")
        if archive.testzip() is not None:
            raise PackageError("Archive CRC verification failed")
    temporary.replace(output)
    report = {"file": str(output), "sha256": sha256(output), "bytes": output.stat().st_size,
              "expanded_bytes": expanded, "choice": choice, "member_sha256": written,
              "crc_verified": True, "gpu_end_to_end_verified": False,
              "official_environment": "Python 3.11, torch 2.7.1 CUDA 12.8"}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = package(args.root, args.assets, args.run, args.output)
    print(json.dumps({"status": "complete", "file": report["file"], "sha256": report["sha256"],
                      "bytes": report["bytes"], "expanded_bytes": report["expanded_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
