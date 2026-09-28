"""Build the four complete, self-contained V5 submission archives."""

# allow: SIZE_OK - atomic four-archive builder and its offline parity runner share one contract.

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Iterator, Mapping
from pathlib import Path, PurePosixPath
from typing import Final

MAX_ZIP_BYTES: Final = 10_000_000_000
MAX_EXPANDED_BYTES: Final = 32_000_000_000
ARMS: Final = ("B1", "B2", "B3", "B4")
EXPERTS: Final = ("voice", "music", "presence")
RUNTIME_FILES: Final = ("v5_submission.py", "v5_experts.py", "v5_fusion.py")
DEPLOY_SCRIPT: Final = '''#!/usr/bin/env python3
"""Run one completed V5 arm from packaged local assets only."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "model" / "runtime"))

from v5_submission import main

if __name__ == "__main__":
    main(ROOT)
'''
# Keep timm's torchvision dependency from replacing the evaluator's torch ABI.
REQUIREMENTS: Final = """--extra-index-url https://download.pytorch.org/whl/cu128
torch==2.7.1+cu128
torchaudio==2.7.1+cu128
torchvision==0.22.1+cu128
imageio-ffmpeg==0.6.0
timm==1.0.15
"""
EXCLUDED_BASE_WEIGHTS: Final = frozenset(
    {"pytorch_model.bin", "model.safetensors", "Cnn14_mAP=0.431.pth"}
)


class PackagingError(ValueError):
    """A source asset or training artifact cannot form a final V5 archive."""


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PackagingError(f"Cannot read completion metadata: {path}") from error
    if not isinstance(value, dict):
        raise PackagingError(f"Completion metadata must be an object: {path}")
    return value


def _completed_checkpoint(job: Path, identity: Mapping[str, str | bool]) -> tuple[Path, dict]:
    """Accept only a fully exposed, digest-pinned final training state."""
    checkpoint = job / "final.pt"
    metadata = _json(job / "training.json")
    if metadata.get("status") != "complete":
        raise PackagingError(f"Training job is not complete: {job.name}")
    full_refit = metadata.get("full_data_refit")
    if not isinstance(full_refit, dict) or full_refit.get("status") != "complete":
        raise PackagingError(f"Training job is not a complete full-data refit: {job.name}")
    if any(metadata.get(key) != value for key, value in identity.items()):
        raise PackagingError(f"Training artifact identity mismatch: {job.name}")
    requested = metadata.get("epochs_requested")
    completed = metadata.get("epochs_completed")
    if not isinstance(requested, int) or requested < 1 or completed != requested:
        raise PackagingError(f"Training epochs are incomplete: {job.name}")
    exposures = metadata.get("exposures")
    if not isinstance(exposures, dict) or not exposures or any(
        not isinstance(value, int) or value != requested for value in exposures.values()
    ):
        raise PackagingError(f"Training exposure coverage is incomplete: {job.name}")
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise PackagingError(f"Final checkpoint is missing or empty: {job.name}")
    actual = _sha256(checkpoint)
    if metadata.get("state_sha256") != actual:
        raise PackagingError(f"Final checkpoint digest mismatch: {job.name}")
    return checkpoint, metadata


def _model_root(baseline: Path) -> Path:
    for candidate in (baseline / "model", baseline):
        if all((candidate / name).is_dir() for name in ("df_arena_1b", "panns", "htdemucs")):
            return candidate
    raise PackagingError("Baseline model assets must contain df_arena_1b, panns, and htdemucs")


def _source_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and not path.name.startswith("."):
            yield path


def _asset_members(
    baseline: Path, sonics: Path, refits: Path, *, temporal: bool,
) -> dict[str, Path]:
    model_root = _model_root(baseline)
    required = (
        model_root / "df_arena_1b" / "config.json",
        model_root / "df_arena_1b" / "modeling_antispoofing.py",
        model_root / "panns" / "class_labels_indices.csv",
        sonics / "config.json",
        sonics / "LICENSE",
        sonics / "provenance.json",
        refits / "data_audit.json",
        baseline.parent / "v4_validation" / "provenance.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    vendor = sonics.parent / "sonics_vendor" / "sonics"
    if not vendor.is_dir():
        missing.append(str(vendor))
    if temporal and not any((model_root / "htdemucs").glob("*.th")):
        missing.append(str(model_root / "htdemucs" / "*.th"))
    if missing:
        raise PackagingError(f"Required offline architecture assets are missing: {missing}")

    members: dict[str, Path] = {}
    for path in model_root.iterdir():
        if path.is_file() and path.name not in EXCLUDED_BASE_WEIGHTS:
            members[f"model/{path.name}"] = path
    for path in _source_files(model_root / "df_arena_1b"):
        if path.name not in EXCLUDED_BASE_WEIGHTS:
            members[f"model/df_arena_1b/{path.relative_to(model_root / 'df_arena_1b').as_posix()}"] = path
    for name in ("class_labels_indices.csv", "component_labels.json"):
        path = model_root / "panns" / name
        if path.is_file():
            members[f"model/panns/{name}"] = path
    for name in ("config.json", "LICENSE", "provenance.json"):
        members[f"model/sonics/{name}"] = sonics / name
    for path in _source_files(vendor):
        if path.suffix == ".py":
            members[f"model/sonics_vendor/sonics/{path.relative_to(vendor).as_posix()}"] = path
    if temporal:
        for path in _source_files(model_root / "htdemucs"):
            members[f"model/htdemucs/{path.relative_to(model_root / 'htdemucs').as_posix()}"] = path
    data_root = baseline.parent / "v4_validation"
    members["model/provenance/final_data_audit.json"] = refits / "data_audit.json"
    members["model/provenance/source_provenance.json"] = data_root / "provenance.json"
    licenses = data_root / "licenses"
    if licenses.is_dir():
        for path in _source_files(licenses):
            members[f"model/provenance/licenses/{path.relative_to(licenses).as_posix()}"] = path
    return members


def _training_artifacts(refits: Path, arms: tuple[str, ...] = ARMS) -> tuple[dict[str, dict[str, tuple[Path, dict]]], dict[str, tuple[Path, dict]]]:
    experts: dict[str, dict[str, tuple[Path, dict]]] = {}
    for arm in ("B1", "B2"):
        if arm not in arms and not (arm == "B2" and set(arms) & {"B3", "B4"}):
            continue
        experts[arm] = {
            kind: _completed_checkpoint(
                refits / "jobs" / f"{arm}_{kind}",
                {"arm": arm, "kind": kind, "augmented": arm == "B2"},
            )
            for kind in EXPERTS
        }
    fusion = {
        arm: _completed_checkpoint(refits / arm, {"name": arm, "joint": arm == "B4"})
        for arm in arms if arm in {"B3", "B4"}
    }
    return experts, fusion


def _write_archive(
    output: Path,
    arm: str,
    assets: Mapping[str, Path],
    expert_states: Mapping[str, tuple[Path, dict]],
    fusion_state: tuple[Path, dict] | None,
) -> Path:
    source = Path(__file__).resolve().parent
    generated: dict[str, bytes] = {
        "script.py": DEPLOY_SCRIPT.encode(),
        "requirements.txt": REQUIREMENTS.encode(),
        "model/runtime/arm.json": json.dumps({"arm": arm}, sort_keys=True).encode(),
    }
    for name in RUNTIME_FILES:
        path = source / name
        if not path.is_file():
            raise PackagingError(f"V5 runtime source is missing: {name}")
        generated[f"model/runtime/{name}"] = path.read_bytes()
    files = dict(assets)
    receipts: dict[str, dict[str, str | int]] = {}
    for kind, (path, metadata) in expert_states.items():
        files[f"model/weights/{kind}.pt"] = path
        receipts[kind] = {
            "sha256": str(metadata["state_sha256"]), "bytes": path.stat().st_size,
        }
    if fusion_state is not None:
        path, metadata = fusion_state
        files["model/weights/fusion.pt"] = path
        receipts["fusion"] = {
            "sha256": str(metadata["state_sha256"]), "bytes": path.stat().st_size,
        }
    generated["model/runtime/weights.json"] = json.dumps(receipts, sort_keys=True).encode()
    expanded = sum(path.stat().st_size for path in files.values()) + sum(map(len, generated.values()))
    if expanded > MAX_EXPANDED_BYTES or expanded > MAX_ZIP_BYTES:
        raise PackagingError(f"V5 {arm} archive would exceed submission limits: {expanded} bytes")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
            for name, path in files.items():
                posix = PurePosixPath(name)
                if posix.is_absolute() or ".." in posix.parts:
                    raise PackagingError(f"Unsafe archive member: {name}")
                archive.write(path, name)
            for name, content in generated.items():
                archive.writestr(name, content)
        if temporary.stat().st_size > MAX_ZIP_BYTES:
            raise PackagingError(f"Built V5 {arm} archive exceeds the ZIP limit")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def build_v5_submissions(
    baseline_assets: Path,
    sonics_assets: Path,
    refit_dir: Path,
    output_dir: Path,
    *, arms: tuple[str, ...] = ARMS,
) -> tuple[Path, ...]:
    """Build selected arms after verifying their required completed refits."""
    if not arms or len(set(arms)) != len(arms) or set(arms) - set(ARMS):
        raise PackagingError("Select unique supported submission arms")
    experts, fusion = _training_artifacts(refit_dir, arms)
    raw_assets = _asset_members(baseline_assets, sonics_assets, refit_dir, temporal=False)
    temporal_assets = (_asset_members(baseline_assets, sonics_assets, refit_dir, temporal=True)
                       if set(arms) & {"B3", "B4"} else {})
    outputs: list[Path] = []
    for arm in arms:
        source_arm = arm if arm in {"B1", "B2"} else "B2"
        outputs.append(_write_archive(
            output_dir / f"V5_{arm}.zip", arm,
            temporal_assets if arm in {"B3", "B4"} else raw_assets,
            experts[source_arm], fusion.get(arm),
        ))
    return tuple(outputs)


def verify_offline_archive(
    archive: Path, data: Path, reference_csv: Path, *, tolerance: float = 1e-4,
) -> dict[str, str | int | float | bool]:
    """Freshly unpack and compare the real evaluator CLI with reference probabilities."""
    with tempfile.TemporaryDirectory(prefix="v5-offline-") as directory:
        root = Path(directory)
        with zipfile.ZipFile(archive) as source:
            source.extractall(root)
        shutil.copytree(data, root / "data")
        environment = os.environ.copy()
        environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        completed = subprocess.run(
            [sys.executable, "script.py"], cwd=root, env=environment,
            capture_output=True, text=True, timeout=3600, check=False,
        )
        if completed.returncode:
            raise PackagingError(f"Packaged inference failed: {completed.stderr[-1000:]}")
        actual_path = root / "output" / "submission.csv"
        with reference_csv.open(encoding="utf-8-sig", newline="") as stream:
            reference = list(csv.DictReader(stream))
        with actual_path.open(encoding="utf-8-sig", newline="") as stream:
            actual = list(csv.DictReader(stream))
        if [row["ID"] for row in actual] != [row["ID"] for row in reference]:
            raise PackagingError("Packaged inference changed evaluator ID order")
        delta = max(
            abs(float(left[column]) - float(right[column]))
            for left, right in zip(actual, reference, strict=True)
            for column in reference[0] if column != "ID"
        )
        if delta > tolerance:
            raise PackagingError(f"Packaged inference parity exceeded tolerance: {delta}")
        return {"passed": True, "rows": len(actual), "max_abs_delta": delta,
                "tolerance": tolerance, "archive_sha256": _sha256(archive)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-assets", type=Path, required=True)
    parser.add_argument("--sonics-assets", type=Path, required=True)
    parser.add_argument("--refit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    args = parser.parse_args()
    for path in build_v5_submissions(
        args.baseline_assets, args.sonics_assets, args.refit_dir, args.output_dir, arms=tuple(args.arms),
    ):
        print(path)


if __name__ == "__main__":
    main()
