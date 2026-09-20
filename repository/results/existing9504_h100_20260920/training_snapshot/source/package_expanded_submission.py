"""Package the locked expanded-data MIL winner with its exact frozen encoders."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from expanded_submission import CANDIDATE, DEMUCS_INPUT_DECODER, SEEDS, SubmissionError, load_heads
from package_mert12_submission import RUNTIME as MERT_RUNTIME
from package_redesign_submission import ASSET_NAMES, _write_member
from redesign_assets import validate_assets
from v6_data import sha256

RUNTIME = (*MERT_RUNTIME, "expanded_submission.py", "localization_models.py", "ast_multitask.py")
ENTRYPOINT = '''import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "model/runtime"))
from expanded_submission import run
if __name__ == "__main__":
    run(ROOT)
'''
# All remaining imports use the competition's preinstalled pinned packages.
REQUIREMENTS = "--extra-index-url https://download.pytorch.org/whl/cu128\ntorchvision==0.22.1+cu128\n"


def package(root: Path, assets: Path, results: Path, output: Path) -> dict:
    hashes = validate_assets(assets)
    identity = json.loads((results / "feature_identity.json").read_text())
    lock_path = results / "train/candidate_lock.json"
    lock = json.loads(lock_path.read_text())
    if set(hashes) != ASSET_NAMES or hashes != identity["all_frozen_assets_sha256"]:
        raise SubmissionError("Packaged encoders differ from expanded training features")
    for name, digest in identity["implementation_sha256"].items():
        if sha256(root / name) != digest:
            raise SubmissionError(f"Training implementation changed: {name}")
    if lock["winner_by_dev_only"] != "mil":
        raise SubmissionError("MIL was not selected on Dev")
    checkpoints = [results / f"train/mil_{seed}.pt" for seed in SEEDS]
    load_heads(checkpoints, "cpu")
    for path in checkpoints:
        if sha256(path) != lock["checkpoint_sha256"][path.name]:
            raise SubmissionError(f"Selected checkpoint changed: {path.name}")
    choice = dict(candidate=CANDIDATE, seeds=list(SEEDS), selected_updates=[96, 72, 72],
                  checkpoints=[f"model/trained/{path.name}" for path in checkpoints],
                  file_fusion="max", selection="dev_only", demucs_input_decoder=DEMUCS_INPUT_DECODER,
                  head_compute="CUDA bfloat16 autocast; CPU float32", full_data_refit=False)
    members = {f"model/runtime/{name}": root / name for name in RUNTIME}
    members.update({f"model/assets/{name}": assets / name for name in sorted(hashes)})
    members["model/assets/asset_hashes.json"] = assets / "asset_hashes.json"
    members.update(zip(choice["checkpoints"], checkpoints, strict=True))
    members["model/trained/candidate_lock.json"] = lock_path
    members["model/trained/feature_identity.json"] = results / "feature_identity.json"
    members["model/trained/config.json"] = results / "train/config.json"
    expected = {name: sha256(path) for name, path in members.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial")
    written: dict[str, str] = {}
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name, path in sorted(members.items()):
            written[name] = _write_member(archive, name, path)
            if written[name] != expected[name]:
                raise SubmissionError(f"Input changed during packaging: {name}")
        for name, value in {"script.py": ENTRYPOINT, "requirements.txt": REQUIREMENTS}.items():
            content = value.encode()
            archive.writestr(zipfile.ZipInfo(name), content)
            written[name] = hashlib.sha256(content).hexdigest()
        manifest = json.dumps(dict(schema_version=1, choice=choice, sha256=written),
                              indent=2, sort_keys=True).encode() + b"\n"
        archive.writestr(zipfile.ZipInfo("model/manifest.json"), manifest)
    written["model/manifest.json"] = hashlib.sha256(manifest).hexdigest()
    with zipfile.ZipFile(temporary) as archive:
        expanded = sum(item.file_size for item in archive.infolist())
        if temporary.stat().st_size > 10_000_000_000 or expanded > 32_000_000_000:
            raise SubmissionError("Archive exceeds competition limits")
        if len(archive.namelist()) != len(written) or set(archive.namelist()) != set(written):
            raise SubmissionError("Archive has missing or duplicate members")
        for name, digest in written.items():
            with archive.open(name) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                    raise SubmissionError(f"Archive hash mismatch: {name}")
        if archive.testzip() is not None:
            raise SubmissionError("Archive CRC mismatch")
    temporary.replace(output)
    report = dict(file=str(output.resolve()), sha256=sha256(output), bytes=output.stat().st_size,
                  expanded_bytes=expanded, choice=choice, member_sha256=written, crc_verified=True,
                  official_l4_validated=False, gpu_end_to_end_verified=False)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = package(Path(__file__).resolve().parent, args.assets, args.results, args.output)
    print(json.dumps({key: value for key, value in report.items() if key != "member_sha256"}, indent=2))
