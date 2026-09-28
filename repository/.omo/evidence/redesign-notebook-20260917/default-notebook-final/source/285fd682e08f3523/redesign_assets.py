"""Safely materialize pinned baseline and MERT assets from an M330 package."""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Final

ASSET_PREFIXES: Final = (
    "model/assets/baseline/model/df_arena_1b/",
    "model/assets/baseline/model/htdemucs/",
    "model/assets/baseline/model/panns/",
    "model/assets/mert/",
)
ASSET_FILES: Final = {"model/assets/baseline/script.py"}
REQUIRED: Final = {
    "baseline/script.py",
    "baseline/model/df_arena_1b/pytorch_model.bin",
    "baseline/model/htdemucs/htdemucs.yaml",
    "baseline/model/panns/Cnn14_mAP=0.431.pth",
    "baseline/model/panns/component_labels.json",
    "mert/config.json",
    "mert/preprocessor_config.json",
    "mert/pytorch_model.bin",
}


class AssetError(ValueError):
    """A package or extracted asset violates the pinned asset contract."""


def sha256(path: Path) -> str:
    """Hash a file without loading large model weights into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _target_name(member: str) -> str | None:
    if member == "model/assets/baseline/script.py":
        return "baseline/script.py"
    for prefix in ASSET_PREFIXES:
        if member.startswith(prefix) and not member.endswith("/"):
            if prefix.startswith("model/assets/baseline/model/"):
                return "baseline/model/" + member.split("/model/", 1)[1]
            return "mert/" + member.removeprefix("model/assets/mert/")
    return None


def validate_assets(assets: Path) -> dict[str, str]:
    """Verify every file against the flat hash manifest required by extraction."""
    manifest_path = assets / "asset_hashes.json"
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise AssetError(f"Unreadable asset hash manifest: {manifest_path}") from error
    if not isinstance(raw, dict) or not raw:
        raise AssetError("asset_hashes.json must be a non-empty path-to-SHA256 mapping")
    manifest: dict[str, str] = {}
    for name, expected in raw.items():
        if not isinstance(name, str) or not isinstance(expected, str):
            raise AssetError("Asset hash entries must be string pairs")
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise AssetError(f"Unsafe asset manifest path: {name}")
        path = assets.joinpath(*relative.parts)
        if path.is_symlink() or not path.resolve().is_relative_to(assets.resolve()):
            raise AssetError(f"Asset path escapes the configured root: {name}")
        if not path.is_file() or sha256(path) != expected:
            raise AssetError(f"Asset digest mismatch: {name}")
        manifest[name] = expected
    missing = REQUIRED.difference(manifest)
    if missing:
        raise AssetError(f"Required assets missing from manifest: {sorted(missing)}")
    return manifest


def prepare_assets(archive: Path, assets: Path, *, expected_sha256: str) -> dict[str, str]:
    """Extract only baseline encoders and MERT from an exactly pinned M330 ZIP."""
    if sha256(archive) != expected_sha256:
        raise AssetError("M330 archive SHA256 differs from the configured pin")
    if assets.exists():
        raise AssetError(f"Asset destination already exists: {assets}")
    with zipfile.ZipFile(archive) as package:
        names = [member.filename for member in package.infolist()]
        if any("\\" in name or PurePosixPath(name).is_absolute() for name in names):
            raise AssetError("M330 package contains a non-portable member path")
        selected_infos = [member for member in package.infolist() if _target_name(member.filename) is not None]
        if len({member.filename for member in selected_infos}) != len(selected_infos):
            raise AssetError("M330 package contains duplicate selected members")
        if any(stat.S_ISLNK(member.external_attr >> 16) for member in selected_infos):
            raise AssetError("M330 package contains a selected symbolic link")
        bad = package.testzip()
        if bad is not None:
            raise AssetError(f"M330 ZIP CRC failure: {bad}")
        try:
            package_manifest = json.loads(package.read("model/manifest.json"))
            expected = package_manifest["sha256"]
        except (KeyError, json.JSONDecodeError) as error:
            raise AssetError("M330 package manifest is missing or invalid") from error
        selected = {
            member.filename: target
            for member in package.infolist()
            if (target := _target_name(member.filename)) is not None
        }
        if any(".." in PurePosixPath(target).parts for target in selected.values()):
            raise AssetError("Unsafe selected asset path in M330 package")
        if len(set(selected.values())) != len(selected):
            raise AssetError("M330 package contains colliding asset paths")
        if not REQUIRED.issubset(selected.values()):
            raise AssetError("M330 package lacks required baseline or MERT assets")
        assets.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="redesign-assets-", dir=assets.parent) as temporary:
            stage = Path(temporary) / "assets"
            hashes: dict[str, str] = {}
            for member, relative in selected.items():
                declared = expected.get(member)
                if not isinstance(declared, str):
                    raise AssetError(f"Package manifest omits selected member: {member}")
                target = stage.joinpath(*PurePosixPath(relative).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with package.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                actual = sha256(target)
                if actual != declared:
                    raise AssetError(f"Package member digest mismatch: {member}")
                hashes[relative] = actual
            (stage / "asset_hashes.json").write_text(
                json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            stage.replace(assets)
    return hashes
