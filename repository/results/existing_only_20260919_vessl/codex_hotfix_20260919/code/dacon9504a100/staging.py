"""Stage and verify every input from the object volume onto the local job disk.

Standard library only (runs in the notebook kernel before the isolated
environment exists).  Contract (AUTHOR_REQUEST.txt):
  * deployment.json ``files`` maps relative filenames -> {key, bytes, sha256, etag};
    ``key`` is relative to the object-volume root (/root-1).
  * dataset/manifest.csv holds all 9504 rows; path=audio/<source_sha256><ext>.
  * audio_upload.json: shards [{key, bytes, sha256, etag}], items [{ID, path,
    audio_sha256}], all_sources_uploaded: true.
  * assets.zip: the 27 frozen files at their asset_hashes.json paths plus
    asset_hashes.json; source.zip: original modules under source/.
Only selected audio members are extracted; every used shard and every file
is hash-verified.  Nothing on the object volume is modified.
"""
from __future__ import annotations

import json
import shutil
import time
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from .common import (PipelineError, atomic_json, checked_members, contained, copy_with_hash,
                     extract_exact, extract_member, make_read_only, now_iso, read_json, safe_relative,
                     sha256_file)
from .dataset import cross_check_split_files, read_manifest, validate_manifest

MUTABLE_SUFFIXES = (".ipynb",)  # Jupyter rewrites notebooks when saving outputs.


def resolve_key(objvol_root: Path, key: str) -> Path:
    return contained(objvol_root, key)


def load_deployment(deployment_root: Path) -> dict[str, Any]:
    path = deployment_root / "deployment.json"
    if not path.is_file():
        raise PipelineError(f"deployment.json not found under {deployment_root}")
    deployment = read_json(path)
    files = deployment.get("files")
    if not isinstance(files, dict) or not files:
        raise PipelineError("deployment.json must contain a non-empty 'files' mapping")
    for name, record in files.items():
        safe_relative(name)
        if not isinstance(record, dict) or not {"key", "bytes", "sha256"} <= set(record):
            raise PipelineError(f"deployment.json files[{name!r}] needs key/bytes/sha256")
    if "runtime_manifest_sha256" not in deployment:
        raise PipelineError("deployment.json lacks runtime_manifest_sha256")
    return deployment


def verify_deployment_files(deployment: Mapping[str, Any], objvol_root: Path) -> dict[str, Any]:
    receipt: dict[str, Any] = {}
    for name, record in sorted(deployment["files"].items()):
        path = resolve_key(objvol_root, record["key"])
        if name.endswith(MUTABLE_SUFFIXES):
            receipt[name] = {"key": record["key"], "verified": False,
                             "reason": "notebook bytes change when Jupyter saves outputs"}
            continue
        if not path.is_file():
            raise PipelineError(f"Deployment file missing: {name} -> {path}")
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != int(record["bytes"]) or digest != record["sha256"]:
            raise PipelineError(f"Deployment bytes changed: {name} size={size} sha256={digest}")
        receipt[name] = {"key": record["key"], "path": str(path), "bytes": size, "sha256": digest,
                         "etag": record.get("etag"), "verified": True}
    return receipt


def find_file(deployment: Mapping[str, Any], objvol_root: Path, deployment_root: Path,
              names: tuple[str, ...]) -> Path | None:
    """Locate a deployment file by exact relative name, then by basename."""
    files = deployment["files"]
    for name in names:
        if name in files:
            return resolve_key(objvol_root, files[name]["key"])
    for name in names:
        matches = [entry for entry in files if PurePosixPath(entry).name == PurePosixPath(name).name]
        if len(matches) == 1:
            return resolve_key(objvol_root, files[matches[0]]["key"])
    for name in names:
        candidate = deployment_root / name
        if candidate.is_file():
            return candidate
    return None


def _local_asset_problems(destination: Path, expected: Mapping[str, str]) -> list[str]:
    """Names whose local copy is missing or differs (asset_hashes.json compared by content)."""
    problems = []
    for name, digest in expected.items():
        path = contained(destination, name)
        if not path.is_file() or sha256_file(path) != digest:
            problems.append(name)
    manifest = destination / "asset_hashes.json"
    try:
        if not manifest.is_file() or json.loads(manifest.read_text()) != dict(expected):
            problems.append("asset_hashes.json")
    except ValueError:
        problems.append("asset_hashes.json")
    return problems


def stage_assets(assets_zip: Path, destination: Path, expected: Mapping[str, str]) -> dict[str, Any]:
    """Extract exactly the 27 frozen files + asset_hashes.json; bytes must equal the contract.

    Re-runs re-hash every local file; a missing or altered local file is re-extracted
    from the (hash-verified) assets.zip instead of failing on it.
    """
    zip_sha = sha256_file(assets_zip)
    with zipfile.ZipFile(assets_zip) as archive:
        members = checked_members(archive)
        if "asset_hashes.json" not in members:
            raise PipelineError("assets.zip lacks asset_hashes.json")
        declared = json.loads(archive.read(members["asset_hashes.json"]))
    if declared != dict(expected):
        raise PipelineError("assets.zip asset_hashes.json differs from the reviewed 27-file contract")
    wanted: dict[str, str | None] = dict(expected)
    wanted["asset_hashes.json"] = None
    marker = destination / ".staged.json"
    state = read_json(marker) if marker.is_file() else None
    if state is None or state.get("assets_zip_sha256") != zip_sha:
        if destination.exists():
            shutil.rmtree(destination)  # job-local copy only; the object volume is never modified
        problems, repaired_only = sorted(wanted), False
    else:
        problems, repaired_only = _local_asset_problems(destination, expected), True
        if not problems:
            return {**state, "reused": True}
    hashes = extract_exact(assets_zip, destination, wanted, only=problems)
    for name in hashes:
        make_read_only(destination / name)
    remaining = _local_asset_problems(destination, expected)
    if remaining:
        raise PipelineError(f"Staged assets still differ after extraction: {remaining}")
    files = {name: sha256_file(destination / name) for name in wanted}
    state = {"assets_zip_sha256": zip_sha, "files": files, "staged_at": now_iso(),
             "repaired": sorted(problems) if repaired_only else []}
    atomic_json(marker, state)
    return state


def stage_source(source_zip: Path, destination: Path, expected: Mapping[str, str]) -> dict[str, Any]:
    """Original modules only; every reviewed module must match its recorded hash."""
    with zipfile.ZipFile(source_zip) as archive:
        members = checked_members(archive)
    present = {name.removeprefix("source/"): name for name in members if name.startswith("source/")}
    missing = sorted(set(expected) - set(present))
    if missing:
        raise PipelineError(f"source.zip lacks reviewed modules: {missing}")
    if destination.exists():
        if all((destination / name).is_file() and sha256_file(destination / name) == digest
               for name, digest in expected.items()):
            return {"modules": dict(expected), "reused": True}
        shutil.rmtree(destination)
    extract_exact(source_zip, destination, dict(expected), strip_prefix="source/", allow_extra=True)
    for name in expected:
        make_read_only(destination / name)
    extra = sorted(set(present) - set(expected))
    return {"modules": dict(expected), "unreviewed_extra_members_not_extracted": extra}


def stage_audio(rows: list[dict[str, str]], upload: Mapping[str, Any], objvol_root: Path,
                data_dir: Path, scratch: Path, log=print) -> dict[str, Any]:
    """Extract only the selected members from hash-verified shards; verify every file."""
    if upload.get("all_sources_uploaded") is not True:
        raise PipelineError("audio_upload.json reports an incomplete upload")
    items = {item["ID"]: item for item in upload.get("items", [])}
    if len(items) != len(upload.get("items", [])):
        raise PipelineError("audio_upload.json items contain duplicate IDs")
    mismatched = [row["ID"] for row in rows if row["ID"] not in items or items[row["ID"]].get("path") != row["path"]
                  or items[row["ID"]].get("audio_sha256") != row["audio_sha256"]]
    extra_items = sorted(set(items) - {row["ID"] for row in rows})
    if mismatched or extra_items:
        raise PipelineError(f"audio_upload items disagree with manifest: mismatched={mismatched[:5]} "
                            f"extra={extra_items[:5]}")
    needed: dict[str, str] = {}
    for row in rows:
        previous = needed.setdefault(row["path"], row["audio_sha256"])
        if previous != row["audio_sha256"]:
            raise PipelineError(f"One audio path maps to two hashes: {row['path']}")
    data_dir.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    satisfied: set[str] = set()
    for relative, digest in needed.items():
        target = contained(data_dir, relative)
        if target.is_file() and sha256_file(target) == digest:
            satisfied.add(relative)
    log(f"[stage_audio] {len(satisfied)}/{len(needed)} files already present and verified")
    shard_receipts = []
    for shard in upload.get("shards", []):
        record = {"key": shard["key"], "bytes": shard["bytes"], "sha256": shard["sha256"], "etag": shard.get("etag")}
        remaining = {path: digest for path, digest in needed.items() if path not in satisfied}
        if not remaining:
            record["status"] = "not_needed_all_files_already_verified"
            shard_receipts.append(record)
            continue
        source = resolve_key(objvol_root, shard["key"])
        local = scratch / "current_shard.zip"
        log(f"[stage_audio] copy+hash {shard['key']} ({int(shard['bytes']) / 1e9:.2f} GB)")
        copy_with_hash(source, local, expected_bytes=int(shard["bytes"]), expected_sha256=shard["sha256"])
        extracted = 0
        with zipfile.ZipFile(local) as archive:
            members = checked_members(archive)
            by_basename: dict[str, list[str]] = {}
            for name in members:
                by_basename.setdefault(PurePosixPath(name).name, []).append(name)
            for relative, digest in remaining.items():
                member = relative if relative in members else None
                if member is None:
                    candidates = by_basename.get(PurePosixPath(relative).name, [])
                    member = candidates[0] if len(candidates) == 1 else None
                if member is None:
                    continue
                extract_member(archive, members[member], contained(data_dir, relative), expected_sha256=digest)
                make_read_only(contained(data_dir, relative))
                satisfied.add(relative)
                extracted += 1
        local.unlink()
        record.update(status="verified_and_used", members_extracted=extracted)
        shard_receipts.append(record)
        log(f"[stage_audio] {shard['key']}: extracted {extracted}; {len(satisfied)}/{len(needed)} verified")
    missing = sorted(set(needed) - satisfied)
    if missing:
        raise PipelineError(f"{len(missing)} selected audio files were not found in any shard: {missing[:5]}")
    unique_bytes = sum(contained(data_dir, relative).stat().st_size for relative in needed)
    row_bytes = sum(contained(data_dir, row["path"]).stat().st_size for row in rows)
    return {"files": len(needed), "rows": len(rows), "audio_bytes_unique_files": unique_bytes,
            "audio_bytes_per_row_sum": row_bytes, "shards": shard_receipts,
            "seconds": time.monotonic() - began, "every_file_sha256_verified": True}


def stage_inputs(config: Mapping[str, Any], log=print) -> dict[str, Any]:
    deployment_root = Path(config["deployment_root"])
    objvol_root = Path(config["objvol_root"])
    job = Path(config["job_root"])
    inputs = job / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    deployment = load_deployment(deployment_root)
    if deployment.get("workspace") not in (None, config["workspace"]):
        raise PipelineError(f"deployment.json workspace {deployment.get('workspace')} != {config['workspace']}")
    pinned = config.get("runtime_manifest_sha256")
    if pinned and deployment["runtime_manifest_sha256"] != pinned:
        raise PipelineError(f"deployment.json runtime_manifest_sha256 {deployment['runtime_manifest_sha256']} "
                            f"!= reviewed {pinned}")
    log("[stage_inputs] verifying every immutable deployment file (bytes + SHA-256)")
    files = verify_deployment_files(deployment, objvol_root)
    manifest_source = find_file(deployment, objvol_root, deployment_root, ("dataset/manifest.csv",))
    upload_source = find_file(deployment, objvol_root, deployment_root, ("audio_upload.json", "dataset/audio_upload.json"))
    assets_zip = find_file(deployment, objvol_root, deployment_root, ("assets.zip",))
    source_zip = find_file(deployment, objvol_root, deployment_root, ("source.zip",))
    for label, path in (("dataset/manifest.csv", manifest_source), ("audio_upload.json", upload_source),
                        ("assets.zip", assets_zip), ("source.zip", source_zip)):
        if path is None or not path.is_file():
            raise PipelineError(f"Required deployment input not found: {label}")
    manifest = inputs / "manifest.csv"
    copy_with_hash(manifest_source, manifest, expected_sha256=deployment["runtime_manifest_sha256"])
    make_read_only(manifest)
    upload_copy = inputs / "audio_upload.json"
    copy_with_hash(upload_source, upload_copy)
    shutil.copyfile(deployment_root / "deployment.json", inputs / "deployment.json")
    rows = read_manifest(manifest)
    if deployment.get("rows") is not None and int(deployment["rows"]) != len(rows):
        raise PipelineError(f"deployment rows={deployment['rows']} manifest rows={len(rows)}")
    report = validate_manifest(rows, config["dataset_expectations"])
    if "split_counts" in deployment and deployment["split_counts"] != report["splits"]:
        raise PipelineError(f"deployment split_counts {deployment['split_counts']} != manifest {report['splits']}")
    documents = {}
    for name, expected_sha in config.get("provenance_csv_sha256", {}).items():
        path = find_file(deployment, objvol_root, deployment_root, (f"dataset/{name}", name))
        if path is None:
            documents[name] = {"status": "not_deployed"}
            continue
        digest = sha256_file(path)
        if digest != expected_sha:
            raise PipelineError(f"{name} SHA-256 {digest} != provenance {expected_sha}")
        copy_with_hash(path, inputs / "dataset" / name, expected_sha256=digest)
        documents[name] = {"status": "verified", "sha256": digest}
    split_files = {split: inputs / "dataset" / f"{split}.csv" for split in ("train", "val", "test")
                   if (inputs / "dataset" / f"{split}.csv").is_file()}
    cross_check = cross_check_split_files(rows, split_files) if split_files else {"status": "split_csvs_not_deployed"}
    if (inputs / "dataset" / "selected.csv").is_file():
        selected = read_manifest(inputs / "dataset" / "selected.csv")
        if sorted(row["ID"] for row in selected) != sorted(row["ID"] for row in rows):
            raise PipelineError("selected.csv ID set differs from the runtime manifest")
        cross_check["selected_csv_ids_match"] = True
    for name in sorted(deployment["files"]):
        if name.startswith("dataset/") and name != "dataset/manifest.csv" and not (inputs / name).is_file():
            source = resolve_key(objvol_root, deployment["files"][name]["key"])
            copy_with_hash(source, contained(inputs, name), expected_sha256=deployment["files"][name]["sha256"])
    log("[stage_inputs] staging frozen assets and original source modules")
    assets = stage_assets(assets_zip, job / "assets", config["asset_hashes"])
    source = stage_source(source_zip, job / "source", config["source_hashes"])
    upload = read_json(upload_copy)
    audio = stage_audio(rows, upload, objvol_root, job / "data", job / "scratch", log=log)
    expected_bytes = config.get("source_audio_bytes")
    if expected_bytes and expected_bytes not in (audio["audio_bytes_unique_files"], audio["audio_bytes_per_row_sum"]):
        raise PipelineError(f"Selected audio bytes {audio['audio_bytes_unique_files']}/"
                            f"{audio['audio_bytes_per_row_sum']} != contract {expected_bytes}")
    receipt = {"deployment_files": files, "manifest_sha256": deployment["runtime_manifest_sha256"],
               "manifest_report": report, "documents": documents, "split_file_cross_check": cross_check,
               "assets": {"zip": str(assets_zip), "zip_sha256": assets["assets_zip_sha256"], "files": len(assets["files"])},
               "source": source, "audio": audio, "finished_at": now_iso(),
               "paths": {"assets_zip": str(assets_zip), "source_zip": str(source_zip)}}
    atomic_json(inputs / "staging_receipt.json", receipt)
    return receipt
