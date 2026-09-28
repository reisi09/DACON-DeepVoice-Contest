"""Build the single handoff ZIP (or the explicit failure ZIP), verified before publication.

Standard library only.  Excludes raw audio and the feature cache except the
explicit parity samples.  Publication is atomic: write .partial locally, verify
every member's SHA-256 and CRC in one pass, rename, copy to the object volume as
.uploading, re-hash the copy, then rename to the final name.
"""
from __future__ import annotations

import json
import os
import shutil
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .common import (CHUNK, PipelineError, atomic_json, canonical_sha, human_bytes, human_seconds, iter_files,
                     json_safe, now_iso, publish_file, read_json, safe_name, sha256_file)
from .constants import FAILURE_NAME, HANDOFF_NAME, LABELS, SEEDS
from .dataset import read_manifest

STORED_SUFFIXES = (".pt", ".bin", ".th", ".pth", ".zip", ".npz", ".flac", ".mp3", ".wav", ".ogg", ".m4a", ".opus")
PACKAGE_DIR = Path(__file__).resolve().parent


def _maybe(path: Path) -> Path | None:
    return path if path.is_file() else None


def collect(config: Mapping[str, Any], status: str) -> list[tuple[str, Path]]:
    job = Path(config["job_root"])
    members: list[tuple[str, Path]] = []

    def add(name: str, path: Path | None) -> None:
        if path is not None and path.is_file():
            members.append((name, path))

    def add_tree(prefix: str, root: Path, pattern: str = "*") -> None:
        if root.is_dir():
            for path in sorted(root.glob(pattern)):
                if path.is_file() and not path.name.endswith((".partial", ".tmp")):
                    members.append((f"{prefix}/{path.name}", path))

    train, eval_dir, cache = Path(config["train_dir"]), Path(config["eval_dir"]), Path(config["cache_dir"])
    for seed in SEEDS:
        for route in ("music", "voice"):
            add(f"model/checkpoints/{route}_{seed}.pt", _maybe(train / f"{route}_{seed}.pt"))
            add(f"model/training/{route}_{seed}_history.json", _maybe(train / f"{route}_{seed}_history.json"))
            add(f"model/training/{route}_{seed}_batch.json", _maybe(train / f"{route}_{seed}_batch.json"))
    for name in ("candidate_lock.json", "run_identity.json", "results.json", "complete.json",
                 "music_validation.csv", "voice_validation.csv", "progress.json"):
        add(f"model/{name}", _maybe(train / name))
    add("model/voice_classifier_initial.pt", _maybe(cache / "voice_classifier_initial.pt"))
    add("model/feature_index.json", _maybe(cache / "index.json"))
    for name in ("val_predictions.csv", "test_predictions.csv"):
        add(f"predictions/{name}", _maybe(eval_dir / name))
    for name in ("val_metrics.json", "test_metrics.json", "test_receipt.json"):
        add(f"metrics/{name}", _maybe(eval_dir / name))
    inputs = job / "inputs"
    for name in ("manifest.csv", "audio_upload.json", "deployment.json", "staging_receipt.json"):
        add(f"dataset/{name}", _maybe(inputs / name))
    if (inputs / "dataset").is_dir():
        for path in iter_files(inputs / "dataset"):
            members.append((f"dataset/deployed/{path.relative_to(inputs / 'dataset').as_posix()}", path))
    # Reviewed composition/provenance/audit documents the code was written against
    # (carried even if the deployment did not list them under dataset/).
    reference = PACKAGE_DIR.parent / "reference" / "dataset"
    if reference.is_dir():
        for path in iter_files(reference):
            members.append((f"dataset/reference/{path.relative_to(reference).as_posix()}", path))
    add("config/run_config.json", _maybe(job / "config" / "run_config.json"))
    add("code/author_code.zip", _maybe(Path(config["code_zip"])) if config.get("code_zip") else None)
    staging = read_json(inputs / "staging_receipt.json") if (inputs / "staging_receipt.json").is_file() else {}
    source_zip = staging.get("paths", {}).get("source_zip")
    add("code/source.zip", _maybe(Path(source_zip)) if source_zip else None)
    notebook = config.get("notebook_path")
    if notebook:
        add(f"code/{Path(notebook).name}", _maybe(Path(notebook)))
    add("environment/requirements.lock.txt", PACKAGE_DIR / "requirements.lock.txt")
    add("environment/requirements.in", PACKAGE_DIR / "requirements.in")
    env_receipt = job / "env" / "receipt.json"
    add("environment/env_receipt.json", _maybe(env_receipt))
    if env_receipt.is_file():
        target = read_json(env_receipt).get("static_ffmpeg", {}).get("target")
        add("environment/bin/ffmpeg", _maybe(Path(target)) if target else None)
    for name in ("pilot_receipt.json", "mert_batch_benchmark.json"):
        add(f"measurements/{name}", _maybe(Path(config["pilot_dir"]) / name))
    for name in ("features_complete.json", "progress.json", "row_timings.jsonl"):
        add(f"measurements/features_{name}", _maybe(Path(config["features_dir"]) / name))
    add_tree("measurements/status", job / "status", "*.json")
    add_tree("logs", job / "logs", "*.log")
    parity = Path(config["parity_dir"])
    add("parity/parity_receipt.json", _maybe(parity / "parity_receipt.json"))
    if (parity / "parity_receipt.json").is_file() and (cache / "index.json").is_file():
        receipt = read_json(parity / "parity_receipt.json")
        index = read_json(cache / "index.json")
        for sample in receipt["samples"]:
            stem = sample["file_stem"]
            suffix = Path(sample["audio_path"]).suffix
            add(f"parity/audio/{stem}{suffix}", _maybe(parity / "data" / sample["audio_path"]))
            add(f"parity/recomputed/{stem}.npz", _maybe(parity / "recomputed" / f"{stem}.npz"))
            entry = index["items"].get(sample["ID"])
            add(f"parity/full_cache/{stem}.npz", _maybe(cache / entry["path"]) if entry else None)
        pilot_cache = Path(config["pilot_dir"]) / "cache"
        if pilot_cache.is_dir():
            add("parity/pilot_cache/index.json", _maybe(pilot_cache / "index.json"))
            for path in sorted(pilot_cache.glob("*.npz")):
                if not path.name.endswith(".tmp.npz"):
                    add(f"parity/pilot_cache/{safe_name(path.stem)}.npz", path)
    if status == "complete":
        assets_zip = staging.get("paths", {}).get("assets_zip")
        add("assets/assets.zip", _maybe(Path(assets_zip)) if assets_zip else None)
    names = [name for name, _ in members]
    if len(names) != len(set(names)):
        raise PipelineError("Duplicate handoff member names")
    return members


def _prediction_ids(path: Path) -> list[str]:
    import csv
    with path.open(newline="", encoding="utf-8") as stream:
        return [row["ID"] for row in csv.DictReader(stream)]


def coverage_report(config: Mapping[str, Any]) -> dict[str, Any]:
    """Exact coverage a complete handoff must prove: 9504 feature rows, every val/test row predicted
    exactly once (manifest order), the lock's six checkpoints byte-identical, parity not mismatched."""
    problems: list[str] = []
    expectations = config["dataset_expectations"]
    rows = read_manifest(Path(config["manifest"]))
    ids = [row["ID"] for row in rows]
    by_split = {split: [row["ID"] for row in rows if row["split"] == split] for split in ("train", "val", "test")}
    report: dict[str, Any] = {"manifest_rows": len(ids), "manifest_sha256": sha256_file(Path(config["manifest"])),
                              "splits": {split: len(values) for split, values in by_split.items()}}
    if len(ids) != int(expectations["rows"]) or len(set(ids)) != len(ids):
        problems.append(f"manifest rows {len(ids)} (unique {len(set(ids))}) != {expectations['rows']}")
    pinned = config.get("runtime_manifest_sha256")
    if pinned and report["manifest_sha256"] != pinned:
        problems.append(f"runtime manifest sha256 {report['manifest_sha256']} != reviewed {pinned}")
    for split, count in expectations["splits"].items():
        if len(by_split[split]) != int(count):
            problems.append(f"{split} rows {len(by_split[split])} != {count}")
    cache_index = Path(config["cache_dir"]) / "index.json"
    complete_path = Path(config["features_dir"]) / "features_complete.json"
    if not cache_index.is_file() or not complete_path.is_file():
        problems.append("feature index or features_complete.json missing")
    else:
        items = read_json(cache_index)["items"]
        complete = read_json(complete_path)
        missing, extra = sorted(set(ids) - set(items)), sorted(set(items) - set(ids))
        report["features"] = {"rows": len(items), "missing": len(missing), "extra": len(extra),
                              "index_sha256": sha256_file(cache_index),
                              "complete_rows": complete.get("rows"), "complete_index_sha256": complete.get("index_sha256")}
        if missing or extra or complete.get("rows") != len(ids) or complete.get("index_sha256") != sha256_file(cache_index):
            problems.append(f"feature coverage not exact: missing={missing[:5]} extra={extra[:5]} "
                            f"complete_rows={complete.get('rows')}")
    report["predictions"] = {}
    for split in ("val", "test"):
        path = Path(config["eval_dir"]) / f"{split}_predictions.csv"
        if not path.is_file():
            problems.append(f"{split}_predictions.csv missing")
            continue
        predicted = _prediction_ids(path)
        exact = predicted == by_split[split]
        report["predictions"][split] = {"rows": len(predicted), "exact_manifest_order": exact,
                                        "sha256": sha256_file(path)}
        if not exact:
            problems.append(f"{split} predictions do not cover exactly the {len(by_split[split])} {split} rows")
    lock_path = Path(config["train_dir"]) / "candidate_lock.json"
    expected_checkpoints = sorted(f"{route}_{seed}.pt" for seed in SEEDS for route in ("music", "voice"))
    if not lock_path.is_file():
        problems.append("candidate_lock.json missing")
    else:
        lock = read_json(lock_path)
        body_ok = canonical_sha({k: v for k, v in lock.items() if k != "lock_body_sha256"}) == lock.get("lock_body_sha256")
        actual = {name: sha256_file(Path(config["train_dir"]) / name) if (Path(config["train_dir"]) / name).is_file()
                  else None for name in expected_checkpoints}
        report["checkpoints"] = {"lock_body_verified": body_ok, "files": actual,
                                 "lock_sha256": sha256_file(lock_path)}
        if not body_ok or sorted(lock.get("checkpoints", {})) != expected_checkpoints or any(
                actual[name] != lock["checkpoints"][name] for name in expected_checkpoints):
            problems.append("checkpoints differ from the six locked checkpoints")
        receipt_path = Path(config["eval_dir"]) / "test_receipt.json"
        if not receipt_path.is_file() or read_json(receipt_path).get("lock_sha256") != sha256_file(lock_path):
            problems.append("test_receipt.json missing or not bound to this lock")
    parity_path = Path(config["parity_dir"]) / "parity_receipt.json"
    parity_status = read_json(parity_path).get("status") if parity_path.is_file() else None
    report["parity_status"] = parity_status
    if parity_status not in ("bitwise", "within_tolerance"):
        problems.append(f"raw-audio parity status {parity_status!r}")
    report["problems"] = problems
    return report


def _metric_line(metrics: Mapping[str, Any], head: str) -> str:
    entry = metrics.get("heads", {}).get(head, {})
    if entry.get("eer") is None:
        return f"| {head} | n/a | n/a | {entry.get('n', 0)} |"
    ci = entry.get("bootstrap_eer", {}).get("ci95")
    ci_text = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "n/a"
    return f"| {head} | {entry['eer']:.4f} {ci_text} | {entry['auc']:.4f} | {entry['n']} |"


def readme(config: Mapping[str, Any], status: str, state: Mapping[str, Any], error: str | None) -> str:
    lines = [f"# {config['run_name']} A100 handoff", "", f"- status: **{status}**", f"- built_at: {now_iso()}",
             f"- workspace: {config['workspace']}; deployment: {config['deployment_root']}", ""]
    if error:
        lines += ["## Failure", "", "```", error[-6000:], "```", ""]
    stages = state.get("stages", {})
    if stages:
        lines += ["## Stages (measured wall time)", "", "| stage | status | seconds (all attempts) |", "|---|---|---|"]
        lines += [f"| {name} | {value.get('status')} | {value.get('cumulative_seconds', value.get('seconds', 0)):.0f} |"
                  for name, value in stages.items()]
        total = sum(a.get("seconds", 0) for value in stages.values() for a in value.get("attempts", []))
        lines += ["", f"Total measured stage time: {human_seconds(total)} (target {config['target_hours']} h; "
                      "reported, not enforced).", ""]
    pilot = Path(config["pilot_dir"]) / "pilot_receipt.json"
    if pilot.is_file():
        receipt = read_json(pilot)
        projection = receipt["projection"]
        lines += ["## Pilot (A100)", "", f"- MERT batch: {receipt['mert_batch']}",
                  f"- projected feature extraction: {projection['projected_feature_hours']:.2f} h over "
                  f"{projection['rows']} rows ({projection['audio_hours']:.1f} h audio)",
                  f"- pilot peak allocated VRAM: {human_bytes(receipt['memory'].get('max_allocated', 0))} of "
                  f"{human_bytes(receipt['memory'].get('total', 0))}", ""]
    for split in ("val", "test"):
        path = Path(config["eval_dir"]) / f"{split}_metrics.json"
        if path.is_file():
            metrics = read_json(path)
            lines += [f"## {split} (masked known-label metrics; bootstrap CI over components)", "",
                      "| head | EER [95% CI] | AUC | n |", "|---|---|---|---|",
                      *[_metric_line(metrics, head) for head in ("file", "voice", "music")], ""]
    parity = Path(config["parity_dir"]) / "parity_receipt.json"
    if parity.is_file():
        lines += [f"## Raw-audio parity: {read_json(parity)['status']}", ""]
    lines += ["## Caveats (carried from the dataset README)", "",
              "- Not an independent test: historical development exposure val 997/1426, test 1001/1426.",
              "- No official/L4 score claim; unknown labels are masked, never imputed.",
              "- Singing (CtrSVDD, 2034 rows) is train-only; singing generalisation is not evaluated.",
              "- Music source/length bias remains (metadata source-only AUC ~0.97 on music val/test).",
              "- Checkpoints were selected on val only; test is scored only after candidate_lock.json (re-scoring is deterministic).",
              "- No submit.zip is included; offline packaging parity must be proven later with parity/ samples.",
              "", "## Layout", "",
              "model/ checkpoints+lock+training evidence; predictions/ and metrics/ val/test; dataset/ manifest, "
              "deployment and staging receipts; code/ author_code.zip, source.zip, notebook; environment/ lock, "
              "env receipt, static ffmpeg; assets/assets.zip (frozen encoders); measurements/ pilot, VRAM, "
              "timings; logs/; parity/ audio+features+recomputed samples; SHA256SUMS.txt covers every member.", ""]
    return "\n".join(lines)


def build(config: Mapping[str, Any], status: str, state: Mapping[str, Any], error: str | None = None,
          log=print) -> dict[str, Any]:
    job = Path(config["job_root"])
    name = HANDOFF_NAME if status == "complete" else FAILURE_NAME
    work = job / "handoff"
    work.mkdir(parents=True, exist_ok=True)
    members = collect(config, status)
    notebook = Path(config["notebook_path"]).name if config.get("notebook_path") else "<notebook path unknown>"
    required = {f"model/checkpoints/{r}_{s}.pt" for r in ("music", "voice") for s in SEEDS} | {
        "model/candidate_lock.json", "model/run_identity.json", "model/feature_index.json",
        "predictions/val_predictions.csv", "predictions/test_predictions.csv",
        "metrics/val_metrics.json", "metrics/test_metrics.json", "metrics/test_receipt.json", "dataset/manifest.csv",
        "code/author_code.zip", f"code/{notebook}", "code/source.zip", "assets/assets.zip",
        "environment/env_receipt.json", "environment/bin/ffmpeg", "environment/requirements.lock.txt",
        "measurements/pilot_receipt.json", "measurements/features_features_complete.json",
        "parity/parity_receipt.json", "config/run_config.json", "dataset/staging_receipt.json",
        "dataset/reference/audit.json", "dataset/reference/provenance.json", "dataset/reference/README.md"}
    missing = sorted(required - {member for member, _ in members})
    if status == "complete":
        if missing:
            raise PipelineError(f"Complete handoff lacks required members: {missing}")
        coverage = coverage_report(config)
        if coverage["problems"]:
            raise PipelineError(f"Complete handoff coverage check failed: {coverage['problems']}")
    else:
        try:
            coverage = coverage_report(config)
        except Exception as problem:  # a failure ZIP must still be written (do not shadow `error`)
            coverage = {"problems": [f"coverage not computable: {type(problem).__name__}: {problem}"[:500]]}
    readme_bytes = readme(config, status, state, error).encode()
    state_bytes = (json.dumps({**state, "handoff_status": status, "error": error}, indent=2, default=str) + "\n").encode()
    digests: dict[str, dict[str, Any]] = {}
    partial = work / (name + ".partial")
    partial.unlink(missing_ok=True)
    free = shutil.disk_usage(work).free
    needed = sum(path.stat().st_size for _, path in members) + 1_000_000_000
    if free < needed:
        raise PipelineError(f"Not enough disk for the handoff ZIP: need {human_bytes(needed)}, free {human_bytes(free)}")
    with zipfile.ZipFile(partial, "w", allowZip64=True) as archive:
        for member, path in sorted(members):
            compression = zipfile.ZIP_STORED if path.suffix.lower() in STORED_SUFFIXES else zipfile.ZIP_DEFLATED
            before = sha256_file(path)
            archive.write(path, member, compress_type=compression)
            digests[member] = {"sha256": before, "bytes": path.stat().st_size, "source": str(path)}
        for member, data in (("HANDOFF_README.md", readme_bytes), ("status/run_state.json", state_bytes)):
            archive.writestr(zipfile.ZipInfo(member, date_time=(2026, 9, 19, 0, 0, 0)), data,
                             compress_type=zipfile.ZIP_DEFLATED)
            digests[member] = {"sha256": __import__("hashlib").sha256(data).hexdigest(), "bytes": len(data),
                               "source": "generated"}
        manifest = {"name": name, "status": status, "built_at": now_iso(), "members": digests,
                    "coverage": json_safe(coverage), "missing_required_members": missing,
                    "excluded": ["raw audio except parity/audio", "feature cache except parity/ npz samples",
                                 "isolated venv", "uv cache"],
                    "labels": list(LABELS)}
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        archive.writestr(zipfile.ZipInfo("handoff_manifest.json", date_time=(2026, 9, 19, 0, 0, 0)),
                         manifest_bytes, compress_type=zipfile.ZIP_DEFLATED)
        digests_for_sums = {**{k: v["sha256"] for k, v in digests.items()},
                            "handoff_manifest.json": __import__("hashlib").sha256(manifest_bytes).hexdigest()}
        sums = "".join(f"{digest}  {member}\n" for member, digest in sorted(digests_for_sums.items())).encode()
        archive.writestr(zipfile.ZipInfo("SHA256SUMS.txt", date_time=(2026, 9, 19, 0, 0, 0)), sums,
                         compress_type=zipfile.ZIP_DEFLATED)
    verify_zip(partial, {**digests_for_sums, "SHA256SUMS.txt": __import__("hashlib").sha256(sums).hexdigest()})
    final_local = work / name
    os.replace(partial, final_local)
    archive_sha = sha256_file(final_local)
    published = Path(config["deployment_root"]) / name
    publication = publish_file(final_local, published)
    if publication["sha256"] != archive_sha:
        raise PipelineError("Published handoff differs from the verified local archive")
    sidecar = {"file": str(published), "local_copy": str(final_local), "sha256": archive_sha,
               "bytes": publication["bytes"], "members": len(digests_for_sums) + 1, "status": status,
               "manifest_member": "handoff_manifest.json", "publish_mode": publication["mode"],
               "crc_and_member_sha256_verified": True, "published_at": now_iso()}
    sidecar_local = work / (name + ".json")
    atomic_json(sidecar_local, sidecar)
    publish_file(sidecar_local, published.with_name(published.name + ".json"))
    log(f"[handoff] {published} sha256={archive_sha} ({human_bytes(sidecar['bytes'])})")
    return sidecar


def verify_zip(path: Path, expected: Mapping[str, str]) -> None:
    """One pass: zipfile checks every CRC at EOF; we check every SHA-256 and the exact member set."""
    import hashlib
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise PipelineError("Handoff ZIP member set differs from its manifest")
        for info in archive.infolist():
            digest = hashlib.sha256()
            with archive.open(info) as stream:
                while chunk := stream.read(CHUNK):
                    digest.update(chunk)
            if digest.hexdigest() != expected[info.filename]:
                raise PipelineError(f"Handoff member hash mismatch: {info.filename}")
