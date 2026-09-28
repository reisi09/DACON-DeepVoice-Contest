"""Build the single handoff ZIP (or the explicit failure ZIP), verified before publication.

Standard library only.  Excludes raw audio and the feature cache except the
explicit parity samples.  Publication is atomic: write .partial locally, verify
every member's SHA-256 and CRC in one pass, rename, copy to the object volume as
.uploading, re-hash the copy, then rename to the final name.
"""
from __future__ import annotations

import csv
import io
import json
import os
import shutil
import zipfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import (CHUNK, PipelineError, atomic_json, canonical_sha, copy_with_hash, disk_free, human_bytes,
                     human_seconds, iter_files, json_safe, now_iso, publish_file, read_json, safe_name,
                     sha256_bytes, sha256_file)
from .constants import FAILURE_NAME, HANDOFF_NAME, LABELS, SEEDS

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


Reader = Callable[[str], "bytes | None"]


def file_reader(members: Sequence[tuple[str, Path]]) -> Reader:
    """Handoff member name -> bytes of the exact file that will be packaged under that name."""
    mapping = dict(members)
    return lambda name: mapping[name].read_bytes() if name in mapping and mapping[name].is_file() else None


def zip_reader(archive: zipfile.ZipFile) -> Reader:
    """Handoff member name -> bytes actually stored in the archive."""
    names = set(archive.namelist())
    return lambda name: archive.read(name) if name in names else None


def coverage_report(config: Mapping[str, Any], read: Reader) -> dict[str, Any]:
    """Exact coverage and evidence binding a complete handoff must prove, computed only
    from handoff members (``read``), so it can be re-run on the packaged bytes:
      * manifest: 9504 unique rows, 6652/1426/1426, runtime SHA pin;
      * features: index covers exactly the manifest; features_complete binds that index;
      * candidate lock: body hash, exactly six checkpoints byte-identical, and the val
        predictions, val metrics, results, run identity, initial fc5, feature index and
        manifest all equal the hashes recorded in the lock;
      * test: receipt bound to this lock; test predictions/metrics equal the receipt hashes;
      * predictions: val/test IDs exactly the split rows in manifest order;
      * parity: bitwise/within_tolerance and bound to this lock.
    """
    problems: list[str] = []
    report: dict[str, Any] = {}

    def load_json(name: str) -> Any:
        data = read(name)
        if data is None:
            problems.append(f"{name} missing")
            return None
        return json.loads(data)

    def digest(name: str) -> str | None:
        data = read(name)
        return None if data is None else sha256_bytes(data)

    def bind(label: str, actual: str | None, expected: Any) -> None:
        if actual is None or actual != expected:
            problems.append(f"{label}: {actual} != {expected}")

    expectations = config["dataset_expectations"]
    manifest_bytes = read("dataset/manifest.csv")
    if manifest_bytes is None:
        return {"problems": ["dataset/manifest.csv missing"]}
    rows = list(csv.DictReader(io.StringIO(manifest_bytes.decode("utf-8"), newline="")))
    ids = [row["ID"] for row in rows]
    by_split = {split: [row["ID"] for row in rows if row["split"] == split] for split in ("train", "val", "test")}
    manifest_sha = sha256_bytes(manifest_bytes)
    report.update(manifest_rows=len(ids), manifest_sha256=manifest_sha,
                  splits={split: len(values) for split, values in by_split.items()})
    if len(ids) != int(expectations["rows"]) or len(set(ids)) != len(ids):
        problems.append(f"manifest rows {len(ids)} (unique {len(set(ids))}) != {expectations['rows']}")
    pinned = config.get("runtime_manifest_sha256")
    if pinned and manifest_sha != pinned:
        problems.append(f"runtime manifest sha256 {manifest_sha} != reviewed {pinned}")
    for split, count in expectations["splits"].items():
        if len(by_split[split]) != int(count):
            problems.append(f"{split} rows {len(by_split[split])} != {count}")

    index, complete = load_json("model/feature_index.json"), load_json("measurements/features_features_complete.json")
    index_sha = digest("model/feature_index.json")
    if index is not None and complete is not None:
        items = index["items"]
        missing, extra = sorted(set(ids) - set(items)), sorted(set(items) - set(ids))
        report["features"] = {"rows": len(items), "missing": len(missing), "extra": len(extra),
                              "index_sha256": index_sha, "complete_rows": complete.get("rows")}
        if missing or extra or complete.get("rows") != len(ids):
            problems.append(f"feature coverage not exact: missing={missing[:5]} extra={extra[:5]} "
                            f"complete_rows={complete.get('rows')}")
        bind("features_complete.index_sha256", index_sha, complete.get("index_sha256"))

    lock, lock_sha = load_json("model/candidate_lock.json"), digest("model/candidate_lock.json")
    expected_checkpoints = sorted(f"{route}_{seed}.pt" for seed in SEEDS for route in ("music", "voice"))
    if lock is not None:
        body_ok = canonical_sha({k: v for k, v in lock.items() if k != "lock_body_sha256"}) == lock.get("lock_body_sha256")
        packaged = {name: digest(f"model/checkpoints/{name}") for name in expected_checkpoints}
        report["checkpoints"] = {"lock_body_verified": body_ok, "files": packaged, "lock_sha256": lock_sha}
        if not body_ok:
            problems.append("candidate_lock.json body hash mismatch")
        if sorted(lock.get("checkpoints", {})) != expected_checkpoints:
            problems.append(f"lock does not cover exactly the six checkpoints: {sorted(lock.get('checkpoints', {}))}")
        for name in expected_checkpoints:
            bind(f"checkpoint {name}", packaged[name], lock.get("checkpoints", {}).get(name))
        run_identity = load_json("model/run_identity.json")
        bind("run_identity.json", canonical_sha(run_identity) if run_identity is not None else None,
             lock.get("run_identity_sha256"))
        for label, member, key in (("val predictions", "predictions/val_predictions.csv", "validation_predictions_sha256"),
                                   ("val metrics", "metrics/val_metrics.json", "validation_metrics_sha256"),
                                   ("results.json", "model/results.json", "results_sha256"),
                                   ("initial fc5", "model/voice_classifier_initial.pt", "voice_classifier_initial_sha256"),
                                   ("feature index", "model/feature_index.json", "feature_index_sha256"),
                                   ("manifest", "dataset/manifest.csv", "manifest_sha256")):
            bind(f"{label} vs candidate_lock", digest(member), lock.get(key))

    test_receipt = load_json("metrics/test_receipt.json")
    test_predictions_sha, test_metrics_sha = digest("predictions/test_predictions.csv"), digest("metrics/test_metrics.json")
    val_predictions_sha = digest("predictions/val_predictions.csv")
    if test_receipt is not None:
        bind("test_receipt.lock_sha256", lock_sha, test_receipt.get("lock_sha256"))
        bind("test predictions vs test_receipt", test_predictions_sha, test_receipt.get("predictions_sha256"))
        bind("test metrics vs test_receipt", test_metrics_sha, test_receipt.get("metrics_sha256"))
    test_metrics, val_metrics = load_json("metrics/test_metrics.json"), load_json("metrics/val_metrics.json")
    if test_metrics is not None:
        bind("test_metrics.predictions_sha256", test_predictions_sha, test_metrics.get("predictions_sha256"))
        bind("test_metrics.lock_sha256", lock_sha, test_metrics.get("lock_sha256"))
    if val_metrics is not None:
        bind("val_metrics.predictions_sha256", val_predictions_sha, val_metrics.get("predictions_sha256"))

    report["predictions"] = {}
    for split in ("val", "test"):
        data = read(f"predictions/{split}_predictions.csv")
        if data is None:
            problems.append(f"{split}_predictions.csv missing")
            continue
        predicted = [row["ID"] for row in csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""))]
        exact = predicted == by_split[split]
        report["predictions"][split] = {"rows": len(predicted), "exact_manifest_order": exact,
                                        "sha256": sha256_bytes(data)}
        if not exact:
            problems.append(f"{split} predictions do not cover exactly the {len(by_split[split])} {split} rows")

    parity = load_json("parity/parity_receipt.json")
    report["parity_status"] = parity.get("status") if parity is not None else None
    if parity is not None:
        if parity.get("status") not in ("bitwise", "within_tolerance"):
            problems.append(f"raw-audio parity status {parity.get('status')!r}")
        bind("parity_receipt.lock_sha256", lock_sha, parity.get("lock_sha256"))
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
    lines = [f"# {config['run_name']} H100 handoff", "", f"- status: **{status}**", f"- built_at: {now_iso()}",
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
        lines += ["## Pilot (H100)", "", f"- MERT batch: {receipt['mert_batch']}",
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
        coverage = coverage_report(config, file_reader(members))
        if coverage["problems"]:
            raise PipelineError(f"Complete handoff coverage check failed: {coverage['problems']}")
    else:
        try:
            coverage = coverage_report(config, file_reader(members))
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
    if status == "complete":  # the evidence gate again, on the bytes actually inside the archive
        with zipfile.ZipFile(partial) as archive:
            packaged = coverage_report(config, zip_reader(archive))
        if packaged["problems"] or json_safe(packaged) != json_safe(coverage):
            partial.unlink(missing_ok=True)
            raise PipelineError(f"Packaged handoff bytes do not reproduce the verified evidence: {packaged['problems']}")
    final_local = work / name
    os.replace(partial, final_local)
    sidecar = publish_handoff(config, final_local, status, len(digests_for_sums) + 1)
    log(f"[handoff] {sidecar['file']} sha256={sidecar['sha256']} ({human_bytes(sidecar['bytes'])})")
    return sidecar


def publish_handoff(config: Mapping[str, Any], final_local: Path, status: str, members: int) -> dict[str, Any]:
    """Durable object-volume copy + sidecar (the notebook's download link is made from the sidecar)."""
    archive_sha = sha256_file(final_local)
    published = Path(config["deployment_root"]) / final_local.name
    publication = publish_file(final_local, published)
    if publication["sha256"] != archive_sha:
        raise PipelineError("Published handoff differs from the verified local archive")
    sidecar = {"file": str(published), "local_copy": str(final_local), "sha256": archive_sha,
               "bytes": publication["bytes"], "members": members, "status": status,
               "manifest_member": "handoff_manifest.json", "publish_mode": publication["mode"],
               "crc_and_member_sha256_verified": True, "published_at": now_iso()}
    sidecar_local = final_local.with_name(final_local.name + ".json")
    atomic_json(sidecar_local, sidecar)
    publish_file(sidecar_local, published.with_name(published.name + ".json"))
    return sidecar


def expose_download(sidecar_path: Path, directory: Path) -> dict[str, Any]:
    """Make the verified handoff ZIP downloadable from the notebook's own directory.

    Jupyter serves only files under its root, and a notebook link resolves relative to
    the notebook's directory, so a link to /root-1/... from a notebook under /root is
    dead.  The ZIP is placed in ``directory`` (the kernel cwd = notebook directory) as a
    hardlink to a SHA-256-verified copy (no extra disk), or as a verified byte copy
    across filesystems.  Symlinks are not used: Jupyter refuses targets outside its root.
    The browser downloads it through Jupyter's streaming /files/ handler; nothing is
    base64-embedded in the notebook.
    """
    sidecar = read_json(Path(sidecar_path))
    expected, size = sidecar["sha256"], int(sidecar["bytes"])
    directory = Path(directory).resolve()
    name = Path(sidecar["file"]).name
    target = directory / name
    mode = None
    if target.is_file() and target.stat().st_size == size and sha256_file(target) == expected:
        mode = "already_present_verified"
    else:
        for candidate in (sidecar.get("local_copy"), sidecar["file"]):
            source = Path(candidate) if candidate else None
            if source is None or not source.is_file() or source.stat().st_size != size or sha256_file(source) != expected:
                continue
            temporary = directory / f".{name}.{os.getpid()}.part"
            temporary.unlink(missing_ok=True)
            try:
                os.link(source, temporary)
                mode = "hardlink_to_verified_copy"
            except OSError:
                if disk_free(directory) < size + 256 * 1024 * 1024:
                    raise PipelineError(f"Not enough space in {directory} for a download copy "
                                        f"({human_bytes(size)}); the durable copy is {sidecar['file']}") from None
                copy_with_hash(source, temporary, expected_bytes=size, expected_sha256=expected)
                mode = "verified_byte_copy"
            os.replace(temporary, target)
            break
        if mode is None:
            raise PipelineError(f"No copy of the handoff matches its sidecar sha256 {expected}: {sidecar_path}")
    return {"path": str(target), "relative_path": name, "sha256": expected, "bytes": size, "mode": mode,
            "durable_copy": sidecar["file"], "directory": str(directory)}


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
