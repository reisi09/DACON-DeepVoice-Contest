"""CPU integration test of the real pipeline code on the synthetic fixture.

Run inside the pinned venv (torch 2.9.1, transformers 4.57.6, ...):
    python tests/test_cpu_integration.py --fixture <dir built by build_fixture.py> --work <scratch dir>

What it protects against (each check names the defect it would catch):
  * staging: shard/member hash mismatch accepted; unselected members extracted;
    traversal/symlink archive members accepted.
  * manifest contract: cross-split provenance leak, label/mask disagreement,
    route target unknown -> must be rejected.
  * features: interrupted extraction corrupts or loses rows -> resume must reach exact
    coverage with every cache validating; pilot and full features must reproduce.
  * training isolation: training must succeed with every test feature file removed
    (proves no test access before the lock); the lock must record 0 test reads.
  * gradient connectivity: trained heads must differ from their initialisation.
  * lock integrity: a checkpoint changed after the lock must be refused at test time.
  * raw-audio parity: recomputed features/predictions must match the cache path.
  * handoff: exact member set, SHA256SUMS agreement, no raw audio/cache beyond the
    parity samples, six checkpoints; failure path must emit a FAILED ZIP and re-raise.
  * CLI safety: stage CLIs must refuse a non-A100 (CPU) configuration.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def expect_error(function, needle: str, *args, **kwargs) -> str:
    try:
        function(*args, **kwargs)
    except Exception as error:  # noqa: BLE001 - the message is asserted
        text = f"{type(error).__name__}: {error}"
        assert needle in text, f"expected {needle!r} in {text[:500]}"
        return text
    raise AssertionError(f"expected an error containing {needle!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    fixture = json.loads((args.fixture / "fixture.json").read_text())
    if args.work.exists():
        shutil.rmtree(args.work)
    job = args.work / "job"
    home = job / "home"
    (home / "panns_data").mkdir(parents=True)
    os.environ.update(HOME=str(home), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", MPLBACKEND="Agg",
                      HF_HOME=str(home / "hf"), NUMBA_CACHE_DIR=str(home / "numba"), PYTHONHASHSEED="0")

    from dacon9504a100 import constants
    from dacon9504a100.common import PipelineError, checked_members, sha256_file
    from dacon9504a100.dataset import DatasetError, read_manifest, validate_manifest
    from dacon9504a100.orchestrator import Pipeline, build_config
    from dacon9504a100.staging import stage_audio, stage_inputs

    # author_code.zip stand-in (the handoff must carry the exact bytes it ran from).
    code_zip = args.work / "author_code.zip"
    with zipfile.ZipFile(code_zip, "w") as archive:
        for path in sorted((ROOT / "dacon9504a100").glob("*")):
            if path.is_file():
                archive.write(path, f"dacon9504a100/{path.name}")
    config = build_config(Path(fixture["deploy"]), job, ROOT, code_zip, None)
    config.update(workspace="fixture", objvol_root=fixture["objvol"], expected_gpu=None, device="cpu",
                  dataset_expectations=fixture["expectations"], asset_hashes=fixture["asset_hashes"],
                  provenance_csv_sha256={}, source_audio_bytes=None, mert_batch_override=4,
                  train_benchmark_override={"microbatch": 256, "window_limit": 10 ** 6}, max_epochs=3, patience=2,
                  parity_max_seconds=60)
    source_hashes = {p.name: sha256_file(p) for p in sorted(args.source_dir.glob("*.py"))}
    assert source_hashes == constants.SOURCE_HASHES, "fixture source must be the reviewed originals"
    config_path = job / "config" / "run_config.json"

    # ---------------------------------------------------------------- staging
    say("staging inputs")
    receipt = stage_inputs(config, log=say)
    assert receipt["audio"]["files"] == fixture["rows"]
    data_files = sorted(p.name for p in (job / "data" / "audio").iterdir())
    assert "unused_member.wav" not in data_files and len(data_files) == fixture["rows"], "only selected members"
    say("staging again (resume path re-verifies every file)")
    again = stage_inputs(config, log=say)
    assert all(s["status"] == "not_needed_all_files_already_verified" for s in again["audio"]["shards"])

    say("negative: tampered shard is rejected by its SHA-256")
    upload = json.loads((job / "inputs" / "audio_upload.json").read_text())
    tampered_root = args.work / "tampered"
    shutil.copytree(fixture["objvol"] + "/shards", tampered_root / "shards")
    shard = tampered_root / "shards" / "audio_000.zip"
    raw = bytearray(shard.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    shard.write_bytes(bytes(raw))
    rows = read_manifest(job / "inputs" / "manifest.csv")
    expect_error(stage_audio, "Copied bytes differ", rows, upload, tampered_root, args.work / "t_data", args.work / "t_scratch")

    say("negative: traversal / symlink members are rejected")
    evil = args.work / "evil.zip"
    with zipfile.ZipFile(evil, "w") as archive:
        archive.writestr("../escape.txt", b"x")
    with zipfile.ZipFile(evil) as archive:
        expect_error(checked_members, "Unsafe relative path", archive)
    link = args.work / "link.zip"
    with zipfile.ZipFile(link, "w") as archive:
        info = zipfile.ZipInfo("audio/link")
        info.external_attr = (0o120777 << 16)
        archive.writestr(info, b"/etc/passwd")
    with zipfile.ZipFile(link) as archive:
        expect_error(checked_members, "symbolic link", archive)

    say("negative: manifest contract rejects leaks and mask disagreements")
    leaked = [dict(r) for r in rows]
    train_row = next(r for r in leaked if r["split"] == "train")
    val_row = next(r for r in leaked if r["split"] == "val")
    val_row["group_id"] = train_row["group_id"]
    expect_error(validate_manifest, "span splits", leaked, fixture["expectations"])
    masked = [dict(r) for r in rows]
    music_row = next(r for r in masked if r["route"] == "music")
    music_row["VOICE_FAKE"] = "1"
    expect_error(validate_manifest, "Mask disagrees", masked, fixture["expectations"])
    unknown_target = [dict(r) for r in rows]
    voice_row = next(r for r in unknown_target if r["route"] == "voice")
    voice_row["VOICE_FAKE"], voice_row["VOICE_FAKE_mask"] = "", "0"
    expect_error(validate_manifest, "Route target VOICE_FAKE unknown", unknown_target, fixture["expectations"])
    parent_leak = [dict(r) for r in rows]
    test_row = next(r for r in parent_leak if r["split"] == "test")
    test_row["parent_ids"] = json.dumps([next(r for r in parent_leak if r["split"] == "train")["ID"]])
    expect_error(validate_manifest, "span splits", parent_leak, fixture["expectations"])

    # Environment stand-in: the test already runs inside the pinned venv.
    import imageio_ffmpeg
    from dacon9504a100.common import atomic_json
    from dacon9504a100.environment import LOCK_FILE, child_environment, link_static_ffmpeg
    atomic_json(job / "env" / "receipt.json", {"lock_sha256": sha256_file(LOCK_FILE), "ffmpeg_ld_library_path": None,
                                               "static_ffmpeg": link_static_ffmpeg(job, imageio_ffmpeg.get_ffmpeg_exe())})
    child_env = child_environment(config)
    assert (home / "panns_data" / "class_labels_indices.csv").is_file(), "PANNs labels must be pre-placed"

    # CLI safety: a CPU configuration must be refused by every stage entry point.
    pipeline = Pipeline(config=config, report_every=1.0)
    os.symlink(sys.prefix, job / "env" / "venv")  # the pinned test venv stands in for JOB/env/venv
    pipeline._gpu_idle = lambda: None  # test-only: no nvidia-smi on this CPU host
    for stage, module, extra in (("pilot", "features", ("pilot",)), ("train", "training", ()),
                                 ("test_eval", "evaluation", ("test",)), ("parity", "parity", ())):
        pipeline.state["stages"][stage] = {"status": "running", "attempts": []}
        expect_error(pipeline._child, "authorized A100", stage, module, *extra)

    # ---------------------------------------------------------------- features
    sys.path.insert(1, str(job / "source"))  # the staged, hash-verified original modules
    from dacon9504a100 import features
    say("pilot (train rows only, fixed batch override on CPU)")
    pilot = features.run_pilot(config)
    assert all(r["split"] == "train" for r in pilot["rows"]), "pilot must use train rows only"
    assert pilot["projection"]["rows"] == fixture["rows"]

    say("full extraction, interrupted after >=3 rows, then resumed")
    runner = [sys.executable, "-c",
              "import json,sys; from dacon9504a100.features import run_full; run_full(json.load(open(sys.argv[1])))",
              str(config_path)]
    env = dict(os.environ, PYTHONPATH=f"{ROOT}{os.pathsep}{job / 'source'}")
    process = subprocess.Popen(runner, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    index_path = job / "cache" / "index.json"
    while process.poll() is None:
        time.sleep(0.5)
        if index_path.is_file() and len(json.loads(index_path.read_text())["items"]) >= 3:
            process.kill()
            break
    process.wait()
    partial = len(json.loads(index_path.read_text())["items"])
    assert 3 <= partial < fixture["rows"], f"interruption point not reached: {partial}"
    result = subprocess.run(runner, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    complete = json.loads((job / "features" / "features_complete.json").read_text())
    assert complete["rows"] == fixture["rows"]
    index = json.loads(index_path.read_text())
    assert set(index["items"]) == {r["ID"] for r in rows}, "exact cache coverage after resume"

    # ---------------------------------------------------------------- training isolation
    from dacon9504a100 import evaluation, training
    say("training with every test feature file removed (must not need them)")
    hidden = args.work / "hidden_test_features"
    hidden.mkdir()
    test_ids = [r["ID"] for r in rows if r["split"] == "test"]
    for identifier in test_ids:
        shutil.move(job / "cache" / index["items"][identifier]["path"], hidden / index["items"][identifier]["path"])
    lock = training.train(config)
    for path in hidden.iterdir():
        shutil.move(path, job / "cache" / path.name)
    assert lock["test_rows_read_before_lock"] == 0 and lock["feature_files_opened_before_lock"]["test"] == 0
    assert sorted(lock["checkpoints"]) == sorted(evaluation.checkpoint_names())
    say("gradient connectivity: trained heads differ from their initialisation")
    import torch
    initial = torch.load(job / "cache" / "voice_classifier_initial.pt", weights_only=True)["state_dict"]
    for seed in constants.SEEDS:
        voice = torch.load(job / "train" / f"voice_{seed}.pt", weights_only=True)["model"]
        assert any(not torch.equal(voice[k], initial[k]) for k in initial), "voice head never updated"
    say("rerun of training after the lock verifies and does not retrain")
    before = {n: sha256_file(job / "train" / n) for n in evaluation.checkpoint_names()}
    training.train(config)
    assert before == {n: sha256_file(job / "train" / n) for n in evaluation.checkpoint_names()}

    say("locked test evaluation")
    test_receipt = evaluation.run_test(config)
    assert test_receipt["test_rows"] == fixture["expectations"]["splits"]["test"]
    metrics = json.loads((job / "eval" / "test_metrics.json").read_text())
    assert metrics["official_style_known_subset"]["score"] is None, "unknown labels must keep the official total undefined"
    assert metrics["presence"]["voice_presence"]["unknown"] > 0
    with (job / "eval" / "test_predictions.csv").open() as stream:
        predicted = list(csv.DictReader(stream))
    assert [p["ID"] for p in predicted] == test_ids
    for p in predicted:
        file_score = max(float(p["VOICE_FAKE_PROB"]) * float(p["VOICE_PRESENT_PROB"]),
                         float(p["MUSIC_FAKE_PROB"]) * float(p["MUSIC_PRESENT_PROB"]))
        assert abs(file_score - float(p["FILE_FAKE_PROB"])) < 1e-12, "FILE must be the unchanged max graph"

    say("negative: a checkpoint changed after the lock is refused")
    target = job / "train" / "music_42.pt"
    backup = target.read_bytes()
    target.write_bytes(backup + b"\0")
    expect_error(evaluation.run_test, "Locked checkpoint changed", config)
    target.write_bytes(backup)

    # ---------------------------------------------------------------- parity
    from dacon9504a100 import parity
    say("raw-audio parity")
    parity_receipt = parity.run_parity(config)
    assert parity_receipt["status"] in ("bitwise", "within_tolerance"), json.dumps(parity_receipt)[:3000]
    assert {s["split"] for s in parity_receipt["samples"]} >= {"val", "test", "train"}

    # ---------------------------------------------------------------- handoff
    from dacon9504a100.handoff import build
    say("handoff ZIP")
    state = {"stages": {"features": {"status": "complete", "attempts": [{"seconds": 1.0, "status": "complete"}]}}}
    sidecar = build(config, "complete", state, None, log=say)
    with zipfile.ZipFile(sidecar["file"]) as archive:
        names = set(archive.namelist())
        sums = dict(reversed(line.split("  ", 1)) for line in archive.read("SHA256SUMS.txt").decode().splitlines())
        for name in names - {"SHA256SUMS.txt"}:
            assert hashlib.sha256(archive.read(name)).hexdigest() == sums[name], name
        assert archive.testzip() is None
    assert {f"model/checkpoints/{r}_{s}.pt" for r in ("music", "voice") for s in constants.SEEDS} <= names
    assert {"config/run_config.json", "dataset/manifest.csv", "dataset/reference/audit.json",
            "dataset/reference/provenance.json", "model/candidate_lock.json"} <= names, "config/provenance/audit/lock"
    audio_members = [n for n in names if n.endswith((".wav", ".flac", ".mp3"))]
    assert audio_members and all(n.startswith("parity/audio/") for n in audio_members), "raw audio only as parity samples"
    npz_members = [n for n in names if n.endswith(".npz")]
    assert npz_members and all(n.startswith("parity/") for n in npz_members), "cache only as parity samples"
    assert len([n for n in npz_members if n.startswith("parity/full_cache/")]) == len(parity_receipt["samples"])
    assert sha256_file(Path(sidecar["file"])) == sidecar["sha256"]

    say("object volume without rename support: publication falls back to a verified direct write")
    from dacon9504a100 import common
    no_rename = args.work / "no_rename_volume"
    real_replace = common.os.replace

    def refusing_replace(source, destination):
        if Path(destination).parent == no_rename:
            raise OSError(95, "Operation not supported")
        return real_replace(source, destination)

    common.os.replace = refusing_replace
    try:
        record = common.publish_file(Path(sidecar["file"]), no_rename / "handoff.zip")
    finally:
        common.os.replace = real_replace
    assert record["mode"] == "direct_write_then_verified" and record["sha256"] == sidecar["sha256"]
    assert sorted(p.name for p in no_rename.iterdir()) == ["handoff.zip"], "no temporary leftovers"

    say("failure path: exception -> FAILED ZIP with status failed, then re-raised")
    failing_config = build_config(Path(fixture["deploy"]), args.work / "failjob", ROOT, code_zip, None)
    failing_config.update({k: config[k] for k in ("workspace", "objvol_root", "expected_gpu", "device",
                                                  "dataset_expectations", "asset_hashes", "provenance_csv_sha256",
                                                  "source_audio_bytes")})
    failing = Pipeline(config=failing_config, report_every=1.0)
    expect_error(failing.run, "", "preflight")
    failed_zip = Path(fixture["deploy"]) / constants.FAILURE_NAME
    with zipfile.ZipFile(failed_zip) as archive:
        state = json.loads(archive.read("status/run_state.json"))
        assert state["handoff_status"] == "failed" and state["error"]
    say("ALL CPU INTEGRATION CHECKS PASSED")


if __name__ == "__main__":
    main()
