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
  * v2 (Codex v1 review):
    - training resume: an epoch-boundary interrupted+resumed WindowMIL run must
      reproduce the uninterrupted run bitwise (weights, predictions, history) and
      continue every RNG stream (Python/NumPy/Torch) exactly; benchmarking must
      not shift any RNG stream;
    - parity: NaN-vs-number and inf must be mismatches (matching NaNs allowed);
      a parity mismatch must raise after saving evidence and block a complete handoff;
    - reuse validation: a stale pilot receipt and a lock whose config/env/evidence
      changed must be refused before any reuse/test access; a missing staged asset
      must be repaired from assets.zip; a wrong runtime manifest SHA pin is refused;
    - handoff: notebook + ffmpeg required; incomplete prediction coverage refused.
  * v3 (Codex v2 review): the complete-handoff gate binds bytes to evidence — val
    predictions/metrics to the lock, test predictions/metrics to test_receipt, test and
    parity receipts to the current lock — on the files to be packaged and again on the
    bytes inside the archive (same-ID value edits, zeroed lock hashes, and a file
    changed between the gate and packaging must all be refused).
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


def check_training_resume(config, rows, job: Path, work: Path) -> None:
    """Epoch-boundary interruption + resume must equal the uninterrupted run exactly.

    A test-only wrapper around the original ``step`` draws from the Python, NumPy
    and Torch generators each call (like any stochastic layer would).  Without
    RNG save/restore the resumed epochs would replay the post-seed streams; with
    it, the draw sequences, weights, predictions and histories must be identical.
    The same draws also show that benchmarking leaves every stream untouched.
    """
    import random

    import numpy as np
    import torch
    from full_bank_batches import loss_weights

    from dacon9504a100 import training
    from dacon9504a100.common import read_json

    index = read_json(job / "cache" / "index.json")
    items = training.read_items(rows, job / "cache", index, ("train", "val"), training.SplitAccess())
    train_items = [x for x in items if x.row["route"] == "music" and x.row["split"] == "train"]
    val_items = [x for x in items if x.row["route"] == "music" and x.row["split"] == "val"]
    weights = loss_weights(train_items)
    means = np.stack([x.features.mean(0) for x in train_items])
    mean, scale = torch.tensor(means.mean(0)), torch.tensor(np.maximum(means.std(0), 1e-6))
    voice = torch.load(job / "cache" / "voice_classifier_initial.pt", map_location="cpu", weights_only=True)
    run_config = {**config, "max_epochs": 5, "patience": 50}
    identity = {"purpose": "resume-equivalence check"}
    real_step, real_benchmark = training.step, training.benchmark
    draws: list[tuple[float, float, float]] = []
    calls = {"n": 0, "stop_at": None}

    class Interrupted(Exception):
        pass

    def stochastic_step(*arguments, **keywords):
        calls["n"] += 1
        if calls["stop_at"] is not None and calls["n"] == calls["stop_at"]:
            raise Interrupted()
        draws.append((random.random(), float(np.random.rand()), float(torch.rand(1))))
        return real_step(*arguments, **keywords)

    def run(output: Path, override=True, stop_at=None):
        output.mkdir(parents=True, exist_ok=True)
        calls.update(n=0, stop_at=stop_at)
        settings = run_config if override else {**run_config, "train_benchmark_override": None}
        return training.train_route_seed("music", 41, train_items, val_items, weights, mean, scale, voice, identity,
                                         output, settings, "cpu")

    steps_per_epoch = -(-len(train_items) // 256)
    training.step = stochastic_step
    try:
        draws.clear()
        uninterrupted, _ = run(work / "resume_a")
        reference_draws = list(draws)
        draws.clear()
        try:
            run(work / "resume_b", stop_at=2 * steps_per_epoch + 1)  # dies at the first step of epoch 3
            raise AssertionError("interruption did not happen")
        except Interrupted:
            pass
        assert torch.load(work / "resume_b" / "music_41_resume.pt", weights_only=True)["epoch"] == 2
        resumed, _ = run(work / "resume_b")
        assert draws == reference_draws, "RNG streams after resume differ from the uninterrupted run"

        def fake_benchmark(*_arguments, **_keywords):  # consumes every generator like a real measurement could
            random.random(), np.random.rand(3), torch.rand(3)
            return 256, 10 ** 6, [{"batch": 256, "status": "ok"}]

        training.benchmark = fake_benchmark
        draws.clear()
        run(work / "resume_c", override=False)
        assert draws == reference_draws, "benchmarking shifted an RNG stream"
    finally:
        training.step, training.benchmark = real_step, real_benchmark
    assert np.array_equal(uninterrupted, resumed), "resumed validation predictions differ"
    a = torch.load(work / "resume_a" / "music_41.pt", weights_only=True)
    b = torch.load(work / "resume_b" / "music_41.pt", weights_only=True)
    assert a["epoch"] == b["epoch"] and a["model"].keys() == b["model"].keys()
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"]), "resumed weights differ"
    last_a = torch.load(work / "resume_a" / "music_41_resume.pt", weights_only=True)
    last_b = torch.load(work / "resume_b" / "music_41_resume.pt", weights_only=True)
    assert all(torch.equal(last_a["model"][k], last_b["model"][k]) for k in last_a["model"]), "final weights differ"
    strip = lambda history: [{k: v for k, v in r.items() if k != "epoch_seconds"} for r in history]  # noqa: E731
    assert strip(last_a["history"]) == strip(last_b["history"]) and len(last_a["history"]) == 5


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
    from dacon9504a100.common import checked_members, sha256_file
    from dacon9504a100.dataset import read_manifest, validate_manifest
    from dacon9504a100.orchestrator import Pipeline, build_config
    from dacon9504a100.staging import stage_audio, stage_inputs

    # author_code.zip stand-in (the handoff must carry the exact bytes it ran from).
    code_zip = args.work / "author_code.zip"
    with zipfile.ZipFile(code_zip, "w") as archive:
        for path in sorted((ROOT / "dacon9504a100").glob("*")):
            if path.is_file():
                archive.write(path, f"dacon9504a100/{path.name}")
    notebook_stand_in = args.work / "DACON_Existing9504_A100.ipynb"
    notebook_stand_in.write_text(json.dumps({"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}))
    config = build_config(Path(fixture["deploy"]), job, ROOT, code_zip, notebook_stand_in)
    deployed_manifest_sha = json.loads((Path(fixture["deploy"]) / "deployment.json").read_text())[
        "runtime_manifest_sha256"]
    config.update(workspace="fixture", objvol_root=fixture["objvol"], expected_gpu=None, device="cpu",
                  runtime_manifest_sha256=deployed_manifest_sha,
                  dataset_expectations=fixture["expectations"], asset_hashes=fixture["asset_hashes"],
                  provenance_csv_sha256={}, source_audio_bytes=None, mert_batch_override=4,
                  train_benchmark_override={"microbatch": 256, "window_limit": 10 ** 6}, max_epochs=3, patience=2,
                  parity_max_seconds=60)
    source_hashes = {p.name: sha256_file(p) for p in sorted(args.source_dir.glob("*.py"))}
    assert source_hashes == constants.SOURCE_HASHES, "fixture source must be the reviewed originals"
    config_path = job / "config" / "run_config.json"

    # ---------------------------------------------------------------- staging
    say("negative: a deployment naming a different runtime manifest SHA is refused")
    expect_error(stage_inputs, "runtime_manifest_sha256", {**config, "runtime_manifest_sha256": "0" * 64}, log=say)
    say("staging inputs")
    receipt = stage_inputs(config, log=say)
    assert receipt["audio"]["files"] == fixture["rows"]
    data_files = sorted(p.name for p in (job / "data" / "audio").iterdir())
    assert "unused_member.wav" not in data_files and len(data_files) == fixture["rows"], "only selected members"
    say("staging again (resume path re-verifies every file)")
    again = stage_inputs(config, log=say)
    assert all(s["status"] == "not_needed_all_files_already_verified" for s in again["audio"]["shards"])

    say("a staged asset deleted locally is re-extracted from assets.zip (not a crash)")
    from dacon9504a100.staging import stage_assets
    victim = job / "assets" / "baseline" / "script.py"
    victim_sha = sha256_file(victim)
    victim.unlink()
    repaired = stage_assets(Path(receipt["paths"]["assets_zip"]), job / "assets", fixture["asset_hashes"])
    assert repaired["repaired"] == ["baseline/script.py"] and sha256_file(victim) == victim_sha

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
    child_environment(config)
    assert (home / "panns_data" / "class_labels_indices.csv").is_file(), "PANNs labels must be pre-placed"

    # CLI safety: a CPU configuration must be refused by every stage entry point.
    pipeline = Pipeline(config=config, report_every=1.0)
    os.symlink(sys.prefix, job / "env" / "venv")  # the pinned test venv stands in for JOB/env/venv
    pipeline._gpu_compute_apps = lambda: []  # test-only: no nvidia-smi on this CPU host (orphan reclaim still runs)
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

    say("negative: a pilot receipt made under a different config is not reused by full extraction")
    expect_error(features.run_full, "Pilot receipt does not match", {**config, "mert_batch_override": 8})

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
    say("training resume: uninterrupted vs interrupted-at-epoch-boundary+resumed WindowMIL")
    check_training_resume(config, rows, job, args.work)

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
    say("negative: changed validation evidence / run config / environment -> lock refused before test opens")
    metrics_path = job / "eval" / "val_metrics.json"
    original_metrics = metrics_path.read_bytes()
    metrics_path.write_bytes(original_metrics.replace(b'"rows"', b'"rows" ', 1))
    expect_error(evaluation.run_test, "Locked validation evidence changed: val_metrics.json", config)
    metrics_path.write_bytes(original_metrics)
    expect_error(evaluation.run_test, "differ from the locked run", {**config, "patience": 7})
    env_path = job / "env" / "receipt.json"
    original_env = env_path.read_bytes()
    env_path.write_text(json.dumps({**json.loads(original_env), "lock_sha256": "0" * 64}))
    expect_error(evaluation.run_test, "environment", config)
    env_path.write_bytes(original_env)
    evaluation.run_test(config)  # restored state verifies again

    # ---------------------------------------------------------------- parity
    from dacon9504a100 import parity
    say("parity comparison: NaN vs number and inf are mismatches; matching NaNs are not")
    import numpy as np
    from dacon9504a100.features import compare_arrays
    tolerance = {"window_truth": 0.0}
    for left, right, expected in (([np.nan, 1.0], [0.0, 1.0], "mismatch"), ([1.0], [np.nan], "mismatch"),
                                  ([np.nan, 1.0], [np.nan, 1.0], "bitwise"), ([np.inf], [np.inf], "mismatch"),
                                  ([1.0, 2.0], [1.0, 2.0 + 1e-9], "mismatch")):
        report = compare_arrays({"window_truth": np.array(left)}, {"window_truth": np.array(right)})
        assert parity.status_for(report, tolerance) == expected, (left, right, report)
    assert parity.prediction_status({"a": parity.prediction_difference(float("nan"), 0.5)}) == "mismatch"
    assert parity.prediction_status({"a": parity.prediction_difference(0.5, 0.5),
                                     "b": parity.prediction_difference(float("nan"), 0.1)}) == "mismatch"

    say("raw-audio parity")
    parity_receipt = parity.run_parity(config)
    assert parity_receipt["status"] in ("bitwise", "within_tolerance"), json.dumps(parity_receipt)[:3000]
    assert {s["split"] for s in parity_receipt["samples"]} >= {"val", "test", "train"}

    say("negative: parity mismatch saves evidence, raises, and blocks a complete handoff")
    from dacon9504a100 import handoff as handoff_module
    from dacon9504a100.handoff import build, collect, coverage_report, file_reader, zip_reader

    def gate_problems(cfg=config):
        """The complete-handoff gate on exactly the files that would be packaged."""
        return coverage_report(cfg, file_reader(collect(cfg, "complete")))["problems"]
    predictions_path = job / "eval" / "test_predictions.csv"
    original_predictions = predictions_path.read_bytes()
    with predictions_path.open(newline="") as stream:
        table = list(csv.DictReader(stream))
    for row in table:
        row["MUSIC_FAKE_PROB"] = str(float(row["MUSIC_FAKE_PROB"]) * 0.5 + 0.25)
    with predictions_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    expect_error(parity.run_parity, "Raw-audio parity mismatch", config)
    assert json.loads((job / "parity" / "parity_receipt.json").read_text())["status"] == "mismatch"
    assert any("parity" in p for p in gate_problems())
    expect_error(build, "coverage check failed", config, "complete", {"stages": {}}, None, log=say)
    predictions_path.write_bytes(original_predictions)
    parity_receipt = parity.run_parity(config)
    assert parity_receipt["status"] in ("bitwise", "within_tolerance")

    say("negative: a test prediction file missing one row fails the coverage check")
    short_eval = args.work / "short_eval"
    shutil.copytree(job / "eval", short_eval)
    lines = (short_eval / "test_predictions.csv").read_text().splitlines(keepends=True)
    (short_eval / "test_predictions.csv").write_text("".join(lines[:-1]))
    assert any("test predictions do not cover" in p for p in gate_problems({**config, "eval_dir": str(short_eval)}))
    assert not gate_problems(), gate_problems()

    say("negative: same IDs, changed bytes -> refused by the complete-handoff gate (evidence-bound hashes)")

    def csv_value(column: str):
        def change(data: bytes) -> bytes:
            rows_ = list(csv.DictReader(data.decode().splitlines(keepends=True)))
            old_value = float(rows_[0][column])
            rows_[0][column] = repr(0.9 if abs(old_value - 0.9) > 1e-6 else 0.1)
            buffer = __import__("io").StringIO(newline="")
            writer = csv.DictWriter(buffer, fieldnames=list(rows_[0]))
            writer.writeheader()
            writer.writerows(rows_)
            changed = buffer.getvalue().encode()
            assert [r["ID"] for r in csv.DictReader(changed.decode().splitlines())] == \
                [r["ID"] for r in csv.DictReader(data.decode().splitlines())], "IDs must be preserved"
            return changed
        return change

    def json_field(key: str, value):
        return lambda data: (json.dumps({**json.loads(data), key: value}, indent=2, sort_keys=True) + "\n").encode()

    for path, transform, label in (
            (job / "eval" / "test_predictions.csv", csv_value("FILE_FAKE_PROB"), "test predictions vs test_receipt"),
            (job / "parity" / "parity_receipt.json", json_field("lock_sha256", "0" * 64), "parity_receipt.lock_sha256"),
            (job / "eval" / "val_predictions.csv", csv_value("VOICE_FAKE_PROB"), "val predictions vs candidate_lock"),
            (job / "eval" / "val_metrics.json", json_field("rows", -1), "val metrics vs candidate_lock"),
            (job / "eval" / "test_metrics.json", json_field("rows", -1), "test metrics vs test_receipt"),
            (job / "eval" / "test_receipt.json", json_field("lock_sha256", "0" * 64), "test_receipt.lock_sha256")):
        original = path.read_bytes()
        path.write_bytes(transform(original))
        try:
            problems = gate_problems()
            assert any(label in p for p in problems), (label, problems)
            expect_error(build, "coverage check failed", config, "complete", {"stages": {}}, None, log=say)
        finally:
            path.write_bytes(original)
    assert not gate_problems(), gate_problems()

    say("negative: a file changed after the gate but before packaging is caught on the packaged bytes")
    real_coverage = handoff_module.coverage_report
    racing_path = job / "eval" / "test_predictions.csv"
    racing_original = racing_path.read_bytes()
    gate_calls = {"n": 0}

    def racing_coverage(cfg, read):
        report = real_coverage(cfg, read)
        gate_calls["n"] += 1
        if gate_calls["n"] == 1:  # after the pre-packaging gate passed
            racing_path.write_bytes(csv_value("FILE_FAKE_PROB")(racing_original))
        return report

    handoff_module.coverage_report = racing_coverage
    try:
        expect_error(build, "Packaged handoff bytes do not reproduce", config, "complete", {"stages": {}}, None,
                     log=say)
    finally:
        handoff_module.coverage_report = real_coverage
        racing_path.write_bytes(racing_original)
    assert gate_calls["n"] == 2
    assert not (job / "handoff" / (constants.HANDOFF_NAME + ".partial")).exists(), "rejected archive left behind"

    # ---------------------------------------------------------------- handoff
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
            "dataset/reference/provenance.json", "model/candidate_lock.json", "code/DACON_Existing9504_A100.ipynb",
            "environment/bin/ffmpeg"} <= names, "config/provenance/audit/lock/notebook/ffmpeg"
    with zipfile.ZipFile(sidecar["file"]) as archive:
        coverage = json.loads(archive.read("handoff_manifest.json"))["coverage"]
    assert coverage["problems"] == [] and coverage["features"]["rows"] == fixture["rows"]
    assert all(v["exact_manifest_order"] for v in coverage["predictions"].values())
    say("packaged-bytes gate: the complete ZIP re-reads clean; one altered packaged member is refused")
    evidence = ["dataset/manifest.csv", "model/feature_index.json", "measurements/features_features_complete.json",
                "model/candidate_lock.json", "model/run_identity.json", "model/results.json",
                "model/voice_classifier_initial.pt", "predictions/val_predictions.csv",
                "predictions/test_predictions.csv", "metrics/val_metrics.json", "metrics/test_metrics.json",
                "metrics/test_receipt.json", "parity/parity_receipt.json",
                *(f"model/checkpoints/{r}_{s}.pt" for r in ("music", "voice") for s in constants.SEEDS)]
    with zipfile.ZipFile(sidecar["file"]) as archive:
        assert coverage_report(config, zip_reader(archive)) == coverage
        packaged = {name: archive.read(name) for name in evidence}
    for altered in (False, True):
        copy = args.work / f"evidence_{altered}.zip"
        with zipfile.ZipFile(copy, "w") as archive:
            for name, data in packaged.items():
                if altered and name == "predictions/test_predictions.csv":
                    data = csv_value("FILE_FAKE_PROB")(data)
                archive.writestr(name, data)
        with zipfile.ZipFile(copy) as archive:
            problems = coverage_report(config, zip_reader(archive))["problems"]
        assert bool(problems) is altered, problems
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
