#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["nbformat>=5.10,<6"]
# ///
"""Build the source-embedded V5 B1--B4 full-refit Colab notebook."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Final

import nbformat as nb

ROOT: Final = Path(__file__).resolve().parent
NOTEBOOK: Final = ROOT / "DACON_236749_final_B1_B4.ipynb"
INPUTS: Final = ROOT / "v4_validation" / "drive_inputs.json"
WORK: Final = "/content/dacon_v5_final"
RUN_ROOT: Final = "/content/drive/MyDrive/DACON_236749/runs_final_v5"
INPUT_NAMES: Final = frozenset(
    {
        "v4_train.zip",
        "v4_dev.zip",
        "v4_sealed.zip",
        "v4_metadata.zip",
        "recovery_sonics_assets.zip",
    }
)
SOURCES: Final = (
    "run_v5_final.py",
    "v5_final_refit.py",
    "package_v5_submission.py",
    "v5_submission.py",
    "v5_dataset.py",
    "v5_augment.py",
    "v5_experts.py",
    "v5_features.py",
    "v5_fusion.py",
    "v5_fusion_training.py",
    "v5_joint.py",
    "v5_reporting.py",
    "v5_training.py",
    "v4_data.py",
    "v4_model.py",
    "v4_model_components.py",
    "v4_training_refit.py",
    "recovery_sonics.py",
    "ast_multitask.py",
    "v2_data.py",
    "v2_scoring.py",
    "graph_features.py",
    "graph_fusion.py",
    "graph_scoring.py",
)
BASELINE_BYTES: Final = 4_562_809_110
BASELINE_SHA256: Final = "cc86f143915c95fd958a0066cbf14b0db9f4c41fd213310c3650b9fe870d0c70"


def code(source: str) -> nb.NotebookNode:
    """Compile generated code before adding it to the notebook."""
    source = textwrap.dedent(source).strip()
    compile(source, "v5_final_notebook_cell", "exec")
    return nb.v4.new_code_cell(source)


def embedded_source(name: str) -> nb.NotebookNode:
    """Compile and embed one member of the offline source closure."""
    source = (ROOT / name).read_text(encoding="utf-8")
    compile(source, name, "exec")
    return nb.v4.new_code_cell(f"%%writefile {WORK}/{name}\n{source}")


def selected_inputs() -> list[dict[str, object]]:
    """Return the complete final-refit archive receipts, including sealed metadata."""
    archives = json.loads(INPUTS.read_text(encoding="utf-8"))
    selected = [archive for archive in archives if archive["name"] in INPUT_NAMES]
    if {archive["name"] for archive in selected} != INPUT_NAMES:
        raise ValueError("Final refit requires exactly train, dev, sealed, metadata, and SONICS receipts")
    return selected


def packages() -> list[str]:
    """Reuse the V5 runtime pins needed by the embedded source closure."""
    return [
        "demucs==4.0.1",
        "panns-inference==0.1.1",
        "transformers==4.57.6",
        "librosa==0.10.2.post1",
        "diffq==0.2.4",
        "einops==0.8.1",
        "timm==1.0.15",
        "scikit-learn==1.8.0",
        "soundfile==0.12.1",
        "imageio-ffmpeg==0.6.0",
        "safetensors==0.8.0",
        "fvcore==0.1.5.post20221221",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allocation-start",
        type=float,
        help="Optional pinned allocation UNIX timestamp; default is the first Colab cell time.",
    )
    args = parser.parse_args()
    if args.allocation_start is not None and args.allocation_start <= 0:
        parser.error("--allocation-start must be a positive UNIX timestamp")
    missing = [name for name in SOURCES if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError("Final notebook source not ready: " + ", ".join(missing))

    cells = [
        nb.v4.new_markdown_cell(
            "# DACON 236749 V5 final B1--B4 refit\n\n"
            "This notebook trains and packages B1--B4 once from the current embedded source. "
            "It does not submit anything to DACON."
        ),
        code(
            """
            import hashlib, importlib.metadata, json, os, shutil, stat, subprocess, sys, threading, time, traceback, zipfile
            from pathlib import Path
            from google.colab import drive, runtime

            PINNED_ALLOCATION_START = __PINNED_ALLOCATION_START__
            FIRST_CELL_STARTED = time.time()
            ALLOCATION_START = float(PINNED_ALLOCATION_START if PINNED_ALLOCATION_START is not None else FIRST_CELL_STARTED)
            ALLOCATION_LIMIT_SECONDS = 7200
            CLEANUP_RESERVE_SECONDS = 180
            SAFE_DEADLINE = ALLOCATION_START + ALLOCATION_LIMIT_SECONDS - CLEANUP_RESERVE_SECONDS
            if SAFE_DEADLINE <= FIRST_CELL_STARTED:
                raise TimeoutError("Final refit safe deadline already elapsed")

            def watchdog_stop():
                try:
                    if "RUN" in globals() and RUN.exists():
                        with (RUN / "budget_events.jsonl").open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps({"event": "watchdog_release_requested", "unix": time.time()}) + "\\n")
                finally:
                    runtime.unassign()

            watchdog = threading.Timer(max(1, SAFE_DEADLINE - time.time()), watchdog_stop)
            watchdog.daemon = True
            watchdog.start()
            try:
                drive.mount("/content/drive")
                DRIVE = Path("/content/drive/MyDrive/DACON_236749")
                WORK = Path("__WORK__")
                WORK.mkdir(parents=True, exist_ok=True)
                RUN = DRIVE / "runs_final_v5" / time.strftime("%Y%m%d_%H%M%S", time.gmtime(FIRST_CELL_STARTED))
                RUN.mkdir(parents=True, exist_ok=False)
            except BaseException:
                runtime.unassign()
                raise

            def write_json(path, value):
                temporary = path.with_suffix(path.suffix + ".tmp")
                temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
                temporary.replace(path)

            def sha256(path):
                with path.open("rb") as stream:
                    return hashlib.file_digest(stream, "sha256").hexdigest()

            def budget_event(event, **extra):
                with (RUN / "budget_events.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"event": event, "unix": time.time(), "elapsed_seconds": time.time() - ALLOCATION_START, **extra}) + "\\n")

            write_json(RUN / "budget.json", {
                "allocation_start_unix": ALLOCATION_START,
                "first_cell_started_unix": FIRST_CELL_STARTED,
                "allocation_limit_seconds": ALLOCATION_LIMIT_SECONDS,
                "cleanup_reserve_seconds": CLEANUP_RESERVE_SECONDS,
                "safe_deadline_unix": SAFE_DEADLINE,
                "policy": "one absolute two-hour allocation including staging, packaging, receipts, and release",
            })
            write_json(RUN / "run_contract.json", {
                "arms": ["B1", "B2", "B3", "B4"],
                "work": str(WORK),
                "offline_after_staging": True,
                "sealed_local_external_holdout": "merged for final refit; official private evaluation is unavailable",
                "dacon_submission": "disabled",
                "builder_execution_state": "not executed by builder",
            })
            budget_event("notebook_started")
            print("RESULTS_DIRECTORY", RUN, flush=True)
            """.replace("__PINNED_ALLOCATION_START__", repr(args.allocation_start)).replace("__WORK__", WORK)
        ),
        nb.v4.new_markdown_cell(
            "## Embedded source closure\n\n"
            "The cells below are written to local Colab storage before the protected staging and offline runner cell."
        ),
    ]
    cells.extend(embedded_source(name) for name in SOURCES)
    cells.extend(
        [
            code("INPUT_ARCHIVES = " + repr(selected_inputs())),
            code("PINNED_PACKAGES = " + repr(packages())),
            code(
                f"""
                BASELINE_BYTES = {BASELINE_BYTES}
                BASELINE_SHA256 = {BASELINE_SHA256!r}
                MIN_FREE_BYTES = 60 * 1024**3

                def safe_extract(archive, destination):
                    root = destination.resolve()
                    with zipfile.ZipFile(archive) as packed:
                        members = packed.infolist()
                        for member in members:
                            relative = Path(member.filename)
                            target = destination / relative
                            if (relative.is_absolute() or ".." in relative.parts or stat.S_ISLNK(member.external_attr >> 16)
                                    or not target.resolve().is_relative_to(root)):
                                raise ValueError(f"Unsafe archive member in {{archive.name}}: {{member.filename}}")
                        for member in members:
                            target = destination / Path(member.filename)
                            if member.is_dir():
                                target.mkdir(parents=True, exist_ok=True)
                                continue
                            target.parent.mkdir(parents=True, exist_ok=True)
                            with packed.open(member) as incoming, target.open("wb") as output:
                                shutil.copyfileobj(incoming, output, 8 * 1024**2)

                def extract_baseline_member(outer, destination):
                    with zipfile.ZipFile(outer) as packed:
                        members = [member for member in packed.infolist() if not member.is_dir() and not Path(member.filename).is_absolute() and ".." not in Path(member.filename).parts and not stat.S_ISLNK(member.external_attr >> 16) and Path(member.filename).name == "baseline_submit.zip"]
                        if len(members) != 1 or members[0].file_size != BASELINE_BYTES:
                            raise ValueError("Expected one bounded baseline_submit.zip member")
                        with packed.open(members[0]) as incoming, destination.open("wb") as output:
                            shutil.copyfileobj(incoming, output, 8 * 1024**2)
                    if destination.stat().st_size != BASELINE_BYTES or sha256(destination) != BASELINE_SHA256:
                        raise ValueError("Official baseline archive receipt mismatch")
                    return destination

                def capture_executed_notebook():
                    try:
                        from google.colab import _message
                        payload = _message.blocking_request("get_ipynb", timeout_sec=30)["ipynb"]
                        payload = json.loads(payload) if isinstance(payload, str) else payload
                        destination = RUN / "executed.ipynb"
                        destination.write_text(json.dumps(payload), encoding="utf-8")
                        write_json(RUN / "executed_notebook_capture.json", {{"status": "captured", "current_cell_may_be_incomplete": True, "sha256": sha256(destination)}})
                    except BaseException:
                        (RUN / "executed_notebook_capture_error.txt").write_text(traceback.format_exc(), encoding="utf-8")

                def record_artifacts():
                    receipt = {{}}
                    for path in sorted(RUN.rglob("*")):
                        if path.is_file() and path.name != "executed.ipynb":
                            receipt[str(path.relative_to(RUN))] = {{"bytes": path.stat().st_size, "sha256": sha256(path)}}
                    write_json(RUN / "drive_receipt.json", receipt)

                def package_receipt_shape(receipt):
                    if not isinstance(receipt, dict):
                        return False
                    offline_validation = receipt.get("offline_validation")
                    return (isinstance(receipt.get("arm"), str) and isinstance(receipt.get("path"), str)
                            and isinstance(receipt.get("bytes"), int) and isinstance(receipt.get("sha256"), str)
                            and isinstance(offline_validation, dict) and offline_validation.get("status") == "passed")

                process = None
                try:
                    available = shutil.disk_usage(WORK).free
                    write_json(RUN / "disk_check.json", {{"path": str(WORK), "available_bytes": available, "required_bytes": MIN_FREE_BYTES}})
                    if available < MIN_FREE_BYTES:
                        raise OSError(f"Insufficient Colab disk: {{available}} < {{MIN_FREE_BYTES}} bytes")
                    (RUN / "code").mkdir(exist_ok=True)
                    for source in WORK.glob("*.py"):
                        shutil.copy2(source, RUN / "code" / source.name)
                    source_hashes = {{path.name: sha256(path) for path in sorted((RUN / "code").glob("*.py"))}}
                    write_json(RUN / "source_hashes.json", source_hashes)
                    write_json(RUN / "input_archives.json", INPUT_ARCHIVES)
                    with (RUN / "install.log").open("w", buffering=1, encoding="utf-8") as log:
                        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PINNED_PACKAGES], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
                    requested = {{package.split("==", 1)[0]: package.split("==", 1)[1] for package in PINNED_PACKAGES}}
                    write_json(RUN / "environment.json", {{"python": sys.version, "requested_packages": requested, "packages": {{name: importlib.metadata.version(name) for name in requested}}}})
                    part_receipts = {{}}
                    for archive in INPUT_ARCHIVES:
                        verified_parts = []
                        for part in archive["parts"]:
                            source = DRIVE / part["name"]
                            if not source.is_file() or source.stat().st_size != part["bytes"] or sha256(source) != part["sha256"]:
                                raise ValueError(f"Drive part receipt mismatch: {{source}}")
                            verified_parts.append({{"name": source.name, "bytes": source.stat().st_size, "sha256": part["sha256"]}})
                        destination = WORK / archive["name"]
                        reusable = destination.is_file() and destination.stat().st_size == archive["bytes"] and sha256(destination) == archive["sha256"]
                        if not reusable:
                            with destination.open("wb") as output:
                                for part in archive["parts"]:
                                    with (DRIVE / part["name"]).open("rb") as incoming:
                                        shutil.copyfileobj(incoming, output, 8 * 1024**2)
                        if destination.stat().st_size != archive["bytes"] or sha256(destination) != archive["sha256"]:
                            raise ValueError(f"Archive receipt mismatch: {{destination.name}}")
                        safe_extract(destination, WORK)
                        part_receipts[archive["name"]] = {{"parts": verified_parts, "archive_bytes": archive["bytes"], "archive_sha256": archive["sha256"], "reused_local_archive": reusable}}
                        budget_event("archive_verified", archive=destination.name, reused_local_archive=reusable)
                    write_json(RUN / "input_part_receipts.json", part_receipts)
                    data = WORK / "v4_validation"
                    required = (data / "train.csv", data / "dev.csv", data / "sealed.csv", WORK / "sonics_assets" / "config.json", WORK / "sonics_assets" / "pytorch_model.bin")
                    if not all(path.is_file() for path in required):
                        raise FileNotFoundError("Verified final-refit inputs are incomplete")
                    open_archive = DRIVE / "open.zip"
                    write_json(RUN / "open_zip_receipt.json", {{"bytes": open_archive.stat().st_size, "sha256": sha256(open_archive), "embedded_baseline_bytes": BASELINE_BYTES, "embedded_baseline_sha256": BASELINE_SHA256}})
                    baseline = WORK / "baseline"
                    safe_extract(extract_baseline_member(open_archive, WORK / "baseline_submit.zip"), baseline)
                    if not (baseline / "script.py").is_file() or not (baseline / "model").is_dir():
                        raise FileNotFoundError("Official baseline script or model is missing")
                    command = [sys.executable, "-u", "run_v5_final.py", "--work", str(WORK), "--output", str(RUN), "--deadline", str(SAFE_DEADLINE)]
                    budget_event("runner_started", command=command)
                    with (RUN / "experiment.log").open("w", buffering=1, encoding="utf-8") as log:
                        process = subprocess.Popen(command, cwd=WORK, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"))
                        while process.poll() is None:
                            if time.time() >= SAFE_DEADLINE:
                                budget_event("runner_terminate_at_safe_deadline")
                                process.terminate()
                                try:
                                    process.wait(timeout=20)
                                except subprocess.TimeoutExpired:
                                    process.kill()
                                    process.wait(timeout=20)
                                raise TimeoutError("Final runner reached the absolute safe allocation boundary")
                            time.sleep(30)
                            lines = (RUN / "experiment.log").read_text(encoding="utf-8", errors="replace").splitlines()
                            print(lines[-1][:1000] if lines else "Staging final refit...", flush=True)
                    if process.returncode:
                        raise RuntimeError(f"Final runner exited with {{process.returncode}}")
                    completion = RUN / "final_completion.json"
                    if not completion.is_file():
                        raise FileNotFoundError("Runner did not write final_completion.json")
                    completion_data = json.loads(completion.read_text(encoding="utf-8"))
                    zip_receipts = completion_data.get("packages")
                    if completion_data.get("status") != "ready_not_submitted" or not isinstance(zip_receipts, list) or len(zip_receipts) != 4:
                        raise ValueError("Runner completion is not a verified four-ZIP result")
                    verified = []
                    arms = set()
                    for receipt in zip_receipts:
                        if not package_receipt_shape(receipt):
                            raise ValueError("Runner ZIP receipt is incomplete")
                        relative = receipt["path"]
                        candidate = Path(relative)
                        archive = (candidate if candidate.is_absolute() else RUN / candidate).resolve()
                        if not archive.is_relative_to(RUN.resolve()):
                            raise ValueError(f"Runner ZIP receipt escapes results directory: {{relative}}")
                        if not archive.is_file() or archive.suffix != ".zip" or archive.stat().st_size != receipt.get("bytes") or sha256(archive) != receipt.get("sha256"):
                            raise ValueError(f"Runner ZIP receipt mismatch: {{relative}}")
                        verified.append(relative)
                        arms.add(receipt["arm"])
                    if arms != {"B1", "B2", "B3", "B4"} or len(set(verified)) != 4:
                        raise ValueError("Runner completion does not cover exactly B1--B4")
                    write_json(RUN / "completion_verified.json", {{"status": "ready_not_submitted", "zip_paths": verified, "packages": zip_receipts, "runner_completion_sha256": sha256(completion)}})
                    budget_event("runner_finished", verified_zip_count=len(verified))
                    print("FINAL_V5_SAVED", RUN, flush=True)
                except BaseException:
                    if "RUN" in globals() and RUN.exists():
                        (RUN / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
                        budget_event("notebook_failure")
                    raise
                finally:
                    try:
                        if "RUN" in globals() and RUN.exists():
                            capture_executed_notebook()
                            record_artifacts()
                            write_json(RUN / "release_requested.json", {{"unix": time.time(), "elapsed_seconds": time.time() - ALLOCATION_START}})
                    finally:
                        watchdog.cancel()
                        runtime.unassign()
                """
            ),
            nb.v4.new_markdown_cell(
                "## Result boundary\n\n"
                "Read `completion_verified.json`, runner receipts, source hashes, archive receipts, logs, and the captured notebook before making a result claim. This notebook does not submit to DACON."
            ),
        ]
    )
    notebook = nb.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"name": NOTEBOOK.name, "gpuType": "A100", "provenance": []},
            "v5_source_state": "source-embedded final refit; builder has not executed Colab",
        },
    )
    nb.validate(notebook)
    nb.write(notebook, NOTEBOOK)


if __name__ == "__main__":
    main()
