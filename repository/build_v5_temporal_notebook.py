#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["nbformat>=5.10,<6"]
# ///
"""Build the source-embedded V5 B3/B4-only Colab notebook; it never runs locally."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Final

import nbformat as nb

ROOT: Final = Path(__file__).resolve().parent
NOTEBOOK: Final = ROOT / "DACON_236749_temporal_only_v5.ipynb"
INPUTS: Final = ROOT / "v4_validation" / "drive_inputs.json"
ORIGIN_RUN: Final = "/content/drive/MyDrive/DACON_236749/runs_compare_v5/20260913_141033"
SOURCES: Final = (
    "v5_resume_temporal.py", "v5_analyze.py", "v5_compare.py", "v5_experts.py",
    "v5_dataset.py", "v5_augment.py", "v5_features.py", "v5_fusion.py",
    "v5_fusion_training.py", "v5_joint.py", "v5_reporting.py", "v5_training.py",
    "v4_data.py", "v4_model.py", "v4_model_components.py", "recovery_sonics.py",
    "ast_multitask.py", "v2_data.py", "v2_scoring.py", "graph_features.py",
    "graph_fusion.py", "graph_scoring.py",
)
INPUT_NAMES: Final = frozenset({"v4_train.zip", "v4_dev.zip", "v4_metadata.zip", "recovery_sonics_assets.zip"})
BASELINE_BYTES: Final = 4_562_809_110
BASELINE_SHA256: Final = "cc86f143915c95fd958a0066cbf14b0db9f4c41fd213310c3650b9fe870d0c70"


def code(source: str) -> nb.NotebookNode:
    """Compile generated notebook code before writing the notebook."""
    source = textwrap.dedent(source).strip()
    compile(source, "v5_temporal_notebook_cell", "exec")
    return nb.v4.new_code_cell(source)


def embedded_source(name: str) -> nb.NotebookNode:
    """Embed one current source file after compiling it locally."""
    source = (ROOT / name).read_text(encoding="utf-8")
    compile(source, name, "exec")
    return nb.v4.new_code_cell(f"%%writefile /content/dacon_v5_temporal/{name}\n{source}")


def selected_inputs() -> list[dict[str, object]]:
    """Select the only allowed input archives for temporal recovery."""
    archives = json.loads(INPUTS.read_text(encoding="utf-8"))
    selected = [archive for archive in archives if archive["name"] in INPUT_NAMES]
    if {archive["name"] for archive in selected} != INPUT_NAMES:
        raise ValueError("Temporal recovery requires exactly train, dev, metadata, and SONICS receipts")
    return selected


def packages() -> list[str]:
    """Reuse the pinned V4 runtime required by the embedded V5 closure."""
    return [
        "demucs==4.0.1", "panns-inference==0.1.1", "transformers==4.57.6",
        "librosa==0.10.2.post1", "diffq==0.2.4", "einops==0.8.1", "timm==1.0.15",
        "scikit-learn==1.8.0", "soundfile==0.12.1", "imageio-ffmpeg==0.6.0",
        "safetensors==0.8.0", "fvcore==0.1.5.post20221221",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocation-start", type=float, help="Optional new allocation UNIX timestamp; default is first-cell time.")
    args = parser.parse_args()
    if args.allocation_start is not None and args.allocation_start <= 0:
        parser.error("--allocation-start must be a positive UNIX timestamp")
    missing = [name for name in SOURCES if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError("Temporal notebook source not ready: " + ", ".join(missing))

    cells = [
        code(
            """
            import hashlib, html, importlib.metadata, json, os, shutil, stat, subprocess, sys, threading, time, traceback, zipfile
            from pathlib import Path
            from google.colab import drive, runtime

            PINNED_ALLOCATION_START = __PINNED_ALLOCATION_START__
            FIRST_CELL_STARTED = time.time()
            ALLOCATION_START = float(PINNED_ALLOCATION_START if PINNED_ALLOCATION_START is not None else FIRST_CELL_STARTED)
            ALLOCATION_LIMIT_SECONDS = 3600
            SAFE_SHUTDOWN_SECONDS = 3480
            SAFE_DEADLINE = ALLOCATION_START + SAFE_SHUTDOWN_SECONDS
            if SAFE_DEADLINE <= FIRST_CELL_STARTED:
                raise TimeoutError("Temporal recovery allocation safe deadline already elapsed")

            def watchdog_stop():
                try:
                    if "RUN" in globals() and RUN.exists():
                        budget_event("watchdog_release_requested")
                finally:
                    runtime.unassign()

            watchdog = threading.Timer(max(1, SAFE_DEADLINE - time.time()), watchdog_stop)
            watchdog.daemon = True
            watchdog.start()
            drive.mount("/content/drive")
            DRIVE = Path("/content/drive/MyDrive/DACON_236749")
            WORK = Path("/content/dacon_v5_temporal")
            WORK.mkdir(exist_ok=True)
            RUN = DRIVE / "runs_compare_v5_temporal" / time.strftime("%Y%m%d_%H%M%S", time.gmtime(FIRST_CELL_STARTED))
            RUN.mkdir(parents=True, exist_ok=False)
            ORIGIN_RUN = Path("__ORIGIN_RUN__")

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
                "safe_shutdown_seconds": SAFE_SHUTDOWN_SECONDS,
                "safe_deadline_unix": SAFE_DEADLINE,
                "deadline_policy": "new absolute 60-minute allocation including setup, with 120-second cleanup reserve",
            })
            write_json(RUN / "temporal_notebook_contract.json", {
                "purpose": "B3/B4 temporal-only recovery",
                "origin_run": str(ORIGIN_RUN),
                "required_authorization": "Additional GPU budget authorization is required before this notebook is executed",
                "execution_state": "not executed by builder",
                "excluded_work": ["B0 fresh inference", "B1/B2 expert retraining", "submission"],
            })
            budget_event("notebook_started")
            print("RESULTS_DIRECTORY", RUN, flush=True)
            """.replace("__PINNED_ALLOCATION_START__", repr(args.allocation_start)).replace("__ORIGIN_RUN__", ORIGIN_RUN)
        ),
        nb.v4.new_markdown_cell(
            "# DACON 236749 V5 B3/B4 temporal-only recovery\n\n"
            "**Requires additional GPU budget authorization before execution. This notebook has not been executed.** "
            "It recreates only the missing B3/B4 temporal stage under a fresh 60-minute allocation and leaves the September 13 run unchanged."
        ),
        nb.v4.new_markdown_cell(
            "## Boundary\n\n"
            "The notebook copies B0–B2 prediction and JSON receipts from the fixed origin run, but never copies its B2 weight files. "
            "The temporal runner reads those pinned weights in place through `--expert-output`; it writes B3/B4 artifacts only under the new run directory."
        ),
        nb.v4.new_markdown_cell("## Embedded execution source\n\nEvery cell below is compiled by the builder and hashed before the temporal-only command starts."),
    ]
    cells.extend(embedded_source(name) for name in SOURCES)
    cells.extend([
        code("INPUT_ARCHIVES = " + repr(selected_inputs())),
        code("PINNED_PACKAGES = " + repr(packages())),
        code(
            f"""
            BASELINE_BYTES = {BASELINE_BYTES}
            BASELINE_SHA256 = {BASELINE_SHA256!r}

            def safe_extract(archive, destination, allow_member=None):
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
                        if allow_member is not None and not allow_member(member.filename):
                            continue
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

            def copy_origin_receipts():
                required = ["comparison.json", "B0_predictions.csv", "B1_predictions.csv", "B2_predictions.csv"]
                copied = {{}}
                for name in required:
                    source = ORIGIN_RUN / name
                    if not source.is_file():
                        raise FileNotFoundError(f"Origin artifact is missing: {{source}}")
                    destination = RUN / name
                    shutil.copy2(source, destination)
                    copied[name] = {{"bytes": destination.stat().st_size, "sha256": sha256(destination)}}
                pinned = {{}}
                for arm in ("B1", "B2"):
                    for kind in ("voice", "music", "presence"):
                        for name in ("training.json", "inference.json", "weights.json"):
                            relative = Path("jobs") / f"{{arm}}_{{kind}}" / name
                            source = ORIGIN_RUN / relative
                            if not source.is_file():
                                raise FileNotFoundError(f"Origin receipt is missing: {{source}}")
                            destination = RUN / relative
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(source, destination)
                            copied[str(relative)] = {{"bytes": destination.stat().st_size, "sha256": sha256(destination)}}
                        weight = json.loads((RUN / "jobs" / f"{{arm}}_{{kind}}" / "weights.json").read_text(encoding="utf-8"))
                        pinned[f"{{arm}}_{{kind}}"] = {{"origin_weight_path": str(ORIGIN_RUN / "jobs" / f"{{arm}}_{{kind}}" / "best.pt"), "sha256": weight["sha256"]}}
                write_json(RUN / "origin_receipt.json", {{"origin_run": str(ORIGIN_RUN), "copied_small_artifacts": copied, "pinned_weights_not_copied": pinned}})

            def analyze_if_comparison_exists():
                if not (RUN / "comparison.json").is_file():
                    return
                remaining = SAFE_DEADLINE - time.time()
                if remaining <= 0:
                    write_json(RUN / "analysis_receipt.json", {{"status": "skipped_deadline"}})
                    return
                command = [sys.executable, "-u", "v5_analyze.py", "--results", str(RUN), "--data", str(WORK / "v4_validation")]
                try:
                    with (RUN / "analysis.log").open("w", buffering=1, encoding="utf-8") as log:
                        subprocess.run(command, cwd=WORK, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=min(180, max(1, int(remaining))))
                    write_json(RUN / "analysis_receipt.json", {{"status": "generated", "command": command}})
                except Exception:
                    (RUN / "analysis_failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
                    write_json(RUN / "analysis_receipt.json", {{"status": "failed", "command": command, "failure": "analysis_failure.txt"}})

            def show_actual_results():
                report = RUN / "analysis.json"
                if report.is_file():
                    from IPython.display import JSON, Image, display
                    display(JSON(json.loads(report.read_text(encoding="utf-8"))))
                    for image in sorted(RUN.rglob("*.png")):
                        display(Image(filename=str(image)))

            def capture_executed_notebook():
                try:
                    from google.colab import _message
                    payload = _message.blocking_request("get_ipynb", timeout_sec=30)["ipynb"]
                    payload = json.loads(payload) if isinstance(payload, str) else payload
                    destination = RUN / "executed.ipynb"
                    destination.write_text(json.dumps(payload), encoding="utf-8")
                    write_json(RUN / "executed_notebook_capture.json", {{"status": "captured", "current_cell_may_be_incomplete": True, "sha256": sha256(destination)}})
                except Exception:
                    (RUN / "executed_notebook_capture_error.txt").write_text(traceback.format_exc(), encoding="utf-8")

            def record_artifacts():
                receipt = {{}}
                for path in sorted(RUN.rglob("*")):
                    if path.is_file() and path.name != "executed.ipynb":
                        receipt[str(path.relative_to(RUN))] = {{"bytes": path.stat().st_size, "sha256": sha256(path)}}
                write_json(RUN / "drive_receipt.json", receipt)

            process = None
            try:
                (RUN / "code").mkdir(exist_ok=True)
                for source in WORK.glob("*.py"):
                    shutil.copy2(source, RUN / "code" / source.name)
                source_hashes = {{path.name: sha256(path) for path in sorted((RUN / "code").glob("*.py"))}}
                write_json(RUN / "source_hashes.json", source_hashes)
                write_json(RUN / "source_bundle_receipt.json", {{"source_count_py": len(source_hashes), "sources": source_hashes}})
                with (RUN / "install.log").open("w", encoding="utf-8") as log:
                    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PINNED_PACKAGES], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
                write_json(RUN / "environment.json", {{"python": sys.version, "requested_packages": PINNED_PACKAGES, "packages": {{name.split("==", 1)[0]: importlib.metadata.version(name.split("==", 1)[0]) for name in PINNED_PACKAGES}}}})
                for archive in INPUT_ARCHIVES:
                    destination = WORK / archive["name"]
                    reusable = destination.is_file() and destination.stat().st_size == archive["bytes"] and sha256(destination) == archive["sha256"]
                    if not reusable:
                        with destination.open("wb") as output:
                            for part in archive["parts"]:
                                source = DRIVE / part["name"]
                                if source.stat().st_size != part["bytes"] or sha256(source) != part["sha256"]:
                                    raise ValueError(f"Drive part receipt mismatch: {{source.name}}")
                                with source.open("rb") as incoming:
                                    shutil.copyfileobj(incoming, output, 8 * 1024**2)
                    if destination.stat().st_size != archive["bytes"] or sha256(destination) != archive["sha256"]:
                        raise ValueError(f"Archive receipt mismatch: {{destination.name}}")
                    safe_extract(destination, WORK, (lambda name: Path(name).name != "sealed.csv") if archive["name"] == "v4_metadata.zip" else None)
                    budget_event("archive_verified", archive=destination.name, reused_local_archive=reusable)
                data = WORK / "v4_validation"
                if (data / "sealed.csv").exists() or not all(path.is_file() for path in (data / "train.csv", data / "dev.csv", WORK / "sonics_assets" / "config.json", WORK / "sonics_assets" / "pytorch_model.bin")):
                    raise FileNotFoundError("Verified temporal-only inputs are incomplete")
                baseline = WORK / "baseline"
                safe_extract(extract_baseline_member(DRIVE / "open.zip", WORK / "embeddedbaseline_submit.zip"), baseline, lambda name: Path(name).parts[0] == "model" or Path(name).name == "script.py" or Path(name).name.startswith("requirements"))
                if not (baseline / "script.py").is_file() or not (baseline / "model").is_dir():
                    raise FileNotFoundError("Official baseline script or model is missing")
                copy_origin_receipts()
                command = [sys.executable, "-u", "v5_resume_temporal.py", "--work", str(WORK), "--output", str(RUN), "--expert-output", str(ORIGIN_RUN), "--deadline", str(SAFE_DEADLINE)]
                budget_event("temporal_runner_started", command=command)
                with (RUN / "experiment.log").open("w", buffering=1, encoding="utf-8") as log:
                    process = subprocess.Popen(command, cwd=WORK, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"))
                    while process.poll() is None:
                        if time.time() >= SAFE_DEADLINE:
                            budget_event("temporal_runner_terminate_at_safe_deadline")
                            process.terminate()
                            try:
                                process.wait(timeout=20)
                            except subprocess.TimeoutExpired:
                                budget_event("temporal_runner_kill_after_terminate_timeout")
                                process.kill()
                                process.wait(timeout=20)
                            raise TimeoutError("Temporal-only runner reached the absolute safe allocation boundary")
                        time.sleep(30)
                        lines = (RUN / "experiment.log").read_text(encoding="utf-8", errors="replace").splitlines()
                        print(lines[-1][:1000] if lines else "Preparing B3/B4 temporal recovery...", flush=True)
                if process.returncode:
                    raise RuntimeError(f"Temporal-only runner exited with {{process.returncode}}")
                budget_event("temporal_runner_finished")
            except Exception:
                (RUN / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
                budget_event("notebook_failure")
                print(traceback.format_exc(), flush=True)
                raise
            finally:
                try:
                    try:
                        analyze_if_comparison_exists()
                    except Exception:
                        (RUN / "analysis_finally_failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
                    try:
                        show_actual_results()
                    except Exception:
                        (RUN / "result_display_failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
                    capture_executed_notebook()
                    record_artifacts()
                    write_json(RUN / "release_requested.json", {{"unix": time.time(), "elapsed_seconds": time.time() - ALLOCATION_START}})
                finally:
                    watchdog.cancel()
                    runtime.unassign()
            """
        ),
        nb.v4.new_markdown_cell("## Takeaways\n\nThis notebook requires a new, explicit GPU-budget authorization before execution. Read `analysis.json`, `REPORT.md`, source hashes, origin receipts, and budget events after a real run."),
    ])
    notebook = nb.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"}, "accelerator": "GPU",
        "colab": {"name": NOTEBOOK.name, "gpuType": "A100", "provenance": []},
        "v5_source_state": "temporal-only preparation; requires additional budget authorization; not executed",
    })
    nb.validate(notebook)
    nb.write(notebook, NOTEBOOK)


if __name__ == "__main__":
    main()
