#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["nbformat>=5.10,<6"]
# ///
"""Build the source-embedded, allocation-bounded V5 comparison notebook."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Final

import nbformat as nb

ROOT: Final = Path(__file__).resolve().parent
NOTEBOOK: Final = ROOT / "DACON_236749_compare_v5.ipynb"
INPUTS: Final = ROOT / "v4_validation" / "drive_inputs.json"
REQUIRED_SOURCES: Final = (
    "v5_compare.py",
    "v5_analyze.py",
    "v5_experts.py",
    "v5_dataset.py",
    "v5_fusion.py",
    "v5_fusion_training.py",
    "v5_joint.py",
    "v5_reporting.py",
    "v5_resume_temporal.py",
    "v5_training.py",
    "v5_features.py",
    "v4_model.py",
    "v4_model_components.py",
    "v4_data.py",
    "recovery_sonics.py",
    "ast_multitask.py",
    "v2_data.py",
    "graph_scoring.py",
    "graph_features.py",
    "graph_fusion.py",
    "v2_scoring.py",
)
OPTIONAL_SOURCES: Final = ("v5_voice.py", "v5_augment.py")
INPUT_NAMES: Final = frozenset(
    {"v4_train.zip", "v4_dev.zip", "v4_metadata.zip", "recovery_sonics_assets.zip"}
)
BASELINE_BYTES: Final = 4_562_809_110
BASELINE_SHA256: Final = "cc86f143915c95fd958a0066cbf14b0db9f4c41fd213310c3650b9fe870d0c70"
G0_SHA256: Final = "bd1c27227829f36e9ae6191db7600920bf6649810f0896a791730b10555f0854"


def code(source: str) -> nb.NotebookNode:
    """Compile a generated code cell before placing it in the notebook."""
    source = textwrap.dedent(source).strip()
    compile(source, "v5_notebook_cell", "exec")
    return nb.v4.new_code_cell(source)


def sources() -> tuple[str, ...]:
    """Return the required V5 closure plus an optional voice implementation."""
    missing = [name for name in REQUIRED_SOURCES if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError("V5 source not ready: " + ", ".join(missing))
    return (*REQUIRED_SOURCES, *(name for name in OPTIONAL_SOURCES if (ROOT / name).is_file()))


def embedded_source(name: str) -> nb.NotebookNode:
    """Embed one compiled repository source file into the Colab workspace."""
    source = (ROOT / name).read_text(encoding="utf-8")
    compile(source, name, "exec")
    return nb.v4.new_code_cell(f"%%writefile /content/dacon_v5/{name}\n{source}")


def selected_inputs() -> list[dict[str, object]]:
    """Keep only the V5 train, dev, metadata, and SONICS receipts."""
    inputs = json.loads(INPUTS.read_text(encoding="utf-8"))
    selected = [archive for archive in inputs if archive["name"] in INPUT_NAMES]
    if {archive["name"] for archive in selected} != INPUT_NAMES:
        raise ValueError("V5 requires exactly train, dev, metadata, and SONICS receipts")
    return selected


def packages(source_names: tuple[str, ...]) -> list[str]:
    """Use the pinned V4 runtime and add PEFT only when V5 imports it."""
    package_list = [
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
    text = "\n".join((ROOT / name).read_text(encoding="utf-8") for name in source_names)
    if "import peft" in text or "from peft " in text:
        package_list.append("peft==0.18.0")
    return package_list


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allocation-start",
        type=float,
        help="Optional original allocation UNIX timestamp; default is first notebook-cell time.",
    )
    args = parser.parse_args()
    if args.allocation_start is not None and args.allocation_start <= 0:
        parser.error("--allocation-start must be a positive UNIX timestamp")

    source_names = sources()
    archive_receipts = selected_inputs()
    pinned_packages = packages(source_names)
    cells = [
        code(
            """
            import csv, hashlib, html, importlib.metadata, json, os, shutil, stat, subprocess, sys, threading, time, traceback, zipfile
            from pathlib import Path
            from google.colab import drive, runtime

            PINNED_ALLOCATION_START = __PINNED_ALLOCATION_START__
            FIRST_CELL_STARTED = time.time()
            ALLOCATION_START = float(PINNED_ALLOCATION_START if PINNED_ALLOCATION_START is not None else FIRST_CELL_STARTED)
            ALLOCATION_LIMIT_SECONDS = 7200
            SAFE_SHUTDOWN_SECONDS = 7080
            SAFE_DEADLINE = ALLOCATION_START + SAFE_SHUTDOWN_SECONDS
            if SAFE_DEADLINE <= FIRST_CELL_STARTED:
                raise TimeoutError("V5 original allocation safe deadline already elapsed")

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
            WORK = Path("/content/dacon_v5")
            WORK.mkdir(exist_ok=True)
            RUN = DRIVE / "runs_compare_v5" / time.strftime("%Y%m%d_%H%M%S", time.gmtime(FIRST_CELL_STARTED))
            RUN.mkdir(parents=True, exist_ok=False)

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
                "deadline_policy": "absolute original allocation clock; setup is included and never resets it",
            })
            write_json(RUN / "notebook_contract.json", {
                "arms": {
                    "B0": "official frozen baseline",
                    "B1": "full voice/music/presence clean",
                    "B2": "same model with augmentation",
                    "B3": "raw/stem temporal",
                    "B4": "nine-state joint temporal",
                },
                "expert_epochs_target": 60,
                "expert_epochs_maximum": 90,
                "actual_budget_truncation": "runner records actual completed epochs, updates, exposure, and any deadline censoring; requested ceilings are not completion claims",
                "sealed_and_ast_policy": "not staged or read",
                "submission_policy": "no submission",
            })
            budget_event("notebook_started")
            print("RESULTS_DIRECTORY", RUN, flush=True)
            """.replace("__PINNED_ALLOCATION_START__", repr(args.allocation_start))
        ),
        nb.v4.new_markdown_cell(
            "# DACON 236749 V5 corrected rerunnable source\n\n"
            "This is corrected rerunnable source, not the executed September 13 run. It executes five receipt-bound comparison arms: B0 official frozen baseline; B1 clean full voice/music/presence; "
            "B2 the same model with augmentation; B3 raw/stem temporal; and B4 nine-state joint temporal. "
            "Experts target 60 epochs with a 90-epoch ceiling. Actual completed work and deadline truncation are runner-written evidence, never inferred."
        ),
        nb.v4.new_markdown_cell(
            "## Data boundaries\n\n"
            "Only V4 train, dev, metadata, and SONICS archives are staged after SHA-256 receipt checks. The V4 sealed archive and AST assets are not staged. "
            "Metadata extraction rejects `sealed.csv`; the original baseline supplies only model, script, and requirements paths."
        ),
        nb.v4.new_markdown_cell(
            "## Embedded execution source\n\n"
            "Each source cell below is compiled by this builder and copied into the run receipt before execution."
        ),
    ]
    cells.extend(embedded_source(name) for name in source_names)
    cells.extend(
        [
            code("INPUT_ARCHIVES = " + repr(archive_receipts)),
            code("PINNED_PACKAGES = " + repr(pinned_packages)),
            nb.v4.new_markdown_cell(
                "## Results\n\n"
                "This corrected rerunnable source displays only JSON tables and images written by its own runner. No result from the September 13 execution is embedded here."
            ),
            code(
                f"""
                BASELINE_BYTES = {BASELINE_BYTES}
                BASELINE_SHA256 = {BASELINE_SHA256!r}
                G0_SHA256 = {G0_SHA256!r}

                def safe_extract(archive, destination, allow_member=None):
                    root = destination.resolve()
                    with zipfile.ZipFile(archive) as packed:
                        members = packed.infolist()
                        for member in members:
                            relative = Path(member.filename)
                            mode = member.external_attr >> 16
                            target = destination / relative
                            if (relative.is_absolute() or ".." in relative.parts or stat.S_ISLNK(mode)
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
                        members = [
                            member for member in packed.infolist()
                            if (not member.is_dir() and not Path(member.filename).is_absolute()
                                and ".." not in Path(member.filename).parts
                                and not stat.S_ISLNK(member.external_attr >> 16)
                                and Path(member.filename).name == "baseline_submit.zip")
                        ]
                        if len(members) != 1 or members[0].file_size != BASELINE_BYTES:
                            raise ValueError(f"Expected one bounded baseline_submit.zip member: {{[(m.filename, m.file_size, m.compress_size) for m in members]}}")
                        with packed.open(members[0]) as incoming, destination.open("wb") as output:
                            shutil.copyfileobj(incoming, output, 8 * 1024**2)
                    if destination.stat().st_size != BASELINE_BYTES or sha256(destination) != BASELINE_SHA256:
                        raise ValueError("Official baseline archive receipt mismatch")
                    return destination

                def capture_executed_notebook():
                    try:
                        from google.colab import _message
                        reply = _message.blocking_request("get_ipynb", timeout_sec=30)
                        payload = reply["ipynb"]
                        if isinstance(payload, str):
                            payload = json.loads(payload)
                        destination = RUN / "executed.ipynb"
                        destination.write_text(json.dumps(payload), encoding="utf-8")
                        backup = RUN / "executed_before_unassign.ipynb"
                        shutil.copy2(destination, backup)
                        write_json(RUN / "executed_notebook_capture.json", {{
                            "status": "captured",
                            "capture_stage": "before_runtime_unassign",
                            "current_cell_may_be_incomplete": True,
                            "sha256": sha256(destination),
                            "backup": backup.name,
                            "backup_sha256": sha256(backup),
                        }})
                    except Exception:
                        (RUN / "executed_notebook_capture_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                        write_json(RUN / "executed_notebook_capture.json", {{"status": "not_captured"}})

                def record_artifacts():
                    receipt = {{}}
                    for path in sorted(RUN.rglob("*")):
                        if path.is_file() and path.name not in {{"executed.ipynb"}}:
                            size = path.stat().st_size
                            receipt[str(path.relative_to(RUN))] = {{"bytes": size, "sha256": sha256(path) if size < 100 * 1024**2 else None,
                                "large_weight_hash_receipt": "See corresponding jobs/*/weights.json" if size >= 100 * 1024**2 else None}}
                    write_json(RUN / "drive_receipt.json", receipt)

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
                            subprocess.run(command, cwd=WORK, stdout=log, stderr=subprocess.STDOUT, check=True,
                                           timeout=min(300, max(1, int(remaining))))
                        write_json(RUN / "analysis_receipt.json", {{"status": "generated", "command": command}})
                        budget_event("analysis_generated")
                    except Exception:
                        (RUN / "analysis_failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
                        write_json(RUN / "analysis_receipt.json", {{"status": "failed", "command": command,
                                                                     "failure": "analysis_failure.txt"}})
                        budget_event("analysis_failed")

                def show_actual_results():
                    summary_paths = [
                        RUN / "comparison_summary.json",
                        RUN / "result_summary.json",
                        RUN / "summary.json",
                        RUN / "comparison.json",
                    ]
                    summary = next((path for path in summary_paths if path.is_file()), None)
                    if summary is None:
                        return
                    payload = json.loads(summary.read_text(encoding="utf-8"))
                    rows = payload.get("arms", payload.get("rows", [])) if isinstance(payload, dict) else payload
                    if isinstance(payload, dict) and not rows:
                        rows = []
                        for arm in ("B0", "B1", "B2", "B3", "B4"):
                            result = payload.get(arm)
                            if not isinstance(result, dict):
                                continue
                            metrics = result.get("metrics", {{}})
                            values = metrics.get("metrics", {{}}) if isinstance(metrics, dict) else {{}}
                            rows.append({{"arm": arm, "status": result.get("status", ""),
                                         "known_subset_proxy": metrics.get("known_subset_proxy", "") if isinstance(metrics, dict) else "", **values}})
                    if isinstance(rows, dict):
                        rows = [{{"arm": name, **value}} if isinstance(value, dict) else {{"arm": name, "value": value}} for name, value in rows.items()]
                    if isinstance(rows, list) and rows and all(isinstance(row, dict) for row in rows):
                        columns = sorted({{key for row in rows for key in row}})
                        body = "".join("<tr>" + "".join(f"<td>{{html.escape(str(row.get(column, '')))}}</td>" for column in columns) + "</tr>" for row in rows)
                        from IPython.display import HTML, Image, display
                        display(HTML("<table><thead><tr>" + "".join(f"<th>{{html.escape(column)}}</th>" for column in columns) + "</tr></thead><tbody>" + body + "</tbody></table>"))
                    else:
                        from IPython.display import JSON, display
                        display(JSON(payload))
                    for image in sorted(RUN.rglob("*.png")):
                        display(Image(filename=str(image)))

                process = None
                try:
                    (RUN / "code").mkdir(exist_ok=True)
                    for source in WORK.glob("*.py"):
                        shutil.copy2(source, RUN / "code" / source.name)
                    source_hashes = {{path.name: sha256(path) for path in sorted((RUN / "code").glob("*.py"))}}
                    write_json(RUN / "source_hashes.json", source_hashes)
                    write_json(RUN / "source_bundle_receipt.json", {{"source_count_py": len(source_hashes), "sources": source_hashes}})
                    write_json(RUN / "input_archives.json", INPUT_ARCHIVES)
                    with (RUN / "install.log").open("w", encoding="utf-8") as log:
                        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PINNED_PACKAGES], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
                    requested = {{package.split("==", 1)[0]: package.split("==", 1)[1] for package in PINNED_PACKAGES}}
                    write_json(RUN / "environment.json", {{
                        "python": sys.version,
                        "requested_packages": requested,
                        "packages": {{name: importlib.metadata.version(name) for name in requested}},
                    }})
                    for archive in INPUT_ARCHIVES:
                        destination = WORK / archive["name"]
                        reusable = destination.is_file() and destination.stat().st_size == archive["bytes"] and sha256(destination) == archive["sha256"]
                        if not reusable:
                            with destination.open("wb") as output:
                                for part in archive["parts"]:
                                    source = DRIVE / part["name"]
                                    if source.stat().st_size != part["bytes"]:
                                        raise ValueError(f"Drive part receipt mismatch: {{source.name}}")
                                    part_hash = hashlib.sha256()
                                    with source.open("rb") as incoming:
                                        while chunk := incoming.read(8 * 1024**2):
                                            part_hash.update(chunk)
                                            output.write(chunk)
                                    if part_hash.hexdigest() != part["sha256"]:
                                        raise ValueError(f"Drive part hash mismatch: {{source.name}}")
                        if destination.stat().st_size != archive["bytes"] or sha256(destination) != archive["sha256"]:
                            raise ValueError(f"Archive receipt mismatch: {{destination.name}}")
                        safe_extract(destination, WORK, (lambda name: Path(name).name != "sealed.csv") if archive["name"] == "v4_metadata.zip" else None)
                        budget_event("archive_verified", archive=destination.name, reused_local_archive=reusable)
                    data = WORK / "v4_validation"
                    if (data / "sealed.csv").exists():
                        raise ValueError("V5 metadata extraction must not materialize sealed.csv")
                    if not all(path.is_file() for path in (data / "train.csv", data / "dev.csv", WORK / "sonics_assets" / "config.json", WORK / "sonics_assets" / "pytorch_model.bin")):
                        raise FileNotFoundError("Verified V5 inputs are incomplete")
                    original = extract_baseline_member(DRIVE / "open.zip", WORK / "embeddedbaseline_submit.zip")
                    baseline = WORK / "baseline"
                    safe_extract(original, baseline, lambda name: Path(name).parts[0] == "model" or Path(name).name == "script.py" or Path(name).name.startswith("requirements"))
                    if not (baseline / "script.py").is_file() or not (baseline / "model").is_dir():
                        raise FileNotFoundError("Official baseline script or model is missing")
                    g0_source = DRIVE / "runs_trainable_v4_matched_recheck" / "20260913_084740" / "G0_predictions.csv"
                    if sha256(g0_source) != G0_SHA256:
                        raise ValueError("Pinned original G0 receipt mismatch")
                    g0 = WORK / "G0_predictions.csv"
                    shutil.copy2(g0_source, g0)
                    if sha256(g0) != G0_SHA256 or len(list(csv.DictReader(g0.open(encoding="utf-8")))) != 210:
                        raise ValueError("Pinned original G0 must be the verified 210-row CSV")
                    command = [sys.executable, "-u", "v5_compare.py", "--work", str(WORK), "--output", str(RUN), "--deadline", str(SAFE_DEADLINE)]
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
                                    budget_event("runner_kill_after_terminate_timeout")
                                    process.kill()
                                    process.wait(timeout=20)
                                raise TimeoutError("V5 runner reached the absolute safe allocation boundary")
                            time.sleep(30)
                            lines = (RUN / "experiment.log").read_text(encoding="utf-8", errors="replace").splitlines()
                            print(lines[-1][:1000] if lines else "Preparing V5 comparison...", flush=True)
                    if process.returncode:
                        raise RuntimeError(f"V5 runner exited with {{process.returncode}}")
                    budget_event("runner_finished")
                    print("V5_SAVED", RUN, flush=True)
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
            nb.v4.new_markdown_cell(
                "## Takeaways\n\n"
                "Read the runner-written arm summaries, `analysis.json`, `REPORT.md`, budget events, source hashes, and receipt manifest before making a comparison claim. This notebook does not submit to DACON."
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
            "v5_source_state": "corrected rerunnable source; not the executed September 13 run",
        },
    )
    nb.validate(notebook)
    nb.write(notebook, NOTEBOOK)


if __name__ == "__main__":
    main()
