"""Notebook-facing stage runner (standard library only; runs in the base kernel).

Stages, in Run All order:
  preflight -> stage_inputs -> environment -> pilot -> features -> train -> test_eval -> parity -> handoff
GPU stages run as child processes of the isolated Python 3.11 venv so the
kernel never holds CUDA memory.  Every stage is resume-safe; any exception
builds DACON_Existing9504_A100_Handoff_FAILED.zip with status "failed" and is
re-raised.  No service start/stop, API keys or submissions happen here.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import constants as C
from .common import (PipelineError, atomic_json, disk_free, human_bytes, human_seconds, now_iso, process_cmdline,
                     publish_file, read_json, sha256_file, tail)

STAGES = ("preflight", "stage_inputs", "environment", "pilot", "features", "train", "test_eval", "parity", "handoff")
GPU_STAGES = {"pilot", "features", "train", "test_eval", "parity"}


def build_config(deployment_root: Path, job_root: Path, code_dir: Path, code_zip: Path | None,
                 notebook_path: Path | None) -> dict[str, Any]:
    job = Path(job_root)
    return {
        "run_name": C.RUN_NAME, "workspace": C.WORKSPACE, "objvol_root": C.OBJVOL_ROOT,
        "deployment_root": str(deployment_root), "job_root": str(job), "code_dir": str(code_dir),
        "code_zip": str(code_zip) if code_zip else None,
        "code_zip_sha256": sha256_file(code_zip) if code_zip and Path(code_zip).is_file() else None,
        "notebook_path": str(notebook_path) if notebook_path else None,
        "expected_gpu": C.EXPECTED_GPU, "device": "cuda", "cuda_visible_devices": "0",
        "dataset_expectations": C.DATASET_EXPECTATIONS, "asset_hashes": C.ASSET_HASHES,
        "source_hashes": C.SOURCE_HASHES, "provenance_csv_sha256": C.PROVENANCE_CSV_SHA256,
        "source_audio_bytes": C.SOURCE_AUDIO_BYTES,
        "manifest": str(job / "inputs" / "manifest.csv"), "data_dir": str(job / "data"),
        "assets_dir": str(job / "assets"), "source_dir": str(job / "source"),
        "env_receipt": str(job / "env" / "receipt.json"), "pilot_dir": str(job / "pilot"),
        "cache_dir": str(job / "cache"), "features_dir": str(job / "features"), "train_dir": str(job / "train"),
        "eval_dir": str(job / "eval"), "parity_dir": str(job / "parity"),
        "max_epochs": 50, "patience": 5, "parity_max_seconds": 90, "target_hours": C.TARGET_HOURS,
        "mert_batch_override": None, "train_benchmark_override": None,
    }


def nvidia_smi(*arguments: str) -> list[list[str]]:
    try:
        result = subprocess.run(["nvidia-smi", *arguments, "--format=csv,noheader,nounits"], capture_output=True,
                                text=True, timeout=60)
    except FileNotFoundError as error:
        raise PipelineError("nvidia-smi not found: this notebook must run on the A100 workspace") from error
    if result.returncode != 0:
        raise PipelineError(f"nvidia-smi failed: {result.stderr.strip()[:500]}")
    return [[cell.strip() for cell in line.split(",")] for line in result.stdout.splitlines() if line.strip()]


def pipeline_children(job: Path) -> list[int]:
    """Our own stage processes (possibly orphaned by a previous kernel)."""
    pids = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit() and int(entry.name) != os.getpid():
            line = process_cmdline(int(entry.name))
            if "-m dacon9504a100." in line and str(job) in line:
                pids.append(int(entry.name))
    return pids


class Pipeline:
    def __init__(self, deployment_root: Path | str = C.DEPLOYMENT_ROOT, job_root: Path | str = C.DEFAULT_JOB_ROOT,
                 code_dir: Path | str | None = None, code_zip: Path | str | None = None,
                 notebook_path: Path | str | None = None, *, config: Mapping[str, Any] | None = None,
                 report_every: float = 300.0) -> None:
        code_dir = Path(code_dir or Path(__file__).resolve().parents[1])
        self.config = dict(config) if config is not None else build_config(
            Path(deployment_root), Path(job_root), code_dir, Path(code_zip) if code_zip else None,
            Path(notebook_path) if notebook_path else None)
        self.job = Path(self.config["job_root"])
        for name in ("status", "logs", "config", "locks"):
            (self.job / name).mkdir(parents=True, exist_ok=True)
        self.config_path = self.job / "config" / "run_config.json"
        atomic_json(self.config_path, self.config)
        self.state_path = self.job / "status" / "run_state.json"
        self.state = read_json(self.state_path) if self.state_path.is_file() else {
            "run_name": self.config["run_name"], "created_at": now_iso(), "stages": {}}
        self.state["status"] = "running"
        self.report_every = report_every
        self._save()

    # ------------------------------------------------------------------ state
    def _save(self) -> None:
        self.state["updated_at"] = now_iso()
        atomic_json(self.state_path, self.state)
        mirror = Path(self.config["deployment_root"]) / "a100_run_status.json"
        try:  # object-volume mirror is best effort; the local state file is authoritative
            publish_file(self.state_path, mirror)
        except (OSError, PipelineError):
            pass

    def log(self, message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with (self.job / "logs" / "orchestrator.log").open("a") as stream:
            stream.write(line + "\n")

    # ------------------------------------------------------------------ public
    def run(self, stage: str) -> Any:
        if stage not in STAGES:
            raise PipelineError(f"Unknown stage {stage}")
        position = STAGES.index(stage)
        for previous in STAGES[:position]:
            if self.state["stages"].get(previous, {}).get("status") != "complete":
                raise PipelineError(f"Stage {stage} requires {previous} to be complete first")
        began = time.monotonic()
        attempts = self.state["stages"].get(stage, {}).get("attempts", [])
        self.state["stages"][stage] = {"status": "running", "started_at": now_iso(), "attempts": attempts}
        self._save()
        self.log(f"=== {stage} ===")
        try:
            result = getattr(self, f"_stage_{stage}")()
        except BaseException as error:
            status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            text = traceback.format_exc()
            seconds = time.monotonic() - began
            self.state["stages"][stage]["attempts"].append({"status": status, "seconds": seconds,
                                                            "finished_at": now_iso()})
            self.state["stages"][stage].update(status=status, finished_at=now_iso(), seconds=seconds,
                                               cumulative_seconds=sum(a["seconds"] for a in
                                                                      self.state["stages"][stage]["attempts"]),
                                               error=text[-8000:])
            self.state["status"] = status
            self._save()
            (self.job / "status" / "failure.txt").write_text(text)
            self.log(f"{stage} {status}: {error}")
            try:
                from .handoff import build
                sidecar = build(self.config, "failed", self.state, text, log=self.log)
                self.log(f"failure ZIP: {sidecar['file']}")
            except Exception as packaging_error:  # never mask the original error
                self.log(f"failure ZIP could not be built: {packaging_error!r}")
            raise
        seconds = time.monotonic() - began
        self.state["stages"][stage]["attempts"].append({"status": "complete", "seconds": seconds, "finished_at": now_iso()})
        self.state["stages"][stage].update(status="complete", finished_at=now_iso(), seconds=seconds,
                                           cumulative_seconds=sum(a["seconds"] for a in
                                                                  self.state["stages"][stage]["attempts"]))
        if stage == "handoff":
            self.state["status"] = "complete"
        self._save()
        self.log(f"{stage} complete in {human_seconds(time.monotonic() - began)}; "
                 f"elapsed stage total {human_seconds(self.total_seconds())}")
        return result

    def total_seconds(self) -> float:
        """Measured wall time over every attempt of every stage (resumed work included once per attempt)."""
        return sum(attempt["seconds"] for value in self.state["stages"].values() for attempt in value.get("attempts", []))

    # ------------------------------------------------------------------ helpers
    def _venv_python(self) -> Path:
        return self.job / "env" / "venv" / "bin" / "python"

    def _child_env(self) -> dict[str, str]:
        from .environment import child_environment
        receipt = read_json(Path(self.config["env_receipt"]))
        return child_environment(self.config, receipt.get("ffmpeg_ld_library_path"))

    def _gpu_idle(self) -> None:
        stale = pipeline_children(self.job)
        for pid in stale:
            self.log(f"terminating orphaned stage process {pid}: {process_cmdline(pid)[:200]}")
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 60
        while pipeline_children(self.job) and time.monotonic() < deadline:
            time.sleep(2)
        for pid in pipeline_children(self.job):
            os.kill(pid, signal.SIGKILL)
        time.sleep(2)
        apps = nvidia_smi("--query-compute-apps=pid,process_name,used_memory")
        if apps:
            raise PipelineError(f"GPU already has compute processes (would distort VRAM measurement/OOM): {apps}")

    def _child(self, stage: str, module: str, *arguments: str, progress: Path | None = None) -> None:
        self._gpu_idle()
        log_path = self.job / "logs" / f"{stage}.log"
        command = [str(self._venv_python()), "-m", f"dacon9504a100.{module}", *arguments, "--config",
                   str(self.config_path)]
        self.log("$ " + " ".join(command))
        with log_path.open("a") as log:
            log.write(f"\n===== {now_iso()} {' '.join(command)}\n")
            log.flush()
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=self._child_env(),
                                       cwd=str(self.job), start_new_session=True)
        last_report = time.monotonic()
        try:
            while process.poll() is None:
                time.sleep(5)
                if time.monotonic() - last_report >= self.report_every:
                    last_report = time.monotonic()
                    self._report(stage, log_path, progress)
        except BaseException:
            self.log(f"stopping {stage} child pid={process.pid}")
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=60)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            raise
        if process.returncode != 0:
            raise PipelineError(f"{stage} child exited with {process.returncode}; last log lines:\n{tail(log_path, 60)}")
        self._report(stage, log_path, progress)

    def _report(self, stage: str, log_path: Path, progress: Path | None) -> None:
        text = tail(log_path, 1)
        if progress is not None and progress.is_file():
            try:
                value = read_json(progress)
            except (OSError, ValueError):
                value = {}
            if stage == "features" and "done" in value:
                memory = value.get("memory", {})
                text = (f"{value['done']}/{value['total']} rows; ETA {human_seconds(value.get('eta_seconds', 0))} "
                        f"(observed/pilot x{value.get('observed_over_pilot_ratio', 0):.2f}); "
                        f"peak VRAM {human_bytes(memory.get('max_allocated', 0))}; last {value.get('last_id')}")
        self.state["stages"][stage]["last_report"] = text[-500:]
        self._save()
        self.log(f"[{stage}] {text}")

    # ------------------------------------------------------------------ stages
    def _stage_preflight(self) -> dict[str, Any]:
        deployment = Path(self.config["deployment_root"])
        report: dict[str, Any] = {"kernel_python": sys.version, "executable": sys.executable,
                                  "deployment_root": str(deployment), "job_root": str(self.job)}
        if sys.version_info < (3, 10):
            raise PipelineError("Kernel Python >= 3.10 is required for the orchestrator")
        if not Path(self.config["objvol_root"]).is_dir():
            raise PipelineError(f"Object volume {self.config['objvol_root']} is not mounted in this workspace")
        if not (deployment / "deployment.json").is_file():
            raise PipelineError(f"deployment.json not found in {deployment}")
        gpus = nvidia_smi("--query-gpu=index,name,memory.total,driver_version,pci.bus_id")
        report["gpus"] = gpus
        if not gpus or self.config["expected_gpu"] not in gpus[0][1]:
            raise PipelineError(f"GPU 0 must be an {self.config['expected_gpu']}: {gpus}")
        self._gpu_idle()
        from .staging import load_deployment
        plan = load_deployment(deployment)
        report["deployment_keys"] = sorted(plan)
        report["deployment_files"] = len(plan["files"])
        from .staging import find_file
        assets_bytes = int(plan["files"].get("assets.zip", {}).get("bytes", 7_000_000_000))
        largest_shard = 0
        upload = find_file(plan, Path(self.config["objvol_root"]), deployment, ("audio_upload.json",))
        if upload is not None and upload.is_file():
            largest_shard = max((int(s["bytes"]) for s in read_json(upload).get("shards", [])), default=0)
        # audio + extracted assets + transient shard copy + venv (~7.5 GB) + transient uv cache (~8 GB)
        # + feature cache (~4 GB) + local handoff (~assets + 1 GB) + 5 GB margin
        need = (C.SOURCE_AUDIO_BYTES + assets_bytes + largest_shard + 7_500_000_000 + 8_000_000_000
                + 4_000_000_000 + assets_bytes + 1_000_000_000 + 5_000_000_000)
        free = disk_free(self.job)
        report.update(job_free_bytes=free, job_required_bytes=need, largest_shard_bytes=largest_shard,
                      assets_zip_bytes=assets_bytes)
        if free < need:
            raise PipelineError(f"{self.job} has {human_bytes(free)} free; need about {human_bytes(need)}")
        report["deployment_free_bytes"] = disk_free(deployment)
        network = {}
        for url in ("https://pypi.org/simple/uv/", "https://files.pythonhosted.org/",
                    "https://github.com/astral-sh/python-build-standalone/releases"):
            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    network[url] = response.status
            except Exception as error:  # recorded; only the environment stage needs packages
                network[url] = f"{type(error).__name__}: {error}"[:200]
        report["network"] = network
        env_ready = Path(self.config["env_receipt"]).is_file()
        if not env_ready and network["https://pypi.org/simple/uv/"] != 200:
            raise PipelineError(f"PyPI is unreachable and no verified environment exists: {network}")
        report["started_at"] = now_iso()
        atomic_json(self.job / "status" / "preflight.json", report)
        self.log(f"GPU {gpus[0][1]} {gpus[0][2]} MiB driver {gpus[0][3]}; free {human_bytes(free)}")
        return report

    def _stage_stage_inputs(self) -> dict[str, Any]:
        from .staging import stage_inputs
        receipt = stage_inputs(self.config, log=self.log)
        report = receipt["manifest_report"]
        self.log(f"manifest verified: {report['splits']} FILE_FAKE {report['file_fake_by_split']}; "
                 f"exposure {report['historical_exposure_by_split']}; singing {report['singing_by_split']}")
        return receipt

    def _stage_environment(self) -> dict[str, Any]:
        from .environment import ensure_environment
        receipt = ensure_environment(self.config, log=self.log)
        probe = receipt["probe"]
        self.log(f"venv {probe['python'].split()[0]} torch {probe['torch']} (CUDA {probe['torch_cuda']}) on "
                 f"{probe['gpu']['name']}; ffmpeg libs {probe.get('ffmpeg_libraries', {}).get('ffmpeg_version')}")
        return receipt

    def _stage_pilot(self) -> dict[str, Any]:
        receipt_path = Path(self.config["pilot_dir"]) / "pilot_receipt.json"
        if receipt_path.is_file():
            receipt = read_json(receipt_path)
            self.log("pilot receipt exists; reusing its measured batch and projection")
        else:
            self._child("pilot", "features", "pilot")
            receipt = read_json(receipt_path)
        projection = receipt["projection"]
        elapsed = self.total_seconds()
        self.log(f"A100 pilot: MERT batch {receipt['mert_batch']}; peak VRAM "
                 f"{human_bytes(receipt['memory'].get('max_allocated', 0))}/{human_bytes(receipt['memory'].get('total', 0))}; "
                 f"projected features {projection['projected_feature_hours']:.2f} h; "
                 f"stages so far {human_seconds(elapsed)}; target {self.config['target_hours']} h "
                 f"(projection only; training/eval time measured later, nothing is dropped)")
        return receipt

    def _stage_features(self) -> dict[str, Any]:
        complete = Path(self.config["features_dir"]) / "features_complete.json"
        self._child("features", "features", "full", progress=Path(self.config["features_dir"]) / "progress.json")
        return read_json(complete)

    def _stage_train(self) -> dict[str, Any]:
        self._child("train", "training", progress=Path(self.config["train_dir"]) / "progress.json")
        lock_path = Path(self.config["train_dir"]) / "candidate_lock.json"
        lock = read_json(lock_path)
        evidence = Path(self.config["deployment_root"]) / "a100_run_evidence"
        published = publish_file(lock_path, evidence / "candidate_lock.json")
        self.state["lock_publication"] = published
        self.log(f"candidate lock {sha256_file(lock_path)} published before test evaluation; "
                 f"test feature files opened before lock: {lock['test_rows_read_before_lock']}")
        return lock

    def _stage_test_eval(self) -> dict[str, Any]:
        self._child("test_eval", "evaluation", "test")
        receipt = read_json(Path(self.config["eval_dir"]) / "test_receipt.json")
        metrics = read_json(Path(self.config["eval_dir"]) / "test_metrics.json")
        heads = metrics["heads"]
        self.log("test (masked, not independent): " + "; ".join(
            f"{h} EER {heads[h]['eer']:.4f}" for h in ("file", "voice", "music") if heads[h].get("eer") is not None))
        return receipt

    def _stage_parity(self) -> dict[str, Any]:
        self._child("parity", "parity")
        receipt = read_json(Path(self.config["parity_dir"]) / "parity_receipt.json")
        self.log(f"raw-audio parity status: {receipt['status']} over {len(receipt['samples'])} samples "
                 f"and {len(receipt['pilot_vs_full'])} pilot rows")
        return receipt

    def _stage_handoff(self) -> dict[str, Any]:
        from .handoff import build
        self.state["status"] = "complete"
        sidecar = build(self.config, "complete", self.state, None, log=self.log)
        self.state["handoff"] = sidecar
        return sidecar
