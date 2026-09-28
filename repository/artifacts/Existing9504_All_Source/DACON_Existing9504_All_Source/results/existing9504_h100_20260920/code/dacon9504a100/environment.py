"""Isolated, hash-pinned Python 3.11 environment for every GPU stage.

The notebook kernel (base image Python) only orchestrates; all torch work runs
in JOB/env/venv created here:
  1. uv 0.8.17 from its PyPI wheel (SHA-256 pinned; no pip needed),
  2. CPython 3.11.13 via ``uv python install`` (fallback: an existing python3.11),
  3. ``uv pip install --require-hashes -r requirements.lock.txt`` (reference pins:
     torch/torchaudio 2.9.1, torchcodec 0.8.1, transformers 4.57.6, numpy 2.2.6, ...),
  4. exact freeze comparison and an in-venv probe (H100, CUDA, imports, the
     torchaudio->torchcodec decoder the pinned HTDemucs loader uses).
torchcodec needs FFmpeg shared libraries; if the image lacks them they are
provisioned (apt-get as root, else conda-forge via a pinned micromamba) and the
choice is recorded.  fairseq is deliberately absent: DF-Arena uses
transformers.Wav2Vec2Model.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import textwrap
import urllib.request
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .common import PipelineError, atomic_json, now_iso, read_json, sha256_file
from .constants import (CONDA_FFMPEG_SPEC, MICROMAMBA_VERSION, PYTHON_VERSION, UV_VERSION, UV_WHEEL_SHA256,
                        UV_WHEEL_URL)

PACKAGE_DIR = Path(__file__).resolve().parent
LOCK_FILE = PACKAGE_DIR / "requirements.lock.txt"
PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s\\;]+)")
IMPORTS = ("torch", "torchaudio", "torchcodec", "transformers", "librosa", "soundfile", "demucs.apply",
           "demucs.pretrained", "demucs.separate", "demucs.audio", "panns_inference", "sklearn", "scipy.special",
           "numpy", "einops", "safetensors", "imageio_ffmpeg", "tqdm")


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def lock_pins(path: Path = LOCK_FILE) -> dict[str, str]:
    pins = {}
    for line in path.read_text().splitlines():
        match = PIN.match(line.strip())
        if match:
            pins[_normalize(match.group(1))] = match.group(2)
    if not pins:
        raise PipelineError("requirements.lock.txt contains no pins")
    return pins


def run(command: list[str], *, env: Mapping[str, str] | None = None, log=print, timeout: float | None = None,
        check: bool = True) -> subprocess.CompletedProcess:
    log("[env] $ " + " ".join(command))
    result = subprocess.run(command, env=None if env is None else dict(env), capture_output=True, text=True,
                            timeout=timeout)
    if check and result.returncode != 0:
        raise PipelineError(f"Command failed ({result.returncode}): {' '.join(command)}\n"
                            f"{result.stdout[-4000:]}\n{result.stderr[-4000:]}")
    return result


def bootstrap_uv(env_dir: Path, log=print) -> Path:
    binary = env_dir / "bin" / "uv"
    if binary.is_file():
        version = run([str(binary), "--version"], log=log).stdout.strip()
        if version.startswith(f"uv {UV_VERSION}"):
            return binary
        binary.unlink()
    wheel = env_dir / "downloads" / UV_WHEEL_URL.rsplit("/", 1)[1]
    wheel.parent.mkdir(parents=True, exist_ok=True)
    if not wheel.is_file() or sha256_file(wheel) != UV_WHEEL_SHA256:
        log(f"[env] downloading uv {UV_VERSION} wheel from PyPI")
        partial = wheel.with_suffix(".partial")
        with urllib.request.urlopen(UV_WHEEL_URL, timeout=120) as response, partial.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        if sha256_file(partial) != UV_WHEEL_SHA256:
            partial.unlink()
            raise PipelineError("uv wheel SHA-256 mismatch")
        partial.replace(wheel)
    with zipfile.ZipFile(wheel) as archive:
        member = f"uv-{UV_VERSION}.data/scripts/uv"
        binary.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source, binary.with_suffix(".partial").open("wb") as target:
            shutil.copyfileobj(source, target)
    binary.with_suffix(".partial").replace(binary)
    binary.chmod(0o755)
    version = run([str(binary), "--version"], log=log).stdout.strip()
    if not version.startswith(f"uv {UV_VERSION}"):
        raise PipelineError(f"Unexpected uv version: {version}")
    return binary


def uv_env(env_dir: Path) -> dict[str, str]:
    environment = dict(os.environ)
    for key in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "CONDA_PREFIX", "UV_PYTHON", "UV_INDEX_URL",
                "UV_EXTRA_INDEX_URL", "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "FORCE_COLOR", "CLICOLOR_FORCE"):
        environment.pop(key, None)
    environment.update(UV_CACHE_DIR=str(env_dir / "uv_cache"), UV_PYTHON_INSTALL_DIR=str(env_dir / "python"),
                       UV_NO_CONFIG="1", UV_PYTHON_DOWNLOADS="manual", UV_HTTP_TIMEOUT="300", NO_COLOR="1")
    return environment


def install_python(uv: Path, env_dir: Path, log=print) -> tuple[Path, str]:
    environment = uv_env(env_dir)
    result = run([str(uv), "python", "install", PYTHON_VERSION], env=environment, log=log, check=False,
                 timeout=1800)
    if result.returncode == 0:
        found = run([str(uv), "python", "find", "--python-preference", "only-managed", PYTHON_VERSION],
                    env=environment, log=log).stdout.strip()
        return Path(found), f"uv-managed CPython {PYTHON_VERSION} (python-build-standalone)"
    log("[env] uv python install failed; looking for an existing python3.11\n" + result.stderr[-2000:])
    for candidate in ("python3.11", "/usr/bin/python3.11", "/usr/local/bin/python3.11"):
        path = shutil.which(candidate)
        if path:
            version = run([path, "-c", "import sys;print('%d.%d.%d'%sys.version_info[:3])"], log=log).stdout.strip()
            if version.startswith("3.11."):
                return Path(path), f"existing system CPython {version} (uv download unavailable)"
    raise PipelineError("No Python 3.11 available: uv python install failed and no python3.11 exists")


def freeze(uv: Path, python: Path, env_dir: Path, log=print) -> dict[str, str]:
    output = run([str(uv), "pip", "freeze", "--python", str(python)], env=uv_env(env_dir), log=log).stdout
    result = {}
    for line in output.splitlines():
        match = PIN.match(line.strip())
        if match:
            result[_normalize(match.group(1))] = match.group(2)
    return result


PROBE = textwrap.dedent('''
    import importlib, importlib.metadata as md, json, os, subprocess, sys, tempfile, time
    expected = sys.argv[1]
    report = {"python": sys.version, "executable": sys.executable, "imports": {}}
    for name in json.loads(sys.argv[2]):
        try:
            importlib.import_module(name)
            report["imports"][name] = "ok"
        except Exception as error:
            report["imports"][name] = f"{type(error).__name__}: {error}"[:600]
    import numpy as np
    import torch
    report["torch"] = torch.__version__
    report["torch_cuda"] = torch.version.cuda
    report["cudnn"] = torch.backends.cudnn.version()
    report["cuda_available"] = torch.cuda.is_available()
    if report["cuda_available"]:
        properties = torch.cuda.get_device_properties(0)
        report["gpu"] = {"name": torch.cuda.get_device_name(0), "count": torch.cuda.device_count(),
                         "capability": list(torch.cuda.get_device_capability(0)),
                         "total_memory": properties.total_memory, "multiprocessors": properties.multi_processor_count}
    try:
        import soundfile as sf, torchaudio, torchcodec
        from torchcodec._core import get_ffmpeg_library_versions
        report["ffmpeg_libraries"] = get_ffmpeg_library_versions()
        signal = (np.random.default_rng(0).standard_normal(16000) * 0.1).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "probe.wav")
            sf.write(path, signal, 16000, subtype="FLOAT")
            waveform, rate = torchaudio.load(path)
        report["torchaudio_torchcodec_decode"] = {"ok": rate == 16000 and waveform.shape == (1, 16000)
                                                  and float((waveform[0].numpy() - signal).__abs__().max()) == 0.0}
    except Exception as error:
        report["torchaudio_torchcodec_decode"] = {"ok": False, "error": f"{type(error).__name__}: {error}"[:1500]}
    import imageio_ffmpeg
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    report["imageio_ffmpeg"] = {"exe": exe, "version": subprocess.run([exe, "-version"], capture_output=True,
                                text=True).stdout.splitlines()[0]}
    report["distributions"] = {d.metadata["Name"]: d.version for d in md.distributions()}
    print("PROBE_JSON=" + json.dumps(report))
''')


def probe(python: Path, environment: Mapping[str, str], expected_gpu: str | None, log=print) -> dict[str, Any]:
    result = run([str(python), "-c", PROBE, expected_gpu or "", json.dumps(IMPORTS)], env=environment, log=log,
                 timeout=900)
    lines = [line for line in result.stdout.splitlines() if line.startswith("PROBE_JSON=")]
    if not lines:
        raise PipelineError("Environment probe produced no report:\n" + result.stdout[-3000:] + result.stderr[-3000:])
    return json.loads(lines[-1].removeprefix("PROBE_JSON="))


def provision_ffmpeg(env_dir: Path, log=print) -> list[dict[str, Any]]:
    """Return candidate library settings, in order: system apt, then conda-forge."""
    attempts: list[dict[str, Any]] = []
    if shutil.which("apt-get") and hasattr(os, "geteuid") and os.geteuid() == 0:
        environment = dict(os.environ, DEBIAN_FRONTEND="noninteractive")
        options = ["-o", "Acquire::Retries=2", "-o", "Acquire::http::Timeout=60"]
        update = run(["apt-get", *options, "update"], env=environment, log=log, check=False, timeout=900)
        install = run(["apt-get", *options, "install", "-y", "--no-install-recommends", "ffmpeg"],
                      env=environment, log=log, check=False, timeout=1800) if update.returncode == 0 else None
        attempts.append({"method": "apt-get install ffmpeg", "ld_library_path": None,
                         "ok": bool(install and install.returncode == 0),
                         "stderr": ((install or update).stderr or "")[-1500:]})
    prefix = env_dir / "ffmpeg"
    try:
        binary = env_dir / "bin" / "micromamba"
        if not binary.is_file():
            base = f"https://github.com/mamba-org/micromamba-releases/releases/download/{MICROMAMBA_VERSION}/micromamba-linux-64"
            partial = binary.with_suffix(".partial")
            binary.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(base, timeout=300) as response, partial.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            with urllib.request.urlopen(base + ".sha256", timeout=60) as response:
                published = response.read().decode().split()[0].strip()
            if sha256_file(partial) != published:
                raise PipelineError("micromamba SHA-256 differs from its published checksum")
            partial.replace(binary)
            binary.chmod(0o755)
        if not (prefix / "lib").is_dir():
            run([str(binary), "create", "-y", "-p", str(prefix), "-c", "conda-forge", "--override-channels",
                 CONDA_FFMPEG_SPEC], env=dict(os.environ, MAMBA_ROOT_PREFIX=str(env_dir / "mamba")), log=log,
                timeout=3600)
        attempts.append({"method": f"micromamba {MICROMAMBA_VERSION} conda-forge {CONDA_FFMPEG_SPEC}",
                         "ld_library_path": str(prefix / "lib"), "ok": True})
    except Exception as error:  # recorded; the caller decides whether any attempt worked
        attempts.append({"method": "micromamba conda-forge ffmpeg", "ld_library_path": None, "ok": False,
                         "error": f"{type(error).__name__}: {error}"[:1500]})
    return attempts


def child_environment(config: Mapping[str, Any], ld_library_path: str | None = None) -> dict[str, str]:
    job = Path(config["job_root"])
    home = job / "home"
    for name in ("", ".cache", "hf", "torch", "numba", "mpl", "panns_data"):
        (home / name).mkdir(parents=True, exist_ok=True)
    # panns_inference reads ~/panns_data/class_labels_indices.csv at import time and
    # would try wget otherwise; place the hash-verified staged asset (baseline does the same).
    labels = job / "assets" / "baseline" / "model" / "panns" / "class_labels_indices.csv"
    target = home / "panns_data" / "class_labels_indices.csv"
    if labels.is_file() and (not target.is_file() or target.read_bytes() != labels.read_bytes()):
        shutil.copyfile(labels, target)
    environment = {key: value for key, value in os.environ.items()
                   if key not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX", "PYTHONSTARTUP",
                                  "JPY_PARENT_PID", "MPLBACKEND")}
    venv = job / "env" / "venv"
    environment.update(
        PATH=os.pathsep.join([str(job / "bin"), str(venv / "bin"), os.environ.get("PATH", "/usr/bin:/bin")]),
        VIRTUAL_ENV=str(venv), HOME=str(home), XDG_CACHE_HOME=str(home / ".cache"), HF_HOME=str(home / "hf"),
        TORCH_HOME=str(home / "torch"), NUMBA_CACHE_DIR=str(home / "numba"), MPLCONFIGDIR=str(home / "mpl"),
        MPLBACKEND="Agg", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
        HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
        PYTHONPATH=os.pathsep.join([str(config["code_dir"]), str(job / "source")]),
        PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0",
        OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", CUBLAS_WORKSPACE_CONFIG=":4096:8",
        CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(config.get("cuda_visible_devices", "0")))
    if ld_library_path:
        # Prepend: conda-forge FFmpeg may need its newer libstdc++; the image's
        # own LD_LIBRARY_PATH (often the mounted NVIDIA driver) is preserved.
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(
            [ld_library_path, *[p for p in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p]])
    return environment


def link_static_ffmpeg(job: Path, exe: str) -> dict[str, str]:
    """Expose imageio-ffmpeg's static binary as `ffmpeg` (original full_bank_run.decoder_path)."""
    target = Path(exe).resolve()
    link = job / "bin" / "ffmpeg"
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        if link.resolve() != target:
            link.unlink()
    if not link.exists():
        link.symlink_to(target)
    return {"path": str(link), "target": str(target), "sha256": sha256_file(target)}


def ensure_environment(config: Mapping[str, Any], log=print) -> dict[str, Any]:
    job = Path(config["job_root"])
    env_dir = job / "env"
    env_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = env_dir / "receipt.json"
    lock_sha = sha256_file(LOCK_FILE)
    pins = lock_pins()
    uv = bootstrap_uv(env_dir, log=log)
    venv = env_dir / "venv"
    python = venv / "bin" / "python"
    previous = read_json(receipt_path) if receipt_path.is_file() else {}
    reuse = (previous.get("lock_sha256") == lock_sha and python.is_file()
             and freeze(uv, python, env_dir, log=log) == pins)
    if reuse:
        interpreter, interpreter_source = Path(previous["base_python"]), previous["base_python_source"]
        log("[env] reusing verified virtual environment")
    else:
        interpreter, interpreter_source = install_python(uv, env_dir, log=log)
        if venv.exists():
            shutil.rmtree(venv)
        run([str(uv), "venv", "--python", str(interpreter), str(venv)], env=uv_env(env_dir), log=log)
        run([str(uv), "pip", "install", "--python", str(python), "--require-hashes", "--no-deps", "--strict",
             "-r", str(LOCK_FILE)], env=uv_env(env_dir), log=log, timeout=7200)
    installed = freeze(uv, python, env_dir, log=log)
    if installed != pins:
        difference = {name: (pins.get(name), installed.get(name)) for name in set(pins) | set(installed)
                      if pins.get(name) != installed.get(name)}
        raise PipelineError(f"Installed packages differ from the lock: {difference}")
    ld_library_path = previous.get("ffmpeg_ld_library_path") if reuse else None
    report = probe(python, child_environment(config, ld_library_path), config["expected_gpu"], log=log)
    provisioning: list[dict[str, Any]] = previous.get("ffmpeg_provisioning", []) if reuse else []
    if not report["torchaudio_torchcodec_decode"]["ok"]:
        log("[env] torchaudio->torchcodec cannot load FFmpeg libraries; provisioning FFmpeg\n"
            + json.dumps(report["torchaudio_torchcodec_decode"])[:1500])
        for attempt in provision_ffmpeg(env_dir, log=log):
            provisioning.append(attempt)
            if not attempt["ok"]:
                continue
            report = probe(python, child_environment(config, attempt["ld_library_path"]), config["expected_gpu"],
                           log=log)
            if report["torchaudio_torchcodec_decode"]["ok"]:
                ld_library_path = attempt["ld_library_path"]
                break
    failures = {name: status for name, status in report["imports"].items() if status != "ok"}
    if failures:
        raise PipelineError(f"Environment imports failed: {failures}")
    if not report["torchaudio_torchcodec_decode"]["ok"]:
        raise PipelineError("torchaudio.load (torchcodec) cannot decode even after FFmpeg provisioning: "
                            + json.dumps({"probe": report["torchaudio_torchcodec_decode"],
                                          "attempts": provisioning})[:3000])
    gpu = report.get("gpu") or {}
    if not report["cuda_available"] or config["expected_gpu"] not in gpu.get("name", ""):
        raise PipelineError(f"Required {config['expected_gpu']} not visible to torch: {gpu or 'CUDA unavailable'}")
    static_ffmpeg = link_static_ffmpeg(job, report["imageio_ffmpeg"]["exe"])
    receipt = {"lock_sha256": lock_sha, "lock_file": str(LOCK_FILE), "uv_version": UV_VERSION,
               "base_python": str(interpreter), "base_python_source": interpreter_source,
               "venv_python": str(python), "packages": installed, "probe": report,
               "ffmpeg_ld_library_path": ld_library_path, "ffmpeg_provisioning": provisioning,
               "static_ffmpeg": static_ffmpeg, "kernel_python": sys.version, "platform": platform.platform(),
               "reused": reuse, "verified_at": now_iso()}
    atomic_json(receipt_path, receipt)
    cache = env_dir / "uv_cache"
    if cache.exists():
        shutil.rmtree(cache, ignore_errors=True)
    return receipt
