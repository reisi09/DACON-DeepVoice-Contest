#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# Run: .venv/bin/python candidate_data_acquire.py PLAN.json
"""Retain checksum-verified source files without admitting them to training."""
from __future__ import annotations

import fcntl
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Final
from urllib.parse import urlparse


ROOT: Final = Path(__file__).resolve().parent
BANK: Final = ROOT / "_workspace/candidate_acquisition_20260919"
OUT: Final = ROOT / "results/candidate_acquisition_20260919"
RESERVE: Final = 30 * 2**30


@dataclass(frozen=True, slots=True)
class SourceFile:
    """Upstream identity and checksum are mandatory, even for an existing file."""

    dataset: str
    path: str
    url: str
    size: int
    checksum: str
    license: str
    source_split: str


def target_path(job: SourceFile) -> Path:
    """Reject escaped output paths or unexpected download origins."""
    target = (BANK / job.path).resolve()
    if not target.is_relative_to(BANK.resolve()) or target == BANK.resolve():
        raise RuntimeError(f"Output path outside bank: {job.path}")
    parsed = urlparse(job.url)
    if parsed.scheme != "https" or parsed.hostname not in {"zenodo.org", "huggingface.co"}:
        raise RuntimeError(f"Unapproved source origin: {job.url}")
    if job.size <= 0:
        raise RuntimeError(f"Invalid upstream byte count: {job.path}")
    return target


def verify_file(path: Path, job: SourceFile) -> str:
    """Catch truncation and corruption before promotion from .part to original."""
    if path.stat().st_size != job.size:
        raise RuntimeError(f"Size mismatch: {job.path}")
    algorithm, expected = job.checksum.split(":", 1)
    if algorithm not in {"md5", "sha256", "git-sha1"}:
        raise RuntimeError(f"Unsupported source checksum: {algorithm}")
    upstream = hashlib.new("sha1" if algorithm == "git-sha1" else algorithm)
    if algorithm == "git-sha1":
        upstream.update(f"blob {job.size}\0".encode())
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 2**20):
            upstream.update(chunk)
            sha.update(chunk)
    if upstream.hexdigest() != expected:
        raise RuntimeError(f"Upstream checksum mismatch: {job.path}")
    return sha.hexdigest()


def acquire(job: SourceFile) -> dict[str, str | int | bool]:
    """Resume only temporary files; refuse to replace any damaged original."""
    target = target_path(job)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_name(target.name + ".download.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return acquire_locked(job, target)


def acquire_locked(job: SourceFile, target: Path) -> dict[str, str | int | bool]:
    """The caller holds the per-original lock throughout transfer and verification."""
    partial = target.with_name(target.name + ".part")
    if not target.exists():
        print(f"download {job.path} ({job.size:,} bytes)", flush=True)
        for attempt in range(8):
            # A closed connection may leave a complete file; never request its empty tail.
            if partial.exists() and partial.stat().st_size >= job.size:
                break
            try:
                subprocess.run([
                    "curl", "--fail", "--location", "--silent", "--show-error",
                    "--proto", "=https", "--proto-redir", "=https",
                    "--connect-timeout", "30", "--speed-time", "180", "--speed-limit", "1024",
                    "--retry", "0", "--continue-at", "-",
                    # HF redirect bodies can exceed tiny JSON files; final size stays exact-checked.
                    "--max-filesize", str(max(job.size, 2**20)), "--output", str(partial), job.url,
                ], check=True)
                break
            except subprocess.CalledProcessError as error:
                if error.returncode not in (5, 6, 7, 18, 22, 28, 52, 55, 56) or attempt == 7:
                    raise
                retained = partial.stat().st_size if partial.exists() else 0
                print(f"retry {job.path}: curl={error.returncode}, retained={retained:,} bytes", flush=True)
                time.sleep(min(60, 15 * (attempt + 1)))
        digest = verify_file(partial, job)
        partial.rename(target)
    else:
        digest = verify_file(target, job)
    print(f"verified {job.path}", flush=True)
    return {**asdict(job), "sha256": digest, "status": "original_checksum_verified",
            "training_admitted": False, "verified_utc": datetime.now(timezone.utc).isoformat()}


def main(plan: Path) -> None:
    """Serialize bank writers; leave partial downloads available for a later retry."""
    OUT.mkdir(parents=True, exist_ok=True)
    BANK.mkdir(parents=True, exist_ok=True)
    jobs = [SourceFile(**row) for row in json.loads(plan.read_text())]
    paths = [target_path(job) for job in jobs]
    if len(set(paths)) != len(paths):
        raise RuntimeError("Multiple jobs target the same original")
    with (BANK / (plan.stem + ".lock")).open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        required = sum(max(0, job.size - (path.with_name(path.name + ".part").stat().st_size
                       if path.with_name(path.name + ".part").exists() else 0))
                       for job, path in zip(jobs, paths) if not path.exists())
        if shutil.disk_usage(BANK).free < required + RESERVE:
            raise RuntimeError(f"Insufficient physical disk space for {required:,} pending bytes plus OS reserve")
        failures = []
        with (OUT / (plan.stem + "_receipts.jsonl")).open("a") as receipt, ThreadPoolExecutor(max_workers=4) as pool:
            pending = {pool.submit(acquire, job): job for job in jobs}
            for future in as_completed(pending):
                job = pending[future]
                try:
                    result = future.result()
                except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
                    failures.append(job.path)
                    result = {"path": job.path, "status": "failed", "error": str(error)}
                    print(f"failed {job.path}: {error}", flush=True)
                receipt.write(json.dumps(result, ensure_ascii=False) + "\n")
                receipt.flush()
        if failures:
            raise RuntimeError(f"Retry incomplete originals: {failures}")


def self_check() -> None:
    """Protect against accepting same-size corruption or writing outside the bank."""
    job = SourceFile("check", "check.bin", "https://zenodo.org/check", 4,
                     "sha256:" + hashlib.sha256(b"good").hexdigest(), "test", "unassigned")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "check.bin"
        path.write_bytes(b"good")
        assert verify_file(path, job) == hashlib.sha256(b"good").hexdigest()
        git_hash = hashlib.sha1(b"blob 4\0good").hexdigest()
        assert verify_file(path, replace(job, checksum="git-sha1:" + git_hash)) == verify_file(path, job)
        path.write_bytes(b"evil")
        try:
            verify_file(path, job)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Same-size corruption was accepted")
    try:
        target_path(replace(job, path="../../outside"))
    except RuntimeError:
        pass
    else:
        raise AssertionError("Output path escaped the bank")
    print("Same-size corruption and output path escape rejected.")


if __name__ == "__main__":
    if sys.argv[1] == "--self-check":
        self_check()
    else:
        main(Path(sys.argv[1]))
