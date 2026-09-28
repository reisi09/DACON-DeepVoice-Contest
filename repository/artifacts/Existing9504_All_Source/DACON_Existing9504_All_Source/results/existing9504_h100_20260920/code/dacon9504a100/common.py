"""Standard-library helpers shared by the kernel-side and child-side stages.

Nothing here imports numpy/torch so the notebook kernel (base image Python)
can use it before the isolated environment exists.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import fcntl
import hashlib
import json
import os
import shutil
import stat
import time
import zipfile
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

CHUNK = 8 * 1024 * 1024


class PipelineError(RuntimeError):
    """A contract, integrity or safety check failed; the run must stop."""


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _fsync_directory(directory: Path) -> None:
    with contextlib.suppress(OSError):
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def atomic_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def json_safe(value: Any) -> Any:
    """Replace NaN/inf floats by strings so evidence can be written with allow_nan=False."""
    import math
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    if isinstance(value, Mapping):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def copy_with_hash(source: Path, target: Path, *, expected_bytes: int | None = None,
                   expected_sha256: str | None = None) -> tuple[int, str]:
    """Stream-copy while hashing the bytes actually read; publish only if they match."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".partial")
    digest, size = hashlib.sha256(), 0
    with Path(source).open("rb") as incoming, temporary.open("wb") as outgoing:
        while chunk := incoming.read(CHUNK):
            outgoing.write(chunk)
            digest.update(chunk)
            size += len(chunk)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    actual = digest.hexdigest()
    if (expected_bytes is not None and size != expected_bytes) or (
            expected_sha256 is not None and actual != expected_sha256):
        temporary.unlink(missing_ok=True)
        raise PipelineError(f"Copied bytes differ from contract: {source} bytes={size} sha256={actual}")
    os.replace(temporary, target)
    return size, actual


def publish_file(source: Path, target: Path) -> dict[str, Any]:
    """Copy onto a possibly object-storage-backed mount and prove the published bytes.

    Preferred: write target.uploading, verify, rename.  Some object-volume FUSE
    mounts cannot rename (or cannot rename large objects); then the bytes are
    written directly to the final name and re-read to verify.  Either way the
    returned record states which mode was used and the verified SHA-256.
    """
    source, target = Path(source), Path(target)
    size, digest = source.stat().st_size, sha256_file(source)
    uploading = target.with_name(target.name + ".uploading")
    try:
        copy_with_hash(source, uploading, expected_bytes=size, expected_sha256=digest)
        os.replace(uploading, target)
        mode = "atomic_rename"
    except OSError:
        for leftover in (uploading, uploading.with_name(uploading.name + ".partial")):
            with contextlib.suppress(OSError):
                leftover.unlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as incoming, target.open("wb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, CHUNK)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        mode = "direct_write_then_verified"
    if target.stat().st_size != size or sha256_file(target) != digest:
        raise PipelineError(f"Published bytes differ from source: {target}")
    return {"path": str(target), "bytes": size, "sha256": digest, "mode": mode}


def safe_relative(name: str) -> PurePosixPath:
    """Reject absolute, parent-escaping, backslash, NUL and empty archive/manifest paths."""
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name:
        raise PipelineError(f"Unsafe relative path: {name!r}")
    relative = PurePosixPath(name)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts) or ":" in relative.parts[0]:
        raise PipelineError(f"Unsafe relative path: {name!r}")
    return relative


def contained(root: Path, relative: str) -> Path:
    """Resolve root/relative and prove it stays under root (symlink-aware)."""
    target = (Path(root) / Path(*safe_relative(relative).parts)).resolve()
    if not target.is_relative_to(Path(root).resolve()):
        raise PipelineError(f"Path escapes its root: {relative}")
    return target


def checked_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    """Map regular-file members after rejecting duplicates, symlinks and traversal names."""
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise PipelineError(f"Archive has duplicate member names: {archive.filename}")
    result: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise PipelineError(f"Archive contains a symbolic link: {info.filename}")
        if info.is_dir():
            safe_relative(info.filename.rstrip("/"))
            continue
        if info.flag_bits & 0x1:
            raise PipelineError(f"Encrypted archive member: {info.filename}")
        safe_relative(info.filename)
        result[info.filename] = info
    return result


def extract_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, target: Path, *,
                   expected_sha256: str | None = None, max_bytes: int | None = None) -> str:
    """Stream one member (CRC is verified by zipfile on EOF) into target atomically."""
    if max_bytes is not None and info.file_size > max_bytes:
        raise PipelineError(f"Archive member exceeds its size bound: {info.filename}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".partial")
    digest, written = hashlib.sha256(), 0
    with archive.open(info) as incoming, temporary.open("wb") as outgoing:
        while chunk := incoming.read(CHUNK):
            written += len(chunk)
            if written > info.file_size:
                raise PipelineError(f"Archive member longer than declared: {info.filename}")
            outgoing.write(chunk)
            digest.update(chunk)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    actual = digest.hexdigest()
    if written != info.file_size or (expected_sha256 is not None and actual != expected_sha256):
        temporary.unlink(missing_ok=True)
        raise PipelineError(f"Archive member content mismatch: {info.filename} sha256={actual}")
    os.replace(temporary, target)
    return actual


def extract_exact(zip_path: Path, destination: Path, expected: Mapping[str, str | None], *,
                  strip_prefix: str = "", allow_extra: bool = False,
                  only: Iterable[str] | None = None) -> dict[str, str]:
    """Extract exactly the expected member set (optionally prefix-stripped) with hashes.

    The archive's member set is always checked against ``expected``; ``only``
    restricts which of those members are (re)written, for repairing a local copy.
    """
    destination.mkdir(parents=True, exist_ok=True)
    extracted: dict[str, str] = {}
    with zipfile.ZipFile(zip_path) as archive:
        members = checked_members(archive)
        mapped: dict[str, zipfile.ZipInfo] = {}
        for name, info in members.items():
            if strip_prefix:
                if not name.startswith(strip_prefix):
                    if allow_extra:
                        continue
                    raise PipelineError(f"Unexpected member outside {strip_prefix}: {name}")
                name = name[len(strip_prefix):]
            if name in mapped:
                raise PipelineError(f"Members collide after prefix stripping: {name}")
            mapped[name] = info
        missing = sorted(set(expected) - set(mapped))
        extra = sorted(set(mapped) - set(expected))
        if missing or (extra and not allow_extra):
            raise PipelineError(f"{zip_path.name}: missing={missing[:10]} unexpected={extra[:10]}")
        selected = set(expected) if only is None else set(only)
        if not selected <= set(expected):
            raise PipelineError(f"Requested members outside the contract: {sorted(selected - set(expected))[:5]}")
        for name, digest in expected.items():
            if name in selected:
                extracted[name] = extract_member(archive, mapped[name], contained(destination, name),
                                                 expected_sha256=digest)
    return extracted


def make_read_only(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def disk_free(path: Path) -> int:
    probe = Path(path)
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free


def iter_files(root: Path) -> Iterator[Path]:
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and not path.is_symlink():
            yield path


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def process_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace")
    except OSError:
        return ""


def process_start_ticks(pid: int) -> int | None:
    """Kernel start time of a running process (clock ticks since boot); distinguishes a
    reused PID.  Zombies/dead entries (state Z/X) count as not running."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] in ("Z", "X", "x"):
            return None
        return int(fields[19])
    except (OSError, IndexError, ValueError):
        return None


def process_identity(pid: int | None = None) -> dict[str, Any]:
    pid = os.getpid() if pid is None else int(pid)
    return {"pid": pid, "start_ticks": process_start_ticks(pid)}


def same_live_process(record: Mapping[str, Any] | None) -> bool:
    """True only if that exact process (pid + start time) is still running."""
    if not record or record.get("pid") is None:
        return False
    ticks = process_start_ticks(int(record["pid"]))
    return ticks is not None and ticks == record.get("start_ticks")


class StageLock:
    """Exclusive per-stage lock for child processes.  It never terminates anyone.

    Two writers on one feature cache or checkpoint directory would corrupt
    resume state, so a second process refuses while the first lives.  Reclaiming
    orphaned stage processes is the controller's job (it can prove the previous
    controller is dead); a lock holder here is always treated as live.
    """

    def __init__(self, path: Path, signature: str) -> None:
        self.path, self.signature = Path(path), signature
        self.stream = None

    def _owner(self) -> dict[str, Any]:
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}

    def acquire(self, *, wait_seconds: float = 30.0) -> None:
        """Wait briefly (a just-terminated holder releases on exit), then refuse."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+")
        deadline = time.monotonic() + wait_seconds
        while True:
            try:
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    owner = self._owner()
                    self.stream.close()
                    self.stream = None
                    raise PipelineError(f"Stage lock {self.path} is held by live process {owner} "
                                        f"({process_cmdline(int(owner.get('pid', 0) or 0))[:200]!r}); refusing to run "
                                        "concurrently and refusing to kill it")
                time.sleep(1)
        self.stream.seek(0)
        self.stream.truncate()
        self.stream.write(json.dumps({**process_identity(), "signature": self.signature, "acquired_at": now_iso()}))
        self.stream.flush()

    def release(self) -> None:
        if self.stream is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            self.stream.close()
            self.stream = None

    def __enter__(self) -> "StageLock":
        self.acquire()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


_CONTROLLER_REGISTRY = "_dacon9504a100_controller_registry"


def _controller_registry() -> dict[str, dict[str, Any]]:
    """Process-wide holder of controller locks.  Lives outside the dacon9504a100
    package so re-running the notebook's bootstrap cell (which purges and
    re-imports the package) finds the lock this kernel already holds."""
    import sys
    import types
    module = sys.modules.get(_CONTROLLER_REGISTRY)
    if module is None:
        module = types.ModuleType(_CONTROLLER_REGISTRY)
        module.locks = {}
        sys.modules[_CONTROLLER_REGISTRY] = module
    return module.locks


class ControllerLock:
    """One live controller (notebook kernel) per job root, taken before any job write.

    * Another live process holds it -> PipelineError; nothing is written, nothing killed.
    * This process already holds it (bootstrap cell re-run) -> ownership moves to the
      new controller object; the old object refuses to run stages.
    * The flock dies with its process, so a dead kernel never blocks a new one.
    """

    def __init__(self, job: Path) -> None:
        self.job = Path(job)
        self.path = self.job / "controller.lock"
        self.key = str(self.path.absolute())
        self.token = object()
        self.newly_acquired = False

    def acquire(self) -> "ControllerLock":
        registry = _controller_registry()
        held = registry.get(self.key)
        if held is not None and not held["stream"].closed:
            held["owner"] = self.token
            return self
        self.job.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+")  # creates the file only if absent; never truncates a holder's record
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.seek(0)
            owner = stream.read()[:500]
            stream.close()
            raise PipelineError(f"Another live controller already runs this job ({self.job}): {owner}. "
                                "Nothing was changed. Use that kernel, or stop it before Run All here.") from None
        stream.seek(0)
        stream.truncate()
        stream.write(json.dumps({**process_identity(), "acquired_at": now_iso(), "kind": "controller"}))
        stream.flush()
        os.fsync(stream.fileno())
        registry[self.key] = {"stream": stream, "owner": self.token, "identity": process_identity()}
        self.newly_acquired = True
        return self

    def owns(self) -> bool:
        held = _controller_registry().get(self.key)
        return held is not None and held["owner"] is self.token and not held["stream"].closed

    def abandon(self) -> None:
        """A takeover in this process failed half-way: nobody owns the held lock until a new takeover."""
        held = _controller_registry().get(self.key)
        if held is not None and held["owner"] is self.token:
            held["owner"] = None

    def release(self) -> None:
        registry = _controller_registry()
        held = registry.get(self.key)
        if held is not None and held["owner"] is self.token:
            with contextlib.suppress(OSError):
                fcntl.flock(held["stream"].fileno(), fcntl.LOCK_UN)
            held["stream"].close()
            registry.pop(self.key, None)


def safe_name(identifier: str) -> str:
    """Portable file stem for an arbitrary manifest ID (IDs may contain ':')."""
    import re
    return re.sub(r"[^A-Za-z0-9_.-]", "_", identifier)[:120] + "__" + sha256_bytes(identifier.encode())[:10]


def human_bytes(value: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024 or unit == "TiB":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}TiB"


def human_seconds(value: float) -> str:
    value = max(0.0, float(value))
    hours, rest = divmod(int(round(value)), 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours:d}h{minutes:02d}m{seconds:02d}s"


def tail(path: Path, lines: int = 40) -> str:
    try:
        data = Path(path).read_bytes()[-200_000:]
    except OSError:
        return ""
    return "\n".join(data.decode(errors="replace").splitlines()[-lines:])


def write_sha256sums(entries: Iterable[tuple[str, str]], path: Path) -> None:
    atomic_write_bytes(path, "".join(f"{digest}  {name}\n" for name, digest in sorted(entries)).encode())
