"""Execute the eight-experiment notebook and always publish a verified archive."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import Final

import nbformat
from nbclient import NotebookClient


MAX_GPU_SECONDS: Final = 14_400


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value: dict[str, int | float | str | bool]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def export_results(output: Path, archive: Path) -> dict[str, int | str | bool]:
    files = sorted(path for path in output.rglob("*") if path.is_file() and not path.is_symlink())
    manifest = {f"output/{path.relative_to(output).as_posix()}": sha256(path) for path in files}
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as packed:
        for path in files:
            packed.write(path, f"output/{path.relative_to(output).as_posix()}")
        packed.writestr("EXPORT_SHA256.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with zipfile.ZipFile(archive) as packed:
        if packed.testzip() is not None:
            raise RuntimeError("result archive CRC verification failed")
        for name, expected in manifest.items():
            if hashlib.sha256(packed.read(name)).hexdigest() != expected:
                raise RuntimeError(f"result archive member verification failed: {name}")
    return {
        "archive_sha256": sha256(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_verified": True,
        "export_member_count": len(files),
    }


def main(config_path: str) -> int:  # noqa: PLR0915
    config_file = Path(config_path)
    config = json.loads(config_file.read_text(encoding="utf-8"))
    root = Path(config["root"])
    shared = Path(config["shared"])
    output = root / "output"
    output.mkdir(parents=True, exist_ok=True)
    started = float(config["started_epoch"])
    deadline = float(config["pause_deadline_epoch"])
    reserve = int(config["export_reserve_seconds"])
    if not 0 < deadline - started <= MAX_GPU_SECONDS or reserve != 600:
        raise RuntimeError("invalid immutable four-hour run boundary")
    if time.time() >= deadline - reserve:
        raise RuntimeError("run started after its export reserve")

    state: dict[str, int | float | str | bool] = {
        "status": "running",
        "archive_ready": False,
        "started_unix": started,
        "hard_deadline_unix": deadline,
        "updated_unix": time.time(),
    }
    atomic_json(shared / "run_state.json", state)
    notebook_name = str(config.get("notebook", "DACON_Eight_H100.ipynb"))
    notebook = nbformat.read(root / "source" / notebook_name, as_version=4)
    executed_path = output / "DACON_Eight_H100_Executed.ipynb"
    status = "failed"
    try:
        timeout = max(1, int(deadline - time.time() - reserve))
        NotebookClient(
            notebook,
            timeout=timeout,
            kernel_name=str(config.get("kernel", "eight-h100")),
            resources={"metadata": {"path": str(root / "source")}},
        ).execute()
        status = "exported"
    except Exception as error:  # noqa: BROAD_EXCEPT_OK - top-level artifact boundary
        state["error"] = f"{type(error).__name__}: {error}"
        (output / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        nbformat.write(notebook, executed_path)
        for path in (root / "run.log", root / "gpu_telemetry.live.csv", config_file):
            if path.exists():
                shutil.copy2(path, output / path.name)
        local_archive = root / "results.zip"
        exported = export_results(output, local_archive)
        shutil.copyfile(local_archive, shared / "results.zip")
        if sha256(shared / "results.zip") != exported["archive_sha256"]:
            raise RuntimeError("published result archive differs from verified local archive")
        receipt: dict[str, int | float | str | bool] = {
            "status": "exported" if status == "exported" else "partial_exported",
            "archive_ready": True,
            **exported,
            "exported_unix": time.time(),
        }
        atomic_json(shared / "export_receipt.json", receipt)
        (shared / "results.zip.sha256").write_text(
            f"{exported['archive_sha256']}  results.zip\n", encoding="utf-8"
        )
        state.update(receipt)
        state["status"] = status
        state["updated_unix"] = time.time()
        atomic_json(shared / "run_state.json", state)
    return 0 if status == "exported" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
