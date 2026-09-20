"""Generate DACON_Existing9504_A100.ipynb bound to one exact author_code.zip (SHA-256).

    python tools/make_notebook.py --author-zip author_code.zip --out DACON_Existing9504_A100.ipynb

The notebook is not inside author_code.zip because it embeds that ZIP's hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

INTRO = """# DACON 236749 — existing-only 9504 rows, A100 end-to-end (Run All)

**What Run All does** (every stage is resume-safe; re-running Run All continues where it stopped):

1. **preflight** — A100 via `nvidia-smi`, object volume `/root-1` and `deployment.json`, no other GPU process, disk budget, package-index reachability.
2. **stage_inputs** — verifies every immutable deployment file (bytes + SHA-256), the runtime manifest SHA, the 9504-row / 6652·1426·1426 / per-split FILE_FAKE balance / exposure / singing-train-only / provenance+content leakage contract; extracts **only the selected audio members** from hash-verified shards and verifies every file; extracts the 27 frozen assets and the original `source/` modules (read-only).
3. **environment** — isolated, hash-pinned Python 3.11 venv (uv 0.8.17; torch/torchaudio 2.9.1, torchcodec 0.8.1, transformers 4.57.6, numpy 2.2.6, … as in the reference requirements; **no fairseq**), exact freeze check, in-venv probe of the A100 and of the torchaudio→torchcodec decoder (FFmpeg libraries provisioned only if missing).
4. **pilot** — A100 VRAM-measured MERT batch (80 % VRAM, ≥90 % throughput rule) and a representative train-only runtime pilot with a projection for all 9504 rows against the ~16 h target (reported, never enforced; no data is dropped).
5. **features** — resume-safe full-duration HTDemucs → MERT-330M layer-12 windows, DF-Arena fc5-input embeddings, PANNs presence for all 9504 rows (exact coverage check).
6. **train** — unchanged WindowMIL (music) and DF-Arena `fc5` (voice) heads, seeds 41/42/43, train-only normalisation, selection on **val only**, then an immutable `candidate_lock.json` (six checkpoint hashes). Test features are never opened before the lock.
7. **test_eval** — verifies the lock, scores test once with the locked heads; masked known-label metrics.
8. **parity** — fresh-process raw-audio re-extraction of representative samples, compared with the cache and the val/test predictions.
9. **handoff** — the single `DACON_Existing9504_A100_Handoff.zip` (checkpoints, predictions, metrics, manifest/provenance/audit, code, notebook, environment lock, assets, measurements, logs, parity samples, SHA256SUMS), CRC+SHA verified and atomically published next to this notebook.

On any exception a `DACON_Existing9504_A100_Handoff_FAILED.zip` with status `failed` is written and the error is re-raised.
This notebook does **not** start/stop services, use API keys, or submit anything to DACON. No cell needs editing.

Caveats carried from the dataset README: val/test are *not* independent (historical exposure 997/1426 and 1001/1426); singing is train-only; music source/length bias remains; no official/L4 score is claimed.
"""

BOOTSTRAP = r'''import hashlib, json, os, shutil, stat, sys, zipfile
from pathlib import Path, PurePosixPath

AUTHOR_CODE_SHA256 = "__AUTHOR_CODE_SHA256__"
NOTEBOOK_NAME = "DACON_Existing9504_A100.ipynb"
DEPLOYMENT_ROOT = Path(os.environ.get("DACON_DEPLOYMENT_ROOT", "/root-1/existing-only-9504-20260919"))
if not (DEPLOYMENT_ROOT / "deployment.json").is_file() and (Path.cwd() / "deployment.json").is_file():
    DEPLOYMENT_ROOT = Path.cwd()  # the notebook lives in the deployment root
JOB_ROOT = Path(os.environ.get("DACON_JOB_ROOT", "/root/dacon_existing9504_a100_20260919"))
_session = os.environ.get("JPY_SESSION_NAME", "")
NOTEBOOK_PATH = Path(_session) if _session.endswith(".ipynb") and Path(_session).is_file() else DEPLOYMENT_ROOT / NOTEBOOK_NAME


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_author_code(archive_path: Path, destination: Path) -> None:
    """Traversal/symlink/duplicate-safe extraction into a fresh directory, then atomic rename."""
    staging = destination.with_name(destination.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("author_code.zip has duplicate members")
        for info in archive.infolist():
            relative = PurePosixPath(info.filename)
            if (relative.is_absolute() or ".." in relative.parts or "\\" in info.filename
                    or stat.S_ISLNK(info.external_attr >> 16)):
                raise RuntimeError(f"Unsafe author_code.zip member: {info.filename}")
            target = (staging / Path(*relative.parts)).resolve()
            if not target.is_relative_to(staging.resolve()):
                raise RuntimeError(f"author_code.zip member escapes: {info.filename}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)
    (staging / ".extracted").write_text(AUTHOR_CODE_SHA256 + "\n")
    if destination.exists():
        shutil.rmtree(destination)
    staging.rename(destination)


CODE_ZIP = DEPLOYMENT_ROOT / "author_code.zip"
if not CODE_ZIP.is_file():
    raise FileNotFoundError(f"author_code.zip must sit next to this notebook in {DEPLOYMENT_ROOT}")
_actual = _sha256(CODE_ZIP)
if _actual != AUTHOR_CODE_SHA256:
    raise RuntimeError(f"author_code.zip SHA-256 {_actual} != notebook-bound {AUTHOR_CODE_SHA256}")
CODE_DIR = JOB_ROOT / "code" / AUTHOR_CODE_SHA256[:16]
_marker = CODE_DIR / ".extracted"
if not _marker.is_file() or _marker.read_text().strip() != AUTHOR_CODE_SHA256:
    _extract_author_code(CODE_ZIP, CODE_DIR)
for _name in [n for n in sys.modules if n == "dacon9504a100" or n.startswith("dacon9504a100.")]:
    del sys.modules[_name]
sys.path.insert(0, str(CODE_DIR))
from dacon9504a100.orchestrator import Pipeline

pipe = Pipeline(DEPLOYMENT_ROOT, JOB_ROOT, CODE_DIR, CODE_ZIP, NOTEBOOK_PATH)
print(f"deployment root : {DEPLOYMENT_ROOT}")
print(f"job root        : {JOB_ROOT}")
print(f"author code     : {CODE_DIR} (sha256 {AUTHOR_CODE_SHA256})")
print(f"notebook        : {NOTEBOOK_PATH}")
print(f"kernel python   : {sys.version.split()[0]} (orchestration only; GPU work runs in the isolated venv)")
'''

FINAL = r'''import json
from IPython.display import FileLink, Markdown, display

_final = DEPLOYMENT_ROOT / "DACON_Existing9504_A100_Handoff.zip"
_sidecar = json.loads(_final.with_name(_final.name + ".json").read_text())
print(json.dumps(_sidecar, indent=2))
display(Markdown(f"**Handoff ZIP** `{_final}` — sha256 `{_sidecar['sha256']}` ({_sidecar['bytes']:,} bytes). "
                 "Return this single file for review/submission preparation."))
try:
    display(FileLink(os.path.relpath(_final, Path.cwd())))
except ValueError:
    display(FileLink(str(_final)))
'''

STAGES = [
    ("preflight", "## 1 · Preflight (A100, mounts, disk, network)"),
    ("stage_inputs", "## 2 · Stage and verify inputs (manifest contract, assets, source, selected audio only)"),
    ("environment", "## 3 · Isolated pinned Python 3.11 environment + in-venv A100/decoder probe"),
    ("pilot", "## 4 · A100 VRAM-measured batch + representative runtime pilot and projection"),
    ("features", "## 5 · Full-duration frozen-encoder features for all 9504 rows (resume-safe; longest stage)"),
    ("train", "## 6 · Train six heads (seeds 41/42/43), select on val, write the immutable lock"),
    ("test_eval", "## 7 · Locked test evaluation (masked known-label metrics)"),
    ("parity", "## 8 · Raw-audio parity samples"),
    ("handoff", "## 9 · Single handoff ZIP (verified, published next to this notebook)"),
]


def cell(kind: str, source: str) -> dict:
    lines = source.splitlines(keepends=True)
    base = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == "code":
        base.update(execution_count=None, outputs=[])
    return base


def build(author_zip: Path, out: Path) -> str:
    digest = hashlib.sha256(author_zip.read_bytes()).hexdigest()
    cells = [cell("markdown", INTRO), cell("markdown", "## 0 · Locate and verify the author code"),
             cell("code", BOOTSTRAP.replace("__AUTHOR_CODE_SHA256__", digest))]
    for stage, title in STAGES:
        cells += [cell("markdown", title), cell("code", f'result_{stage} = pipe.run("{stage}")')]
    cells += [cell("markdown", "## Download link"), cell("code", FINAL)]
    notebook = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "dacon9504a100": {"author_code_sha256": digest, "generated_by": "tools/make_notebook.py"}},
        "nbformat": 4, "nbformat_minor": 5}
    for index, item in enumerate(notebook["cells"]):
        item["id"] = f"cell-{index:02d}"
    out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return digest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-zip", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.author_zip, args.out))
