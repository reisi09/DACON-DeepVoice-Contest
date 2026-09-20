"""Execute the generated notebook in a real Jupyter kernel on a non-GPU host.

    python tests/test_notebook_bootstrap.py --notebook DACON_Existing9504_A100.ipynb \
        --author-zip author_code.zip --deploy <fixture>/objvol/deploy --work <scratch>

Needs nbclient + ipykernel in the interpreter running it (use the image's own
kernel Python, e.g. 3.13).  What it protects against:
  * the notebook cannot find/verify/extract author_code.zip or import the Pipeline
    in the actual kernel Python (bootstrap cell must succeed);
  * a different author_code.zip is silently used (hash binding must refuse it);
  * Run All on a host without the A100 object volume continues or fails silently
    (the preflight cell must raise, and a FAILED handoff ZIP with status "failed"
    carrying the exact author_code.zip bytes and the notebook must be published).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

FAILURE_NAME = "DACON_Existing9504_A100_Handoff_FAILED.zip"
NOTEBOOK_NAME = "DACON_Existing9504_A100.ipynb"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(deploy: Path, target: Path, notebook: Path, author_zip: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for entry in deploy.iterdir():
        if entry.name.startswith(("DACON_", "a100_run")) or entry.name in ("author_code.zip", NOTEBOOK_NAME):
            continue
        (target / entry.name).symlink_to(entry.resolve())
    shutil.copyfile(notebook, target / NOTEBOOK_NAME)
    shutil.copyfile(author_zip, target / "author_code.zip")


def execute(deploy: Path, job: Path) -> tuple[nbformat.NotebookNode, BaseException | None]:
    os.environ.update(DACON_DEPLOYMENT_ROOT=str(deploy), DACON_JOB_ROOT=str(job))
    notebook = nbformat.read(deploy / NOTEBOOK_NAME, as_version=4)
    nbformat.validate(notebook)
    client = NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(deploy)}})
    try:
        client.execute()
    except CellExecutionError as error:
        return notebook, error
    return notebook, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--author-zip", type=Path, required=True)
    parser.add_argument("--deploy", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    bound = json.loads(args.notebook.read_text())["metadata"]["dacon9504a100"]["author_code_sha256"]
    assert bound == sha256(args.author_zip), "notebook is not bound to this author_code.zip"

    print("negative: a different author_code.zip must be refused by the bootstrap cell")
    tampered = args.work / "tampered_deploy"
    prepare(args.deploy, tampered, args.notebook, args.author_zip)
    with (tampered / "author_code.zip").open("ab") as stream:
        stream.write(b"\0")
    _, error = execute(tampered, args.work / "tampered_job")
    assert error is not None and "notebook-bound" in str(error), f"tampered code accepted: {error}"
    assert not (args.work / "tampered_job" / "code").exists(), "tampered code must not be extracted"

    print("Run All on a host without /root-1 and without the A100")
    deploy = args.work / "deploy"
    prepare(args.deploy, deploy, args.notebook, args.author_zip)
    notebook, error = execute(deploy, args.work / "job")
    bootstrap = next(c for c in notebook.cells if c.cell_type == "code")
    printed = "".join(o.get("text", "") for o in bootstrap.outputs)
    assert bound in printed and "kernel python" in printed, f"bootstrap cell did not complete: {bootstrap.outputs}"
    assert error is not None and "/root-1 is not mounted" in str(error), f"preflight did not fail: {error}"
    executed = [c for c in notebook.cells if c.cell_type == "code" and c.get("execution_count")]
    assert len(executed) == 2, "Run All must stop at the failing preflight cell"
    failure = deploy / FAILURE_NAME
    sidecar = json.loads((deploy / (FAILURE_NAME + ".json")).read_text())
    assert sidecar["status"] == "failed" and sidecar["sha256"] == sha256(failure)
    with zipfile.ZipFile(failure) as archive:
        assert archive.testzip() is None
        state = json.loads(archive.read("status/run_state.json"))
        assert state["handoff_status"] == "failed" and state["stages"]["preflight"]["status"] == "failed"
        assert hashlib.sha256(archive.read("code/author_code.zip")).hexdigest() == bound
        assert f"code/{NOTEBOOK_NAME}" in archive.namelist()
    status = json.loads((deploy / "a100_run_status.json").read_text())
    assert status["status"] == "failed", "progress mirror on the deployment root must show the failure"
    print("NOTEBOOK BOOTSTRAP CHECKS PASSED")


if __name__ == "__main__":
    main()
