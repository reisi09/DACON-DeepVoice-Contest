"""The generated FINAL cell must give a link a root-scoped Jupyter server can actually serve.

    python tests/test_download_link.py --notebook DACON_Existing9504_A100.ipynb \
        --author-zip author_code.zip --work <scratch>

Needs nbformat, nbclient, ipykernel and jupyter_server in the interpreter (no GPU,
no network).  A small handoff ZIP is published by the real publish_handoff code;
the notebook's own bootstrap and FINAL cells run in a real kernel; a local
jupyter_server restricted to a root directory then serves the link.

What it protects against:
  * v2 defect: the link pointed at the object-volume copy (../root-1/...), outside a
    /root-scoped server -> the "download" fails after a 16 h run.  Case A puts the
    notebook under the server root with deployment and job dirs OUTSIDE it: the
    old-style link must fail, the new link must stream bytes whose SHA-256 equals
    the sidecar, and the local copy must be a hardlink (no multi-GB duplication).
  * case B: the notebook sits in the deployment directory itself (inside the root):
    the published ZIP is reused in place and served.
  * cross-filesystem fallback (os.link fails with EXDEV) and a stale same-name file
    in the notebook directory: replaced by a hash-verified copy.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import posixpath
import re
import secrets
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dacon9504a100 import handoff  # noqa: E402
from dacon9504a100.constants import HANDOFF_NAME  # noqa: E402

NOTEBOOK_NAME = "DACON_Existing9504_A100.ipynb"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never route localhost through a proxy


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def small_handoff(job: Path, deploy: Path) -> dict:
    local = job / "handoff" / HANDOFF_NAME
    local.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(local, "w") as archive:
        archive.writestr("README.txt", "small stand-in handoff for the download-link check\n")
        archive.writestr("payload.bin", secrets.token_bytes(3 * 1024 * 1024))
    return handoff.publish_handoff({"deployment_root": str(deploy)}, local, "complete", 2)


def final_only_notebook(generated: Path, target: Path) -> None:
    """The generated bootstrap + FINAL cells, byte-for-byte (the GPU stage cells are skipped)."""
    notebook = nbformat.read(generated, as_version=4)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    notebook.cells = [code[0], code[-1]]
    assert "AUTHOR_CODE_SHA256 =" in code[0].source and "expose_download" in code[-1].source
    target.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, target)


def execute(notebook_path: Path, deploy: Path, job: Path) -> nbformat.NotebookNode:
    os.environ.update(DACON_DEPLOYMENT_ROOT=str(deploy), DACON_JOB_ROOT=str(job))
    notebook = nbformat.read(notebook_path, as_version=4)
    NotebookClient(notebook, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": str(notebook_path.parent)}}).execute()
    return notebook


def link_href(notebook: nbformat.NotebookNode) -> str:
    for output in notebook.cells[-1].outputs:
        html = output.get("data", {}).get("text/html", "")
        match = re.search(r"<a href=['\"]([^'\"]+)['\"]", html)
        if match:
            return match.group(1)
    raise AssertionError(f"FINAL cell produced no link: {notebook.cells[-1].outputs}")


def files_url(notebook_dir_in_root: str, href: str) -> str:
    """How Jupyter resolves a relative link in a notebook output: against the notebook's
    directory, served by the /files/ handler (JupyterLab resolver / classic redirect)."""
    return "/files/" + quote(posixpath.normpath(posixpath.join(notebook_dir_in_root, href)))


class Server:
    def __init__(self, root: Path, log: Path) -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        self.token = secrets.token_hex(16)
        self.log = log.open("w")
        self.process = subprocess.Popen(
            [sys.executable, "-m", "jupyter_server", f"--ServerApp.root_dir={root}", f"--ServerApp.port={self.port}",
             "--ServerApp.port_retries=0", "--ServerApp.ip=127.0.0.1", "--ServerApp.open_browser=False",
             "--ServerApp.allow_root=True",
             f"--IdentityProvider.token={self.token}"], stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 90
        while True:
            try:
                if self.get("/api/status")[0] == 200:
                    break
            except OSError:
                pass
            if time.monotonic() > deadline or self.process.poll() is not None:
                raise RuntimeError(f"jupyter_server did not start: {log.read_text()[-2000:]}")
            time.sleep(0.5)

    def get(self, path: str) -> tuple[int, bytes]:
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}",
                                         headers={"Authorization": f"token {self.token}"})
        try:
            with OPENER.open(request, timeout=60) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, b""

    def stop(self) -> None:
        self.process.terminate()
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.kill()
        self.log.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--author-zip", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    if args.work.exists():
        shutil.rmtree(args.work)

    print("case A: notebook under a root-scoped server; deployment and job dirs outside the root")
    root_a = args.work / "server_root_a"
    notebook_dir = root_a / "uploaded"
    deploy_a, job_a = args.work / "objvol" / "existing-only-9504-20260919", args.work / "job_a"
    deploy_a.mkdir(parents=True)
    shutil.copyfile(args.author_zip, deploy_a / "author_code.zip")
    sidecar = small_handoff(job_a, deploy_a)
    final_only_notebook(args.notebook, notebook_dir / NOTEBOOK_NAME)
    notebook = execute(notebook_dir / NOTEBOOK_NAME, deploy_a, job_a)
    href = link_href(notebook)
    assert not href.startswith(("/", "http")) and ".." not in PurePosixPath(href).parts, href
    local = notebook_dir / href
    assert os.path.samefile(local, sidecar["local_copy"]), "expected a hardlink to the verified local copy"
    server = Server(root_a, args.work / "server_a.log")
    try:
        status, body = server.get(files_url("uploaded", href))
        assert status == 200 and sha256(body) == sidecar["sha256"] and len(body) == sidecar["bytes"], status
        old_href = os.path.relpath(sidecar["file"], notebook_dir)  # the v2 FINAL cell's link
        assert old_href.startswith(".."), old_href
        old_status, _ = server.get(files_url("uploaded", old_href))
        assert old_status != 200, "the v2-style link unexpectedly worked; case A does not reproduce the defect"
    finally:
        server.stop()

    print("case B: notebook inside the deployment directory, which is inside the server root")
    root_b = args.work / "server_root_b"
    deploy_b, job_b = root_b / "existing-only-9504-20260919", args.work / "job_b"
    deploy_b.mkdir(parents=True)
    shutil.copyfile(args.author_zip, deploy_b / "author_code.zip")
    sidecar_b = small_handoff(job_b, deploy_b)
    final_only_notebook(args.notebook, deploy_b / NOTEBOOK_NAME)
    notebook = execute(deploy_b / NOTEBOOK_NAME, deploy_b, job_b)
    href = link_href(notebook)
    assert href == HANDOFF_NAME and os.path.samefile(deploy_b / href, sidecar_b["file"])
    server = Server(root_b, args.work / "server_b.log")
    try:
        status, body = server.get(files_url(deploy_b.name, href))
        assert status == 200 and sha256(body) == sidecar_b["sha256"], status
    finally:
        server.stop()

    print("cross-filesystem fallback and a stale same-name file are replaced by a verified copy")
    elsewhere = args.work / "other_filesystem"
    elsewhere.mkdir()
    (elsewhere / HANDOFF_NAME).write_bytes(b"stale bytes from an older run")
    real_link = os.link

    def cross_device(*_arguments, **_keywords):
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    os.link = cross_device
    try:
        record = handoff.expose_download(deploy_a / (HANDOFF_NAME + ".json"), elsewhere)
    finally:
        os.link = real_link
    copied = (elsewhere / HANDOFF_NAME).read_bytes()
    assert record["mode"] == "verified_byte_copy" and sha256(copied) == sidecar["sha256"]
    assert not os.path.samefile(elsewhere / HANDOFF_NAME, sidecar["local_copy"])
    assert sorted(p.name for p in elsewhere.iterdir()) == [HANDOFF_NAME], "temporary files left behind"
    print(json.dumps({"case_b_href": href, "v2_style_link_http_status": old_status, "case_a_sha256": sidecar["sha256"]}))
    print("DOWNLOAD LINK CHECKS PASSED")


if __name__ == "__main__":
    main()
