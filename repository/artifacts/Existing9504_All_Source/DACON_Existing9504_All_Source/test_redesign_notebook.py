#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["nbclient>=0.10,<1", "nbformat>=5.10,<6"]
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly (no venv, no pip install needed):
#      uv run test_redesign_notebook.py cpu OUTPUT_ROOT EXECUTED.ipynb
# 3. Or make executable and run:
#      chmod +x test_redesign_notebook.py && ./test_redesign_notebook.py cpu OUTPUT_ROOT EXECUTED.ipynb
# ──────────────────
"""Execute the notebook default or CPU algorithm-QA path top to bottom."""

from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Literal

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


class NotebookExecutionError(RuntimeError):
    """Raised when an executed notebook contradicts its requested mode."""


def execute(mode: Literal["default", "cpu", "expired"], root: Path, output: Path) -> dict[str, str | bool | float]:
    """Execute a fresh notebook and verify the mode-specific result boundary."""
    source = Path(__file__).with_name("DACON_Redesign_H100_Experiments.ipynb")
    notebook = nbformat.read(source, as_version=4)
    if mode == "cpu":
        changed = False
        for cell in notebook.cells:
            if cell.cell_type == "code" and '"cpu_integration": False' in cell.source:
                cell.source = cell.source.replace('"cpu_integration": False', '"cpu_integration": True')
                changed = True
        if not changed:
            raise NotebookExecutionError("CPU integration flag cell was not found")
    if mode == "expired":
        changed = False
        for cell in notebook.cells:
            if cell.cell_type == "code" and 'costly = FLAGS["extract_features"]' in cell.source:
                cell.source = 'FLAGS["extract_features"] = True\n' + cell.source
                changed = True
        if not changed:
            raise NotebookExecutionError("costly-stage cell was not found")
        (root / "results").mkdir(parents=True, exist_ok=True)
        (root / "results" / "gpu_budget.json").write_text(json.dumps({
            "started_unix": 0.0,
            "hard_deadline_unix": 1.0,
            "stage_deadline_unix": 0.0,
        }), encoding="utf-8")
    os.environ["DACON_REDESIGN_ROOT"] = str(root.resolve())
    client = NotebookClient(notebook, timeout=240, kernel_name="python3")
    try:
        executed = client.execute()
    except CellExecutionError:
        if mode != "expired":
            raise
        executed = client.nb
    else:
        if mode == "expired":
            raise NotebookExecutionError("expired GPU budget did not interrupt the costly stage")
    nbformat.write(executed, output)
    progress = json.loads((root / "results" / "progress.json").read_text(encoding="utf-8"))
    if mode == "expired":
        readback = json.loads((root / "exports" / "interrupted_readback.json").read_text(encoding="utf-8"))
        archive = root / "exports" / "interrupted_results.zip"
        with zipfile.ZipFile(archive) as packed:
            if packed.testzip() is not None:
                raise NotebookExecutionError("interrupted archive CRC readback failed")
        if progress["stages"]["extract_features"]["status"] != "interrupted" or readback.get("verified") is not True:
            raise NotebookExecutionError("expired budget did not record interruption and verified export")
        return {
            "mode": mode,
            "executed": True,
            "algorithm_qa_only": False,
            "export_verified": True,
            "save_load_prediction_max_difference": 0.0,
        }
    integration_path = root / "cpu_integration" / "integration.json"
    if mode == "cpu":
        integration = json.loads(integration_path.read_text(encoding="utf-8"))
        if integration.get("status") != "algorithm_qa_only" or integration.get("actual_data_performance_verified") is not False:
            raise NotebookExecutionError("CPU fixture crossed the performance-evidence boundary")
        return {
            "mode": mode,
            "executed": True,
            "algorithm_qa_only": True,
            "export_verified": bool(integration["export_verified"]),
            "save_load_prediction_max_difference": float(integration["save_load_prediction_max_difference"]),
        }
    if integration_path.exists() or progress["stages"]["cpu_integration"]["status"] != "not_run":
        raise NotebookExecutionError("default mode unexpectedly executed CPU integration")
    return {"mode": mode, "executed": True, "algorithm_qa_only": False, "export_verified": False, "save_load_prediction_max_difference": 0.0}


def main() -> None:
    """Run the requested notebook mode."""
    if len(sys.argv) != 4 or sys.argv[1] not in {"default", "cpu", "expired"}:
        raise NotebookExecutionError("usage: test_redesign_notebook.py default|cpu|expired OUTPUT_ROOT EXECUTED.ipynb")
    result = execute(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
