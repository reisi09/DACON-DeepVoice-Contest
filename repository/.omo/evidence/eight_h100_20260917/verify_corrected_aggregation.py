from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results/eight_h100_20260917"
OUTPUT = RESULTS / "downloaded/unpacked/output"
CORRECTED = RESULTS / "corrected_summary/final_aggregation.corrected.json"
RECEIPT = RESULTS / "corrected_summary/correction_receipt.json"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


corrected = load(CORRECTED)
sealed = load(OUTPUT / "final_aggregation.json")
receipt = load(RECEIPT)
archive = RESULTS / "downloaded/results.zip"
notebook = OUTPUT / "DACON_Eight_H100_Executed.ipynb"
assert sealed["dataset"]["component_count"] == 178
assert corrected["dataset"]["component_count"] == 256
assert corrected["experiments"]["E8"]["fixed_updates"] is None
assert corrected["experiments"]["E8"]["fit_kind"] == "non_iterative_closed_form_real_only_gaussian"
assert sha256(archive) == receipt["immutable_archive"]["sha256"] == "681d41de10397d31a371daaa276a4a90d19f04789b85e01fdd6f936399dd23d2"
assert sha256(notebook) == receipt["immutable_executed_notebook"]["sha256"] == "bb28445e88a4d963aa7969fddba03e2c88badf4d969249e2456aca6b08acee91"
assert sha256(CORRECTED) == receipt["authoritative_corrected_summary"]["sha256"]
assert receipt["numerical_scores_changed"] is False
print("VALID global=256 sealed=178 E8=null archive_and_notebook_immutable=true")
