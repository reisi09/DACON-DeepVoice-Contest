from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from eight_fixed_io import connected_components


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results/eight_h100_20260917"
SEALED = RESULTS / "downloaded/unpacked/output"
CORRECTED = RESULTS / "corrected_summary"


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_corrected_summary_uses_actual_report_populations_and_noniterative_e8() -> None:
    # Given: the immutable downloaded reports, prepared manifest, and verified population correction.
    corrected = _json(CORRECTED / "final_aggregation.corrected.json")
    receipt = _json(CORRECTED / "correction_receipt.json")
    sidecar = _json(RESULTS / "postrun_e2/fixed_experiment_population_denominators.json")
    with (ROOT / "_workspace/eight_h100_20260917/dataset/manifest.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    # When: population metadata is resolved from the complete provenance graph and raw reports.
    global_components = len(set(connected_components(rows).values()))
    reports = {
        f"E{number}": _json(SEALED / ("e7/report.json" if number == 7 else f"experiment_{number}.json"))
        for number in range(1, 9)
    }

    # Then: the correction matches each real experiment contract without changing numerical evidence.
    assert global_components == corrected["dataset"]["component_count"] == 256
    assert corrected["supersedes"]["sha256"] == _sha256(SEALED / "final_aggregation.json")
    for name, report in reports.items():
        entry = corrected["experiments"][name]
        assert entry["status"] == report["status"] == "complete"
        assert entry["component_count"] == report["component_count"]
        assert entry["fixed_updates"] == report["fixed_updates"]
        assert entry["source_report_sha256"] == _sha256(
            SEALED / ("e7/report.json" if name == "E7" else f"experiment_{name[1:]}.json")
        )
        assert entry["label_denominators"] == report["label_denominators"]
        if name in sidecar["populations"]:
            assert entry["branch_population_denominators"] == sidecar["populations"][name]
            assert entry["raw_report_metadata_scope"] == "generic_top_level_not_branch_authoritative"
        else:
            assert "branch_population_denominators" not in entry
            assert entry["raw_report_metadata_scope"] == "experiment_native"
    assert corrected["experiments"]["E2"]["component_count"] == 4
    assert corrected["experiments"]["E5"]["component_count"] == 4
    assert corrected["experiments"]["E7"]["component_count"] == 10
    assert corrected["experiments"]["E8"]["fixed_updates"] is None
    assert corrected["experiments"]["E8"]["fit_kind"] == "non_iterative_closed_form_real_only_gaussian"
    assert receipt["numerical_scores_changed"] is False
    assert receipt["immutable_archive_changed"] is False
