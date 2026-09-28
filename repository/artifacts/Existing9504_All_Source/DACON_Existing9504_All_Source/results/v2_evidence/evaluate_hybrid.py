"""Reproduce the dev-locked hybrid comparison without fitting or choosing on test."""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from graph_features import atomic_json, digest
from package_hybrid_submission import route_predictions
from run_experiment import read_csv, write_csv
from run_v2 import compare_fresh, open_locked_test
from v2_scoring import _bootstrap, _proxy


def main():
    run = ROOT / "results/v2_colab_20260912_174330"
    lock = json.loads((run / "hybrid_candidate_lock.json").read_text())
    assert digest(run / "hybrid_selection.json") == lock["selection_sha256"]
    assert lock["selected"] == "G0_AST_MUSIC"
    rows = open_locked_test(run / "hybrid_candidate_lock.json", ROOT / "v2_validation/locked_test/manifest.csv")
    g0 = read_csv(run / "G0_test_predictions.csv")
    ast = read_csv(run / "AST_LAST_BLOCK_test_predictions.csv")
    routed = route_predictions(g0, ast)
    write_csv(run / "G0_AST_MUSIC_test_predictions.csv", routed)
    report = compare_fresh(rows, {"G0": g0, "G0_AST_MUSIC": routed})
    uncertainty = {}
    for name in report:
        subset = rows if name == "combined" else [r for r in rows if r["split"] == name]
        baseline = {r["ID"]: r for r in g0}
        candidate = {r["ID"]: r for r in routed}
        uncertainty[name] = {"g0_proxy": _proxy(subset, baseline)["value"],
                             "hybrid_proxy": _proxy(subset, candidate)["value"],
                             "paired_bootstrap": _bootstrap(subset, baseline, candidate)}
    atomic_json(run / "hybrid_comparison.json", {"fresh_test": report, "uncertainty": uncertainty,
        "selection_basis": "dev-only separate lock; protocol registered before root read fresh results",
        "candidate_lock_sha256": digest(run / "hybrid_candidate_lock.json"),
        "scope": "partial-label external validation; not a DACON leaderboard score"})
    print(json.dumps(uncertainty))


if __name__ == "__main__":
    main()
