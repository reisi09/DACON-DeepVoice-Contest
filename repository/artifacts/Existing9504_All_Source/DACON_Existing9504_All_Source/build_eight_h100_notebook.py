"""Build the ordered, auditable eight-experiment H100 notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "DACON_Eight_H100.ipynb"


def markdown(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(source.strip())


def code(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(source.strip())


def build() -> None:
    cells = [
        markdown(
            """
# Eight controlled audio-deepfake diagnostics on one H100

This notebook executes E1-E8 in the preregistered order. Feature extraction is
inference with frozen pretrained encoders. E1/E3/E4/E6/E8 fit small diagnostic
heads, and E7 learns LoRA adapters. These are different evidence types.

Labels remain source/construction metadata unless the prepared manifest says
otherwise. Results are diagnostics on the recorded populations, not an official
DACON leaderboard score. Missing labels stay unknown and are never imputed.
Reports distinguish exposed historical diagnostics, previously used `fresh_*`
cohorts, the source-confounded FMA-real versus SONICS-fake stress panel, and
constructed probes. None is described as a fresh holdout.
"""
        ),
        markdown(
            """
## Context and fixed execution contract

The CPU-prepared dataset is validated here; it is not rebuilt on paid GPU time.
All test scoring happens only after development/calibration selection is sealed.
The run has an immutable four-hour deadline with a 600-second export reserve.
"""
        ),
        code(
            """
import json, os, time
from pathlib import Path

import torch

CONFIG_PATH = Path(os.environ.get("EIGHT_CONFIG", "/root/eight_h100_20260917/config.json"))
config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
root = Path(config["root"])
dataset = root / "source" / "dataset"
assert 0 < config["pause_deadline_epoch"] - config["started_epoch"] <= 14_400
assert config["export_reserve_seconds"] == 600
assert time.time() < config["pause_deadline_epoch"] - config["export_reserve_seconds"]
assert torch.cuda.device_count() == 1
assert "H100" in torch.cuda.get_device_name(0)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
{"config": config, "gpu": torch.cuda.get_device_name(0), "dataset": str(dataset)}
"""
        ),
        markdown("## Validate the prepared dataset"),
        code(
            """
from eight_contract import validate_manifest

dataset_receipt = validate_manifest(dataset)
dataset_receipt
"""
        ),
        markdown("## Extract shared features once"),
        code(
            """
import eight_features
import eight_lora_train

feature_preflight = eight_features.preflight(CONFIG_PATH)
lora_preflight = eight_lora_train.preflight(CONFIG_PATH)
{"features": feature_preflight, "lora": lora_preflight}
"""
        ),
        code(
            """

feature_receipt = eight_features.extract(CONFIG_PATH)
feature_receipt
"""
        ),
        markdown("## E1: balanced frozen-feature reference"),
        code(
            """
import eight_ablation

e1 = eight_ablation.run(CONFIG_PATH, 1)
e1
"""
        ),
        markdown("## E2 and E5: paired mixture and sparse-event probes"),
        code(
            """
import eight_probes

e2 = eight_probes.run(CONFIG_PATH, 2)
e5 = eight_probes.run(CONFIG_PATH, 5)
{"E2": e2, "E5": e5}
"""
        ),
        markdown("## E3, E4, E6 and E8: fixed-feature controlled comparisons"),
        code(
            """
fixed_results = {number: eight_ablation.run(CONFIG_PATH, number) for number in (3, 4, 6, 8)}
fixed_results
"""
        ),
        markdown("## E7: frozen and LoRA comparisons"),
        code(
            """
e7 = eight_lora_train.run(CONFIG_PATH)
e7
"""
        ),
        markdown(
            """
## Results

The chart below is generated only from numeric values present in sealed result
JSON files. It is descriptive: metrics with different units are not combined.
The individual reports and per-file predictions are the audit source.
"""
        ),
        code(
            """
import csv
import matplotlib.pyplot as plt
from eight_fixed_io import connected_components

manifest_rows = list(csv.DictReader((dataset / "manifest.csv").open(encoding="utf-8")))
component_count = len(set(connected_components(manifest_rows).values()))
label_denominators = {
    label: {split: {"known": sum(row[label] in {"0", "1"} and row["split"] == split for row in manifest_rows),
                    "unknown": sum(row[label] == "" and row["split"] == split for row in manifest_rows)}
            for split in ("train", "calibration", "dev", "test")}
    for label in ("FILE_FAKE", "VOICE_FAKE", "MUSIC_FAKE", "VOICE_PRESENT", "MUSIC_PRESENT")
}
source_counts = {}
probe_counts = {}
for row in manifest_rows:
    source_counts[row["source"]] = source_counts.get(row["source"], 0) + 1
    probe_counts[row["probe_family"]] = probe_counts.get(row["probe_family"], 0) + 1

paths = {int(path.stem.split("_")[-1]): path for path in (root / "output").glob("experiment_*.json")}
paths[7] = root / "output" / "e7" / "report.json"
claims = {
    1: ("balanced-sampler head difference", "same prepared labels/features and fixed updates", "historical/source-confounded diagnostic"),
    2: ("raw-minus-recovered-vocal score shift", "paired identical windows", "separator-view diagnostic"),
    3: ("CE-plus-consistency difference", "same transformed views and fixed updates", "unseen-chain diagnostic within recorded sources"),
    4: ("fixed spectral fusion difference", "dev-selected fusion on grouped split", "complementarity diagnostic"),
    5: ("pooling-rule sparse-event sensitivity", "same windows with calibration-locked thresholds", "constructed-composite diagnostic"),
    6: ("encoder-layer head difference", "same grouped split, heads, and updates", "layer probe only"),
    7: ("CE-plus-KL LoRA difference", "same views, schedules, seeds, and initialization", "bounded speech adaptation diagnostic"),
    8: ("REAL-only distance behavior", "train-REAL fit and calibration-fixed polarity", "shifted-REAL false-positive diagnostic"),
}
aggregation = {"official_score": None, "fresh_holdout_claim": False, "robustness_claim": False,
               "dataset": {"rows": len(manifest_rows), "component_count": component_count,
                           "label_denominators": label_denominators, "source_counts": source_counts,
                           "probe_family_counts": probe_counts}, "experiments": {}}
prediction_counts = []
for number in range(1, 9):
    path = paths[number]
    report = json.loads(path.read_text(encoding="utf-8"))
    prediction_files = sorted((root / "output").glob(f"**/*prediction*E{number}*"))
    if number == 7:
        prediction_files = sorted((root / "output" / "e7" / "predictions").glob("*.json"))
    prediction_count = 0
    for prediction in prediction_files:
        if prediction.suffix == ".csv":
            prediction_count += max(0, sum(1 for _ in prediction.open(encoding="utf-8")) - 1)
        elif prediction.suffix == ".json":
            value = json.loads(prediction.read_text(encoding="utf-8"))
            prediction_count += len(value) if isinstance(value, list) else 0
    lock = root / "output" / f"selection_E{number}_locked.json"
    if number == 7:
        lock = root / "output" / "e7" / "selection_locked.json"
    selection_locked = lock.is_file() if number != 2 else "predeclared_no_selection"
    estimand, scope, limit = claims[number]
    aggregation["experiments"][f"E{number}"] = {
        "status": report.get("status", "missing_status"), "report": str(path),
        "prediction_files": [str(path) for path in prediction_files], "prediction_rows": prediction_count,
        "component_count": report.get("component_count"),
        "label_denominators": report.get("label_denominators", {}),
        "raw_report_metadata_scope": report.get(
            "population_scope",
            "report-defined; not a substitute for branch-specific population denominators",
        ),
        "bootstrap_unit": report.get("bootstrap_unit"), "population": report.get("population"),
        "fixed_updates": report.get("fixed_updates", report.get("fixed_step_counts")),
        "class_exposure": report.get("class_exposure"),
        "seeds": report.get("seeds", []),
        "selection_locked": report.get("selection_locked", selection_locked),
        "estimand": estimand, "comparison_scope": scope, "claim_limit": limit,
    }
    prediction_counts.append((f"E{number}", prediction_count))

(root / "output" / "final_aggregation.json").write_text(json.dumps(aggregation, indent=2, sort_keys=True) + "\\n")
figure, axis = plt.subplots(figsize=(9, 4.8))
axis.bar([name for name, _ in prediction_counts], [count for _, count in prediction_counts], color="#226B8A")
axis.set_ylabel("Exported per-file prediction rows")
axis.set_title("Durable prediction evidence by experiment")
axis.grid(axis="y", alpha=0.25)
figure.tight_layout()
figure.savefig(root / "output" / "prediction_evidence_counts.png", dpi=160)
plt.show()
aggregation
"""
        ),
        markdown(
            """
## Takeaways and evidence boundary

Interpret only comparisons whose report status is complete and whose population,
split/component counts, label-mask denominators, component-bootstrap unit,
fixed updates/class exposure/seeds, locked selection, estimand, comparison scope,
claim limit, and uncertainty are recorded. A failed or data-insufficient
experiment counts only as explicit partial evidence. No result establishes broad
robustness, a fresh holdout, or an official score. Historical/pretraining overlap
remains unknown unless a report supplies direct evidence.
"""
        ),
    ]
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Eight H100", "language": "python", "name": "eight-h100"},
            "language_info": {"name": "python", "version": "3.11"},
        },
    )
    nbformat.validate(notebook)
    nbformat.write(notebook, NOTEBOOK)


if __name__ == "__main__":
    build()
