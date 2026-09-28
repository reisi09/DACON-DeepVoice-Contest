# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "torch"]
# ///
# How to run: PYTHONPATH=. .venv/bin/python results/precision_audit_20260918/training_reproduce.py
"""Replay stored MERT12 inputs/heads; no training, GPU, or source modifications."""
from collections import Counter
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from eight_heads import FrozenHead
from mert12_full_refit import coverage_draws, eligible_rows


def main() -> None:
    """Protect sampling coverage and measure its saved-head in-sample consequence."""
    torch.set_num_threads(2)
    rows = eligible_rows(Path("_workspace/eight_h100_20260917/dataset/manifest.csv"))
    base = Path("results/mert12_a100_20260918/downloaded/unpacked/output/mert12_full")
    receipt = json.loads((base / "full_refit_receipt.json").read_text())
    omitted = set(receipt["checkpoints"][0]["before_zero_exposure_ids"])
    for seed in (41, 42, 43):
        schedule, coverage = coverage_draws(rows, seed)
        assert coverage["after_counts_by_id"] == receipt["checkpoints"][seed - 41]["after_counts_by_id"]
        counts = Counter(index for batch in schedule for index in batch)
        last = {index: step for step, batch in enumerate(schedule) for index in batch}
        subset = [index for index, row in enumerate(rows) if row["ID"] in omitted]
        print(json.dumps({"seed": seed, "unique": len(counts),
                          "prefix_only_rows": len(subset),
                          "prefix_only_labels": dict(Counter(rows[index]["label"] for index in subset)),
                          "prefix_only_last_step": max(last[index] for index in subset),
                          "prefix_only_max_count": max(counts[index] for index in subset)}))
    vectors = []
    with zipfile.ZipFile("_workspace/mert12_a100_20260918/upload/features.zip") as archive:
        for row in rows:
            with np.load(io.BytesIO(archive.read(str(row["ID"]) + ".npz")), allow_pickle=False) as saved:
                vectors.append(saved["mert_l12"][saved["valid_samples"] > 0].mean(0))
    matrix = np.stack(vectors).astype(np.float32)
    predictions = []
    for seed in (41, 42, 43):
        checkpoint = torch.load(base / f"head_seed{seed}.pt", map_location="cpu", weights_only=True)
        model = FrozenHead(1024).eval()
        model.load_state_dict(checkpoint["model"], strict=True)
        standardized = (matrix - checkpoint["mean"].numpy()) / checkpoint["scale"].numpy()
        with torch.inference_mode():
            predictions.append(model(torch.from_numpy(standardized))[0].sigmoid().numpy())
    probability = np.mean(predictions, axis=0)
    errors = [str(row["ID"]) for row, score in zip(rows, probability, strict=True)
              if int(score >= 0.5) != int(row["label"])]
    print(json.dumps({"scope": "training-set diagnostic; not generalization evidence",
                      "errors_at_0_5": len(errors), "errors_in_prefix_only_rows": len(set(errors) & omitted)}))


if __name__ == "__main__":
    main()
