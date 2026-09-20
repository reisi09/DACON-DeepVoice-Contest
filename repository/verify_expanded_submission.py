"""Check actual packaged predictions against direct learned MIL arithmetic."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from expanded_submission import SEEDS, combine, load_heads, predict_music
from graph_scoring import PREDICTION_COLUMNS, validate_predictions
from v6_data import sha256


def verify(qa: Path, training: Path, output: Path) -> None:
    torch.set_num_threads(4)
    with (qa / "output/submission.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    with (qa / "sample_submission.csv").open() as stream:
        ids = [row["ID"] for row in csv.DictReader(stream)]
    validate_predictions(rows, ids)
    with (qa / "reordered_subset/output/submission.csv").open() as stream:
        subset = list(csv.DictReader(stream))
    assert [row["ID"] for row in subset] == ids[:2][::-1]
    original = {row["ID"]: row for row in rows}
    independence_error = max(abs(float(row[key]) - float(original[row["ID"]][key]))
                             for row in subset for key in PREDICTION_COLUMNS)
    assert independence_error < 1e-7, f"File order/subset affected predictions: {independence_error}"
    paths = [training / f"mil_{seed}.pt" for seed in SEEDS]
    heads = load_heads(paths, "cpu")
    maximum = 0.0
    alternative_gap = 0.0
    probabilities = []
    for index, row in enumerate(rows):
        with np.load(qa / f"features_{index}.npz") as saved:
            features = saved["features"]
        values = torch.from_numpy(features)
        direct, mean_baseline = [], []
        with torch.inference_mode():
            for path in paths:
                state = torch.load(path, map_location="cpu", weights_only=True)["model"]
                normalized = (values - state["mean"]) / state["scale"]
                def mlp(x: torch.Tensor) -> torch.Tensor:
                    hidden = F.gelu(F.linear(x, state["head.network.0.weight"], state["head.network.0.bias"]))
                    return F.linear(hidden, state["head.network.2.weight"], state["head.network.2.bias"]).squeeze(-1)
                logits = mlp(normalized)
                score = logits.logsumexp(0) - torch.tensor(len(logits), dtype=torch.float32).log()
                direct.append(float(score.sigmoid()))
                mean_baseline.append(float(mlp(normalized.mean(0)).sigmoid()))
        expected = float(np.mean(direct))
        loaded = predict_music(heads, features, "cpu")
        duplicated = predict_music(heads, np.tile(features, (2, 1)), "cpu")
        maximum = max(maximum, abs(expected - float(row["MUSIC_FAKE_PROB"])),
                      abs(expected - loaded), abs(expected - duplicated))
        alternative_gap = max(alternative_gap, abs(expected - float(np.mean(mean_baseline))))
        numbers = [float(row[key]) for key in PREDICTION_COLUMNS]
        assert abs(numbers[0] - combine(numbers[1], numbers[2], (numbers[3], numbers[4]))[0]) < 1e-9
        probabilities.append(dict(ID=row["ID"],music_probability=expected,windows=len(features)))
    assert maximum < 1e-6, f"Packaged/direct learned MIL mismatch: {maximum}"
    assert alternative_gap > 1e-4, "Fixture does not distinguish MIL from the old mean-before-head path"
    # Score another file between identical calls; no prior-file state may affect outputs.
    with np.load(qa / "features_0.npz") as saved:
        first = saved["features"]
    assert abs(predict_music(heads, first, "cpu") - probabilities[0]["music_probability"]) < 1e-6
    report = dict(status="passed",files=len(rows),max_probability_error=maximum,
                  tolerance=1e-6,mean_pooling_alternative_gap=alternative_gap,
                  checkpoint_sha256={path.name:sha256(path) for path in paths},
                  probabilities=probabilities,order_subset_max_probability_error=independence_error,
                  device="cpu",official_l4_validated=False,gpu_prediction_parity_verified=False,
                  submission_sha256=sha256(qa / "output/submission.csv"))
    output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--qa",type=Path,required=True)
    parser.add_argument("--training",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    verify(args.qa,args.training,args.output)
