#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly (no venv, no pip install needed):
#      uv run redesign_notebook_results.py RESULTS/PREDICTIONS
# 3. Or make executable and run:
#      chmod +x redesign_notebook_results.py && ./redesign_notebook_results.py RESULTS/PREDICTIONS
# ──────────────────
"""Exact candidate/seed/ensemble metric readout for the experiment notebook."""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from graph_scoring import official_metrics


def _metrics(path: Path) -> dict[str, float | None]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    identifiers = list(artifact["predictions"])
    labels = [{"ID": identifier, **artifact["labels"][identifier]} for identifier in identifiers]
    predictions = [{"ID": identifier, **artifact["predictions"][identifier]} for identifier in identifiers]
    report = official_metrics(labels, predictions)
    return {"score": report["score"], **report["metrics"]}


def summarize_predictions(directory: Path) -> list[dict[str, str | float | None]]:
    """Report seed metrics, seed mean/SD, and probability-ensemble metrics separately."""
    rows: list[dict[str, str | float | None]] = []
    grouped: dict[str, list[dict[str, float | None]]] = defaultdict(list)
    ensembles: list[tuple[str, dict[str, float | None]]] = []
    for path in sorted(directory.glob("*.json")):
        match = re.fullmatch(r"(.+)_seed(\d+)", path.stem)
        metrics = _metrics(path)
        if match:
            arm, seed = match.groups()
            grouped[arm].append(metrics)
            rows.append({"candidate": arm, "kind": "seed", "seed": seed, **metrics})
        elif path.stem.endswith("_ensemble"):
            ensembles.append((path.stem.removesuffix("_ensemble"), metrics))
    metric_names = ("score", "file_eer", "voice_eer", "music_eer", "voice_presence_auc", "music_presence_auc")
    for arm, values in sorted(grouped.items()):
        aggregate: dict[str, str | float | None] = {"candidate": arm, "kind": "seed_mean_sd", "seed": None}
        for name in metric_names:
            observed = [row[name] for row in values]
            if any(value is None for value in observed):
                aggregate[f"{name}_mean"] = None
                aggregate[f"{name}_sd"] = None
            else:
                array = np.asarray(observed, dtype=float)
                aggregate[f"{name}_mean"] = float(array.mean())
                aggregate[f"{name}_sd"] = float(array.std())
        rows.append(aggregate)
    for arm, metrics in ensembles:
        rows.append({"candidate": arm, "kind": "probability_ensemble", "seed": None, **metrics})
    return rows


def main() -> None:
    """Print the exact JSON readout."""
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/predictions")
    print(json.dumps(summarize_predictions(directory), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
