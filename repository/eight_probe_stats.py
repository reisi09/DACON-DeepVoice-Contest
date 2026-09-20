"""Small provenance and threshold helpers shared by E2/E5 diagnostics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

import numpy as np

from eight_metrics import threshold_at_fpr


def component_ids(rows: list[dict[str, str]]) -> tuple[dict[str, str], int]:
    """Return provenance-connected component IDs from pipe-delimited atoms."""
    parent: dict[str, str] = {}

    def find(atom: str) -> str:
        parent.setdefault(atom, atom)
        while parent[atom] != atom:
            parent[atom] = parent[parent[atom]]
            atom = parent[atom]
        return atom

    for row in rows:
        atoms = row["group_id"].split("|")
        root = find(atoms[0])
        for atom in atoms[1:]:
            other = find(atom)
            if other != root:
                parent[other] = root
    identifiers = {row["ID"]: find(row["group_id"].split("|")[0]) for row in rows}
    return identifiers, len(set(identifiers.values()))


def e2_threshold_effects(predictions: list[dict], target_fpr: float = 0.05) -> Mapping:
    """Separate REAL false-positive and FAKE recall changes at raw calibration thresholds."""
    thresholds: dict[str, float] = {}
    effects: dict[str, dict[str, float | int | str]] = {}
    for model in ("df", "anti"):
        for pool in ("mean", "max", "top30"):
            calibration = [item for item in predictions if item["model"] == model and item["pool"] == pool
                           and item["split"] == "calibration" and item["voice_label"] in {"0", "1"}]
            if not any(item["voice_label"] == "0" for item in calibration):
                continue
            threshold = threshold_at_fpr([int(item["voice_label"]) for item in calibration],
                                         [float(item["raw_score"]) for item in calibration], target_fpr)
            thresholds[f"{model}/{pool}"] = threshold
            grouped: dict[str, list[dict]] = defaultdict(list)
            for item in predictions:
                if item["model"] == model and item["pool"] == pool and item["voice_label"] in {"0", "1"}:
                    key = "/".join((model, pool, item["split"], item["source"], item["generator"],
                                    item["variant"], "voice_real" if item["voice_label"] == "0" else "voice_fake"))
                    grouped[key].append(item)
            for key, values in grouped.items():
                raw_rate = float(np.mean([float(item["raw_score"]) >= threshold for item in values]))
                recovered_rate = float(np.mean([float(item["vocals_score"]) >= threshold for item in values]))
                effects[key] = {"n": len(values), "metric": "fpr" if values[0]["voice_label"] == "0" else "recall",
                                "raw_rate": raw_rate, "recovered_rate": recovered_rate,
                                "recovered_minus_raw": recovered_rate - raw_rate}
    return {"target_fpr": target_fpr, "raw_calibration_thresholds": thresholds, "effects": effects}
