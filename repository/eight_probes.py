"""Paired E2 and sparse-insert E5 diagnostics over shared feature windows."""

from __future__ import annotations

import csv
import json
import math
import time
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import numpy as np

from eight_contract import dataset_manifest, read_manifest
from eight_metrics import binary_summary, threshold_at_fpr
from eight_probe_stats import component_ids, e2_threshold_effects

MODELS: Final = ("df", "anti")
POOLS: Final = ("mean", "max", "top30")


class ProbeError(RuntimeError):
    """Probe inputs cannot support the predeclared paired diagnostic."""


def pool_probabilities(probabilities: np.ndarray) -> dict[str, float]:
    """Apply the three fixed E5 rules to one shared probability schedule."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ProbeError("Probability pooling requires one finite non-empty vector")
    count = max(1, math.ceil(0.3 * len(values)))
    return {"mean": float(values.mean()), "max": float(values.max()), "top30": float(np.sort(values)[-count:].mean())}


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ProbeError(f"Refusing to write an empty diagnostic table: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _load(root: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, np.ndarray]]]:
    rows = [dict(row) for row in read_manifest(dataset_manifest(root))]
    records: dict[str, dict[str, np.ndarray]] = {}
    for row in rows:
        path = root / "output/features" / f"{row['ID']}.npz"
        if not path.exists():
            raise ProbeError(f"Missing shared feature record: {row['ID']}")
        with np.load(path, allow_pickle=False) as saved:
            identity = json.loads(str(saved["identity_json"]))
            if identity["audio_sha256"] != row["audio_sha256"]:
                raise ProbeError(f"Feature/audio identity mismatch: {row['ID']}")
            records[row["ID"]] = {name: saved[name].copy() for name in saved.files if name != "identity_json"}
    return rows, records


def _mark(config: dict, stage: str, **extra: str | int | float) -> None:
    value = {"stage": stage, "time": time.time(), **extra}
    root = Path(config["root"])
    _atomic_json(root / "output/progress.json", value)
    _atomic_json(Path(config["shared"]) / "progress.json", value)


def _run_e2(
    config: dict, rows: list[dict[str, str]], records: dict[str, dict[str, np.ndarray]], components: dict[str, str],
) -> Mapping:
    began = time.time()
    selected = [row for row in rows if row["probe_family"] in {"speech", "swap", "partial"}
                or (row["probe_family"] == "core" and row["kind"] in {"speech", "voice"})]
    component_count = len({components[row["ID"]] for row in selected})
    predictions: list[dict] = []
    for row in selected:
        record = records[row["ID"]]
        for model in MODELS:
            raw, vocals = pool_probabilities(record[f"{model}_raw"]), pool_probabilities(record[f"{model}_vocals"])
            for pooling in POOLS:
                predictions.append({
                    "ID": row["ID"], "pair_id": row["pair_id"], "split": row["split"],
                    "probe_family": "speech" if row["probe_family"] == "core" else row["probe_family"],
                    "voice_mask": "exact_voice" if row["VOICE_FAKE"] in {"0", "1"} else "unknown_voice",
                    "voice_label": row["VOICE_FAKE"], "model": model, "pool": pooling,
                    "raw_score": raw[pooling], "vocals_score": vocals[pooling],
                    "vocals_minus_raw": vocals[pooling] - raw[pooling], "component_id": components[row["ID"]],
                    "variant": row["variant"], "music_present": row["MUSIC_PRESENT"],
                    "source": row["source"], "generator": row["generator"],
                    "vocal_parent_id": json.loads(row["parent_ids"])[0],
                })
    if not predictions:
        report = {"status": "data_insufficient", "population": 0, "reason": "no speech/swap/partial rows",
                  "seeds": [], "fixed_step_counts": 0, "selection_objective": None,
                  "data_limitations": ["no paired rows"], "per_file_predictions": None,
                  "wall_time_seconds": time.time() - began, "causal_comparison": None,
                  "label_denominators": {}, "component_count": 0, "bootstrap_unit": "provenance component",
                  "fixed_updates": 0, "class_exposure": {}, "selection_locked": True,
                  "estimand": "recovered-vocal minus raw detector score", "comparison_scope": "none", "claim_limit": "no data"}
    else:
        grouped: dict[str, list[float]] = defaultdict(list)
        references = {(item["ID"], item["model"], item["pool"]): item["raw_score"] for item in predictions}
        reference_grouped: dict[str, list[float]] = defaultdict(list)
        for item in predictions:
            label = "voice_real" if item["voice_label"] == "0" else "voice_fake" if item["voice_label"] == "1" else "voice_unknown"
            key = "/".join((item["model"], item["pool"], item["split"], item["source"], item["generator"],
                            item["variant"], item["probe_family"], label))
            grouped[key].append(float(item["vocals_minus_raw"]))
            reference = references.get((item["vocal_parent_id"], item["model"], item["pool"]))
            item["voice_reference_score"] = "" if reference is None else reference
            item["raw_minus_voice_reference"] = "" if reference is None else float(item["raw_score"]) - float(reference)
            item["vocals_minus_voice_reference"] = "" if reference is None else float(item["vocals_score"]) - float(reference)
            if reference is not None:
                reference_grouped[key].append(float(item["vocals_score"]) - float(reference))
        exact = [row for row in selected if row["VOICE_FAKE"] in {"0", "1"}]
        reference_coverage = sum(item["variant"] != "base" and item["voice_reference_score"] != "" for item in predictions)
        report = {
            "status": "complete" if reference_coverage else "data_insufficient", "population": len(selected), "paired_rows": len(predictions),
            "comparison": "primary same-window raw mixture versus recovered vocal; vocals_minus_raw also measures separator shift",
            "summaries": {key: {"n": len(values), "mean_delta": float(np.mean(values)), "median_delta": float(np.median(values))}
                          for key, values in sorted(grouped.items())},
            "voice_reference_summaries": {key: {"n": len(values), "mean_delta": float(np.mean(values))}
                                          for key, values in sorted(reference_grouped.items())},
            "voice_reference_nonbase_rows": reference_coverage,
            "calibration_fixed_label_effects": e2_threshold_effects(predictions, float(config.get("e2_target_fpr", 0.05))),
            "seeds": [], "fixed_step_counts": 0,
            "selection_objective": None, "data_limitations": [
                "unknown_voice rows are reported separately and never assigned a voice truth label",
                "background vocal presence is not independently reviewed; recovered-vocal changes are not pure backing isolation",
            ],
            "per_file_predictions": "predictions_E2.csv", "wall_time_seconds": time.time() - began,
            "causal_comparison": "raw and recovered-vocals views use identical window starts and the same frozen detector",
            "label_denominators": {"voice_known": len(exact), "voice_unknown": len(selected) - len(exact),
                                   "voice_real": sum(row["VOICE_FAKE"] == "0" for row in exact),
                                   "voice_fake": sum(row["VOICE_FAKE"] == "1" for row in exact)},
            "component_count": component_count, "bootstrap_unit": "provenance component",
            "fixed_updates": 0, "class_exposure": {"same_frozen_models": len(selected)}, "selection_locked": True,
            "estimand": "within-file recovered-vocal minus raw score, plus same-pair voice-reference shifts",
            "comparison_scope": "core speech (including processed variants), swap, and partial diagnostic rows; core music excluded",
            "claim_limit": "separator shift diagnostic; backing vocal presence is unreviewed and no isolated-backing claim is made",
        }
        _write_csv(Path(config["root"]) / "output/predictions_E2.csv", predictions)
    _atomic_json(Path(config["root"]) / "output/experiment_2.json", report)
    return report


def _event_stratum(row: dict[str, str]) -> tuple[float, str]:
    events = json.loads(row["event_intervals"])
    fake = [event for event in events if str(event.get("label", "")) in {"1", "fake"}]
    inserted = sum(float(event["end"]) - float(event["start"]) for event in fake)
    if not fake:
        return 0.0, "real_control"
    duration, midpoint = float(row["duration_seconds"]), (float(fake[0]["start"]) + float(fake[-1]["end"])) / 2
    position = "start" if midpoint < duration / 3 else "end" if midpoint > 2 * duration / 3 else "middle"
    return inserted, position


def _run_e5(
    config: dict, rows: list[dict[str, str]], records: dict[str, dict[str, np.ndarray]], components: dict[str, str],
) -> Mapping:
    began = time.time()
    source = [row for row in rows if row["probe_family"] == "partial" and row["FILE_FAKE"] in {"0", "1"}]
    component_count = len({components[row["ID"]] for row in source})
    predictions: list[dict] = []
    for row in source:
        inserted, position = _event_stratum(row)
        item = {"ID": row["ID"], "split": row["split"], "label": int(row["FILE_FAKE"]),
                "duration_seconds": float(row["duration_seconds"]), "insert_seconds": inserted,
                "position": position, "component_id": components[row["ID"]]}
        for model in MODELS:
            item.update({f"{model}_{pool}": score for pool, score in pool_probabilities(records[row["ID"]][f"{model}_raw"]).items()})
        predictions.append(item)
    calibration = [item for item in predictions if item["split"] == "calibration"]
    dev = [item for item in predictions if item["split"] == "dev"]
    if (not calibration or not dev or not any(item["label"] == 0 for item in calibration)
            or {item["label"] for item in dev} != {0, 1}):
        report = {"status": "data_insufficient", "population": len(source),
                  "reason": "E5 requires calibration REAL rows and labeled dev rows before test is opened",
                  "seeds": [], "fixed_step_counts": 0, "selection_objective": None,
                  "data_limitations": ["sealed test was not opened"], "per_file_predictions": None,
                  "wall_time_seconds": time.time() - began, "causal_comparison": None,
                  "label_denominators": {"real": sum(item["label"] == 0 for item in predictions),
                                         "fake": sum(item["label"] == 1 for item in predictions)},
                  "component_count": component_count,
                  "bootstrap_unit": "provenance component", "fixed_updates": 0,
                  "class_exposure": {}, "selection_locked": False,
                  "estimand": "sparse-fake recall at calibration-fixed FPR", "comparison_scope": "partial probes",
                  "claim_limit": "sealed test not opened"}
        _atomic_json(Path(config["root"]) / "output/experiment_5.json", report)
        return report
    target_fpr = float(config.get("e5_target_fpr", 0.05))
    thresholds, dev_reports, selected = {}, {}, {}
    for model in MODELS:
        candidates = []
        for pool in POOLS:
            key = f"{model}_{pool}"
            threshold = threshold_at_fpr([item["label"] for item in calibration], [item[key] for item in calibration], target_fpr)
            summary = binary_summary([item["label"] for item in dev], [item[key] for item in dev], threshold)
            thresholds[key], dev_reports[key] = threshold, summary
            candidates.append((float(summary["recall"] if summary["recall"] is not None else -1), -float(summary["fpr"] or 0), -POOLS.index(pool), pool))
        selected[model] = max(candidates)[-1]
    lock = {"experiment": 5, "selected_pool_by_model": selected, "thresholds": thresholds,
            "selection_data": "calibration thresholds then dev recall/fpr only", "test_used": False, "time": time.time()}
    lock_path = Path(config["root"]) / "output/selection_E5_locked.json"
    _atomic_json(lock_path, lock)
    sealed = [item for item in predictions if item["split"] == "test"]
    test_reports = {key: binary_summary([item["label"] for item in sealed], [item[key] for item in sealed], threshold)
                    for key, threshold in thresholds.items()} if sealed else {}
    strata: dict[str, dict[str, int | float | None]] = {}
    for model, pool in selected.items():
        key = f"{model}_{pool}"
        for duration, inserted, position in sorted({(item["duration_seconds"], item["insert_seconds"], item["position"]) for item in sealed}):
            subset = [item for item in sealed if (item["duration_seconds"], item["insert_seconds"], item["position"]) == (duration, inserted, position)]
            strata[f"{model}/{duration:g}s/{inserted:g}s/{position}"] = binary_summary(
                [item["label"] for item in subset], [item[key] for item in subset], thresholds[key])
    long_real = {key: summary["fpr"] for key, summary in test_reports.items() if summary["negative_n"]}
    _write_csv(Path(config["root"]) / "output/predictions_E5.csv", predictions)
    report = {
        "status": "complete", "population": len(source), "calibration_n": len(calibration), "dev_n": len(dev), "sealed_test_n": len(sealed),
        "pooling_rules": list(POOLS), "threshold_target_fpr": target_fpr, "thresholds": thresholds,
        "dev_selection": dev_reports, "selected_pool_by_model": selected, "sealed_test": test_reports,
        "duration_insert_position": strata, "long_real_fpr": long_real,
        "selection_lock": str(lock_path), "seeds": [], "fixed_step_counts": 0,
        "selection_objective": "maximize dev sparse-fake recall, then lower dev FPR, fixed pool order tie-break",
        "data_limitations": ["constructed composites are diagnostics, not natural long recordings"],
        "per_file_predictions": "predictions_E5.csv", "wall_time_seconds": time.time() - began,
        "causal_comparison": "mean/max/top30 pool exactly the same detector windows",
        "label_denominators": {"real": sum(item["label"] == 0 for item in predictions),
                               "fake": sum(item["label"] == 1 for item in predictions)},
        "component_count": component_count,
        "bootstrap_unit": "provenance component", "fixed_updates": 0,
        "class_exposure": {split: {"real": sum(item["split"] == split and item["label"] == 0 for item in predictions),
                                    "fake": sum(item["split"] == split and item["label"] == 1 for item in predictions)}
                           for split in ("calibration", "dev", "test")},
        "selection_locked": True, "estimand": "sparse-fake recall and long-REAL FPR at calibration-fixed thresholds",
        "comparison_scope": "same windows across mean/max/top30, stratified by duration, insert length, and position",
        "claim_limit": "constructed composites only; durations, positions, and windows are not independent observations",
    }
    _atomic_json(Path(config["root"]) / "output/experiment_5.json", report)
    return report


def run(config_path: str | Path, experiment: int | None = None) -> dict[str, dict]:
    """Run E2, E5, or both without permitting any test-dependent selection."""
    if experiment not in {None, 2, 5}:
        raise ProbeError("Probe experiment must be 2, 5, or omitted")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    rows, records = _load(Path(config["root"]))
    components, _ = component_ids(rows)
    result: dict[str, dict] = {}
    if experiment in {None, 2}:
        _mark(config, "experiment_2")
        result["E2"] = _run_e2(config, rows, records, components)
    if experiment in {None, 5}:
        _mark(config, "experiment_5")
        result["E5"] = _run_e5(config, rows, records, components)
    return result
