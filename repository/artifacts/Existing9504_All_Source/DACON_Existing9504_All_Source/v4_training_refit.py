"""Recipe lock, complete eligible-data refit, and one-way sealed evaluation."""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import torch

from v4_batches import music_supervision_rows
from v4_data import load_manifest
from v4_model import load_model, parameter_groups
from v4_selection import (accept_sealed, mark_sealed_opened, write_recipe_lock,
    write_sealed_inference_receipt, write_weight_lock)
from v4_training_io import atomic_json, budget_event, digest, g0_predictions
from v4_training_trials import TrialSpec, predict_rows, train_trial, write_predictions


def _hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path): digest(path) for path in paths if path.is_file()}


def _last_exposures(result: Mapping) -> dict[str, int]:
    reports = result.get("reports", [])
    return {} if not reports else {str(key): int(value) for key, value in reports[-1]["cumulative_exposures"].items()}


def refit_and_seal(args, stage_deadlines: Mapping[str, float], selected: str, parent: str,
                   main: Mapping[str, Mapping], chosen: Mapping[str, float], selection_path: Path,
                   train_rows: list[dict[str, str]], dev_rows: list[dict[str, str]],
                   g0: list[dict[str, str | float]], assets: Mapping[str, Path],
                   device: torch.device, output: Path, budget_path: Path) -> dict:
    provenance = json.loads((args.data / "provenance.json").read_text(encoding="utf-8"))
    provenance_by_id = {row["ID"]: row for row in provenance["sources"]}
    resolved = []
    for row in [*train_rows, *dev_rows]:
        full_path = str(provenance_by_id[row["ID"]].get("full_path", ""))
        resolved.append({**row, "path": full_path or row["path"],
                         "refit_path_source": "full_original" if full_path else "declared_derived"})
    eligible = music_supervision_rows(resolved)
    identifiers = [row["ID"] for row in eligible]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("eligible train+dev refit IDs must be unique")
    eligible_manifest = output / "eligible_refit_manifest.csv"
    with eligible_manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("ID", "path", "path_source", "sha256"))
        writer.writeheader()
        writer.writerows({"ID": row["ID"], "path": row["path"],
                          "path_source": row["refit_path_source"],
                          "sha256": digest(args.data / row["path"])} for row in eligible)
    kind = "ast" if selected == "P2" or (selected == "P4" and parent == "P2") else "sonics"
    rate = chosen[kind]
    stage_epochs = ({parent: int(main[parent]["best_epoch"]),
                     "P4": int(main["P4"]["best_epoch"])} if selected == "P4"
                    else {selected: int(main[selected]["best_epoch"])})
    epoch_count = sum(stage_epochs.values())
    data_files = [args.data / name for name in ("train.csv", "dev.csv", "protocol.json",
                                                 "provenance.json", "integrity_report.json")]
    code_files = [Path(name) for name in ("train_v4.py", "v4_model.py",
        "v4_model_components.py", "v4_training.py", "v4_training_data.py",
        "v4_training_trials.py", "v4_training_refit.py", "v4_training_io.py", "v4_batches.py",
        "v4_data.py", "v4_evaluation.py", "v4_selection.py", "graph_features.py",
        "graph_fusion.py", "graph_scoring.py", "v2_scoring.py", "recovery_sonics.py",
        "ast_multitask.py", "v2_data.py")]
    recipe = {"selected": selected, "parent": parent or selected,
              "treatment": "parent_then_temporal_bag" if selected == "P4" else "full_fine_tune",
              "learning_rate": {"encoder": rate, "head": 3 * rate if kind == "sonics" else 1e-4},
              "scheduler": {"warmup_epochs": 1, "horizon_epochs": 30, "minimum_multiplier": .1},
              "epoch_count": epoch_count, "stage_epochs": stage_epochs,
              "data_hashes": _hashes(data_files), "source_hashes": _hashes(code_files),
              "pooling": "normalized_logsumexp_temperature_2" if selected == "P4" else "mean_probability",
              "seeds": {"training": 20260913, "bootstrap": 20260913},
              "gate_thresholds": {"D": .005, "eer_regression": .01, "auc_regression": .01,
                  "bootstrap_win": .80, "bootstrap_p10": 0, "bootstrap_valid": .90,
                  "controlled_file_eer": .02, "refit_safety_factor": 1.25}}
    recipe_path = output / "recipe_lock.json"
    write_recipe_lock(recipe_path, selection_path, eligible_manifest, recipe)
    refit_started = time.time()
    refit_results: list[dict] = []

    def fit(spec: TrialSpec, state: Path | None = None) -> dict:
        model_asset = assets[spec.kind] / ("pytorch_model.bin" if spec.kind == "sonics" else "model.safetensors")
        contract_files = [args.data / "train.csv", args.data / "dev.csv", eligible_manifest,
                          model_asset, *code_files]
        spec = replace(spec, contract_hash="|".join(digest(path) for path in contract_files))
        loaded = load_model(spec.kind, assets[spec.kind], device, state_path=state)
        groups = parameter_groups(loaded)
        result = train_trial(loaded, (groups.encoder, groups.head), spec, eligible, dev_rows, g0,
                             args.data, output / "refit" / spec.name,
                             min(args.deadline, stage_deadlines["refit"]))
        refit_results.append(result)
        return result

    if selected == "P4":
        parent_spec = TrialSpec("parent", kind, rate, 3 * rate if kind == "sonics" else 1e-4,
                                stage_epochs[parent], p3=parent == "P3", patience=0)
        first = fit(parent_spec)
        if first.get("status") == "completed":
            final = fit(TrialSpec("P4", kind, rate, 3 * rate if kind == "sonics" else 1e-4,
                                  stage_epochs["P4"], p4=True, patience=0),
                        Path(str(first["final_weights"])))
        else:
            final = first
    else:
        final = fit(TrialSpec(selected, kind, rate, 3 * rate if kind == "sonics" else 1e-4,
                              epoch_count, p3=selected == "P3", patience=0))
    complete = (len(refit_results) == len(stage_epochs) and all(
        result.get("status") == "completed" and int(result.get("epochs_completed", 0)) == epochs
        for result, epochs in zip(refit_results, stage_epochs.values(), strict=True)))
    actual: dict[str, int] = {identifier: 0 for identifier in identifiers}
    for result in refit_results:
        for identifier, count in _last_exposures(result).items():
            if identifier in actual:
                actual[identifier] += count
    exposures_complete = complete and all(actual[identifier] == epoch_count for identifier in identifiers)
    refit_manifest = output / "refit_manifest.csv"
    with refit_manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("ID", "expected_exposures", "actual_exposures"))
        writer.writeheader()
        writer.writerows({"ID": identifier, "expected_exposures": epoch_count,
                          "actual_exposures": actual[identifier]} for identifier in identifiers)
    audit = {"completed": complete, "exposures_complete": exposures_complete,
             "prescribed_epochs": epoch_count,
             "completed_epochs": sum(int(result.get("epochs_completed", 0)) for result in refit_results),
             "started_unix": refit_started, "completed_unix": time.time(), "stages": refit_results}
    atomic_json(output / "final_weight_audit.json", audit)
    if not complete or not exposures_complete:
        budget_event(budget_path, "refit_incomplete")
        return {"final": "G0", "reason": "refit_incomplete", "sealed_opened": False}
    weights = Path(str(final["final_weights"]))
    weight_lock = output / "weight_lock.json"
    write_weight_lock(weight_lock, recipe_path, weights, code_files, refit_manifest, audit)
    opened = output / "sealed_opened.json"
    mark_sealed_opened(opened, weight_lock, weights)
    sealed_started = time.time()
    sealed_rows = load_manifest(args.data, "sealed")
    sealed_g0 = g0_predictions(sealed_rows, args.data, args.baseline, output / "sealed",
                               min(args.deadline, stage_deadlines["sealed"]))
    loaded = load_model(kind, assets[kind], device, state_path=weights)
    candidate = predict_rows(loaded, sealed_rows, args.data, sealed_g0, p4=selected == "P4",
                             deadline=min(args.deadline, stage_deadlines["sealed"]))
    candidate_path = output / f"{selected}_sealed_predictions.csv"
    write_predictions(candidate_path, candidate)
    g0_path = output / "sealed" / "G0_predictions.csv"
    receipt = output / "sealed_inference_receipt.json"
    write_sealed_inference_receipt(receipt, weight_lock, weights, opened, g0_path, candidate_path,
                                   sealed_started, time.time())
    acceptance = accept_sealed(sealed_rows, selected, weight_lock, weights, opened, receipt)
    atomic_json(output / "sealed_acceptance.json", acceptance)
    budget_event(budget_path, "sealed_complete", accepted=bool(acceptance["accepted"]))
    return {"final": acceptance["final"], "selection": acceptance, "sealed_opened": True}
