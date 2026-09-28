#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "torch", "torchaudio", "safetensors", "transformers", "imageio-ffmpeg", "soundfile", "librosa", "einops", "timm", "fvcore", "scikit-learn", "demucs", "panns-inference"]
# ///
# How to run: uv run train_v4.py --help
"""Run the fixed, budgeted V4 full-fine-tuning comparison."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import torch
import soundfile as sf

from v4_batches import music_supervision_rows
from v4_data import load_manifest
from v4_model import load_model, parameter_groups
from v4_selection import (accept_sealed, mark_sealed_opened, select_development,
    write_recipe_lock, write_sealed_inference_receipt, write_weight_lock)
from v4_training_io import atomic_json, budget_event, digest, g0_predictions, read_predictions
from v4_training_refit import refit_and_seal
from v4_training_trials import (TrialSpec, profile_update, reset_training_seed, train_trial,
                                write_predictions)

SEED = 20_260_913
STAGE_END_MINUTES = {"profile": 8, "g0": 20, "probes": 35, "main": 60, "temporal": 80,
                     "refit": 95, "sealed": 105, "export": 110, "shutdown": 118}
TRAINING_SOURCES = tuple(Path(name) for name in ("train_v4.py", "v4_model.py",
    "v4_model_components.py", "v4_training.py", "v4_training_data.py",
    "v4_training_trials.py", "v4_training_refit.py", "v4_training_io.py", "v4_batches.py",
    "v4_data.py", "v4_evaluation.py", "v4_selection.py", "graph_features.py",
    "graph_fusion.py", "graph_scoring.py", "v2_scoring.py", "recovery_sonics.py",
    "ast_multitask.py", "v2_data.py"))


def _budget_schedule(now: float, hard: float) -> dict[str, float]:
    """Scale every stage into the actual time left without extending the allocation."""
    if hard <= now:
        raise TimeoutError("V4 deadline already reached")
    remaining = hard - now
    total = 60 * STAGE_END_MINUTES["shutdown"]
    return {stage: now + remaining * (60 * minutes / total)
            for stage, minutes in STAGE_END_MINUTES.items()}


def _effective_deadline(provided: float, budget: dict, allocation_start: float) -> float:
    safe_seconds = float(budget.get("safe_shutdown_seconds", 7080))
    declared = float(budget.get("safe_deadline_unix", provided))
    return min(provided, declared, allocation_start + safe_seconds)


def _write_epoch_artifacts(output: Path, outcomes: dict[str, dict]) -> None:
    fields = ("pipeline", "epoch", "optimizer_steps", "rank_score", "status")
    with (output / "epoch_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, outcome in outcomes.items():
            for report in outcome.get("reports", []):
                writer.writerow({"pipeline": name, "epoch": report["epoch"],
                    "optimizer_steps": report["optimizer_steps"], "rank_score": report["rank_score"],
                    "status": outcome["status"]})
    atomic_json(output / "exposures.json", {name: {
        "cumulative_counts": (value.get("reports") or [{}])[-1].get("cumulative_exposures", {}),
        "records": value.get("exposure_records_path", "")} for name, value in outcomes.items()})


def run(args: argparse.Namespace) -> dict:
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    budget_path = output / "budget_events.jsonl"
    runner_start = time.time()
    allocation_start = runner_start
    budget = {}
    budget_file = output / "budget.json"
    if budget_file.is_file():
        budget = json.loads(budget_file.read_text())
        allocation_start = float(budget["allocation_start_unix"])
    args.deadline = _effective_deadline(args.deadline, budget, allocation_start)
    stages = _budget_schedule(runner_start, args.deadline)
    atomic_json(output / "stage_budget_plan.json", {"allocation_start_unix": allocation_start,
        "runner_start_unix": runner_start, "setup_elapsed_seconds": max(0, runner_start - allocation_start),
        "hard_deadline_unix": args.deadline, "runner_seconds_remaining": args.deadline - runner_start,
        "stages": {name: {"canonical_ceiling_minutes": STAGE_END_MINUTES[name],
                           "deadline_unix": deadline,
                           "seconds_after_runner_start": deadline - runner_start}
                   for name, deadline in stages.items()}})
    budget_event(budget_path, "runner_started", deadline=args.deadline,
                 setup_elapsed_seconds=max(0, runner_start - allocation_start))
    train_rows = load_manifest(args.data, "train")
    dev_rows = load_manifest(args.data, "dev")
    train = music_supervision_rows(train_rows)
    if not train:
        raise ValueError("train.csv has no justified music supervision")
    eligible_ids = {row["ID"] for row in train}
    eligibility = [{"ID": row["ID"], "eligible_music_supervision": row["ID"] in eligible_ids,
                    "reason": "known_music_present_and_authenticity" if row["ID"] in eligible_ids
                    else "not_a_known_music_present_authenticity_target"} for row in train_rows]
    atomic_json(output / "training_eligibility.json", {"manifest_rows": len(train_rows),
                "eligible_rows": len(train), "excluded_rows": len(train_rows) - len(train),
                "rows": eligibility})
    durations = sorted(((sf.info(args.data / row["path"]).duration, row) for row in dev_rows),
                       key=lambda item: item[0])
    p95_row = durations[max(0, math.ceil(0.95 * len(durations)) - 1)][1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assets = {"sonics": args.sonics, "ast": args.ast}
    profiles = []
    for kind in ("sonics", "ast"):
        reset_training_seed(SEED)
        loaded = load_model(kind, assets[kind], device)
        groups = parameter_groups(loaded)
        profiles.append(profile_update(loaded, (groups.encoder, groups.head), train_rows, args.data,
                                       p95_row, output / "profile_codec",
                                       stages["profile"]))
        del loaded
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    atomic_json(output / "throughput_profile.json", profiles)
    budget_event(budget_path, "profile_complete", p95_file=p95_row["ID"])
    g0 = g0_predictions(dev_rows, args.data, args.baseline, output / "dev",
                        stages["g0"])
    budget_event(budget_path, "g0_complete", files=len(g0))
    by_kind = {str(profile["kind"]): profile for profile in profiles}
    steps = math.ceil(len(train) / 32)
    probe_seconds = sum(3 * (60 * float(by_kind[kind]["update_seconds"]) * 16
        + len(dev_rows) * float(by_kind[kind]["full_file_inference_seconds"]))
        for kind in ("sonics", "ast"))
    minimum_main_seconds = 5 * steps * 16 * sum(float(by_kind[kind]["update_seconds"])
        for kind in ("sonics", "ast", "sonics")) + 15 * len(dev_rows) * max(
        float(profile["full_file_inference_seconds"]) for profile in profiles)
    atomic_json(output / "pre_search_forecast.json", {"probe_seconds": probe_seconds,
        "minimum_main_seconds": minimum_main_seconds,
        "projected_search_seconds": probe_seconds + minimum_main_seconds,
        "seconds_until_temporal_ceiling": max(0.0, stages["temporal"] - time.time()),
        "fits_before_temporal_ceiling": probe_seconds + minimum_main_seconds <= max(
            0.0, stages["temporal"] - time.time()),
        "policy": "fixed full validation and layer scope; hard stage boundaries censor incomplete trials"})
    outcomes: dict[str, dict] = {}

    def execute(spec: TrialSpec, deadline: float, state: Path | None = None) -> dict:
        model_asset = assets[spec.kind] / ("pytorch_model.bin" if spec.kind == "sonics" else "model.safetensors")
        contract_files = [args.data / "train.csv", args.data / "dev.csv",
                          output / "dev" / "G0_predictions.csv", model_asset, *TRAINING_SOURCES]
        contract_hash = "|".join(digest(path) for path in contract_files)
        spec = replace(spec, contract_hash=contract_hash)
        reset_training_seed(spec.seed)
        loaded = load_model(spec.kind, assets[spec.kind], device, state_path=state)
        groups = parameter_groups(loaded)
        try:
            result = train_trial(loaded, (groups.encoder, groups.head), spec, train_rows, dev_rows,
                                 g0, args.data, output / "training" / spec.name, deadline)
        finally:
            del loaded
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        outcomes[spec.name] = result
        return result

    probe_end = stages["probes"]
    probe_scores: dict[str, list[tuple[float, float]]] = {"sonics": [], "ast": []}
    for kind, rates in (("sonics", (1e-5, 3e-5, 1e-4)), ("ast", (1e-6, 3e-6, 1e-5))):
        for rate in rates:
            name = f"probe_{kind}_{rate:.0e}"
            result = execute(TrialSpec(name, kind, rate, 3 * rate if kind == "sonics" else 1e-4,
                                       epochs=30, max_updates=60, patience=0), probe_end)
            if (result["status"] == "completed" and int(result["updates"]) == 60
                    and math.isfinite(float(result["best_score"]))):
                probe_scores[kind].append((float(result["best_score"]), rate))
    chosen = {kind: max(values, key=lambda item: (item[0], -item[1]))[1]
              for kind, values in probe_scores.items() if values}
    budget_event(budget_path, "probes_complete", completed=sum(map(len, probe_scores.values())))
    main_start, main_end = time.time(), stages["main"]
    main: dict[str, dict] = {}
    p3_epoch_seconds = steps * 16 * (float(by_kind["sonics"]["update_seconds"])
        + float(by_kind["sonics"]["p3_preparation_seconds"]))
    p3_slot_seconds = max(0.0, main_end - main_start) / 3
    p1_epoch_cap = min(30, max(5, math.floor(p3_slot_seconds / max(1e-9, 1.25 * p3_epoch_seconds))))
    atomic_json(output / "augmentation_budget.json", {"p3_profiled_epoch_seconds": p3_epoch_seconds,
        "reserved_seconds": p3_slot_seconds, "safety_factor": 1.25,
        "p1_planned_epoch_cap": p1_epoch_cap,
        "five_epoch_pair_forecast_fits": 5 * 1.25 * p3_epoch_seconds <= p3_slot_seconds})
    if "sonics" in chosen:
        main["P1"] = execute(TrialSpec("P1", "sonics", chosen["sonics"], 3 * chosen["sonics"],
                                         p1_epoch_cap),
                               main_start + max(0, main_end - main_start) / 3)
    if "ast" in chosen:
        main["P2"] = execute(TrialSpec("P2", "ast", chosen["ast"], 1e-4, 30),
                               main_start + 2 * max(0, main_end - main_start) / 3)
    if main.get("P1", {}).get("status") == "completed":
        epochs = int(main["P1"]["epochs_completed"])
        main["P3"] = execute(TrialSpec("P3", "sonics", chosen["sonics"], 3 * chosen["sonics"],
                                         epochs, p3=True, patience=0), main_end)
        if (main["P3"].get("status") == "completed" and
                (main["P3"]["run_total_updates"] != main["P1"]["completed_epoch_updates"] or
                 main["P3"]["run_total_window_exposures"] !=
                 main["P1"]["completed_epoch_window_exposures"])):
            main["P3"]["status"] = "invalid_unmatched_exposure"
    complete_main = {name: value for name, value in main.items()
        if value.get("status") == "completed"
        and (name == "P3" or int(value.get("epochs_completed", 0)) >= 5)}
    parent = max(complete_main, key=lambda name: (float(complete_main[name]["best_score"]),
                 -(("P1", "P2", "P3").index(name))), default="")
    if parent:
        parent_kind = "ast" if parent == "P2" else "sonics"
        parent_rate = chosen[parent_kind]
        parent_weights = Path(str(complete_main[parent]["weights"]))
        p4_steps = math.ceil((math.ceil(len(train) / 2) + len(train) // 2) / 4) * 5
        temporal_end = stages["temporal"]
        continued = execute(TrialSpec(f"{parent}_continued", parent_kind, parent_rate,
                          3 * parent_rate if parent_kind == "sonics" else 1e-4, 5,
                          max_updates=p4_steps, p4=True, p4_control=True, patience=0),
                          (time.time() + temporal_end) / 2, parent_weights)
        main["P4"] = execute(TrialSpec("P4", parent_kind, parent_rate,
                             3 * parent_rate if parent_kind == "sonics" else 1e-4,
                             5, max_updates=p4_steps, p4=True, patience=0), temporal_end, parent_weights)
        atomic_json(output / "p4_comparison.json", {"parent": parent,
            "parent_checkpoint": str(parent_weights), "optimizer_updates": p4_steps,
            "treatment": "partial_fake_bags_with_known_window_auxiliary",
            "control": "same_P4_window_schedule_with_matched_real_splices_mean_parent_loss_no_aux",
            "treatment_run_updates": main["P4"].get("run_total_updates", 0),
            "control_run_updates": continued.get("run_total_updates", 0),
            "treatment_window_exposures": main["P4"].get("run_total_window_exposures", 0),
            "control_window_exposures": continued.get("run_total_window_exposures", 0)})
        if (main["P4"].get("status") == "completed" and
                (main["P4"]["run_total_updates"] != continued["run_total_updates"] or
                 main["P4"]["run_total_window_exposures"] != continued["run_total_window_exposures"])):
            main["P4"]["status"] = "invalid_unmatched_exposure"
    candidates: dict[str, list[dict[str, str | float]]] = {}
    runtime: dict[str, float] = {}
    projections: dict[str, float] = {}
    for name in ("P1", "P2", "P3", "P4"):
        result = main.get(name)
        minimum_epochs = 5 if name in ("P1", "P2", "P4") else 1
        if (not result or result.get("status") != "completed"
                or int(result.get("epochs_completed", 0)) < minimum_epochs):
            continue
        prediction_path = output / "training" / name / f"epoch_{int(result['best_epoch']):03d}_dev_predictions.csv"
        predictions = read_predictions(prediction_path)
        write_predictions(output / f"{name}_dev_predictions.csv", predictions)
        candidates[name] = predictions
        runtime[name] = float(result["seconds"])
        updates = max(1, int(result["updates"]))
        projections[name] = runtime[name] / updates * math.ceil(updates * len(music_supervision_rows([*train_rows, *dev_rows])) / len(train))
        if name == "P4":
            parent_result = main[parent]
            parent_updates = max(1, int(parent_result["updates"]))
            projections[name] += (float(parent_result["seconds"]) / parent_updates
                                  * math.ceil(parent_updates * len(music_supervision_rows([*train_rows, *dev_rows])) / len(train)))
    profile = {name: {"search_seconds": runtime[name], "updates": main[name]["updates"],
                       "projected_complete_refit_seconds": projections[name]} for name in candidates}
    atomic_json(output / "throughput_forecast.json", profile)
    reserved = max(0.0, stages["refit"] - max(time.time(), stages["temporal"]))
    selection = select_development(dev_rows, g0, candidates, runtime, projections, reserved)
    selection_path = output / "selection.json"
    atomic_json(selection_path, selection)
    statuses = {"G0": {"status": "completed", "files": len(g0)}}
    statuses.update({name: {"status": main.get(name, {}).get("status", "not_run"),
                            "reason": "no_equal_completed_evidence" if name not in main else ""}
                     for name in ("P1", "P2", "P3", "P4")})
    atomic_json(output / "pipeline_status.json", statuses)
    trials_path = output / "trials.jsonl"
    temporary = trials_path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for outcome in outcomes.values():
            stream.write(json.dumps(outcome, sort_keys=True, allow_nan=False) + "\n")
        for name, status in statuses.items():
            if name == "G0" or name not in main:
                stream.write(json.dumps({"name": name, **status}, sort_keys=True) + "\n")
    temporary.replace(trials_path)
    _write_epoch_artifacts(output, outcomes)
    selected = str(selection["selected"])
    if selected == "G0":
        budget_event(budget_path, "retained_g0", reason="no_candidate_passed")
        return {"final": "G0", "selection": selection, "sealed_opened": False}
    return refit_and_seal(args, stages, selected, parent, main, chosen, selection_path,
                          train_rows, dev_rows, g0, assets, device, output, budget_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("data", "baseline", "sonics", "ast", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    try:
        result = run(args)
        atomic_json(args.output / "finished.json", {**result, "completed_unix": time.time(),
                    "actual_gpu_metrics": "measured_in_this_run" if torch.cuda.is_available() else "pending"})
    except Exception:  # noqa: BROAD_EXCEPT_OK
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "failure.txt").write_text(__import__("traceback").format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
