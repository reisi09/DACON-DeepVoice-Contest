"""E7: deterministic AntiDeepfake head/LoRA controls with locked evaluation."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import gc, hashlib, json, os, time

import numpy as np
import torch

from antideepfake_runtime import AntiDeepfake, REVISION, verify_native
from eight_lora import (adapter_state_dict, frozen_parameter_checksum, inject_lora,
                        load_adapter_state_dict, reinitialize_adapters, tensor_checksum)
from eight_lora_runtime import (E7Error, HOP, WINDOW, balanced_schedule, load_views,
                                logits, make_windows, predictions, set_adapter_training,
                                smoke, summary, teacher_cache, train)


def _atomic_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _teacher_binding(directory: Path, train_views: list) -> Mapping[str, str | list[dict]]:
    records = [{"ID": view.row["ID"], "audio_sha256": view.row["audio_sha256"],
                "start": view.start, "window": WINDOW} for view in train_views]
    payload = {"native_revision": REVISION, "weights_sha256": _sha256(directory / "model.safetensors"),
               "preprocess": "AntiDeepfake.forward layer_norm; zero-padded 64600 window", "views": records}
    payload["binding_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return payload


def _paired_test_changes(all_predictions: dict, summaries: dict, seeds: list[int]) -> Mapping[str, dict]:
    """Report exact per-seed/per-speaker paired changes without a population CI."""
    result: dict[str, dict] = {}
    for seed in seeds:
        for effect, control, candidate in (
            ("lora_effect", f"head_only_ce_seed{seed}", f"lora_ce_seed{seed}"),
            ("kl_effect", f"lora_ce_seed{seed}", f"lora_ce_kl_seed{seed}"),
        ):
            baseline = {row["ID"]: row for row in all_predictions[control]["test"]}
            changes = []
            for row in all_predictions[candidate]["test"]:
                old = baseline[row["ID"]]
                speaker = next((atom for atom in row["component_id"].split("|")
                                if atom.startswith("speaker:")), row["component_id"])
                changes.append({"ID": row["ID"], "speaker": speaker,
                                "score_delta": row["score"] - old["score"],
                                "absolute_error_delta": abs(row["score"] - row["label"])
                                                        - abs(old["score"] - old["label"])})
            speakers = sorted({row["speaker"] for row in changes})
            result[f"{effect}_seed{seed}"] = {
                "files": len(changes), "speakers": len(speakers), "ci95": None,
                "test_eer_delta": summaries[candidate]["test"]["eer"]
                                  - summaries[control]["test"]["eer"],
                "per_speaker": {speaker: {"n": sum(row["speaker"] == speaker for row in changes),
                    "mean_score_delta": float(np.mean([row["score_delta"] for row in changes
                                                       if row["speaker"] == speaker])),
                    "mean_absolute_error_delta": float(np.mean([row["absolute_error_delta"] for row in changes
                                                                if row["speaker"] == speaker]))}
                    for speaker in speakers},
            }
    return result


def preflight(config_path: str | Path) -> Mapping[str, str | int | dict]:
    """Run one real H100 adapter update and rollback before expensive extraction."""
    config = json.loads(Path(config_path).read_text())
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise E7Error("E7 preflight requires the authorized H100 workspace")
    root, device = Path(config["root"]), "cuda"
    directory = Path(config.get("antideepfake", root / "source" / "antideepfake"))
    parity = verify_native(directory, device)
    splits, population = load_views(config)
    model = AntiDeepfake(directory, device).eval()
    modules = inject_lora(model, rank=8, alpha=16, expected_layers=48)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if len(modules) != 96 or trainable != 1_968_642:
        raise E7Error(f"Unexpected E7 topology: modules={len(modules)}, trainable={trainable}")
    initial = reinitialize_adapters(model, int(config["seeds"][0]))
    result = {"status": "admitted", "native_parity": parity,
              "gpu": torch.cuda.get_device_name(0), "population": population,
              "qv_modules": len(modules), "trainable_parameters": trainable,
              "gradient_and_rollback": smoke(model, splits["train"], initial, device)}
    path = root / "output" / "e7" / "admission_E7.json"
    _atomic_json(path, result)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run(config_path: str | Path) -> Mapping[str, str | int | float | bool | dict | list | None]:
    """Run frozen, head-only, LoRA-CE, and LoRA-CE+KL before locked test scoring."""
    started = time.time()
    config = json.loads(Path(config_path).read_text())
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise E7Error("E7 requires the authorized H100 workspace")
    deadline = float(config["pause_deadline_epoch"]) - float(config.get("export_reserve_seconds", 600))
    if time.time() >= deadline:
        raise TimeoutError("E7 allocation deadline has expired")
    root, device = Path(config["root"]), "cuda"
    output = root / "output" / "e7"
    output.mkdir(parents=True, exist_ok=True)
    directory = Path(config.get("antideepfake", root / "source" / "antideepfake"))

    parity = verify_native(directory, device)
    splits, population = load_views(config)
    model = AntiDeepfake(directory, device).eval()
    modules = inject_lora(model, rank=8, alpha=16, expected_layers=48)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if len(modules) != 96 or trainable != 1_968_642:
        raise E7Error(f"Unexpected E7 topology: modules={len(modules)}, trainable={trainable}")
    pristine = adapter_state_dict(model)
    pristine_checksum = tensor_checksum(pristine)
    frozen_before = frozen_parameter_checksum(model)

    teacher_train, train_embeddings = teacher_cache(model, splits["train"], device)
    binding = _teacher_binding(directory, splits["train"])
    cache_path = output / "teacher_train_logits.pt"
    torch.save({"logits": teacher_train, "binding_sha256": binding["binding_sha256"]}, cache_path)
    _atomic_json(output / "teacher_binding.json", binding)
    teacher_file = predictions(splits["train"], teacher_train)
    contradictions = [row["ID"] for row in teacher_file
                      if (row["score"] >= 0.5) != bool(row["label"])]

    seeds = [int(seed) for seed in config["seeds"]]
    initial_by_seed = {seed: reinitialize_adapters(model, seed) for seed in seeds}
    smoke_report = smoke(model, splits["train"], initial_by_seed[seeds[0]], device)
    load_adapter_state_dict(model, pristine)
    if tensor_checksum(adapter_state_dict(model)) != pristine_checksum:
        raise E7Error("Pristine frozen reference state did not restore after GPU smoke")
    frozen_logits = {split: logits(model, splits[split], device) for split in ("calibration", "dev")}
    all_predictions: dict[str, dict[str, list[dict]]] = {"frozen": {}}
    summaries: dict[str, dict[str, dict]] = {"frozen": {}}
    for split in ("calibration", "dev"):
        all_predictions["frozen"][split] = predictions(splits[split], frozen_logits[split])
        summaries["frozen"][split] = summary(all_predictions["frozen"][split])

    labels = [int(view.row["VOICE_FAKE"]) for view in splits["train"]]
    steps, batch_size = int(config["lora_steps"]), int(config["lora_batch_size"])
    arms = (("head_only_ce", 0.0, False), ("lora_ce", 0.0, True), ("lora_ce_kl", 0.1, True))
    checkpoints: dict[str, Path] = {}
    training: dict[str, dict] = {}
    for arm, kl_weight, adapters in arms:
        set_adapter_training(model, adapters)
        for seed in seeds:
            key = f"{arm}_seed{seed}"
            curve, train_report = train(
                model, splits["train"], teacher_train, initial_by_seed[seed],
                balanced_schedule(labels, steps, batch_size, seed), kl_weight, device, deadline,
                train_embeddings if not adapters else None)
            checkpoint = output / "checkpoints" / f"{key}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            probe_before_reload = logits(model, splits["dev"][:2], device)
            torch.save(adapter_state_dict(model), checkpoint)
            saved_state = torch.load(checkpoint, map_location=device, weights_only=True)
            if tensor_checksum(saved_state) != train_report["adapter_checksum"]:
                raise E7Error(f"Serialized adapter checksum changed: {key}")
            load_adapter_state_dict(model, saved_state)
            reload_exact = torch.equal(logits(model, splits["dev"][:2], device), probe_before_reload)
            if not reload_exact:
                raise E7Error(f"Reloaded checkpoint changed real dev logits: {key}")
            checkpoints[key] = checkpoint
            curve_path = output / "curves" / f"{key}.json"
            _atomic_json(curve_path, curve)
            all_predictions[key], summaries[key] = {}, {}
            for split in ("calibration", "dev"):
                rows = predictions(splits[split], logits(model, splits[split], device))
                all_predictions[key][split] = rows
                summaries[key][split] = summary(rows)
            training[key] = train_report | {"arm": arm, "seed": seed, "kl_weight": kl_weight,
                                             "adapters_trainable": adapters,
                                             "reloaded_dev_logits_exact": reload_exact,
                                             "checkpoint": str(checkpoint), "curve": str(curve_path)}

    arm_dev = {arm: float(np.mean([summaries[f"{arm}_seed{seed}"]["dev"]["eer"]
                                   for seed in seeds])) for arm, _, _ in arms}
    chosen = min(arm_dev, key=lambda name: (arm_dev[name], name))
    thresholds = {key: summaries[key]["calibration"]["threshold"] for key in summaries}
    selection = {"locked_before_test": True, "selection_objective": "lowest mean dev voice EER",
                 "chosen_arm": chosen, "mean_dev_eer": arm_dev,
                 "calibration_thresholds": thresholds, "seeds": seeds}
    selection_path = output / "selection_locked.json"
    _atomic_json(selection_path, selection)

    load_adapter_state_dict(model, pristine)
    if tensor_checksum(adapter_state_dict(model)) != pristine_checksum:
        raise E7Error("Pristine frozen reference state did not restore before test")
    frozen_test = logits(model, splits["test"], device)
    all_predictions["frozen"]["test"] = predictions(splits["test"], frozen_test)
    summaries["frozen"]["test"] = summary(all_predictions["frozen"]["test"], thresholds["frozen"])
    for key, checkpoint in checkpoints.items():
        load_adapter_state_dict(model, torch.load(checkpoint, map_location=device, weights_only=True))
        rows = predictions(splits["test"], logits(model, splits["test"], device))
        all_predictions[key]["test"] = rows
        summaries[key]["test"] = summary(rows, thresholds[key])
    prediction_paths: dict[str, str] = {}
    for key, values in all_predictions.items():
        path = output / "predictions" / f"{key}.json"
        _atomic_json(path, [row for split in ("calibration", "dev", "test") for row in values[split]])
        prediction_paths[key] = str(path)
    paired_changes = _paired_test_changes(all_predictions, summaries, seeds)

    set_adapter_training(model, True)
    frozen_after = frozen_parameter_checksum(model)
    if frozen_after != frozen_before:
        raise E7Error("Frozen encoder byte checksum changed during E7")
    denominators = {split: {str(label): sum(row["label"] == label
                                            for row in all_predictions["frozen"][split])
                            for label in (0, 1)} for split in ("calibration", "dev", "test")}
    report = {
        "status": "complete", "experiment": "E7", "native_parity": parity,
        "population": population, "label_denominators": denominators,
        "component_count": sum(population["speakers"].values()),
        "bootstrap_unit": "provenance-connected group_id; no population CI asserted for two-speaker test",
        "fixed_updates": {"frozen": {"per_seed": 0, "total": 0},
                          **{arm: {"per_seed": steps, "total": steps * len(seeds)}
                             for arm, _, _ in arms}},
        "head_only_cached_encoder_windows": len(train_embeddings),
        "class_exposure": {arm: {"per_seed": {"real_windows": steps, "fake_windows": steps},
                                  "total": {"real_windows": steps * len(seeds),
                                            "fake_windows": steps * len(seeds)}}
                           for arm, _, _ in arms},
        "seeds": seeds, "selection_locked": True,
        "estimand": {"lora_effect": "lora_ce minus head_only_ce",
                     "kl_effect": "lora_ce_kl minus lora_ce"},
        "comparison_scope": "trusted core-speech VOICE_FAKE labels; train base/telephone/mp3_32k; holdout base/opus_telephone",
        "claim_limit": "diagnostic only: test has two speakers; pretraining overlap and historical exposure are unknown",
        "labels": "VOICE_FAKE remains ground truth; native CE target=1-label because class 0 is FAKE",
        "batch_size": batch_size, "window_samples": WINDOW, "hop_samples": HOP,
        "short_audio_padding": "zero",
        "teacher": {"checkpoint": str(directory / "model.safetensors"), "native_revision": REVISION,
                    "cache": str(cache_path), "binding": binding, "temperature": 1.0,
                    "kl": "two-class categorical KL (Bernoulli-equivalent)",
                    "contradicting_files": contradictions, "contradiction_count": len(contradictions),
                    "contradiction_rate": len(contradictions) / len(teacher_file)},
        "topology": {"rank": 8, "alpha": 16, "qv_modules": len(modules),
                     "adapter_parameters": 1_966_080, "classifier_parameters": 2_562,
                     "trainable_parameters": trainable, "gradient_checkpointing": False,
                     "deterministic_eval_mode_with_gradients": True,
                     "frozen_reference_state_checksum": pristine_checksum,
                     "frozen_parameter_checksum_before": frozen_before,
                     "frozen_parameter_checksum_after": frozen_after},
        "gpu_smoke": smoke_report, "training": training, "summaries": summaries,
        "paired_test_changes": paired_changes,
        "uncertainty": {"ci95": None, "reason": "test contains two speakers"},
        "preflight_receipt": str(output / "admission_E7.json") if (output / "admission_E7.json").exists() else None,
        "selection": selection, "selection_lock_path": str(selection_path),
        "per_file_prediction_paths": prediction_paths, "wall_seconds": time.time() - started,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "data_limitations": ["core speech only", "test labels used only after selection lock",
                             "zero-padded common windows are not claimed as native bag parity",
                             "test speaker population is too small for a population CI"],
        "causal_comparison": "paired arms share seed-specific initialization, minibatches, class balance, 200 updates, optimizer, and views",
    }
    _atomic_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    import sys
    print(json.dumps(run(sys.argv[1]), indent=2))
