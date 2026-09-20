"""CPU-only post-run E2 Music/File response analysis over frozen E1 heads."""
# noqa: SIZE_OK - one standalone post-run CLI preserves the preregistered analysis boundary.
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from eight_contract import read_manifest, sha256_file
from eight_ablation import AblationContractError, music_model_rows
from eight_fixed_io import JsonValue, connected_components
from eight_heads import FrozenHead
from eight_metrics import binary_summary, threshold_at_fpr

ARMS = {"parent": "parent", "balanced": "source_generator_parent"}
SEEDS = (41, 42, 43)


class AnalysisError(RuntimeError):
    """Downloaded E1/E2 artifacts violate the preregistered post-run contract."""


def _atomic_json(path: Path, value: Mapping[str, JsonValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _vector(output: Path, row: Mapping[str, str], key: str) -> np.ndarray:
    path = output / "features" / f"{row['ID']}.npz"
    if not path.exists():
        raise AnalysisError(f"missing feature record: {row['ID']}")
    with np.load(path, allow_pickle=False) as saved:
        identity = json.loads(str(saved["identity_json"]))
        valid = np.asarray(saved["valid_samples"]) > 0
        value = np.asarray(saved[key], dtype=np.float32)
    if identity.get("audio_sha256") != row["audio_sha256"] or value.ndim != 2 or valid.shape != (len(value),):
        raise AnalysisError(f"feature identity or shape changed: {row['ID']}/{key}")
    if not valid.any() or not np.isfinite(value[valid]).all():
        raise AnalysisError(f"feature has no finite valid windows: {row['ID']}/{key}")
    return value[valid].mean(0)


def _normalizer(output: Path, rows: list[dict[str, str]], kind: str, key: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if kind == "music":
        try:
            candidates = music_model_rows(rows, ("base",))
        except AblationContractError as error:
            raise AnalysisError(str(error)) from error
        train = [row for row in candidates if row["split"] == "train"]
    else:
        train = [row for row in rows if row["split"] == "train" and row["kind"] == kind
                 and row["variant"] == "base" and row["VOICE_FAKE"] in {"0", "1"}]
    label = "MUSIC_FAKE" if kind == "music" else "VOICE_FAKE"
    if not train or {str(row[label]) for row in train} != {"0", "1"}:
        raise AnalysisError(f"{kind} normalizer requires both labeled training classes")
    matrix = np.stack([_vector(output, row, key) for row in train])
    mean, scale = matrix.mean(0), matrix.std(0)
    scale[scale < 1e-6] = 1.0
    return mean, scale, [row["ID"] for row in train]


def _load_heads(output: Path, branch: str, arm: str, width: int) -> list[FrozenHead]:
    models = []
    for seed in SEEDS:
        path = output / "checkpoints" / f"E1_{branch}_{arm}_seed{seed}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        expected = {"input_width": width, "steps": 500, "batch_size": 32, "seed": seed,
                    "mode": ARMS[arm], "consistency_weight": 0.0, "architecture": "mlp"}
        if any(payload.get(key) != value for key, value in expected.items()):
            raise AnalysisError(f"checkpoint metadata changed: {path.name}")
        counts, digest = payload.get("draw_counts", {}), str(payload.get("draw_sha256", ""))
        if (counts.get("class_0") != 8000 or counts.get("class_1") != 8000 or len(digest) != 64
                or not set(digest) <= set("0123456789abcdef")):
            raise AnalysisError(f"checkpoint exposure metadata changed: {path.name}")
        model = FrozenHead(width, "mlp")
        model.load_state_dict(payload["model"], strict=True); model.eval(); models.append(model)
    return models


def _probability(models: Sequence[FrozenHead], value: np.ndarray) -> float:
    tensor = torch.as_tensor(value, dtype=torch.float32)[None]
    with torch.inference_mode():
        return float(np.mean([model(tensor)[0].sigmoid().item() for model in models]))


def _score(
    output: Path, row: dict[str, str], arm: str, models: dict[str, dict[str, list[FrozenHead]]],
    normalizers: dict[str, tuple[np.ndarray, np.ndarray]], component: str,
) -> dict[str, str | float]:
    music = _probability(models[arm]["music"], (_vector(output, row, "mert_l24") - normalizers["music"][0]) / normalizers["music"][1])
    voice = _probability(models[arm]["voice"], (_vector(output, row, "anti_l24") - normalizers["voice"][0]) / normalizers["voice"][1])
    with np.load(output / "features" / f"{row['ID']}.npz", allow_pickle=False) as saved:
        valid, presence = np.asarray(saved["valid_samples"]) > 0, np.asarray(saved["presence"], dtype=np.float64)
    if presence.shape != (len(valid), 2) or not np.isfinite(presence[valid]).all():
        raise AnalysisError(f"invalid presence record: {row['ID']}")
    vp, mp = presence[valid].max(0).tolist()
    voice_evidence, music_evidence = vp * voice, mp * music
    return {"ID": row["ID"], "split": row["split"], "pair_id": row["pair_id"], "variant": row["variant"],
            "arm": arm, "component_id": component, "FILE_FAKE": row["FILE_FAKE"],
            "VOICE_FAKE": row["VOICE_FAKE"], "MUSIC_FAKE": row["MUSIC_FAKE"],
            "MUSIC_PRESENT": row["MUSIC_PRESENT"], "music_score": music, "voice_score": voice,
            "music_presence": mp, "voice_presence": vp, "music_evidence": music_evidence,
            "voice_evidence": voice_evidence, "file_score": 1 - (1 - voice_evidence) * (1 - music_evidence)}


def _background(variant: str) -> tuple[str, str]:
    for background in ("sonics_fake", "fma_a", "fma_b", "noise"):
        prefix = f"swap_{background}_"
        if variant.startswith(prefix):
            return background, variant.removeprefix(prefix)
    raise AnalysisError(f"unknown E2 swap variant: {variant}")


def _contrasts(scored: list[dict[str, str | float]]) -> list[dict[str, str | float]]:
    grouped: dict[tuple[str, str, str, str], dict[str, dict[str, str | float]]] = defaultdict(dict)
    for item in scored:
        background, level = _background(str(item["variant"]))
        if background != "noise":
            grouped[(str(item["arm"]), str(item["split"]), str(item["pair_id"]), level)][background] = item
    result = []
    for (arm, split, pair, level), values in grouped.items():
        if set(values) != {"fma_a", "fma_b", "sonics_fake"}:
            continue
        fake, real_a, real_b = values["sonics_fake"], values["fma_a"], values["fma_b"]
        if real_a["FILE_FAKE"] != real_b["FILE_FAKE"] or real_a["MUSIC_FAKE"] != real_b["MUSIC_FAKE"]:
            raise AnalysisError(f"two-real-backing labels disagree: {pair}/{level}")
        row: dict[str, str | float] = {"arm": arm, "split": split, "pair_id": pair, "level": level,
            "component_id": str(fake["component_id"]), "file_case": "truth_changing" if real_a["FILE_FAKE"] == "0" else "already_fake"}
        for key in ("music_score", "music_evidence", "voice_score", "voice_evidence", "file_score"):
            row[f"delta_{key}"] = float(fake[key]) - 0.5 * (float(real_a[key]) + float(real_b[key]))
            row[f"control_{key}"] = float(real_a[key]) - float(real_b[key])
        result.append(row)
    return result


def _mean_delta(rows: list[dict[str, str | float]], key: str) -> dict[str, int | float | None]:
    values = [float(row[key]) for row in rows]
    components = sorted({str(row["component_id"]) for row in rows})
    interval = None
    if len(components) >= 5:
        grouped = {component: [float(row[key]) for row in rows if str(row["component_id"]) == component]
                   for component in components}
        rng = np.random.default_rng(41)
        boot = [float(np.mean([value for component in rng.choice(components, len(components), replace=True)
                               for value in grouped[str(component)]])) for _ in range(1000)]
        interval = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
    return {"pairs": len(values), "components": len(components),
            "mean": float(np.mean(values)) if values else None, "median": float(np.median(values)) if values else None,
            "positive_rate": float(np.mean(np.asarray(values) > 0)) if values else None, "ci95": interval,
            "uncertain": len(components) < 5}


def _write_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def run(dataset: str | Path, output: str | Path, analysis_dir: str | Path, target_fpr: float = 0.05) -> dict[str, JsonValue]:
    """Analyze calibration first, lock, then open dev/test swap features."""
    dataset, output, destination = Path(dataset), Path(output), Path(analysis_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows = [dict(row) for row in read_manifest(dataset / "manifest.csv")]
    components = connected_components(rows)
    normalizers_full = {"music": _normalizer(output, rows, "music", "mert_l24"),
                        "voice": _normalizer(output, rows, "speech", "anti_l24")}
    normalizers = {name: values[:2] for name, values in normalizers_full.items()}
    np.savez_compressed(destination / "normalizers.npz", **{f"{name}_{part}": value
                        for name, values in normalizers.items() for part, value in zip(("mean", "scale"), values, strict=True)})
    checkpoint_hashes = {path.name: sha256_file(path) for path in sorted((output / "checkpoints").glob("E1_*_seed*.pt"))
                         if any(path.name.startswith(f"E1_{branch}_{arm}_") for branch in ("music", "voice") for arm in ARMS)}
    if len(checkpoint_hashes) != 12:
        raise AnalysisError("exactly twelve main E1 checkpoints are required")
    receipt = {"manifest_sha256": sha256_file(dataset / "manifest.csv"), "normalizer_sha256": sha256_file(destination / "normalizers.npz"),
               "checkpoint_sha256": checkpoint_hashes,
               "normalizers": {name: {"key": "mert_l24" if name == "music" else "anti_l24", "train_ids": values[2]}
                               for name, values in normalizers_full.items()}}
    _atomic_json(destination / "normalizer_receipt.json", receipt)
    models = {arm: {"music": _load_heads(output, "music", arm, len(normalizers["music"][0])),
                    "voice": _load_heads(output, "voice", arm, len(normalizers["voice"][0]))} for arm in ARMS}
    swaps = [row for row in rows if row["probe_family"] == "swap"]
    calibration_rows = [row for row in swaps if row["split"] == "calibration"]
    calibration = [_score(output, row, arm, models, normalizers, components[row["ID"]]) for row in calibration_rows for arm in ARMS]
    thresholds: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        arm_rows = [row for row in calibration if row["arm"] == arm]
        music_real = [row for row in arm_rows if row["MUSIC_FAKE"] == "0" and _background(str(row["variant"]))[0].startswith("fma_")]
        file_real = [row for row in arm_rows if row["FILE_FAKE"] == "0" and _background(str(row["variant"]))[0].startswith("fma_")]
        if not music_real or not file_real:
            raise AnalysisError(f"calibration lacks REAL FMA controls: {arm}")
        thresholds[arm] = {"music": threshold_at_fpr([0] * len(music_real), [float(row["music_score"]) for row in music_real], target_fpr),
                           "file": threshold_at_fpr([0] * len(file_real), [float(row["file_score"]) for row in file_real], target_fpr)}
    lock = {"status": "locked_before_heldout_feature_read", "target_fpr": target_fpr, "thresholds": thresholds,
            "calibration_ids": sorted({str(row["ID"]) for row in calibration}), "arms": list(ARMS),
            "threshold_calibration_ids": {arm: {
                "music": sorted(str(row["ID"]) for row in calibration if row["arm"] == arm and row["MUSIC_FAKE"] == "0" and _background(str(row["variant"]))[0].startswith("fma_")),
                "file": sorted(str(row["ID"]) for row in calibration if row["arm"] == arm and row["FILE_FAKE"] == "0" and _background(str(row["variant"]))[0].startswith("fma_"))}
                for arm in ARMS},
            "selection": "none; both arms and all three seeds averaged", "test_used": False, "time": time.time(), **receipt}
    lock_path = destination / "selection_E2_music_file_postrun_locked.json"; _atomic_json(lock_path, lock)
    heldout_rows = [row for row in swaps if row["split"] in {"dev", "test"}]
    heldout = [_score(output, row, arm, models, normalizers, components[row["ID"]]) for row in heldout_rows for arm in ARMS]
    scored = calibration + heldout; contrasts = _contrasts(scored)
    reports: dict[str, dict[str, JsonValue]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for split in ("dev", "test"):
            selected = [row for row in scored if row["arm"] == arm and row["split"] == split]
            music = [row for row in selected if row["MUSIC_FAKE"] in {"0", "1"}]
            truth = [row for row in contrasts if row["arm"] == arm and row["split"] == split and row["file_case"] == "truth_changing"]
            file_rows = [row for row in selected if any(row["pair_id"] == item["pair_id"] for item in truth)
                         and _background(str(row["variant"]))[0] != "noise"]
            reports[arm][split] = {"music_threshold_metrics": binary_summary([int(row["MUSIC_FAKE"]) for row in music], [float(row["music_score"]) for row in music], thresholds[arm]["music"]),
                "file_threshold_metrics_truth_changing": binary_summary([int(row["FILE_FAKE"]) for row in file_rows], [float(row["file_score"]) for row in file_rows], thresholds[arm]["file"]),
                "music_response": _mean_delta([row for row in contrasts if row["arm"] == arm and row["split"] == split], "delta_music_score"),
                "music_real_backing_control": _mean_delta([row for row in contrasts if row["arm"] == arm and row["split"] == split], "control_music_score"),
                "truth_changing_file": _mean_delta(truth, "delta_file_score"),
                "file_real_backing_control": _mean_delta(truth, "control_file_score"),
                "already_fake_file": _mean_delta([row for row in contrasts if row["arm"] == arm and row["split"] == split and row["file_case"] == "already_fake"], "delta_file_score")}
    _write_csv(destination / "predictions_E2_music_file.csv", scored)
    _write_csv(destination / "contrasts_E2_music_file.csv", contrasts)
    report: dict[str, JsonValue] = {"status": "complete", "analysis": "postrun_cpu_only", "arms": reports,
        "selection_lock": str(lock_path), "normalizer_receipt": str(destination / "normalizer_receipt.json"),
        "noise_rows_excluded_from_truth_contrasts": sum("noise" in row["variant"] for row in swaps),
        "estimand": "fake SONICS backing minus mean of two real FMA backings for the same vocal and level",
        "claim_limit": "constructed source-confounded diagnostic; separator and presence effects retained; no official score",
        "official_score": None}
    _atomic_json(destination / "experiment_2_music_file_postrun.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path); parser.add_argument("output", type=Path); parser.add_argument("analysis_dir", type=Path)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    arguments = parser.parse_args(); run(arguments.dataset, arguments.output, arguments.analysis_dir, arguments.target_fpr)


if __name__ == "__main__":
    main()
