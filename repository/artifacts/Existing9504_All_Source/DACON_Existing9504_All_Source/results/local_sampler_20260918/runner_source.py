#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "torch", "scikit-learn"]
# ///
# ─── How to run ───
# Reuse this checkout's installed environment: .venv/bin/python local_sampler_compare.py
# Alternatively: uv run local_sampler_compare.py
# ──────────────────
"""Matched CPU sampler comparison on the original, untouched E6 split."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import platform
from collections import Counter
from pathlib import Path
from typing import Final
import zipfile

import numpy as np
import torch
from torch.nn import functional as functional

from eight_fixed_io import atomic_json, connected_components, write_predictions
from eight_heads import FrozenHead, balanced_draws, predict
from eight_metrics import binary_summary, grouped_bootstrap_delta
from mert12_full_refit import eligible_rows, sha256
from results.local_sampler_20260918 import legacy_eight_heads as legacy

OUTPUT: Final = Path("results/local_sampler_20260918")
MANIFEST: Final = Path("_workspace/eight_h100_20260917/dataset/manifest.csv")
FEATURES: Final = Path("_workspace/mert12_a100_20260918/upload/features.zip")
HISTORICAL: Final = Path("results/eight_h100_20260917/downloaded/unpacked/output")
SEEDS: Final = (41, 42, 43)
SPLITS: Final = ("train", "dev", "calibration", "test")


def main() -> None:
    """Train both arms identically except for the corrected parent schedule."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    with MANIFEST.open(newline="") as stream:
        manifest_rows = list(csv.DictReader(stream))
    components = connected_components(manifest_rows)
    rows = eligible_rows(MANIFEST)
    for row in rows:
        row["component_id"] = components[str(row["ID"])]
    split_indices = {split: np.asarray([i for i, row in enumerate(rows) if row["split"] == split])
                     for split in SPLITS}
    assert {split: len(indices) for split, indices in split_indices.items()} == {
        "train": 831, "dev": 164, "calibration": 200, "test": 127}
    component_splits: dict[str, set[str]] = {}
    for row in rows:
        component_splits.setdefault(str(row["component_id"]), set()).add(str(row["split"]))
    assert all(len(splits) == 1 for splits in component_splits.values()), "provenance split leakage"
    vectors, lineage = [], []
    asset_identities = set()
    with zipfile.ZipFile(FEATURES) as archive:
        assert set(archive.namelist()) == {f"{row['ID']}.npz" for row in rows}, "feature ID population changed"
        for row in rows:
            payload = archive.read(f"{row['ID']}.npz")
            with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
                identity = json.loads(str(saved["identity_json"]))
                values = np.asarray(saved["mert_l12"], dtype=np.float32)
                valid = np.asarray(saved["valid_samples"]) > 0
            assert identity["ID"] == row["ID"] and identity["audio_sha256"] == row["audio_sha256"]
            assert values.shape[1] == 1024 and valid.any() and np.isfinite(values[valid]).all()
            asset_identities.add(json.dumps(identity["baseline_assets_sha256"], sort_keys=True))
            vectors.append(values[valid].mean(0))
            lineage.append({**row, "feature_sha256": hashlib.sha256(payload).hexdigest(),
                            "valid_windows": int(valid.sum())})
    assert len(asset_identities) == 1, "mixed frozen feature assets"
    raw = np.stack(vectors).astype(np.float32)
    train_indices = split_indices["train"]
    train_rows = [rows[i] for i in train_indices]
    mean, scale = raw[train_indices].mean(0), raw[train_indices].std(0)
    scale[scale < 1e-6] = 1
    matrix = (raw - mean) / scale
    np.savez_compressed(OUTPUT / "normalization.npz", mean=mean, scale=scale,
                        fit_ids=np.asarray([row["ID"] for row in train_rows]))
    atomic_json(OUTPUT / "input_lineage.json", {"rows": lineage,
                "frozen_asset_sha256": json.loads(next(iter(asset_identities)))})
    atomic_json(OUTPUT / "comparison_lock.json", {
        "feature_zip_sha256": sha256(FEATURES), "manifest_sha256": sha256(MANIFEST),
        "legacy_source_sha256": sha256(Path(legacy.__file__)),
        "corrected_source_sha256": sha256(Path("eight_heads.py")),
        "runner_sha256": sha256(Path(__file__)), "normalization_fit_split": "train831 only",
        "normalization_sha256": sha256(OUTPUT / "normalization.npz"),
        "raw_matrix_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
        "steps": 500, "batch_size": 32, "seeds": list(SEEDS), "architecture": "1024->128 GELU->1",
        "optimizer": "AdamW(lr=0.001,weight_decay=0.01)", "gradient_clip_norm": 1.0,
        "device": "cpu", "torch_version": torch.__version__, "numpy_version": np.__version__,
        "python": platform.python_version(), "cpu_threads": 2, "test_used_for_selection": False,
        "selection": "fixed matched comparison; no hyperparameter or checkpoint selection",
        "scope": "historical held-out diagnostic, already inspected previously; not fresh validation or official score"})
    x = torch.from_numpy(matrix[train_indices])
    y = torch.tensor([row["label"] for row in train_rows], dtype=torch.float32)
    columns, reports, exposure, losses, phase_reports = {}, {}, [], [], {}
    historical_checks = {}
    for arm, sampler in (("legacy", legacy.balanced_draws), ("corrected", balanced_draws)):
        reports[arm], phase_reports[arm] = {}, {}
        for seed in SEEDS:
            draws = sampler(train_rows, steps=500, batch_size=32, seed=seed, mode="source_generator_parent")
            assert all(sum(int(train_rows[i]["label"]) for i in batch) == 16 for batch in draws)
            flat = [i for batch in draws for i in batch]
            draw_hash = hashlib.sha256("\n".join(str(train_rows[i]["ID"]) for i in flat).encode()).hexdigest()
            np.savez_compressed(OUTPUT / f"{arm}_seed{seed}_schedule.npz", indices=np.asarray(draws))
            phase_counts = [Counter(i for batch in draws[start:start + 100] for i in batch)
                            for start in range(0, 500, 100)]
            counts = Counter(flat)
            for i, row in enumerate(train_rows):
                exposure.append({"arm": arm, "seed": seed, "ID": row["ID"], "label": row["label"],
                                 "source": row["source"], "generator": row["generator"], "parent_id": row["parent_id"],
                                 "total": counts[i], **{f"steps_{p*100+1}_{(p+1)*100}": c[i]
                                                       for p, c in enumerate(phase_counts)}})
            phase_reports[arm][str(seed)] = [{"steps": [p*100+1, (p+1)*100], "unique_rows": len(c),
                "zero_exposure_rows": len(train_rows)-len(c), "class_draws": dict(Counter(
                    str(train_rows[i]["label"]) for i, count in c.items() for _ in range(count)))}
                for p, c in enumerate(phase_counts)]
            torch.manual_seed(seed)
            model = FrozenHead(1024)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
            for step, indices in enumerate(draws):
                chosen = torch.as_tensor(indices)
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(x[chosen])
                loss = functional.binary_cross_entropy_with_logits(logits, y[chosen])
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                losses.append({"arm": arm, "seed": seed, "step": step+1,
                               "bce": float(loss.detach()), "gradient_norm_before_clip": float(norm)})
            model.eval()
            score = predict(model, matrix)
            columns[f"{arm}_seed{seed}"] = score
            torch.save({"model": model.state_dict(), "mean": torch.from_numpy(mean), "scale": torch.from_numpy(scale),
                        "draw_sha256": draw_hash, "seed": seed, "steps": 500, "batch_size": 32},
                       OUTPUT / f"{arm}_seed{seed}.pt")
            reports[arm][str(seed)] = {"draw_sha256": draw_hash, "unique_train_rows": len(counts),
                "unseen_train_ids": [str(row["ID"]) for i, row in enumerate(train_rows) if counts[i] == 0],
                "split_metrics": {split: binary_summary([int(rows[i]["label"]) for i in indices], score[indices], 0.5)
                                  for split, indices in split_indices.items()}}
            if arm == "legacy":
                checkpoint_path = HISTORICAL / f"checkpoints/E6_mert_l12_seed{seed}.pt"
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                assert checkpoint["draw_sha256"] == draw_hash, "legacy draw order differs from original E6"
                historical_model = FrozenHead(1024).eval()
                historical_model.load_state_dict(checkpoint["model"], strict=True)
                replay_score = predict(historical_model, matrix)
                columns[f"historical_replay_seed{seed}"] = replay_score
                with (HISTORICAL / "predictions_E6_MERT.csv").open(newline="") as stream:
                    original = {row["ID"]: float(row[f"mert_l12_seed{seed}"]) for row in csv.DictReader(stream)}
                historical_checks[str(seed)] = {"checkpoint_sha256": sha256(checkpoint_path),
                    "draw_hash_exact_match": True,
                    "stored_test_replay_max_abs_error": max(abs(float(replay_score[i])-original[str(rows[i]["ID"])])
                                                           for i in split_indices["test"]),
                    "retrained_cpu_vs_saved_model_max_abs_error": float(np.max(np.abs(score-replay_score)))}
                if seed == 41:
                    reference = legacy.fit_head("legacy_reference_seed41", train_rows, matrix[train_indices],
                        OUTPUT / "reference", steps=500, batch_size=32, seed=seed, mode="source_generator_parent")
                    assert np.array_equal(score, predict(reference.model, matrix)), "training instrumentation changed legacy result"
            print(json.dumps({"arm": arm, "seed": seed, "unique_rows": len(counts),
                              "dev_eer": reports[arm][str(seed)]["split_metrics"]["dev"]["eer"]}), flush=True)
        columns[f"{arm}_ensemble"] = np.mean([columns[f"{arm}_seed{seed}"] for seed in SEEDS], axis=0)
    ensembles, paired = {}, {}
    for split, indices in split_indices.items():
        ensembles[split] = {}
        for arm in ("legacy", "corrected"):
            score = columns[f"{arm}_ensemble"]
            selected = [rows[i] for i in indices]
            ensembles[split][arm] = {**binary_summary([int(row["label"]) for row in selected], score[indices], 0.5),
                "real_false_positive_ids_at_0_5": [str(rows[i]["ID"]) for i in indices
                                                     if rows[i]["label"] == 0 and score[i] >= 0.5],
                "source_slices": {source: binary_summary(
                    [int(rows[i]["label"]) for i in indices if rows[i]["source"] == source],
                    [float(score[i]) for i in indices if rows[i]["source"] == source], 0.5)
                    for source in sorted({str(row["source"]) for row in selected})}}
        if split in ("dev", "test"):
            records = [{**rows[i], "control": float(columns["legacy_ensemble"][i]),
                        "candidate": float(columns["corrected_ensemble"][i])} for i in indices]
            delta = grouped_bootstrap_delta(records, "control", "candidate", n_boot=1000, seed=41)
            paired[split] = {**delta, "invalid_resamples": delta["resamples"]-delta["valid"],
                "invalid_fraction": (delta["resamples"]-delta["valid"])/delta["resamples"] if delta["resamples"] else None,
                "scope": "paired fixed-model probability-ensemble group bootstrap; excludes training and layer-selection uncertainty"}
    for name, records in (("exposure.csv", exposure), ("losses.csv", losses)):
        with (OUTPUT / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    write_predictions(OUTPUT / "predictions.csv", rows, columns)
    atomic_json(OUTPUT / "comparison.json", {"seeds": reports, "ensemble": ensembles,
        "paired_group_bootstrap": paired, "phase_exposure": phase_reports,
        "historical_replay": historical_checks, "official_score": None, "test_used_for_selection": False,
        "threshold_0_5_scope": "diagnostic only; not calibrated or selected", "legacy_loop_parity_seed41": "bitwise identical"})
    print(json.dumps({"status": "complete", "output": str(OUTPUT), "paired": paired}), flush=True)


if __name__ == "__main__":
    main()
