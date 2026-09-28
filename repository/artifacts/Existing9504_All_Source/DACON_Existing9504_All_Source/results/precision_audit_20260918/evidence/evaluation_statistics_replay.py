# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scikit-learn", "torch"]
# ///
# How to run from the repository root (uses its existing dependencies):
# .venv/bin/python results/precision_audit_20260918/evidence/evaluation_statistics_replay.py
"""Replay saved-sampler binding and evaluation-population evidence without training."""

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from eight_ablation import music_model_rows
from eight_fixed_io import connected_components, model_rows
from eight_heads import balanced_draws
from eight_metrics import comparison_status, grouped_bootstrap_delta

EVIDENCE = Path(__file__).resolve().parent
OUTPUT = ROOT / "results/eight_h100_20260917/downloaded/unpacked/output"
MANIFEST = ROOT / "_workspace/eight_h100_20260917/dataset/manifest.csv"
with MANIFEST.open() as stream:
    rows = list(csv.DictReader(stream))

sampler = []
bindings = []
input_paths = {MANIFEST}
for branch in ("music", "voice"):
    cohort = (music_model_rows(rows, ("base",)) if branch == "music" else
              model_rows([row for row in rows if row["kind"] == "speech" and row["variant"] == "base"]))
    train = [row for row in cohort if row["split"] == "train"]
    for mode in ("parent", "source_generator_parent"):
        seeds = (41,) if mode == "parent" else (41, 42, 43)
        for seed in seeds:
            flat = [index for batch in balanced_draws(train, steps=500, batch_size=32, seed=seed, mode=mode)
                    for index in batch]
            selected = [train[index] for index in flat]
            digest = hashlib.sha256("\n".join(row["ID"] for row in selected).encode()).hexdigest()
            arm = "parent" if mode == "parent" else "balanced"
            names = [f"E1_{branch}_{arm}_seed{seed}"]
            if mode == "source_generator_parent":
                names += ([f"E4_mert_seed{seed}", f"E4_spectral_seed{seed}",
                           *[f"E6_mert_l{layer}_seed{seed}" for layer in (6, 12, 24)]] if branch == "music" else
                          [f"E6_anti_l{layer}_seed{seed}" for layer in (12, 24, 48)])
            for name in names:
                checkpoint = OUTPUT / "checkpoints" / (name + ".pt")
                input_paths.add(checkpoint)
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                assert (payload["steps"], payload["batch_size"], payload["seed"], payload["mode"]) == (500, 32, seed, mode)
                assert payload["draw_sha256"] == digest, f"saved draw sequence changed: {checkpoint}"
                if mode == "source_generator_parent":
                    bindings.append({"checkpoint": str(checkpoint.relative_to(ROOT)), "branch": branch, "seed": seed,
                                     "mode": mode, "available_parents": len({row["parent_id"] for row in train}),
                                     "drawn_parents": len({row["parent_id"] for row in selected}),
                                     "computed_draw_sha256": digest, "saved_draw_sha256": payload["draw_sha256"],
                                     "digest_matches": True})
            if seed == 41:
                cells = []
                for label, source, generator in sorted({(row["label"], row["source"], row["generator"]) for row in train}):
                    expected = {row["parent_id"] for row in train
                                if (row["label"], row["source"], row["generator"]) == (label, source, generator)}
                    counts = Counter(row["parent_id"] for row in selected
                                     if (row["label"], row["source"], row["generator"]) == (label, source, generator))
                    cells.append({"label": label, "source": source, "generator": generator,
                                  "available_parents": len(expected), "drawn_parents": len(counts),
                                  "never_drawn_parents": len(expected - counts.keys()),
                                  "draw_count_range_all_parents": [min(counts[parent] for parent in expected), max(counts[parent] for parent in expected)],
                                  "draws": sum(counts.values())})
                sampler.append({"branch": branch, "mode": mode, "train_rows": len(train),
                                "train_parents": len({row["parent_id"] for row in train}),
                                "drawn_parents": len({row["parent_id"] for row in selected}),
                                "checkpoint": str((OUTPUT / "checkpoints" / (names[0] + ".pt")).relative_to(ROOT)),
                                "draw_sha256": digest, "saved_draw_sha256": digest,
                                "digest_matches": True, "cells": cells})

probe_populations = {}
for experiment in ("E2", "E5"):
    prediction_path = OUTPUT / f"predictions_{experiment}.csv"
    input_paths.add(prediction_path)
    with prediction_path.open() as stream:
        predictions = list(csv.DictReader(stream))
    probe_populations[experiment] = {}
    for split in sorted({row["split"] for row in predictions}):
        selected = [row for row in predictions if row["split"] == split]
        labels_by_id = {row["ID"]: row.get("label", row.get("voice_label")) for row in selected}
        probe_populations[experiment][split] = {"table_rows": len(selected), "unique_ids": len(labels_by_id),
            "components": len({row["component_id"] for row in selected}),
            "label_counts": dict(Counter(labels_by_id.values()))}

components = connected_components(rows)
splits = defaultdict(set)
for row in rows:
    splits[components[row["ID"]]].add(row["split"])
population_report = {"global_components": len(splits), "cross_split_components": sum(len(value) > 1 for value in splits.values()), "populations": {}}
for kind in ("music", "voice"):
    selected = [row for row in rows if row["split"] == "test" and row["probe_family"] == "core" and row["variant"] == "base"
                and (row["MUSIC_PRESENT"] == "1" and row["MUSIC_FAKE"] in ("0", "1") if kind == "music"
                     else row["kind"] == "speech" and row["VOICE_FAKE"] in ("0", "1"))]
    label = "MUSIC_FAKE" if kind == "music" else "VOICE_FAKE"
    sizes = Counter(components[row["ID"]] for row in selected)
    population_report["populations"][kind] = {"rows": len(selected), "components": len(sizes),
        "component_row_sizes": sorted(sizes.values()),
        "label_component_support": {value: len({components[row["ID"]] for row in selected if row[label] == value}) for value in ("0", "1")},
        "label_source_rows": dict(Counter(row[label] + "|" + row["source"] for row in selected))}

# Protect against dropping rare-class invalid resamples without disclosing their fraction.
rare = [{"label": int(index == 0), "component_id": str(index), "control": 0.1 if index == 0 else 0.9,
         "candidate": 0.9 if index == 0 else 0.1} for index in range(5)]
bootstrap = grouped_bootstrap_delta(rare, "control", "candidate", n_boot=1000, seed=41)
population_report["rare_class_bootstrap_reproducer"] = {**bootstrap, "invalid": bootstrap["resamples"] - bootstrap["valid"],
    "comparison_status": comparison_status(bootstrap, {}, {}),
    "description": "Five independent groups; only one positive group. Illustrates missing invalid-fraction gate, not observed production result."}

assert len(bindings) == 30
assert population_report["cross_split_components"] == 0
for filename, payload in (("sampler", sampler), ("sampler_binding", bindings),
                          ("probe_populations", probe_populations), ("populations_bootstrap", population_report)):
    (EVIDENCE / f"evaluation_statistics_{filename}.json").write_text(json.dumps(payload, indent=2) + "\n")

input_paths.update(ROOT / filename for filename in ("eight_ablation.py", "eight_fixed_io.py", "eight_heads.py", "eight_metrics.py"))
provenance = {"replay_command": ".venv/bin/python results/precision_audit_20260918/evidence/evaluation_statistics_replay.py",
              "input_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(input_paths)},
              "bound_balanced_checkpoints": len(bindings), "torch_version": torch.__version__,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(EVIDENCE / "evaluation_statistics_replay_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
print(json.dumps({"bound_checkpoints": len(bindings), "global_components": len(splits),
                  "cross_split_components": population_report["cross_split_components"],
                  "sampler_parent_coverage": [{key: row[key] for key in ("branch", "mode", "train_parents", "drawn_parents")} for row in sampler]}))
