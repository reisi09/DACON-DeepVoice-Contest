#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.7.1", "numpy==2.4.6"]
# ///
# ─── How to run ───
# From the repository, reuse the installed CPU-capable runtime:
# .venv/bin/python results/precision_audit_20260918/evidence/evaluation_oof_replay.py
# Optional isolated runtime: uv run --script evaluation_oof_replay.py
# Install uv if needed: curl -LsSf https://astral.sh/uv/install.sh | sh
# Reads saved artifacts only; writes evaluation_oof_verification.json beside this file.
# ──────────────────
"""Replay saved OOF population, leakage, checkpoint, and ensemble checks on CPU."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import platform
import shlex
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    """Hash an input without loading the entire checkpoint into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    """Write reproducible evidence and fail if a measured OOF invariant breaks."""
    script = Path(__file__).resolve()
    root = script.parents[3]
    run = root / "results/redesign_h100_20260917"
    base = run / "downloaded/unpacked/output/experiments"
    source_path = run / "sources.csv"
    with source_path.open(encoding="utf-8", newline="") as stream:
        sources = {row["ID"]: row for row in csv.DictReader(stream)}
    audit_path = run / "downloaded/unpacked/evidence/data_audit_copy.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    summary_path = base / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    input_paths = [source_path, audit_path, summary_path, script]
    report = {
        "replay_command": shlex.join([str(root / ".venv/bin/python"), str(script)]),
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "numpy": np.__version__, "device": "cpu", "inference_run": False},
        "scope": "Saved A/B head OOF provenance and arithmetic; no score or quality claim",
        "accepted_input_rows": len(sources), "split_rows": audit["split_rows"],
        "voice_status": summary["voice"], "seed_artifacts": [], "ensembles": [],
    }
    all_checks = []
    for path in sorted((base / "oof").glob("*_seed*.json")):
        artifact = json.loads(path.read_text(encoding="utf-8"))
        validation_path = base / "predictions" / path.name
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        input_paths.extend([path, validation_path])
        identifiers = set(artifact["predictions"])
        groups = {group for row in artifact["groups"].values() for group in row}
        validation_groups = {group for row in validation["groups"].values() for group in row}
        held_counts = Counter(row for fold in artifact["folds"] for row in fold["held_out_ids"])
        checks = {
            "train_population_count": len(identifiers) == audit["split_rows"]["train"],
            "known_source_ids": identifiers <= sources.keys(),
            "all_rows_held_once": set(held_counts) == identifiers and set(held_counts.values()) == {1},
            "artifact_sections_same_ids": all(set(artifact[key]) == identifiers for key in
                                              ("labels", "groups", "strata", "window_scores")),
            "global_validation_rows_disjoint": not identifiers & validation["predictions"].keys(),
            "global_validation_groups_disjoint": not groups & validation_groups,
            "training_manifest_matches_audit": artifact["upstream_identity"]["training_manifest_sha256"]
                                               == audit["split_manifest_sha256"]["train"],
        }
        item = {
            "path": str(path.relative_to(root)), "sha256": sha256(path),
            "rows": len(identifiers), "groups": len(groups),
            "held_count_distribution": dict(Counter(held_counts.values())),
            "labels": {key: dict(Counter(str(row[key]) for row in artifact["labels"].values()))
                       for key in next(iter(artifact["labels"].values()))},
            "strata": {key: dict(Counter(row[key] for row in artifact["strata"].values()))
                       for key in ("kind", "domain", "view")},
            "validation_overlap_rows": len(identifiers & validation["predictions"].keys()),
            "validation_overlap_groups": len(groups & validation_groups),
            "checks": checks, "folds": [], "current_source_hash_comparison": {},
        }
        for name, saved_digest in artifact["upstream_identity"]["source_sha256"].items():
            current = root / name
            input_paths.append(current)
            item["current_source_hash_comparison"][name] = {
                "saved_sha256": saved_digest, "current_sha256": sha256(current),
                "matches": saved_digest == sha256(current),
            }
        for fold in artifact["folds"]:
            rows = [set(fold[key]) for key in ("held_out_ids", "train_ids", "earlystop_ids")]
            partitions = [set(fold[key]) for key in
                          ("held_out_groups", "trained_groups", "earlystop_groups")]
            pairs = list(itertools.combinations(range(3), 2))
            row_overlaps = [len(rows[i] & rows[j]) for i, j in pairs]
            group_overlaps = [len(partitions[i] & partitions[j]) for i, j in pairs]
            lineage = []
            for section in rows:
                lineage.append({
                    "parent_ids": {parent for identifier in section
                                   for parent in json.loads(sources[identifier]["parent_ids"])},
                    **{key: {sources[identifier][key] for identifier in section
                             if sources[identifier][key]}
                       for key in ("artist_id", "speaker_id", "audio_sha256")},
                })
            source_overlaps = {key: [len(lineage[i][key] & lineage[j][key]) for i, j in pairs]
                               for key in lineage[0]}
            checkpoint = base / "oof_checkpoints" / path.stem.split("_")[0] / \
                path.stem.split("seed")[1] / f"fold{fold['fold']}" / "best.pt"
            input_paths.append(checkpoint)
            checkpoint_report = torch.load(checkpoint, map_location="cpu", weights_only=True)["report"]
            fold_checks = {
                "rows_pairwise_disjoint": not any(row_overlaps),
                "groups_pairwise_disjoint": not any(group_overlaps),
                "full_population_partition": set.union(*rows) == identifiers,
                "source_identities_pairwise_disjoint": not any(sum(source_overlaps.values(), [])),
                "row_group_membership": all(set(artifact["groups"][identifier]) <= partitions[index]
                                            for index, section in enumerate(rows) for identifier in section),
                "checkpoint_hash_matches": sha256(checkpoint) == fold["checkpoint_sha256"],
                "checkpoint_train_ids_match": set(checkpoint_report["train_ids"]) == rows[1],
                "checkpoint_earlystop_ids_match": set(checkpoint_report["earlystop_ids"]) == rows[2],
                "checkpoint_selected_epoch_matches": checkpoint_report["selected_epoch"] == fold["selected_epoch"],
            }
            item["folds"].append({
                "fold": fold["fold"], "partition_order": ["held_out", "fitting", "earlystop"],
                "row_counts": [len(part) for part in rows],
                "group_counts": [len(part) for part in partitions],
                "overlap_pairs": pairs, "row_overlaps": row_overlaps,
                "group_overlaps": group_overlaps, "source_identity_overlaps": source_overlaps,
                "missing_population_ids": sorted(identifiers - set.union(*rows)),
                "extra_population_ids": sorted(set.union(*rows) - identifiers),
                "checkpoint": str(checkpoint.relative_to(root)),
                "checkpoint_sha256": sha256(checkpoint), "selected_epoch": fold["selected_epoch"],
                "checks": fold_checks,
            })
            all_checks.extend(fold_checks.values())
        all_checks.extend(checks.values())
        report["seed_artifacts"].append(item)
    for arm in ("A", "B"):
        path = base / "oof" / f"{arm}_ensemble.json"
        input_paths.append(path)
        artifact = json.loads(path.read_text(encoding="utf-8"))
        members = artifact["oof_members"]
        originals = [json.loads((base / "oof" / f"{arm}_seed{seed}.json").read_text(encoding="utf-8"))
                     for seed in (41, 42, 43)]
        arithmetic_checks = []
        errors = {}
        for section in ("predictions", "window_scores"):
            max_error = 0.0
            for identifier, values in artifact[section].items():
                for key, value in values.items():
                    expected = np.mean(np.stack([member[section][identifier][key] for member in members]), axis=0)
                    actual = np.asarray(value)
                    arithmetic_checks.append(bool(np.array_equal(actual, expected)))
                    max_error = max(max_error, float(np.max(np.abs(actual - expected))))
            errors[section] = max_error
        checks = {"saved_members_match": members == originals,
                  "member_means_exact": all(arithmetic_checks)}
        all_checks.extend(checks.values())
        report["ensembles"].append({"path": str(path.relative_to(root)), "sha256": sha256(path),
                                    "member_count": len(members), "maximum_absolute_errors": errors,
                                    "checks": checks})
    report["input_sha256"] = {str(path.relative_to(root)): sha256(path) for path in sorted(set(input_paths))}
    report["all_measured_invariants_pass"] = all(all_checks)
    output = script.with_name("evaluation_oof_verification.json")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "all_measured_invariants_pass": all(all_checks)}))
    raise SystemExit(0 if all(all_checks) else 1)


if __name__ == "__main__":
    main()
