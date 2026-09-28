"""Five-score predictions and masked metrics from the locked six-head ensemble.

Derived from source/full_bank_evaluate.py.  The score graph is unchanged:
  music  = mean over seeds of sigmoid(logmeanexp_windows(WindowMIL MLP logits))
  voice  = mean over seeds of max_segment softmax(fc5)[fake]  (0 if the vocal stem is silent)
  FILE   = max(voice * VOICE_PRESENT_PROB, music * MUSIC_PRESENT_PROB)   (PANNs presence)
No fitted calibration and no test-driven choice.  Val and test are scored by
the same function; test is only reachable after candidate_lock.json exists.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from eight_metrics import eer as sklearn_eer
from full_bank_features import row_identity, validate_cache
from graph_scoring import PREDICTION_COLUMNS, SUBMISSION_COLUMNS, _roc_metrics, official_metrics, validate_predictions
from localization_models import WindowMIL

from .common import PipelineError, StageLock, atomic_json, canonical_sha, now_iso, read_json, sha256_file
from .constants import LABELS, SEEDS
from .dataset import ROUTE_LABEL, exposure_flag, is_singing, read_manifest
from .features import identity_key
from .gpu import configure_numerics, require_device

EXTRA_COLUMNS = ("route", "split", "source", "generator", "input_kind", "historical_development_exposure",
                 "component_id", *LABELS, *(f"{n}_mask" for n in LABELS))


def checkpoint_names() -> list[str]:
    return [f"{route}_{seed}.pt" for seed in SEEDS for route in ("music", "voice")]


def load_heads(train_dir: Path, device: str, expected: Mapping[str, str] | None = None,
               expected_train_ids: Mapping[str, Sequence[str]] | None = None):
    """Reload the six checkpoints exactly (hash-verified against the lock when given)."""
    music, voice, hashes, fake_index = [], [], {}, None
    for seed in SEEDS:
        for route, collection in (("music", music), ("voice", voice)):
            path = train_dir / f"{route}_{seed}.pt"
            digest = sha256_file(path)
            if expected is not None and expected.get(path.name) != digest:
                raise PipelineError(f"Checkpoint differs from the lock: {path.name}")
            record = torch.load(path, map_location="cpu", weights_only=True)
            if record["route"] != route or record["seed"] != seed:
                raise PipelineError(f"Checkpoint route/seed mismatch: {path.name}")
            if expected_train_ids is not None and sorted(record["train_ids"]) != sorted(expected_train_ids[route]):
                raise PipelineError(f"Checkpoint train_ids differ from the train split: {path.name}")
            state = record["model"]
            if not all(torch.isfinite(value).all() for value in state.values()):
                raise PipelineError(f"Nonfinite checkpoint: {path.name}")
            model = WindowMIL(state["mean"], state["scale"]) if route == "music" else nn.Linear(1280, 2)
            model.load_state_dict(state, strict=True)
            collection.append(model.to(device).eval().requires_grad_(False))
            if fake_index is not None and record["fake_label_index"] != fake_index:
                raise PipelineError("Voice fake-label index differs across seeds")
            fake_index = record["fake_label_index"]
            hashes[path.name] = digest
    return music, voice, int(fake_index), hashes


@torch.no_grad()
def score_arrays(arrays: Mapping[str, np.ndarray], heads, device: str) -> dict[str, Any]:
    """Five scores (+ per-seed extras) for one file's cached/recomputed feature arrays."""
    music_heads, voice_heads, fake_index, _ = heads
    features = torch.from_numpy(np.asarray(arrays["mert_windows"])).to(device)
    embeddings = torch.from_numpy(np.asarray(arrays["voice_embeddings"])).to(device)
    presence = np.asarray(arrays["presence"]).astype(np.float64)
    silent = bool(arrays["voice_silent"])
    music_scores, voice_scores = [], []
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        for head in music_heads:
            logits, _ = head.head((features - head.mean) / head.scale)
            music_scores.append(float((logits.float().logsumexp(0) - np.log(len(features))).sigmoid()))
        for head in voice_heads:
            voice_scores.append(0. if silent else float(head(embeddings).float().softmax(-1)[:, fake_index].max()))
    music, voice = float(np.mean(music_scores)), float(np.mean(voice_scores))
    values = [max(voice * presence[0], music * presence[1]), voice, music, float(presence[0]), float(presence[1])]
    record: dict[str, Any] = dict(zip(PREDICTION_COLUMNS, values, strict=True))
    record.update({f"MUSIC_FAKE_PROB_seed{seed}": value for seed, value in zip(SEEDS, music_scores)})
    record.update({f"VOICE_FAKE_PROB_seed{seed}": value for seed, value in zip(SEEDS, voice_scores)})
    record["VOICE_FAKE_PROB_df_arena_original_fc5"] = float(arrays["voice_original"])
    record["voice_stem_silent"] = silent
    return record


def load_cached(cache_dir: Path, row: Mapping[str, str], index: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], str]:
    entry = index["items"][row["ID"]]
    path = (cache_dir / entry["path"]).resolve()
    if not path.is_relative_to(cache_dir.resolve()):
        raise PipelineError("Feature path escapes the cache")
    validate_cache(path, row_identity(row, identity_key(index["identity"])), entry["sha256"])
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}, entry["sha256"]


def predict_rows(rows: Sequence[Mapping[str, str]], cache_dir: Path, heads, device: str) -> list[dict[str, Any]]:
    index = read_json(cache_dir / "index.json")
    output = []
    for row in rows:
        arrays, digest = load_cached(cache_dir, row, index)
        output.append({"ID": row["ID"], **score_arrays(arrays, heads, device), "feature_sha256": digest})
    validate_predictions([{k: r[k] for k in SUBMISSION_COLUMNS} for r in output], [row["ID"] for row in rows])
    return output


def write_predictions(path: Path, rows: Sequence[Mapping[str, str]], predictions: Sequence[Mapping[str, Any]]) -> str:
    extra = [name for name in predictions[0] if name not in SUBMISSION_COLUMNS]
    fields = [*SUBMISSION_COLUMNS, *extra, *EXTRA_COLUMNS]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row, prediction in zip(rows, predictions, strict=True):
            writer.writerow({**prediction, **{name: row.get(name, "") for name in EXTRA_COLUMNS}})
    temporary.replace(path)
    return sha256_file(path)


def _eer_auc(labels: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    labels = [int(value) for value in labels]
    positives = sum(labels)
    if not labels or positives in (0, len(labels)):
        return {"n": len(labels), "positives": positives, "eer": None, "auc": None,
                "note": "single class; EER/AUC undefined"}
    eer, auc = _roc_metrics(labels, [float(value) for value in scores])
    return {"n": len(labels), "positives": positives, "eer": float(eer), "auc": float(auc)}


def _bootstrap(rows, labels, scores, groups, *, resamples: int = 1000, seed: int = 41) -> dict[str, Any]:
    by_group = defaultdict(list)
    for index, group in enumerate(groups):
        by_group[group].append(index)
    keys = sorted(by_group)
    if len(keys) < 5 or len(set(labels)) < 2:
        return {"components": len(keys), "ci95": None}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        chosen = [i for g in rng.choice(len(keys), len(keys), replace=True) for i in by_group[keys[g]]]
        sample = [labels[i] for i in chosen]
        if 0 < sum(sample) < len(sample):
            values.append(_roc_metrics(sample, [scores[i] for i in chosen])[0])
    return {"components": len(keys), "resamples": resamples, "valid": len(values),
            "ci95": [float(np.quantile(values, .025)), float(np.quantile(values, .975))] if values else None,
            "unit": "leakage component (component_id), with replacement"}


def head_view(rows, predictions, head: str):
    """(labels, scores, rows) for one head on its known, applicable subset (masks respected)."""
    label, column = {"file": ("FILE_FAKE", "FILE_FAKE_PROB"), "voice": ("VOICE_FAKE", "VOICE_FAKE_PROB"),
                     "music": ("MUSIC_FAKE", "MUSIC_FAKE_PROB")}[head]
    present = {"voice": "VOICE_PRESENT", "music": "MUSIC_PRESENT"}.get(head)
    selected = [(row, p) for row, p in zip(rows, predictions, strict=True)
                if row[f"{label}_mask"] == "1" and (present is None or (row[f"{present}_mask"] == "1" and row[present] == "1"))]
    return [int(r[label]) for r, _ in selected], [float(p[column]) for _, p in selected], [r for r, _ in selected]


def metrics_report(rows: Sequence[Mapping[str, str]], predictions: Sequence[Mapping[str, Any]], *,
                   thresholds: Mapping[str, float] | None = None) -> dict[str, Any]:
    submission = [{k: p[k] for k in SUBMISSION_COLUMNS} for p in predictions]
    report: dict[str, Any] = {"rows": len(rows), "official_style_known_subset": official_metrics(rows, submission),
                              "heads": {}, "presence": {}, "per_seed": {}, "exposure_subsets": {}, "strata": [],
                              "baseline_df_arena_original_fc5": {},
                              "interpretation": ["historical development exposure: not an independent test",
                                                 "no official/L4 score; unknown labels are masked, not imputed",
                                                 "singing (CtrSVDD) is train-only; no singing evaluation"]}
    for head in ("file", "voice", "music"):
        labels, scores, subset = head_view(rows, predictions, head)
        entry = _eer_auc(labels, scores)
        entry["mask"] = ("FILE_FAKE known" if head == "file" else
                         f"{head.upper()}_FAKE known and {head.upper()}_PRESENT known == 1")
        entry["bootstrap_eer"] = _bootstrap(subset, labels, scores, [r["component_id"] for r in subset])
        if thresholds and head in thresholds:
            threshold = thresholds[head]
            negatives = [s for s, y in zip(scores, labels) if y == 0]
            positives = [s for s, y in zip(scores, labels) if y == 1]
            entry["at_locked_validation_eer_threshold"] = {
                "threshold": threshold,
                "fpr": float(np.mean([s >= threshold for s in negatives])) if negatives else None,
                "fnr": float(np.mean([s < threshold for s in positives])) if positives else None}
        report["heads"][head] = entry
    for name, column in (("voice_presence", "VOICE_PRESENT"), ("music_presence", "MUSIC_PRESENT")):
        known = [(int(r[column]), float(p[f"{column}_PROB"])) for r, p in zip(rows, predictions) if r[f"{column}_mask"] == "1"]
        report["presence"][name] = {**_eer_auc([k for k, _ in known], [s for _, s in known]),
                                    "unknown": sum(r[f"{column}_mask"] == "0" for r in rows)}
    for head in ("voice", "music"):
        labels, _, subset = head_view(rows, predictions, head)
        index = {id(r): i for i, r in enumerate(rows)}
        for seed in SEEDS:
            column = f"{head.upper()}_FAKE_PROB_seed{seed}"
            scores = [float(predictions[index[id(r)]][column]) for r in subset]
            report["per_seed"][f"{head}_{seed}"] = _eer_auc(labels, scores)
    labels, _, subset = head_view(rows, predictions, "voice")
    lookup = {r["ID"]: p for r, p in zip(rows, predictions)}
    report["baseline_df_arena_original_fc5"]["voice"] = _eer_auc(
        labels, [float(lookup[r["ID"]]["VOICE_FAKE_PROB_df_arena_original_fc5"]) for r in subset])
    for flag in (True, False):
        chosen = [(r, p) for r, p in zip(rows, predictions) if exposure_flag(r) is flag]
        name = "historical_development_exposure" if flag else "no_recorded_exposure"
        sub_rows, sub_predictions = [r for r, _ in chosen], [p for _, p in chosen]
        report["exposure_subsets"][name] = {"rows": len(chosen), **{
            head: _eer_auc(*head_view(sub_rows, sub_predictions, head)[:2]) for head in ("file", "voice", "music")}}
    cells = defaultdict(list)
    for row, prediction in zip(rows, predictions):
        head = row["route"]
        cells[(head, row.get("source", ""), row.get("generator", ""))].append(
            (int(row[ROUTE_LABEL[head]]), float(prediction[f"{head.upper()}_FAKE_PROB"]),
             float(prediction["FILE_FAKE_PROB"])))
    for (route, source, generator), values in sorted(cells.items()):
        labels = [v[0] for v in values]
        report["strata"].append({"route": route, "source": source, "generator": generator, "rows": len(values),
                                 "positives": sum(labels),
                                 "route_score_mean": float(np.mean([v[1] for v in values])),
                                 "file_score_mean": float(np.mean([v[2] for v in values])),
                                 **{k: v for k, v in _eer_auc(labels, [x[1] for x in values]).items()
                                                    if k in ("eer", "auc")}})
    report["singing_rows"] = sum(is_singing(r) for r in rows)
    return report


def validation_thresholds(rows, predictions) -> dict[str, float]:
    """EER operating points on validation, fixed in the lock before any test scoring."""
    thresholds = {}
    for head in ("file", "voice", "music"):
        labels, scores, _ = head_view(rows, predictions, head)
        result = sklearn_eer(labels, scores)
        if result["threshold"] is not None:
            thresholds[head] = float(result["threshold"])
    return thresholds


def evaluate_split(config: Mapping[str, Any], split: str, *, lock: Mapping[str, Any] | None) -> dict[str, Any]:
    if split == "test" and lock is None:
        raise PipelineError("Test evaluation requires the candidate lock")
    device = config["device"]
    rows = [row for row in read_manifest(Path(config["manifest"])) if row["split"] == split]
    all_rows = read_manifest(Path(config["manifest"]))
    train_ids = {route: [r["ID"] for r in all_rows if r["split"] == "train" and r["route"] == route]
                 for route in ("music", "voice")}
    heads = load_heads(Path(config["train_dir"]), device, lock["checkpoints"] if lock else None, train_ids)
    predictions = predict_rows(rows, Path(config["cache_dir"]), heads, device)
    eval_dir = Path(config["eval_dir"])
    prediction_sha = write_predictions(eval_dir / f"{split}_predictions.csv", rows, predictions)
    thresholds = lock.get("validation_eer_thresholds") if lock else None
    report = metrics_report(rows, predictions, thresholds=thresholds if split == "test" else None)
    report.update({"split": split, "checkpoints": heads[3], "predictions_sha256": prediction_sha,
                   "lock_sha256": sha256_file(Path(config["train_dir"]) / "candidate_lock.json") if lock else None,
                   "feature_index_sha256": sha256_file(Path(config["cache_dir"]) / "index.json"),
                   "evaluated_at": now_iso()})
    atomic_json(eval_dir / f"{split}_metrics.json", report)
    return {"report": report, "predictions": predictions, "rows": rows}


def verify_lock(config: Mapping[str, Any]) -> dict[str, Any]:
    train_dir = Path(config["train_dir"])
    lock = read_json(train_dir / "candidate_lock.json")
    body = {k: v for k, v in lock.items() if k != "lock_body_sha256"}
    if canonical_sha(body) != lock.get("lock_body_sha256"):
        raise PipelineError("candidate_lock.json body hash mismatch")
    for name, digest in lock["checkpoints"].items():
        if sha256_file(train_dir / name) != digest:
            raise PipelineError(f"Locked checkpoint changed: {name}")
    if sorted(lock["checkpoints"]) != sorted(checkpoint_names()):
        raise PipelineError("Lock must cover exactly the six route/seed checkpoints")
    if sha256_file(Path(config["manifest"])) != lock["manifest_sha256"]:
        raise PipelineError("Manifest changed after the lock")
    if sha256_file(Path(config["cache_dir"]) / "index.json") != lock["feature_index_sha256"]:
        raise PipelineError("Feature index changed after the lock")
    return lock


def run_test(config: Mapping[str, Any]) -> dict[str, Any]:
    info = require_device(config["device"], config["expected_gpu"])
    configure_numerics()
    lock = verify_lock(config)
    result = evaluate_split(config, "test", lock=lock)
    receipt = {"device": info, "lock_sha256": sha256_file(Path(config["train_dir"]) / "candidate_lock.json"),
               "lock_locked_at": lock["locked_at"], "test_rows": len(result["rows"]),
               "predictions_sha256": result["report"]["predictions_sha256"],
               "metrics_sha256": sha256_file(Path(config["eval_dir"]) / "test_metrics.json"),
               "checkpoint_selection_used_test": False, "evaluated_at": now_iso()}
    atomic_json(Path(config["eval_dir"]) / "test_receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("test",))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    if config["device"] != "cuda" or config["expected_gpu"] != "A100":
        raise PipelineError("The CLI runs only on the authorized A100 (device=cuda)")
    with StageLock(Path(config["job_root"]) / "locks" / "evaluation.lock", "dacon9504a100.evaluation"):
        run_test(config)


if __name__ == "__main__":
    main()
