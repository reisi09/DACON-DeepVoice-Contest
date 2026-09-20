"""Exact ROC diagnostics and development-only model selection for V2."""

from __future__ import annotations

import argparse, csv, json, random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from graph_scoring import official_metrics, validate_predictions

Scalar = str | int | float | None
Row = Mapping[str, Scalar]
PredictionRows = Mapping[str, Sequence[Row]]
WEIGHTS: Final = {"file_eer": .45, "voice_eer": .18, "music_eer": .27, "voice_presence_auc": .05, "music_presence_auc": .05}
EER_SPECS: Final = (("file", "FILE_FAKE", "FILE_FAKE_PROB", None), ("voice", "VOICE_FAKE", "VOICE_FAKE_PROB", "VOICE_PRESENT"), ("music", "MUSIC_FAKE", "MUSIC_FAKE_PROB", "MUSIC_PRESENT"))


def _label(row: Row, name: str) -> int | None:
    value = row.get(name)
    if value is None or value == "":
        return None
    if value not in (0, 1, "0", "1"):
        raise ValueError(f"invalid binary label: {name}")
    return int(value)


def _role(row: Row) -> str:
    role = row.get("role", row.get("split"))
    if not isinstance(role, str) or not role:
        raise ValueError("manifest rows require role or split")
    return role


def _prediction_map(rows: Sequence[Row], manifest: Sequence[Row]) -> dict[str, Row]:
    expected = [str(row["ID"]) for row in manifest]
    parsed = validate_predictions(rows, expected)
    return {row["ID"]: row for row in parsed}


def _eligible(rows: Sequence[Row], predictions: Mapping[str, Row], target: str,
              score: str, mask: str | None) -> tuple[list[int], list[float], int]:
    targets: list[int] = []
    scores: list[float] = []
    unknown = 0
    for row in rows:
        if mask is not None:
            present = _label(row, mask)
            if present is None:
                unknown += 1
                continue
            if present == 0:
                continue
        truth = _label(row, target)
        if truth is None:
            unknown += 1
            continue
        targets.append(truth)
        scores.append(float(predictions[str(row["ID"])][score]))
    return targets, scores, unknown


def _roc(targets: Sequence[int], scores: Sequence[float]) -> dict[str, Scalar] | None:
    positives = sum(targets)
    negatives = len(targets) - positives
    if not positives or not negatives:
        return None
    ranked = sorted(zip(scores, targets, strict=True), reverse=True)
    points: list[tuple[float | None, int, int]] = [(None, 0, 0)]
    tp = fp = 0
    index = 0
    while index < len(ranked):
        threshold = ranked[index][0]
        while index < len(ranked) and ranked[index][0] == threshold:
            tp += ranked[index][1]
            fp += 1 - ranked[index][1]
            index += 1
        points.append((threshold, tp, fp))
    def distance(point: tuple[float | None, int, int]) -> float:
        return abs(point[2] / negatives - (1 - point[1] / positives))
    threshold, tp, fp = min(points, key=distance)
    fpr, fnr = fp / negatives, 1 - tp / positives
    auc = sum((right[2] / negatives - left[2] / negatives) *
              (right[1] / positives + left[1] / positives) / 2
              for left, right in zip(points, points[1:]))
    return {"eer": (fpr + fnr) / 2, "auc": auc, "threshold": threshold,
            "fpr": fpr, "fnr": fnr, "fp": fp, "fn": positives - tp,
            "tp": tp, "tn": negatives - fp, "n": len(targets)}


def _detail_report(rows: Sequence[Row], predictions: Mapping[str, Row]) -> dict[str, dict[str, Scalar]]:
    details: dict[str, dict[str, Scalar]] = {}
    for name, target, score, mask in EER_SPECS:
        targets, scores, unknown = _eligible(rows, predictions, target, score, mask)
        point = _roc(targets, scores)
        details[name] = {"unknown_relevant": unknown, "n": len(targets),
                         "point": point}
    return details


def _pair_changes(rows: Sequence[Row], baseline: Mapping[str, Row], candidate: Mapping[str, Row],
                  target: str, score: str, mask: str | None) -> dict[str, int]:
    labels, scores_a, _ = _eligible(rows, baseline, target, score, mask)
    _, scores_b, _ = _eligible(rows, candidate, target, score, mask)
    positives = [index for index, value in enumerate(labels) if value == 1]
    negatives = [index for index, value in enumerate(labels) if value == 0]
    flips = worsened = 0
    for positive in positives:
        for negative in negatives:
            before = scores_a[positive] - scores_a[negative]
            after = scores_b[positive] - scores_b[negative]
            if (before > 0 > after) or (before < 0 < after):
                flips += 1
                worsened += int(before > 0 > after)
    return {"pairs": len(positives) * len(negatives), "flips": flips,
            "positive_to_negative_worsened": worsened}


def audit_forensics(manifest_rows: Sequence[Row], predictions_by_name: PredictionRows) -> dict[str, object]:
    """Report all available roles without converting a partial proxy into a score."""
    maps = {name: _prediction_map(rows, manifest_rows) for name, rows in predictions_by_name.items()}
    if "G0" not in maps:
        raise ValueError("G0 predictions are required for forensics")
    roles: dict[str, list[Row]] = defaultdict(list)
    for row in manifest_rows:
        roles[_role(row)].append(row)
    report: dict[str, object] = {"metric": "nearest_abs_fpr_minus_fnr", "roles": {}}
    for role, rows in sorted(roles.items()):
        models: dict[str, object] = {}
        for name, prediction in maps.items():
            models[name] = {"official_proxy_only": official_metrics(rows, [prediction[str(row["ID"])] for row in rows]),
                            "roc": _detail_report(rows, prediction)}
        comparisons: dict[str, object] = {}
        for name, prediction in maps.items():
            if name == "G0":
                continue
            per_head: dict[str, object] = {}
            for head, target, score, mask in EER_SPECS:
                base = models["G0"]["roc"][head]["point"]
                changed = models[name]["roc"][head]["point"]
                per_head[head] = {"baseline": base, "candidate": changed,
                                  "auc_delta": None if base is None or changed is None else changed["auc"] - base["auc"],
                                  "pair_changes": _pair_changes(rows, maps["G0"], prediction, target, score, mask)}
            comparisons[name] = per_head
        report["roles"][role] = {"models": models, "comparisons": comparisons}
    return report


def _proxy(rows: Sequence[Row], predictions: Mapping[str, Row]) -> dict[str, object]:
    metrics = official_metrics(rows, [predictions[str(row["ID"])] for row in rows])
    usable = {name: weight for name, weight in WEIGHTS.items() if metrics["metrics"][name] is not None}
    total = sum(usable.values())
    values = metrics["metrics"]
    proxy = None if not total else sum(weight / total * ((1 - float(values[name])) if name.endswith("eer") else float(values[name]))
                                      for name, weight in usable.items())
    return {"value": proxy, "proxy_weights": usable, "metric_report": metrics,
            "is_partial": bool(metrics["partial"])}


def _resampled_proxy(rows: Sequence[Row], predictions: Mapping[str, Row]) -> float | None:
    """Score a bootstrap sample whose repeated IDs cannot enter CSV validation."""
    values: dict[str, float] = {}
    for metric, target, score, mask in EER_SPECS:
        targets, scores, _ = _eligible(rows, predictions, target, score, mask)
        point = _roc(targets, scores)
        if point is not None:
            values[f"{metric}_eer"] = point["eer"]
    for metric, target, score in (("voice_presence", "VOICE_PRESENT", "VOICE_PRESENT_PROB"),
                                  ("music_presence", "MUSIC_PRESENT", "MUSIC_PRESENT_PROB")):
        targets, scores, _ = _eligible(rows, predictions, target, score, None)
        point = _roc(targets, scores)
        if point is not None:
            values[f"{metric}_auc"] = point["auc"]
    usable = {name: weight for name, weight in WEIGHTS.items() if name in values}
    total = sum(usable.values())
    return None if not total else sum(weight / total * ((1 - values[name]) if name.endswith("eer") else values[name])
                                     for name, weight in usable.items())


def _components(rows: Sequence[Row]) -> list[list[int]]:
    parent: dict[str, str] = {}
    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value
    def join(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left
    for row in rows:
        raw_group = row.get("group_id")
        if not isinstance(raw_group, str) or not raw_group:
            raise ValueError("manifest rows require nonempty group_id constituents")
        groups = raw_group.split("|")
        if any(not group for group in groups):
            raise ValueError("manifest rows require nonempty group_id constituents")
        for group in groups[1:]:
            join(groups[0], group)
    units: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        units[find(str(row["group_id"]).split("|")[0])].append(index)
    return list(units.values())


def _bootstrap(rows: Sequence[Row], baseline: Mapping[str, Row], candidate: Mapping[str, Row]) -> dict[str, object]:
    units = _components(rows)
    rng = random.Random(20260913)
    deltas: list[float] = []
    for _ in range(2000):
        sampled = [row for unit in (rng.choice(units) for _ in units) for index in unit for row in [rows[index]]]
        base, changed = _resampled_proxy(sampled, baseline), _resampled_proxy(sampled, candidate)
        if base is not None and changed is not None:
            deltas.append(changed - base)
    deltas.sort()
    return {"seed": 20260913, "resamples": 2000, "lineage_components": len(units),
            "valid_resamples": len(deltas), "uncomputable_resamples": 2000 - len(deltas),
            "win_rate": None if not deltas else sum(delta > 0 for delta in deltas) / len(deltas),
            "p10_delta": None if not deltas else deltas[max(0, (len(deltas) - 1) // 10)]}


def select_candidate(manifest_rows: Sequence[Row], predictions_by_name: PredictionRows) -> dict[str, object]:
    """Select a V2 candidate only from the dedicated development selector role."""
    if any(_role(row) != "dev_select" for row in manifest_rows):
        raise ValueError("selection accepts development roles only")
    maps = {name: _prediction_map(rows, manifest_rows) for name, rows in predictions_by_name.items()}
    if "G0" not in maps:
        raise ValueError("G0 predictions are required for selection")
    models: dict[str, dict[str, object]] = {"G0": _proxy(manifest_rows, maps["G0"])}
    candidates = [name for name in ("AST_FROZEN", "AST_LAST_BLOCK", "G0_AST_MUSIC") if name in maps]
    for name in candidates:
        current = _proxy(manifest_rows, maps[name])
        bootstrap = _bootstrap(manifest_rows, maps["G0"], maps[name])
        current["bootstrap"] = bootstrap
        base_metrics = models["G0"]["metric_report"]["metrics"]
        new_metrics = current["metric_report"]["metrics"]
        regression = any(float(new_metrics[key]) > float(base_metrics[key]) + .02 for key in WEIGHTS if key.endswith("eer") and new_metrics[key] is not None and base_metrics[key] is not None)
        companion = any(_roc(*_eligible(manifest_rows, maps[name], target, score, mask)[:2])["auc"] < _roc(*_eligible(manifest_rows, maps["G0"], target, score, mask)[:2])["auc"] - .01 for _, target, score, mask in EER_SPECS if _roc(*_eligible(manifest_rows, maps[name], target, score, mask)[:2]) is not None and _roc(*_eligible(manifest_rows, maps["G0"], target, score, mask)[:2]) is not None)
        gain = None if current["value"] is None or models["G0"]["value"] is None else float(current["value"]) - float(models["G0"]["value"])
        gates = {"proxy_gain": gain is not None and gain >= .005,
                 "bootstrap_complete": bootstrap["valid_resamples"] == 2000,
                 "bootstrap_win_rate": bootstrap["win_rate"] is not None and bootstrap["win_rate"] >= .70,
                 "bootstrap_p10": bootstrap["p10_delta"] is not None and bootstrap["p10_delta"] >= -.005,
                 "eer_regression": not regression, "companion_auc_regression": not companion}
        current["gates"] = {**gates, "proxy_gain_delta": gain, "passed": all(gates.values())}
        models[name] = current
    passed = [name for name in candidates if models[name]["gates"]["passed"]]
    selected = max(passed, key=lambda name: (models[name]["value"], -candidates.index(name)), default="G0")
    failed = {name: [gate for gate, value in models[name]["gates"].items() if value is False]
              for name in candidates if name not in passed}
    return {"selected": selected, "reason": "all_gates_passed" if selected != "G0" else "no_candidate_passed_all_gates",
            "failed_gates": failed, "models": models}


def _read_csv(path: Path) -> list[dict[str, Scalar]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _load_predictions(directory: Path, manifest: Sequence[Row]) -> dict[str, list[Row]]:
    files = {name: directory / f"{name}_predictions.csv" for name in ("G0", "G1", "G2", "G3", "G4")}
    if all(path.is_file() for path in files.values()):
        return {name: _read_csv(path) for name, path in files.items()}
    from graph_fusion import graph_predictions
    features = json.loads((directory / "features.json").read_text())
    fusion = json.loads((directory / "fusion.json").read_text())
    by_id = {feature["ID"]: graph_predictions(feature, fusion) for feature in features}
    return {name: [by_id[str(row["ID"])][name] for row in manifest] for name in files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("diagnose")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = _read_csv(args.manifest)
    report = audit_forensics(manifest, _load_predictions(args.predictions, manifest))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
