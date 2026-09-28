"""Run E1/E3/E4/E6/E8 frozen-feature experiments."""
# noqa: SIZE_OK - one public dispatcher keeps notebook experiment numbering explicit.
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from eight_density import RealOnlyGaussian
from eight_fixed_io import (
    JsonValue, atomic_json, evaluate, matrix, model_rows, read_config, read_rows, standardize, write_predictions,
)
from eight_heads import HeadResult, fit_head, predict
from eight_metrics import comparison_status, grouped_bootstrap_delta, threshold_at_fpr

EXPERIMENTS = (1, 3, 4, 6, 8)


class AblationContractError(ValueError):
    """Fixed experiment inputs or arm schedules violate the sealed design."""


def music_model_rows(
    rows: list[dict[str, str]], variants: tuple[str, ...], *, omit_train_variants: tuple[str, ...] = (),
) -> list[dict[str, str | int]]:
    """Select labeled core-music rows without relying on the source-format ``kind`` field."""
    selected = [
        {**row, "label": int(row["MUSIC_FAKE"])}
        for row in rows
        if row["probe_family"] == "core"
        and row["MUSIC_PRESENT"] == "1"
        and row["MUSIC_FAKE"] in {"0", "1"}
        and row["variant"] in variants
        and not (row["split"] == "train" and row["variant"] in omit_train_variants)
    ]
    if not selected:
        raise AblationContractError("core music selector returned no labeled rows")
    for split in ("train", "calibration", "dev", "test"):
        population = [row for row in selected if row["split"] == split]
        expected_variants = set(variants).difference(omit_train_variants if split == "train" else ())
        if {row["label"] for row in population} != {0, 1}:
            raise AblationContractError(f"core music {split} population lacks a truth class")
        if {str(row["variant"]) for row in population} != expected_variants:
            raise AblationContractError(f"core music {split} population lacks a planned codec view")
    return selected


def _split(rows: list[dict[str, str | int]], name: str) -> list[dict[str, str | int]]:
    return [row for row in rows if row["split"] == name]


def _fit_predict(
    root: Path, rows: list[dict[str, str | int]], key: str, name: str, config: dict[str, JsonValue],
    *, mode: str, consistency: float = 0.0, width: int | None = None,
    architecture: str = "mlp",
) -> tuple[dict[int, tuple[HeadResult, np.ndarray, np.ndarray]], list[dict[str, str | int]], list[dict[str, str | int]], np.ndarray, np.ndarray]:
    train, dev = _split(rows, "train"), _split(rows, "dev")
    raw_train, raw_dev = matrix(root, train, key, width), matrix(root, dev, key, width)
    mean, scale = raw_train.mean(axis=0), raw_train.std(axis=0)
    scale[scale < 1e-6] = 1.0
    train_x, dev_x = (raw_train - mean) / scale, (raw_dev - mean) / scale
    pairs: list[tuple[int, int]] = []
    if consistency:
        by_pair: dict[str, list[int]] = {}
        for index, row in enumerate(train):
            if row.get("pair_id"):
                by_pair.setdefault(str(row["pair_id"]), []).append(index)
        for indices in by_pair.values():
            base = [index for index in indices if train[index]["variant"] == "base"]
            processed = [index for index in indices if train[index]["variant"] in ("telephone", "mp3_32k")]
            pairs.extend((base[0], index) for index in processed if base)
        if not pairs:
            raise AblationContractError("E3 requires split-local base/processed pairs")
    fitted: dict[int, tuple[HeadResult, np.ndarray, np.ndarray]] = {}
    for seed in [int(value) for value in config.get("seeds", [41, 42, 43])]:
        result = fit_head(name + f"_seed{seed}", train, train_x, root / "output/checkpoints", steps=int(config.get("head_steps", 500)),
                          batch_size=int(config.get("head_batch_size", 32)), seed=seed, mode=mode,
                          paired_indices=pairs, consistency_weight=consistency, architecture=architecture)
        fitted[seed] = (result, predict(result.model, train_x), predict(result.model, dev_x))
    return fitted, train, dev, mean, scale


def _lock(root: Path, number: int, payload: dict[str, JsonValue]) -> Path:
    path = root / f"output/selection_E{number}_locked.json"
    atomic_json(path, {**payload, "experiment": number, "test_used": False, "locked_epoch": time.time()})
    return path


def _paired_ci(rows: list[dict[str, str | int]], control: np.ndarray, candidate: np.ndarray) -> dict[str, JsonValue]:
    records = [{**row, "component_id": row["component_id"], "control": float(a), "candidate": float(b)}
               for row, a, b in zip(rows, control, candidate, strict=True)]
    return grouped_bootstrap_delta(records, "control", "candidate", n_boot=1000, seed=41)


def _held(
    root: Path, rows: list[dict[str, str | int]], key: str, mean: np.ndarray, scale: np.ndarray,
    width: int | None = None,
) -> tuple[list[dict[str, str | int]], np.ndarray]:
    test = _split(rows, "test")
    return test, (matrix(root, test, key, width) - mean) / scale


def _seed_summary(values: list[float]) -> dict[str, JsonValue]:
    return {"mean": float(np.mean(values)), "std": float(np.std(values)),
            "range": [float(np.min(values)), float(np.max(values))], "n_seeds": len(values)}


def experiment_1(root: Path, config: dict[str, JsonValue], all_rows: list[dict[str, str]]) -> dict[str, JsonValue]:
    branches: dict[str, JsonValue] = {}
    contexts = {}
    for branch, kind, key in (("music", "music", "mert_l24"), ("voice", "speech", "anti_l24")):
        rows = (music_model_rows(all_rows, ("base",)) if branch == "music" else
                model_rows([row for row in all_rows if row["kind"] == kind and row["variant"] == "base"]))
        control, train, dev, mean, scale = _fit_predict(root, rows, key, f"E1_{branch}_parent", config, mode="parent")
        balanced, _, _, _, _ = _fit_predict(root, rows, key, f"E1_{branch}_balanced", config, mode="source_generator_parent")
        report: dict[str, JsonValue] = {"seeds": {}, "same_updates": True, "same_class_exposure": True}
        for seed in sorted(control):
            control_dev, candidate_dev = control[seed][2], balanced[seed][2]
            control_report, candidate_report = evaluate(dev, control_dev), evaluate(dev, candidate_dev)
            delta = _paired_ci(dev, control_dev, candidate_dev)
            report["seeds"][str(seed)] = {"control": control_report, "balanced": candidate_report,
                                                   "paired_delta": delta,
                                                   "adoption_status": comparison_status(delta, control_report["strata"], candidate_report["strata"]),
                                                   "updates_each": int(config.get("head_steps", 500)),
                                                   "control_draw_sha256": control[seed][0].draw_sha256,
                                                   "balanced_draw_sha256": balanced[seed][0].draw_sha256,
                                                   "class_draw_counts": balanced[seed][0].draw_counts}
        report["seed_summary"] = {
            "control_dev_eer": _seed_summary([float(report["seeds"][str(seed)]["control"]["eer"]) for seed in control]),
            "balanced_dev_eer": _seed_summary([float(report["seeds"][str(seed)]["balanced"]["eer"]) for seed in balanced]),
        }
        generators = sorted({str(row["generator"]) for row in train if row.get("generator")})
        sensitivity = {}
        for generator in generators:
            sensitivity_rows = [row for row in rows if
                                (row["split"] == "train" and str(row.get("generator", "")) != generator)
                                or (row["split"] == "dev" and (str(row.get("generator", "")) == generator or int(row["label"]) == 0))]
            sensitivity_dev = _split(sensitivity_rows, "dev")
            sensitivity_train = _split(sensitivity_rows, "train")
            if ({int(row["label"]) for row in sensitivity_dev} != {0, 1}
                    or {int(row["label"]) for row in sensitivity_train} != {0, 1}):
                sensitivity[generator] = {"status": "data_insufficient",
                                          "reason": "held-generator fold lacks both training or evaluation labels",
                                          "train_rows": len(sensitivity_train), "eval_rows": len(sensitivity_dev)}
                continue
            held_control, held_train, held_dev, _, _ = _fit_predict(
                root, sensitivity_rows, key, f"E1_{branch}_holdout_{generator}_parent", config, mode="parent")
            held_balanced, _, _, _, _ = _fit_predict(
                root, sensitivity_rows, key, f"E1_{branch}_holdout_{generator}_balanced", config, mode="source_generator_parent")
            sensitivity[generator] = {"status": "complete", "train_excluded_generator": generator,
                                      "train_rows": len(held_train), "eval_rows": len(held_dev),
                                      "eval_sources": sorted({str(row["source"]) for row in held_dev}),
                                      "source_confounded": True,
                                      "seeds": {str(seed): {
                                          "control": evaluate(held_dev, held_control[seed][2]),
                                          "balanced": evaluate(held_dev, held_balanced[seed][2]),
                                          "paired_delta": _paired_ci(held_dev, held_control[seed][2], held_balanced[seed][2]),
                                      } for seed in held_control}}
        report["generator_holdout_sensitivity"] = sensitivity
        branches[branch] = report
        contexts[branch] = (rows, key, mean, scale, control, balanced)
    _lock(root, 1, {"selection": "predeclared parent-uniform vs source-generator-parent sampler; all seeds retained"})
    for branch, (rows, key, mean, scale, control, balanced) in contexts.items():
        test, test_x = _held(root, rows, key, mean, scale)
        columns = {}
        for seed in control:
            control_score = predict(control[seed][0].model, test_x)
            candidate_score = predict(balanced[seed][0].model, test_x)
            columns[f"parent_seed{seed}"] = control_score; columns[f"balanced_seed{seed}"] = candidate_score
            branches[branch]["seeds"][str(seed)]["test"] = {
                "control": evaluate(test, control_score), "balanced": evaluate(test, candidate_score),
                "paired_delta": _paired_ci(test, control_score, candidate_score),
            }
        write_predictions(root / f"output/predictions_E1_{branch}.csv", test, columns)
    return {"status": "complete", "branches": branches, "official_score": None,
            "limitations": ["historical and fresh sources are diagnostic", "source/generator labels can be confounded"]}


def experiment_3(root: Path, config: dict[str, JsonValue], all_rows: list[dict[str, str]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    contexts = {}
    for branch, kind, key in (("music", "music", "mert_l24"), ("voice", "speech", "anti_l24")):
        variants = ("base", "telephone", "mp3_32k", "opus_telephone")
        if branch == "music":
            rows = music_model_rows(all_rows, variants, omit_train_variants=("opus_telephone",))
        else:
            eligible = [row for row in all_rows if row["kind"] == kind and row["variant"] in variants
                        and not (row["split"] == "train" and row["variant"] == "opus_telephone")]
            rows = model_rows(eligible)
        ce, _, dev, mean, scale = _fit_predict(root, rows, key, f"E3_{branch}_ce", config, mode="variant_parent")
        consistent, _, _, _, _ = _fit_predict(root, rows, key, f"E3_{branch}_consistent", config, mode="variant_parent", consistency=0.1)
        unseen = [index for index, row in enumerate(dev) if row["variant"] == "opus_telephone"]
        seed_reports = {}
        selected_rows = [dev[i] for i in unseen]
        for seed in ce:
            if ce[seed][0].draw_sha256 != consistent[seed][0].draw_sha256:
                raise AblationContractError("E3 arms received different CE examples")
            control_report = evaluate(selected_rows, ce[seed][2][unseen])
            candidate_report = evaluate(selected_rows, consistent[seed][2][unseen])
            delta = _paired_ci(selected_rows, ce[seed][2][unseen], consistent[seed][2][unseen])
            seed_reports[str(seed)] = {"ce": control_report, "ce_plus_consistency": candidate_report,
                                       "paired_delta": delta,
                                       "adoption_status": comparison_status(delta, control_report["strata"], candidate_report["strata"]),
                                       "draw_sha256": ce[seed][0].draw_sha256,
                                       "class_draw_counts": ce[seed][0].draw_counts}
        result[branch] = {"seeds": seed_reports, "train_normalization_only": True, "equal_ce_exposure": True}
        result[branch]["seed_summary"] = {
            "ce_dev_eer": _seed_summary([float(seed_reports[str(seed)]["ce"]["eer"]) for seed in ce]),
            "consistent_dev_eer": _seed_summary([float(seed_reports[str(seed)]["ce_plus_consistency"]["eer"]) for seed in ce]),
        }
        contexts[branch] = (rows, key, mean, scale, ce, consistent)
    _lock(root, 3, {"selection": "fixed CE versus CE+0.1 consistency; unseen Opus+telephone evaluation"})
    for branch, (rows, key, mean, scale, ce, consistent) in contexts.items():
        test, test_x = _held(root, rows, key, mean, scale)
        chosen = [i for i, row in enumerate(test) if row["variant"] == "opus_telephone"]
        selected_rows = [test[i] for i in chosen]; columns = {}
        for seed in ce:
            control_all = predict(ce[seed][0].model, test_x); candidate_all = predict(consistent[seed][0].model, test_x)
            control, candidate = control_all[chosen], candidate_all[chosen]
            columns[f"ce_seed{seed}"] = control_all; columns[f"consistent_seed{seed}"] = candidate_all
            result[branch]["seeds"][str(seed)]["test_unseen_chain"] = {
                "ce": evaluate(selected_rows, control), "ce_plus_consistency": evaluate(selected_rows, candidate),
                "paired_delta": _paired_ci(selected_rows, control, candidate)}
            result[branch]["seeds"][str(seed)]["test_by_variant"] = {
                variant: {"ce": evaluate([row for row in test if row["variant"] == variant], control_all[[i for i, row in enumerate(test) if row["variant"] == variant]]),
                          "ce_plus_consistency": evaluate([row for row in test if row["variant"] == variant], candidate_all[[i for i, row in enumerate(test) if row["variant"] == variant]])}
                for variant in sorted({str(row["variant"]) for row in test})}
        write_predictions(root / f"output/predictions_E3_{branch}.csv", test, columns)
    return {"status": "complete", "branches": result, "official_score": None}


def experiment_4(root: Path, config: dict[str, JsonValue], all_rows: list[dict[str, str]]) -> dict[str, JsonValue]:
    rows = music_model_rows(all_rows, ("base",))
    mert, _, dev, mert_mean, mert_scale = _fit_predict(root, rows, "mert_l24", "E4_mert", config, mode="source_generator_parent", architecture="linear")
    spectral, _, _, spectral_mean, spectral_scale = _fit_predict(root, rows, "spectrum_stem", "E4_spectral", config, mode="source_generator_parent", width=384, architecture="linear")
    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    seed_reports: dict[str, JsonValue] = {}
    selected: dict[str, float] = {}
    for seed in mert:
        candidates = {weight: (1 - weight) * mert[seed][2] + weight * spectral[seed][2] for weight in grid}
        weight = min(grid, key=lambda value: (evaluate(dev, candidates[value])["eer"] is None,
                                               1.0 if evaluate(dev, candidates[value])["eer"] is None else evaluate(dev, candidates[value])["eer"], value))
        selected[str(seed)] = weight
        seed_reports[str(seed)] = {"mert": evaluate(dev, mert[seed][2]), "spectral": evaluate(dev, spectral[seed][2]),
                                   "fusion": evaluate(dev, candidates[weight]), "selected_spectral_weight": weight,
                                   "paired_delta": _paired_ci(dev, mert[seed][2], candidates[weight])}
    _lock(root, 4, {"selected_spectral_weight": selected, "grid": grid, "feature": "spectrum_stem[:384]",
                    "n_fft": 2048, "selection_split": "dev"})
    test, mert_x = _held(root, rows, "mert_l24", mert_mean, mert_scale)
    _, spectral_x = _held(root, rows, "spectrum_stem", spectral_mean, spectral_scale, 384)
    columns = {}
    for seed in mert:
        mert_score, spectral_score = predict(mert[seed][0].model, mert_x), predict(spectral[seed][0].model, spectral_x)
        fusion = (1 - selected[str(seed)]) * mert_score + selected[str(seed)] * spectral_score
        columns[f"mert_seed{seed}"] = mert_score; columns[f"spectral_seed{seed}"] = spectral_score; columns[f"fusion_seed{seed}"] = fusion
        seed_reports[str(seed)]["test"] = {"mert": evaluate(test, mert_score), "spectral": evaluate(test, spectral_score),
                                                  "fusion": evaluate(test, fusion), "paired_delta": _paired_ci(test, mert_score, fusion)}
    write_predictions(root / "output/predictions_E4.csv", test, columns)
    return {"status": "complete", "dev": seed_reports,
            "seed_summary": {name: _seed_summary([float(seed_reports[str(seed)][name]["eer"]) for seed in mert])
                             for name in ("mert", "spectral", "fusion")}, "official_score": None}


def experiment_6(root: Path, config: dict[str, JsonValue], all_rows: list[dict[str, str]]) -> dict[str, JsonValue]:
    report: dict[str, JsonValue] = {}
    chosen: dict[str, str] = {}
    contexts = {}
    for family, kind, keys in (("MERT", "music", ("mert_l6", "mert_l12", "mert_l24")),
                               ("Anti", "speech", ("anti_l12", "anti_l24", "anti_l48"))):
        rows = (music_model_rows(all_rows, ("base",)) if family == "MERT" else
                model_rows([row for row in all_rows if row["kind"] == kind and row["variant"] == "base"]))
        candidates = {key: _fit_predict(root, rows, key, f"E6_{key}", config, mode="source_generator_parent") for key in keys}
        summaries = {key: _seed_summary([float(evaluate(dev, fitted[seed][2])["eer"]) for seed in fitted])
                     for key, (fitted, _, dev, _, _) in candidates.items()}
        means = {key: float(value["mean"]) for key, value in summaries.items()}
        chosen[family] = min(keys, key=lambda key: (means[key], key))
        report[family] = {"mean_dev_eer": means, "seed_summary": summaries, "selected": chosen[family],
                          "seeds": list(config.get("seeds", [41, 42, 43]))}
        contexts[family] = (rows, candidates)
    _lock(root, 6, {"selected_layers": chosen, "selection_split": "dev", "same_head_and_updates": True})
    for family, (rows, candidates) in contexts.items():
        columns = {}; test_rows = None; tests = {}
        for key, (fitted, _, _, mean, scale) in candidates.items():
            test, test_x = _held(root, rows, key, mean, scale); test_rows = test
            tests[key] = {}
            for seed in fitted:
                score = predict(fitted[seed][0].model, test_x)
                columns[f"{key}_seed{seed}"] = score; tests[key][str(seed)] = evaluate(test, score)
        report[family]["test"] = tests
        if test_rows is not None:
            write_predictions(root / f"output/predictions_E6_{family}.csv", test_rows, columns)
    return {"status": "complete", "families": report, "official_score": None}


def experiment_8(root: Path, config: dict[str, JsonValue], all_rows: list[dict[str, str]]) -> dict[str, JsonValue]:
    rows = music_model_rows(all_rows, ("base",))
    train_real = [row for row in _split(rows, "train") if row["label"] == 0]
    calibration = [row for row in _split(rows, "calibration") if row["label"] == 0]
    dev = _split(rows, "dev")
    train_x, calibration_x, dev_x = standardize(matrix(root, train_real, "mert_l24"), matrix(root, calibration, "mert_l24"), matrix(root, dev, "mert_l24"))
    model = RealOnlyGaussian(0.1).fit(train_x, np.zeros(len(train_real), dtype=np.int64), [str(row["ID"]) for row in train_real])
    model.save(root / "output/checkpoints/E8_real_only.npz")
    threshold = threshold_at_fpr(np.zeros(len(calibration), dtype=np.int64), model.score(calibration_x), 0.05)
    scores = model.score(dev_x)
    _lock(root, 8, {"threshold": threshold, "threshold_target_real_fpr": 0.05, "fit_ids": list(model.fit_ids_),
                    "fit_labels": "REAL only", "polarity": "distance high=fake", "selection_split": "calibration"})
    test = _split(rows, "test")
    # REAL-train normalization parameters are embedded by standardizing against train_x's source cohort.
    raw_train = matrix(root, train_real, "mert_l24"); mean, scale = raw_train.mean(axis=0), raw_train.std(axis=0)
    scale[scale < 1e-6] = 1.0
    test_scores = model.score((matrix(root, test, "mert_l24") - mean) / scale)
    write_predictions(root / "output/predictions_E8.csv", dev + test, {"distance": np.concatenate([scores, test_scores])})
    return {"status": "complete", "dev": evaluate(dev, scores, threshold), "threshold": threshold, "fit_rows": len(train_real),
            "test": evaluate(test, test_scores, threshold), "calibration_real_rows": len(calibration),
            "optional_fusion": None, "official_score": None}


def run(config_path: str, experiment_number: int | None = None) -> dict[str, JsonValue]:
    """Run one or all frozen-feature experiments with independent selection locks."""
    config = read_config(config_path)
    root = Path(str(config["root"])); rows = read_rows(root)
    selected = EXPERIMENTS if experiment_number is None else (experiment_number,)
    reports: dict[str, JsonValue] = {}
    functions = {1: experiment_1, 3: experiment_3, 4: experiment_4, 6: experiment_6, 8: experiment_8}
    for number in selected:
        if number not in functions:
            raise AblationContractError(f"unsupported fixed-feature experiment: {number}")
        started = time.time(); report = functions[number](root, config, rows)
        known = model_rows(rows)
        lock = root / f"output/selection_E{number}_locked.json"
        report = {**report, "experiment": number, "wall_seconds": time.time() - started,
                  "population": "known component label within the experiment modality/variant",
                  "label_denominators": {"real": sum(row["label"] == 0 for row in known),
                                         "fake": sum(row["label"] == 1 for row in known)},
                  "component_count": len({str(row["component_id"]) for row in known}),
                  "bootstrap_unit": "provenance-connected component; CI conditional on fixed fitted models",
                  "fixed_updates": None if number == 8 else int(config.get("head_steps", 500)),
                  "class_exposure": ("REAL-only fit" if number == 8 else
                                     f"{int(config.get('head_batch_size', 32)) // 2} REAL + {int(config.get('head_batch_size', 32)) // 2} FAKE per batch{int(config.get('head_batch_size', 32))}"),
                  "seeds": [] if number == 8 else config.get("seeds", [41, 42, 43]),
                  "selection_locked": str(lock),
                  "estimand": "paired candidate minus reference EER; lower is better",
                  "comparison_scope": "same manifest split, modality, labels and frozen feature cache",
                  "claim_limit": "diagnostic historical/fresh-source evidence; no official score or authenticity generalization"}
        atomic_json(root / f"output/experiment_{number}.json", report); reports[str(number)] = report
    return reports
