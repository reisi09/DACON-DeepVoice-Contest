from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from eight_ablation import AblationContractError, music_model_rows
from eight_contract import read_manifest
from eight_density import RealOnlyGaussian
from eight_fixed_io import connected_components
from eight_heads import balanced_draws, consistency_objective, consistency_penalty
from eight_metrics import binary_summary, eer, grouped_bootstrap_delta
from graph_scoring import official_metrics


def test_actual_manifest_music_population_uses_component_truth_not_kind() -> None:
    # Given: the built 7,101-row manifest, whose music sources are represented as kind=mixed.
    roots = (Path(__file__).parent / "source/dataset/manifest.csv",
             Path(__file__).parent / "_workspace/eight_h100_20260917/dataset/manifest.csv")
    manifest = next(path for path in roots if path.exists())
    rows = [dict(row) for row in read_manifest(manifest)]

    # When: the fixed experiments select their core music population.
    selected = music_model_rows(rows, ("base", "telephone", "mp3_32k", "opus_telephone"),
                                omit_train_variants=("opus_telephone",))

    # Then: every evaluated split retains both truth classes and the planned codec controls.
    expected_variants = {"train": {"base", "telephone", "mp3_32k"},
                         "calibration": {"base", "telephone", "mp3_32k", "opus_telephone"},
                         "dev": {"base", "telephone", "mp3_32k", "opus_telephone"},
                         "test": {"base", "telephone", "mp3_32k", "opus_telephone"}}
    for split, variants in expected_variants.items():
        population = [row for row in selected if row["split"] == split]
        assert {row["label"] for row in population} == {0, 1}
        assert {row["variant"] for row in population} == variants
        assert all(row["probe_family"] == "core" and row["MUSIC_PRESENT"] == "1"
                   and row["label"] == int(row["MUSIC_FAKE"]) for row in population)
        assert all({row["source"] for row in population if row["label"] == label} for label in (0, 1))
        assert any(row["generator"] for row in population if row["label"] == 1)

    try:
        music_model_rows([{**rows[0], "probe_family": "swap"}], ("base",))
    except AblationContractError as error:
        assert "no labeled rows" in str(error)
    else:
        raise AssertionError("an empty core-music selector was accepted")


def test_official_eer_uses_full_tied_score_roc() -> None:
    # Given: tied scores whose full ROC has a unique nearest EER point.
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.9, 0.8, 0.8, 0.1])

    # When: the experiment metric is evaluated.
    measured = eer(labels, scores)

    # Then: it matches the published nearest-point rule with tied thresholds.
    assert measured["eer"] == 0.75
    assert measured["n"] == 4
    labels_rows = [{"ID": str(i), "FILE_FAKE": int(label), "VOICE_FAKE": "", "MUSIC_FAKE": "",
                    "VOICE_PRESENT": 0, "MUSIC_PRESENT": 0} for i, label in enumerate(labels)]
    prediction_rows = [{"ID": str(i), "FILE_FAKE_PROB": float(score), "VOICE_FAKE_PROB": 0.5,
                        "MUSIC_FAKE_PROB": 0.5, "VOICE_PRESENT_PROB": 0.0, "MUSIC_PRESENT_PROB": 0.0}
                       for i, score in enumerate(scores)]
    assert official_metrics(labels_rows, prediction_rows)["metrics"]["file_eer"] == measured["eer"]


def test_sampler_equalizes_class_and_source_without_view_skew() -> None:
    # Given: duplicated views and unequal source counts inside each class.
    rows = [
        {"ID": f"a{i}", "parent_id": "p0", "source": "A", "generator": "", "label": 0}
        for i in range(8)
    ] + [
        {"ID": "b", "parent_id": "p1", "source": "B", "generator": "", "label": 0},
        {"ID": "c", "parent_id": "p2", "source": "C", "generator": "G1", "label": 1},
        {"ID": "d", "parent_id": "p3", "source": "D", "generator": "G2", "label": 1},
    ]

    # When: a full fixed-step draw schedule is created.
    draws = balanced_draws(rows, steps=500, batch_size=32, seed=41, mode="source_generator_parent")
    selected = [rows[index] for batch in draws for index in batch]

    # Then: exposure is exactly fixed and class/source cells are balanced.
    assert len(selected) == 16_000
    assert sum(row["label"] == 0 for row in selected) == 8_000
    assert sum(row["source"] == "A" for row in selected) == 4_000
    assert sum(row["source"] == "B" for row in selected) == 4_000


def test_sampler_does_not_starve_sources_beyond_one_batch() -> None:
    # Given: more source strata than one class half-batch can contain.
    rows = [{"ID": f"r{i}", "parent_id": f"r{i}", "source": f"R{i}", "generator": "", "label": 0}
            for i in range(20)] + [
        {"ID": f"f{i}", "parent_id": f"f{i}", "source": f"F{i}", "generator": f"G{i}", "label": 1}
        for i in range(20)]

    # When: two batches advance the global stratum cursor.
    draws = balanced_draws(rows, steps=2, batch_size=32, seed=41, mode="source_generator_parent")
    selected = {rows[index]["source"] for batch in draws for index in batch}

    # Then: every stratum is reached instead of resetting at source 0 each batch.
    assert len(selected) == 40


def test_e3_variant_exposure_differs_by_at_most_one_per_class() -> None:
    # Given: the sealed three-view E3 cohort.
    rows = [{"ID": f"{label}_{variant}", "parent_id": f"p{label}", "source": "s",
             "generator": "g" if label else "", "variant": variant, "label": label}
            for label in (0, 1) for variant in ("base", "telephone", "mp3_32k")]

    # When: 500 x 32 examples are scheduled.
    draws = balanced_draws(rows, steps=500, batch_size=32, seed=41, mode="variant_parent")

    # Then: indivisible exposure is deterministic and differs by at most one within each class.
    for label in (0, 1):
        counts = [sum(rows[index]["label"] == label and rows[index]["variant"] == variant
                      for batch in draws for index in batch)
                  for variant in ("base", "telephone", "mp3_32k")]
        assert max(counts) - min(counts) <= 1


def test_consistency_penalty_reaches_both_paired_logits() -> None:
    # Given: paired clean and transformed logits.
    clean = torch.tensor([0.2, -0.4], requires_grad=True)
    transformed = torch.tensor([-0.3, 0.7], requires_grad=True)
    labels = torch.tensor([1.0, 0.0])

    # When: CE plus paired probability/feature consistency is differentiated.
    loss = consistency_objective(clean, transformed, labels, weight=0.1)
    loss.backward()

    # Then: both views receive real optimization signal.
    assert clean.grad is not None and float(clean.grad.abs().sum()) > 0
    assert transformed.grad is not None and float(transformed.grad.abs().sum()) > 0

    clean = torch.tensor([0.2, -0.4], requires_grad=True)
    transformed = torch.tensor([-0.3, 0.7], requires_grad=True)
    penalty = consistency_penalty(clean, transformed, clean[:, None], transformed[:, None], weight=0.1)
    penalty.backward()
    assert clean.grad is not None and transformed.grad is not None


def test_real_only_density_fit_rejects_fake_contamination() -> None:
    # Given: a nominal set containing one fake row.
    model = RealOnlyGaussian(shrinkage=0.1)
    features = np.array([[0.0, 0.0], [0.1, -0.1], [8.0, 8.0]])

    # When/Then: fitting cannot silently consume the fake example.
    try:
        model.fit(features, np.array([0, 0, 1]), ["r0", "r1", "f0"])
    except ValueError as error:
        assert "REAL-only" in str(error)
    else:
        raise AssertionError("fake contamination was accepted")


def test_group_bootstrap_marks_single_component_uncertain() -> None:
    # Given: paired predictions but only one independent provenance component.
    rows = [
        {"label": 0, "control": 0.2, "candidate": 0.1, "component_id": "only"},
        {"label": 1, "control": 0.8, "candidate": 0.9, "component_id": "only"},
    ]

    # When: paired uncertainty is requested.
    report = grouped_bootstrap_delta(rows, "control", "candidate", n_boot=50, seed=41)

    # Then: no population CI is asserted from one component.
    assert report["ci95"] is None
    assert report["uncertain"] is True
    assert binary_summary([0, 1], [0.2, 0.8])["eer"] == 0.0


def test_global_components_merge_transitive_contributors() -> None:
    # Given: two rows connected only through a shared intermediate atom.
    rows = [
        {"ID": "x", "group_id": "a|b", "parent_ids": '["a", "b"]'},
        {"ID": "y", "group_id": "b|c", "parent_ids": '["b", "c"]'},
        {"ID": "z", "group_id": "d", "parent_ids": '["d"]'},
    ]

    # When: components are resolved over the complete manifest.
    components = connected_components(rows)

    # Then: the transitive chain is one bootstrap unit and d remains independent.
    assert components["x"] == components["y"]
    assert components["x"] != components["z"]
    assert len(set(components.values())) == 2
