"""Checks protect time alignment, independent inference, and dev-only fitting."""

from copy import deepcopy
import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from graph_fusion import feature_vector, fit_fusion, graph_predictions


def feature(identifier="a", level=0.1):
    return {"ID": identifier, "starts": [0, 64600],
            "presence": [[0.95, 0.05], [0.8, 0.1]],
            "voice_fake": [level, min(level + 0.1, 1.0)],
            "music_fake": [0.1, 0.2], "raw_fake": [level, level],
            "stem_rms": [0.1, 0.1], "duration": 8.075}


def development_data():
    features = [feature(str(i), 0.05 + i / 25) for i in range(20)]
    labels = [{"ID": str(i), "group_id": f"source-{i}", "split": "dev",
               "FILE_FAKE": int(i >= 10), "VOICE_FAKE": int(i >= 10),
               "MUSIC_FAKE": "", "VOICE_PRESENT": 1, "MUSIC_PRESENT": 0}
              for i in range(20)]
    return features, labels


def test_time_alignment_removes_cross_window_evidence():
    # Given presence and fake evidence occur in different windows.
    sample = feature()
    sample.update(presence=[[1.0, 0.0], [0.0, 0.0]],
                  voice_fake=[0.0, 1.0], music_fake=[0.0, 0.0])
    # When the graph performs temporally aligned fusion.
    rows = graph_predictions(sample)
    # Then G0 exactly reproduces baseline inflation and G2 excludes it.
    assert rows["G0"]["FILE_FAKE_PROB"] == 1.0
    assert rows["G2"]["FILE_FAKE_PROB"] == 0.0


def test_top_two_reduces_an_isolated_peak():
    # Given one isolated fake score among three windows.
    sample = feature()
    sample.update(starts=[0, 64600, 129200], duration=12.1125,
                  presence=[[1.0, 0.0]] * 3, voice_fake=[0.1, 1.0, 0.2],
                  music_fake=[0.0] * 3, raw_fake=[0.0] * 3)
    # When top-two aggregation replaces the maximum.
    rows = graph_predictions(sample)
    # Then it averages the strongest two pieces of evidence.
    assert rows["G1"]["FILE_FAKE_PROB"] == pytest.approx(0.6)
    assert rows["G0"]["FILE_FAKE_PROB"] == 1.0


def test_raw_path_participates_only_in_hybrid_and_learned_graphs():
    # Given identical separated stems but different original-waveform evidence.
    low, high = feature(), feature()
    high["raw_fake"] = [0.99, 0.99]
    # When both files are predicted independently.
    before, after = graph_predictions(low), graph_predictions(high)
    # Then G3 changes, while graphs without the raw path remain identical.
    assert after["G3"]["FILE_FAKE_PROB"] > before["G3"]["FILE_FAKE_PROB"]
    assert after["G3"]["VOICE_FAKE_PROB"] > before["G3"]["VOICE_FAKE_PROB"]
    assert all(before[name] == after[name] for name in ("G0", "G1", "G2"))


def test_predictions_are_independent_of_file_order_and_batch_membership():
    # Given files with different evidence and a saved fusion model.
    features, labels = development_data()
    fusion = fit_fusion(features, labels)
    # When inference order and batch membership change.
    forward = {f["ID"]: graph_predictions(f, fusion) for f in features}
    reverse = {f["ID"]: graph_predictions(f, fusion) for f in reversed(features)}
    # Then no cross-file statistics can affect a file's prediction.
    assert forward == reverse
    assert forward["3"] == graph_predictions(features[3], fusion)


def test_json_inference_matches_real_sklearn_and_excludes_holdout():
    # Given development data and adversarial held-out labels/features.
    features, labels = development_data()
    heldout = feature("heldout", 0.99)
    heldout_label = {**labels[0], "ID": "heldout", "group_id": "heldout-source",
                    "split": "holdout", "FILE_FAKE": 0, "VOICE_FAKE": 0}
    x = np.stack([feature_vector(f) for f in sorted(features, key=lambda f: f["ID"])])
    y_by_id = {row["ID"]: row["FILE_FAKE"] for row in labels}
    y = [y_by_id[f["ID"]] for f in sorted(features, key=lambda f: f["ID"])]
    scaler = StandardScaler().fit(x)
    reference = LogisticRegression(C=0.1, max_iter=1000, random_state=42).fit(scaler.transform(x), y)
    # When only the development split fits the serialized model.
    fusion = json.loads(json.dumps(fit_fusion(features + [heldout], labels + [heldout_label])))
    # Then plain-NumPy deployment agrees with sklearn without held-out influence.
    expected = reference.predict_proba(scaler.transform([feature_vector(heldout)]))[0, 1]
    assert graph_predictions(heldout, fusion)["G4"]["FILE_FAKE_PROB"] == pytest.approx(expected, abs=5.1e-11)
    assert fusion == fit_fusion(features, labels)
    assert fusion["heads"]["MUSIC_FAKE_PROB"]["status"] == "fallback"
    assert graph_predictions(heldout, fusion)["G4"]["MUSIC_FAKE_PROB"] == graph_predictions(heldout)["G0"]["MUSIC_FAKE_PROB"]


def test_true_component_presence_masks_head_training():
    # Given an absent-voice development sample with a misleading voice label.
    features, labels = development_data()
    absent = feature("absent", 0.99)
    absent_label = {**labels[0], "ID": "absent", "group_id": "absent-source",
                    "VOICE_PRESENT": 0, "VOICE_FAKE": 0}
    # When the component head is fitted.
    original = fit_fusion(features, labels)
    extended = fit_fusion(features + [absent], labels + [absent_label])
    # Then absent components never enter the voice head's scaler or labels.
    assert original["heads"]["VOICE_FAKE_PROB"] == extended["heads"]["VOICE_FAKE_PROB"]


@pytest.mark.parametrize("group", ["source-0", "source-0|new-music-source"])
def test_source_overlap_is_rejected_before_fitting(group):
    # Given development and holdout examples derived from the same source.
    features, labels = development_data()
    leaked = {**labels[0], "ID": "copy", "split": "holdout", "group_id": group}
    # When training is requested, then the leakage boundary rejects it.
    with pytest.raises(ValueError, match="group"):
        fit_fusion(features, labels + [leaked])


def test_extreme_fitted_logits_stay_finite_and_bounded():
    # Given a trained model whose intercepts saturate both sigmoid tails.
    features, labels = development_data()
    fusion = fit_fusion(features, labels)
    fusion["heads"]["FILE_FAKE_PROB"].update(coef=[0.0] * 11, intercept=10000.0)
    fusion["heads"]["VOICE_FAKE_PROB"].update(coef=[0.0] * 11, intercept=-10000.0)
    # When the offline NumPy path evaluates it with floating errors enabled.
    with np.errstate(over="raise", invalid="raise"):
        row = graph_predictions(features[0], fusion)["G4"]
    # Then both limiting probabilities are valid submission scores.
    assert row["FILE_FAKE_PROB"] == 1.0
    assert row["VOICE_FAKE_PROB"] == 0.0


@pytest.mark.parametrize("field,value", [("voice_fake", [float("nan"), 0.1]),
    ("music_fake", [0.1]), ("raw_fake", [-0.1, 0.1]),
    ("presence", [[0.1, 0.2], [0.1, float("inf")]])])
def test_invalid_cached_evidence_is_rejected(field, value):
    # Given a corrupted or temporally mismatched inference cache.
    sample = deepcopy(feature())
    sample[field] = value
    # When prediction is requested, then invalid evidence cannot enter a CSV.
    with pytest.raises(ValueError):
        graph_predictions(sample)
