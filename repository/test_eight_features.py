from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from eight_contract import MANIFEST_FIELDS
from eight_feature_schema import PROFILE_KEYS, feature_profiles, required_keys
from eight_features import FeatureError, _Engine, _resolve_parent_hashes, _validate_cache, window_layout
from eight_probe_stats import component_ids, e2_threshold_effects
from eight_probes import _run_e2, pool_probabilities, run
from eight_spectral import spectrum_features


def test_window_layout_covers_tail_once_without_repeating_audio() -> None:
    # Given: audio just over two common hops long.
    samples = 100_000

    # When: aligned windows are scheduled.
    starts, valid = window_layout(samples)

    # Then: the exact tail is included once and padding is explicit.
    assert starts.tolist() == [0, 32_300, 35_400]
    assert valid.tolist() == [64_600, 64_600, 64_600]
    assert len(set(starts.tolist())) == len(starts)
    short_starts, short_valid = window_layout(12_345)
    assert short_starts.tolist() == [0]
    assert short_valid.tolist() == [12_345]


def test_sparse_probability_pooling_distinguishes_local_insert() -> None:
    # Given: one localized fake-like window among ten windows.
    probabilities = np.asarray([0.05] * 9 + [0.95], dtype=np.float32)

    # When: all predeclared E5 pooling rules use the same windows.
    pooled = pool_probabilities(probabilities)

    # Then: max and top-30 retain sparse evidence that the mean dilutes.
    assert np.isclose(pooled["mean"], 0.14)
    assert np.isclose(pooled["max"], 0.95)
    assert np.isclose(pooled["top30"], 0.35)
    assert pooled["max"] > pooled["top30"] > pooled["mean"]


def test_spectrum_features_localize_known_frequency_perturbation() -> None:
    # Given: equal-amplitude 1.2 kHz and 6.2 kHz tones.
    sample_rate = 16_000
    time = np.arange(64_600, dtype=np.float32) / sample_rate
    low = np.sin(2 * np.pi * 1_200 * time).astype(np.float32)
    high = np.sin(2 * np.pi * 6_200 * time).astype(np.float32)

    # When: the fixed 1-7 kHz representation is computed.
    low_features = spectrum_features(low[None, :])
    high_features = spectrum_features(high[None, :])

    # Then: energy moves from the lower to the upper half of the declared band.
    assert low_features.shape == high_features.shape == (1, 448)
    assert int(np.argmax(low_features[0, -64:])) < 32
    assert int(np.argmax(high_features[0, -64:])) >= 32
    assert np.isfinite(low_features).all() and np.isfinite(high_features).all()


def test_e5_selection_uses_dev_before_opposing_sealed_test(tmp_path: Path) -> None:
    # Given: dev favors mean pooling while the sealed test favors max pooling.
    dataset = tmp_path / "source/dataset"
    features = tmp_path / "output/features"
    dataset.mkdir(parents=True)
    features.mkdir(parents=True)
    cases = [
        ("cal_real", "calibration", 0, [0.1, 0.9]),
        ("cal_fake", "calibration", 1, [0.6, 0.6]),
        ("dev_real", "dev", 0, [0.1, 0.1]),
        ("dev_fake", "dev", 1, [0.6, 0.6]),
        ("test_real", "test", 0, [0.1, 0.1]),
        ("test_fake", "test", 1, [0.0, 0.95]),
    ]
    rows = []
    for identifier, split, label, probabilities in cases:
        digest = hashlib.sha256(identifier.encode()).hexdigest()
        events = [] if label == 0 else [{"component": "voice", "start": 5.0, "end": 5.5, "label": 1}]
        row = {name: "" for name in MANIFEST_FIELDS}
        row.update(ID=identifier, path=f"audio/{identifier}.wav", parent_id=identifier,
                   parent_ids=json.dumps([identifier]), group_id=identifier, split=split,
                   source="fixture", generator="fixture" if label else "", kind="mixed",
                   variant="partial", probe_family="partial", pair_id=identifier,
                   FILE_FAKE=str(label), VOICE_FAKE=str(label), MUSIC_FAKE="0",
                   VOICE_PRESENT="1", MUSIC_PRESENT="1", audio_sha256=digest,
                   duration_seconds="30", event_intervals=json.dumps(events))
        rows.append(row)
        values = np.asarray(probabilities, dtype=np.float32)
        np.savez_compressed(features / f"{identifier}.npz", df_raw=values, anti_raw=values,
                            identity_json=np.asarray(json.dumps({"audio_sha256": digest})))
    with (dataset / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"root": str(tmp_path), "shared": str(tmp_path / "shared")}), encoding="utf-8")

    # When: E5 locks on calibration/dev and only then summarizes test.
    report = run(config, 5)["E5"]

    # Then: the opposing test distribution cannot change the dev-selected rule.
    assert report["selected_pool_by_model"] == {"df": "mean", "anti": "mean"}
    assert report["sealed_test"]["anti_mean"]["recall"] == 0.0
    assert report["sealed_test"]["anti_max"]["recall"] == 1.0
    assert json.loads((tmp_path / "output/selection_E5_locked.json").read_text())["test_used"] is False


def test_parent_identity_resolves_manifest_and_external_source_hashes(tmp_path: Path) -> None:
    # Given: one derived parent in the manifest and one retained external source.
    dataset = tmp_path / "source/dataset"
    dataset.mkdir(parents=True)
    with (dataset / "sources.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("ID", "source_sha256", "group_id", "source", "historical"))
        writer.writeheader()
        writer.writerow({"ID": "external", "source_sha256": "e" * 64, "group_id": "g", "source": "fixture", "historical": "0"})
    rows = [
        {"ID": "parent", "audio_sha256": "p" * 64, "parent_ids": json.dumps(["external"])},
        {"ID": "derived", "audio_sha256": "d" * 64, "parent_ids": json.dumps(["parent", "external"])},
    ]

    # When: cache provenance is resolved.
    resolved = _resolve_parent_hashes(tmp_path, rows)

    # Then: the derived identity binds both exact contributing hashes.
    assert resolved["derived"] == {"parent": "p" * 64, "external": "e" * 64}


def test_global_provenance_bridge_connects_filtered_probe_rows() -> None:
    # Given: a non-probe row bridges two probe provenance groups.
    rows = [
        {"ID": "left", "group_id": "a|b"},
        {"ID": "bridge", "group_id": "b|c"},
        {"ID": "right", "group_id": "c|d"},
    ]

    # When: components are formed over the full manifest.
    identifiers, count = component_ids(rows)

    # Then: filtering out the bridge later cannot split one connected component.
    assert count == 1
    assert identifiers["left"] == identifiers["right"]


def test_e2_threshold_effects_do_not_cancel_real_and_fake_changes() -> None:
    # Given: recovery raises REAL false positives while preserving FAKE recall.
    rows = []
    for split, label, raw, recovered in (("calibration", "0", 0.1, 0.2), ("calibration", "1", 0.6, 0.4)):
        rows.append({"model": "df", "pool": "max", "split": split, "voice_label": label,
                     "source": "bg", "generator": "", "variant": "mix_0dB",
                     "raw_score": raw, "vocals_score": recovered})

    # When: the calibration-fixed paired effects are reported.
    effects = e2_threshold_effects(rows)["effects"]

    # Then: REAL FPR and FAKE recall remain distinct estimands.
    real = effects["df/max/calibration/bg//mix_0dB/voice_real"]
    fake = effects["df/max/calibration/bg//mix_0dB/voice_fake"]
    assert real["metric"] == "fpr" and real["recovered_minus_raw"] == 1.0
    assert fake["metric"] == "recall" and fake["recovered_minus_raw"] == 0.0


def test_e2_swap_joins_core_vocal_parent_without_duplicate_control(tmp_path: Path) -> None:
    # Given: a swap points at the retained core vocal through parent_ids[0].
    core = {"ID": "core_voice", "pair_id": "", "split": "calibration", "probe_family": "core",
            "VOICE_PRESENT": "1", "VOICE_FAKE": "0", "kind": "speech", "variant": "base",
            "MUSIC_PRESENT": "0", "group_id": "speaker", "source": "speech", "generator": "",
            "parent_ids": json.dumps(["external_voice"])}
    swap = core | {"ID": "swap_mix", "pair_id": "e2:core_voice", "probe_family": "swap",
                   "variant": "mix_0dB", "MUSIC_PRESENT": "1", "group_id": "background|speaker",
                   "source": "FMA", "parent_ids": json.dumps(["core_voice", "background"])}
    records = {
        "core_voice": {"df_raw": np.asarray([0.1]), "df_vocals": np.asarray([0.1]),
                       "anti_raw": np.asarray([0.1]), "anti_vocals": np.asarray([0.1])},
        "swap_mix": {"df_raw": np.asarray([0.4]), "df_vocals": np.asarray([0.2]),
                     "anti_raw": np.asarray([0.5]), "anti_vocals": np.asarray([0.3])},
    }
    components, _ = component_ids([core, swap])
    config = {"root": str(tmp_path), "shared": str(tmp_path / "shared")}

    # When: E2 resolves the same-gain voice-only control.
    report = _run_e2(config, [core, swap], records, components)

    # Then: every model/pool swap row has a real parent join and the run is complete.
    assert report["status"] == "complete"
    assert report["voice_reference_nonbase_rows"] == 6


def test_df_windows_use_pinned_one_dimensional_native_contract_in_order() -> None:
    # Given: the pinned DF wrapper, which inserts its own batch dimension.
    import torch

    class PinnedDfSeam:
        def __init__(self) -> None:
            self.first_samples: list[float] = []

        def __call__(self, *, input_values: torch.Tensor) -> dict[str, torch.Tensor]:
            assert input_values.ndim == 1
            first = input_values[0]
            self.first_samples.append(float(first))
            return {"logits": torch.stack((first, -first))[None]}

    model = PinnedDfSeam()
    engine = object.__new__(_Engine)
    engine.torch, engine.device, engine.deadline, engine.batch_size = torch, torch.device("cpu"), float("inf"), 2
    engine.df, engine.df_fake = model, 0
    windows = np.stack([np.full(64_600, value, dtype=np.float32) for value in (-2.0, 0.0, 2.0)])

    # When: aligned windows are passed through DF-Arena.
    probabilities = engine._df(windows)

    # Then: each native call is 1-D and output order still matches window order.
    assert model.first_samples == [-2.0, 0.0, 2.0]
    assert np.allclose(probabilities, [0.01798621, 0.5, 0.98201376], atol=1e-7)


def test_sealed_manifest_roles_cover_every_consumer_without_unused_modalities() -> None:
    # Given: the final sealed 7,101-row manifest and its actual experiment consumers.
    manifest = Path(__file__).parent / "_workspace/eight_h100_20260917/dataset/manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    # When: one global role map assigns exact per-row extraction keys.
    profiles = feature_profiles(rows)
    counts = {profile: list(profiles.values()).count(profile) for profile in PROFILE_KEYS}

    # Then: every sealed row is covered and every consumer key is present without global dummy arrays.
    assert len(rows) == len(profiles) == 7_101
    assert counts == {"core_music_base": 1322, "core_music_processed": 3135,
                      "core_speech_base": 512, "core_speech_processed": 1324,
                      "swap": 160, "partial": 648}
    for row in rows:
        keys = required_keys(profiles[row["ID"]])
        if row["probe_family"] == "partial":
            assert keys == {"starts", "valid_samples", "padding_samples", "raw_mask",
                            "df_raw", "df_vocals", "anti_raw", "anti_vocals"}
        elif row["probe_family"] == "swap":
            assert {"df_raw", "df_vocals", "anti_raw", "anti_vocals", "mert_l24", "anti_l24", "presence"} <= keys
        elif row["kind"] == "speech":
            assert {"df_raw", "df_vocals", "anti_raw", "anti_vocals", "anti_l24"} <= keys
            assert ({"anti_l12", "anti_l48"} <= keys) == (row["variant"] == "base")
        else:
            assert "mert_l24" in keys and not {"df_raw", "anti_raw", "presence"} & keys
            assert ({"mert_l6", "mert_l12", "spectrum_stem"} <= keys) == (row["variant"] == "base")
        assert "spectrum_raw" not in keys


def test_role_pruning_matches_full_lightweight_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: deterministic lightweight seams for every expensive model/view.
    import torch
    import eight_features

    raw = np.linspace(-1.0, 1.0, 8_000, dtype=np.float32)

    class Baseline:
        @staticmethod
        def load_audio(_path: Path) -> np.ndarray:
            return raw.copy()

        @staticmethod
        def separate_voice_and_music(_path: Path, _separator: object, _device: object) -> tuple[np.ndarray, np.ndarray]:
            return raw * 0.5, raw * 0.25

    engine = object.__new__(_Engine)
    engine.base, engine.separator, engine.device, engine.torch = Baseline(), object(), torch.device("cpu"), torch
    engine.deadline, engine.batch_size = float("inf"), 8

    def anti(windows: np.ndarray, keep: bool) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        score = np.full(len(windows), 0.37, dtype=np.float32)
        layers = {layer: np.full((len(windows), 1280), layer / 100, dtype=np.float32) for layer in (12, 24, 48)} if keep else {}
        return score, layers

    def mert(windows: np.ndarray, layers: tuple[int, ...]) -> dict[int, np.ndarray]:
        return {layer: np.full((len(windows), 1024), layer / 100, dtype=np.float32) for layer in layers}

    engine._anti = anti
    engine._mert = mert
    engine._df = lambda windows: np.full(len(windows), 0.61, dtype=np.float32)
    engine._presence = lambda windows: np.tile(np.asarray([[0.2, 0.8]], dtype=np.float32), (len(windows), 1))
    monkeypatch.setattr(eight_features, "spectrum_features", lambda windows: np.tile(np.arange(448, dtype=np.float32), (len(windows), 1)))

    # When: each sparse role runs independently and a full reference runs every seam.
    outputs = {profile: engine.extract(Path("fixture.wav"), profile) for profile in PROFILE_KEYS}
    starts, valid = window_layout(len(raw))
    raw_windows = eight_features._windows(raw, starts)
    vocal_windows = eight_features._windows(raw * 0.5, starts)
    stem_windows = eight_features._windows(raw * 0.25, starts)
    anti_raw, anti_layers = anti(raw_windows, True)
    full = {"starts": starts, "valid_samples": valid, "padding_samples": 64_600 - valid,
            "raw_mask": np.ones(len(starts), dtype=np.uint8), "stem_mask": np.ones(len(starts), dtype=np.uint8),
            "presence": engine._presence(raw_windows), "df_raw": engine._df(raw_windows),
            "df_vocals": engine._df(vocal_windows), "anti_raw": anti_raw,
            "anti_vocals": anti(vocal_windows, False)[0], "spectrum_stem": eight_features.spectrum_features(stem_windows)}
    full.update({f"mert_l{layer}": value for layer, value in mert(stem_windows, (6, 12, 24)).items()})
    full.update({f"anti_l{layer}": value for layer, value in anti_layers.items()})

    # Then: pruning only omits arrays; every retained numerical value equals full inference.
    for profile, output in outputs.items():
        assert set(output) == required_keys(profile)
        for key, value in output.items():
            assert np.array_equal(value, full[key]), (profile, key)


def test_cache_identity_rejects_stale_feature_profile(tmp_path: Path) -> None:
    # Given: a numerically valid partial cache made under a different feature profile identity.
    arrays = {"starts": np.asarray([0]), "valid_samples": np.asarray([100]),
              "padding_samples": np.asarray([64_500]), "raw_mask": np.asarray([1], dtype=np.uint8),
              **{key: np.asarray([0.5], dtype=np.float32) for key in ("df_raw", "df_vocals", "anti_raw", "anti_vocals")}}
    old_identity = {"ID": "row", "feature_profile": "partial"}
    expected_identity = {"ID": "row", "feature_profile": "core_speech_processed"}
    path = tmp_path / "row.npz"
    np.savez_compressed(path, **arrays, identity_json=np.asarray(json.dumps(old_identity, sort_keys=True)))

    # When/Then: cache reuse fails before stale arrays can masquerade as the expected role.
    with pytest.raises(FeatureError, match="Stale feature cache identity"):
        _validate_cache(path, expected_identity, "core_speech_processed")
