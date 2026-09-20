"""Protect true interval alignment and native AST preprocessing without a GPU."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torchaudio.compliance.kaldi import fbank
from transformers import ASTFeatureExtractor

from eight_audio import SR, manifest_row, write
from localization_dataset import construct_training_partials
from localization_features import frame_truth
from mert12_submission import HOP, WINDOW, window_layout, windows


def test_interval_fraction_requires_complete_known_music_truth() -> None:
    # Given: an event crossing fbank boundaries; voice annotations cannot label music.
    events = [{"component": "music", "start": .01, "end": .02, "label": 1},
              {"component": "voice", "start": 0, "end": 1, "label": 1}]
    partial = {"variant": "partial_fake", "event_intervals": json.dumps(events)}
    explicit = {"variant": "base", "MUSIC_FAKE": "1", "event_intervals": json.dumps([
        {"component": "music", "start": .01, "end": .035, "label": 1}])}
    unknown = {"variant": "base", "MUSIC_FAKE": "1", "event_intervals": "[]"}
    starts, valid = np.asarray([0]), np.asarray([800])

    # When: constructed, explicitly annotated, and unknown source rows are mapped.
    edited, mask = frame_truth(partial, starts, valid)
    observed, _ = frame_truth(explicit, starts, valid)
    missing, _ = frame_truth(unknown, starts, valid)

    # Then: occupancy is fractional; incomplete/absent truth stays unknown.
    np.testing.assert_allclose(edited[0, :3], [.4, .4, 0], rtol=0, atol=1e-7)
    assert observed[0, 1] == 1
    assert np.isnan(observed[0, [0, 2]]).all()
    assert np.isnan(missing[mask]).all()
    assert np.isnan(edited[~mask]).all()


def test_frame_mask_and_normalization_match_native_ast_processor() -> None:
    # Given: the actual 64600-sample MERT12 window and AudioSet AST processor.
    waveform = (np.random.default_rng(41).standard_normal(WINDOW) * .01).astype(np.float32)
    processor = ASTFeatureExtractor()
    row = {"variant": "base", "event_intervals": json.dumps([
        {"component": "music", "start": 0, "end": WINDOW / SR, "label": 1}])}

    # When: labels and native Kaldi/AST features are extracted from the same audio.
    targets, mask = frame_truth(row, np.asarray([0]), np.asarray([len(waveform)]))
    native = fbank(torch.from_numpy(waveform)[None], sample_frequency=SR,
                   window_type="hanning", num_mel_bins=128).numpy()
    processed = processor(waveform, sampling_rate=SR, return_tensors="np").input_values[0]

    # Then: validity follows the real extractor, including normalized right padding.
    np.testing.assert_array_equal(mask[0], np.arange(1024) < len(native))
    np.testing.assert_array_equal(targets[mask], np.ones(len(native), dtype=np.float32))
    np.testing.assert_allclose(processed[:len(native)], processor.normalize(native), rtol=0, atol=0)
    np.testing.assert_allclose(processed[len(native):], -processor.mean / (2 * processor.std),
                               rtol=0, atol=1e-7)
    assert np.isnan(targets[~mask]).all()


def test_exact_tail_window_retains_absolute_event_alignment() -> None:
    # Given: an eight-second file with an event that only its exact tail can observe.
    audio = np.arange(8 * SR, dtype=np.float32)
    event_start = len(audio) - 100
    row = {"variant": "partial_fake", "event_intervals": json.dumps([
        {"component": "music", "start": event_start / SR, "end": len(audio) / SR, "label": 1}])}
    starts = window_layout(len(audio))

    # When: baseline windows and interval masks share the same sample starts.
    segments = windows(audio, starts)
    truth, mask = frame_truth(row, starts, np.minimum(WINDOW, len(audio) - starts))
    last_frame = np.flatnonzero(mask[-1])[-1]
    frame_left = int(starts[-1]) + last_frame * 160
    overlap = min(frame_left + 400, len(audio)) - max(frame_left, event_start)

    # Then: the non-hop-aligned tail is preserved and truth stays on the audio timeline.
    assert starts[-1] == len(audio) - WINDOW and starts[-1] % HOP != 0
    np.testing.assert_array_equal(segments[-1], audio[-WINDOW:])
    assert np.all(truth[:-1][mask[:-1]] == 0)
    np.testing.assert_allclose(truth[-1, last_frame], overlap / 400, rtol=0, atol=1e-7)
    assert truth[-1, last_frame] > 0


def test_dataset_partial_and_sham_variants_reach_frame_supervision(tmp_path: Path) -> None:
    # Given: real source audio so the actual builder emits its own interval schema.
    source, output = tmp_path / "source", tmp_path / "constructed"
    samples = np.arange(8 * SR, dtype=np.float32) / SR
    rows = []
    for name, label, frequency in (("bed_a", 0, 91), ("bed_b", 0, 173), ("fake", 1, 279)):
        path = source / "audio" / f"{name}.wav"
        write(path, (.2 * np.sin(2 * np.pi * frequency * samples)).astype(np.float32))
        rows.append(manifest_row(name, path, [f"{name}_source"], {f"{name}_atom"}, "train",
                                 "suno" if label else "FMA", "suno" if label else "REAL",
                                 "music", "base", "core", "", (str(label), "", str(label), "", "1"), [], source))

    # When: genuine start/middle/end fake-sham pairs enter the feature label path.
    built = construct_training_partials(rows, source, output, pairs=3, seed=41)
    starts = window_layout(8 * SR)
    aligned = [frame_truth(row, starts, np.full(len(starts), WINDOW)) for row in built]

    # Then: every real-bed frame is supervised and only declared fake intervals are positive.
    for row, (truth, mask) in zip(built, aligned, strict=True):
        event = json.loads(row["event_intervals"])[0]
        left = starts[:, None] + np.arange(1024)[None] * 160
        occupied = np.maximum(0, np.minimum(left + 400, round(event["end"] * SR))
                              - np.maximum(left, round(event["start"] * SR))) / 400
        expected = occupied * event["label"]
        np.testing.assert_allclose(truth[mask], expected[mask], rtol=0, atol=1e-7)
        assert np.isfinite(truth[mask]).all() and np.isnan(truth[~mask]).all()
        assert bool(np.any(truth[mask] > 0)) == bool(event["label"])
