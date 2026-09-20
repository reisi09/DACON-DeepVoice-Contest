"""Resume the unchanged V5 final models from validated local audio/features."""
import hashlib
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

import v5_features as features

_original_prepare = features.prepare_views
_original_encode = features.encode_expert_views

def cached_prepare(examples, baseline, cache, deadline, *, training):
    result, missing = {}, []
    for example in examples:
        waveform = features.read_audio(example.path)
        offset = 0
        if training and len(waveform) > 320000:
            offset = int(hashlib.sha256(example.id.encode()).hexdigest()[:16], 16) % (len(waveform) - 320000 + 1)
            waveform = waveform[offset:offset + 320000]
        paths = [cache / f"{example.id}_{kind}.wav" for kind in ("raw", "voice", "music")]
        try:
            valid = all(sf.info(p).frames == len(waveform) and sf.info(p).samplerate == 16000 for p in paths)
            valid = valid and np.array_equal(sf.read(paths[0], dtype="float32")[0], waveform)
        except (OSError, RuntimeError):
            valid = False
        if valid:
            result[example.id] = features.AudioViews(*paths, offset / 16000)
        else:
            missing.append(example)
    print("RESUME_AUDIO_CACHE", len(result), "missing", len(missing), flush=True)
    if missing:
        result.update(_original_prepare(missing, baseline, cache, deadline, training=training))
    return result

def cached_encode(model, examples, views, kind, output, deadline):
    output.mkdir(parents=True, exist_ok=True)
    missing = []
    reused = 0
    for example in examples:
        path = output / f"{example.id}.npz"
        view = views[example.id]
        expected = np.asarray(features.window_starts(sf.info(view.raw).frames, 64600), dtype=np.float32) / 16000 + view.offset_seconds
        try:
            with np.load(path, allow_pickle=False) as data:
                valid = (np.array_equal(data["times"], expected)
                         and data["features"].ndim == 2
                         and len(data["features"]) == len(expected)
                         and np.isfinite(data["features"]).all())
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            valid = False
        if valid:
            reused += 1
        else:
            missing.append(example)
    print("RESUME_FEATURE_CACHE", kind, reused, "missing", len(missing), flush=True)
    if missing:
        with tempfile.TemporaryDirectory(prefix=f"{kind}_resume_", dir=output.parent) as temporary:
            fresh = Path(temporary) / "features"
            _original_encode(model, missing, views, kind, fresh, deadline)
            for path in fresh.glob("*.npz"):
                path.replace(output / path.name)

features.prepare_views = cached_prepare
features.encode_expert_views = cached_encode

if __name__ == "__main__":
    from run_v5_final import main
    main()
