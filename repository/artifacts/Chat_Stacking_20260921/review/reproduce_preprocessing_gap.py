"""Reproduce acceptance of incompatible gain policies; no audio/GPU required.

Run with PYTHONPATH pointing to returned/src. This demonstrates a defect;
after repair, load_records must reject the mixed preprocessing policy.
"""
import tempfile
from pathlib import Path

from dacon_stacking.common import digest, write_json
from dacon_stacking.features import SCHEMA, write_extended
from dacon_stacking.workflow import load_records


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        rows = []
        index = {"schema": "dacon.stacking.cache_index.v1", "items": {}}
        for i, preserve in enumerate((False, True)):
            row = {"ID": str(i), "audio_sha256": f"{i + 1:064x}", "base_exposure": {"head_train": "no"}, "preserve_gain": preserve}
            rows.append(row)
            provenance = {
                "audio_sha256": row["audio_sha256"],
                "base_exposure": row["base_exposure"],
                "code_sha256": {"runtime": "a" * 64},
                "checkpoint_sha256": {"heads": "b" * 64},
                "numeric_mode": "cuda_bfloat16_heads_fp32_encoder_features_no_tf32",
                "feature_schema": SCHEMA,
                "timebase": {"sr": 16000},
                "preprocessing": {"sr": 16000, "mono": True, "default_peak_cap": 0.6, "preserve_gain": preserve},
            }
            record = {"ID": row["ID"], "features": {"Pv": 0.8, "Pm": 0.7, "Fv": 0.2, "Fm": 0.3},
                      "baseline": {"FILE_FAKE_PROB": 0.21, "VOICE_PRESENT_PROB": 0.8, "MUSIC_PRESENT_PROB": 0.7, "VOICE_FAKE_PROB": 0.2, "MUSIC_FAKE_PROB": 0.3}}
            path = root / f"{i}.npz"
            sidecar_hash = write_extended(path, record, provenance)
            index["items"][row["ID"]] = {"path": path.name, "sidecar_sha256": sidecar_hash, "identity": digest(provenance)}
        write_json(root / "index.json", index)
        accepted = load_records(root / "index.json", rows)
        print({"defect_reproduced": True, "accepted_preserve_gain": [r["provenance"]["preprocessing"]["preserve_gain"] for r in accepted]})
        rows[1]["preserve_gain"] = False
        load_records(root / "index.json", rows)
        print({"manifest_false_cache_true_accepted": True})


if __name__ == "__main__":
    main()
