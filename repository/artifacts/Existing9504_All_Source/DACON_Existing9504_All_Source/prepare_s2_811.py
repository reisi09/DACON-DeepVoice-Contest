"""Create only the 80/10/10 S2 data partition from the sealed source pool."""
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "torchaudio", "transformers", "numpy", "soundfile"]
# ///
# Run: .venv/bin/python prepare_s2_811.py

import collections
import csv
import hashlib
import json
from pathlib import Path
import runpy
from typing import Final

ROOT: Final = Path(__file__).resolve().parent
OUTPUT: Final = ROOT / "results/s2_811_h100_20260916"


def prepare() -> None:
    """Change only the group split ratio, preserving S2 source exclusions."""
    source_root = ROOT / "results/fullft8_h100x1_20260916/downloaded"
    worker = source_root / "batch_worker.py"
    original = worker.read_bytes()
    if hashlib.sha256(original).hexdigest() != "9c076487890c6e438a9deac80a83be0e04ce4b96d512e6219612efd0cbfbd727":
        message = "Sealed worker changed"
        raise ValueError(message)
    text = original.decode()
    if text.count("len(ordered) * .2") != 1:
        message = "Expected exactly one group-ratio setting"
        raise ValueError(message)
    changed = text.replace("len(ordered) * .2", "len(ordered) * .1")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ns = runpy.run_path(str(worker), run_name="s2_split_source")
    start = changed.index("def make_splits(")
    end = changed.index("\ndef credit_estimate", start)
    exec(compile(changed[start:end], "s2_811_split_only", "exec"), ns)
    runtime = json.loads((source_root / "h100x1_fullft8_run/runtime.json").read_text())
    original_splits = runtime["splits"]
    splits = ns["make_splits"](original_splits["S0"])
    selected = splits["S2"]
    ns["require_disjoint"](selected)
    rows = original_splits["S0"]["train"]
    actual_ids = {row["ID"] for part in selected.values() for row in part}
    excluded = [row for row in rows if row["ID"] not in actual_ids]
    assert all(ns["family"](row) == "audioldm" for row in excluded)
    assert actual_ids | {row["ID"] for row in excluded} == {row["ID"] for row in rows}
    assert selected == ns["make_splits"](original_splits["S0"])["S2"]
    counts = {part: {"files": len(pool), "parent_groups": len(ns["groups"](pool)),
                     "music_labels": dict(collections.Counter(row["MUSIC_FAKE"] for row in pool)),
                     "sources": dict(collections.Counter(row["source"] for row in pool))}
              for part, pool in selected.items()}
    assert [counts[part]["parent_groups"] for part in ["train", "tune", "test"]] == [49, 6, 6]
    manifest = {"requested_ratio": [8, 1, 1], "allocation_unit": "connected original-parent groups",
                "source_pool": "Original S0 train pool, identical to the original S1/S2 split construction",
                "source_files": len(rows), "counts": counts, "excluded_AudioLDM_train_tune": len(excluded),
                "split_seed": "split41", "training_seed": 41, "split": selected,
                "checks": {"parent_disjoint": True, "file_hash_disjoint": True, "pcm_hash_disjoint": True,
                           "reproducible": True, "complete_pool_accounting": True},
                "historical_warning": "Exploratory rerun after prior model/test inspection, not an untouched final test.",
                "test_size_warning": "80/10/10 reduces test size versus the previous 60/20/20 grouping."}
    (OUTPUT / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for part, pool in [*selected.items(), ("excluded_audioldm", excluded)]:
        name = "validation" if part == "tune" else part
        records = [dict(row, original_split=row["split"], split=name,
                        path=str(ROOT / "v4_validation" / row["path"].split("/v4_validation/")[-1]))
                   for row in pool]
        assert all(Path(row["path"]).is_file() for row in records)
        with (OUTPUT / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
    print(json.dumps({"counts": counts, "excluded": len(excluded)}, indent=2))


if __name__ == "__main__":
    prepare()
