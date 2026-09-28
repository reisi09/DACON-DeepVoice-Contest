"""Atomic artifacts and the one-time frozen G0 prediction cache."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from graph_features import Extractor, extract_cached
from graph_fusion import graph_predictions


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value: Mapping | Sequence) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def g0_predictions(
    rows: Sequence[Mapping[str, str]],
    data_root: Path,
    baseline: Path,
    output: Path,
    deadline: float,
) -> list[dict[str, str | float]]:
    """Compute complete G0 once, then release its heavyweight models."""
    prediction_path = output / "G0_predictions.csv"
    paths = [data_root / row["path"] for row in rows]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("G0 manifest references missing audio")
    dependency_paths = [baseline / "script.py", baseline / "model" / "SHA256SUMS.txt",
                        Path("graph_features.py"), Path("graph_fusion.py"), Path("v2_data.py")]
    if any(not path.is_file() for path in dependency_paths):
        raise FileNotFoundError("G0 dependency receipt cannot be constructed")
    receipt = {"rows": [{"ID": str(row["ID"]), "path": str(row["path"]),
                          "sha256": digest(path)} for row, path in zip(rows, paths, strict=True)],
               "dependencies": {str(path): digest(path) for path in dependency_paths}}
    receipt["cache_key"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True).encode()).hexdigest()
    receipt_path = output / "g0_receipt.json"
    if (prediction_path.is_file() and (output / "g0_parity.json").is_file()
            and receipt_path.is_file()
            and json.loads(receipt_path.read_text(encoding="utf-8")) == receipt):
        cached = read_predictions(prediction_path)
        if [row["ID"] for row in cached] != [row["ID"] for row in rows]:
            raise ValueError("G0 cache IDs do not match the manifest")
        return cached
    extractor = Extractor(baseline / "script.py", include_raw=False)
    try:
        features = extract_cached(extractor, paths, output / "g0_cache" / receipt["cache_key"][:16], deadline)
        if len(features) != len(rows):
            raise TimeoutError(f"G0 coverage incomplete: {len(features)}/{len(rows)}")
        predictions = []
        for row, feature in zip(rows, features, strict=True):
            feature["ID"] = row["ID"]
            predictions.append(graph_predictions(feature)["G0"])
        parity = []
        for row, prediction in [(row, predictions[index]) for index, row in enumerate(rows)
                                if row.get("panel") == "music_real"][:2]:
            path = data_root / row["path"]
            audio = extractor.base.load_audio(path)
            vp, mp = extractor.base.predict_presence(extractor.panns, extractor.vi, extractor.mi, audio)
            voice, music = extractor.base.separate_voice_and_music(path, extractor.demucs, extractor.device)
            vf = extractor.base.predict_fake(extractor.df, extractor.fake_index, voice, extractor.device)
            mf = extractor.base.predict_fake(extractor.df, extractor.fake_index, music, extractor.device)
            expected = [extractor.base.combine_file_fake_score(vf, mf, vp, mp), vf, mf, vp, mp]
            actual = [float(prediction[name]) for name in ("FILE_FAKE_PROB", "VOICE_FAKE_PROB",
                "MUSIC_FAKE_PROB", "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB")]
            delta = max(abs(left - right) for left, right in zip(expected, actual, strict=True))
            if delta > 1e-5:
                raise AssertionError(f"G0 parity failed for {row['ID']}: {delta}")
            parity.append({"ID": row["ID"], "max_probability_delta": delta})
        atomic_json(output / "g0_parity.json", parity)
    finally:
        extractor.close()
    with prediction_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(predictions)
    atomic_json(receipt_path, receipt)
    return predictions


def budget_event(path: Path, event: str, **fields: str | int | float | bool) -> None:
    append_jsonl(path, {"event": event, "unix": time.time(), **fields})
