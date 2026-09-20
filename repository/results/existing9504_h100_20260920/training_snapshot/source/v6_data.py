"""Source/PCM-disjoint expert and fusion data, with reproducible mixtures."""

import csv
import hashlib
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def keys(row: dict[str, str]) -> set[str]:
    return {"group:" + g for g in row["group_id"].split("|")} | {
        "pcm:" + row["pcm_sha256"], "id:" + row["ID"]}


def assert_disjoint(left: list[dict], right: list[dict]) -> None:
    shared = set().union(*(keys(r) for r in left)) & set().union(*(keys(r) for r in right))
    if shared:
        raise ValueError(f"Source/PCM/ID leakage between partitions: {sorted(shared)[:5]}")


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len({r["ID"] for r in rows}) != len(rows):
        raise ValueError(f"Empty manifest or duplicate IDs: {path}")
    for row in rows:
        if not row.get("group_id") or any(not x for x in row["group_id"].split("|")):
            raise ValueError(f"Missing source group: {row['ID']}")
        for label in ("FILE_FAKE", "VOICE_FAKE", "MUSIC_FAKE", "VOICE_PRESENT", "MUSIC_PRESENT"):
            if row[label] not in ("", "0", "1"):
                raise ValueError(f"Invalid label: {row['ID']}/{label}")
        audio_path = (path.parent / row["path"]).resolve()
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        if not len(audio) or not np.isfinite(audio).all():
            raise ValueError(f"Invalid audio: {audio_path}")
        row.update(path=str(audio_path), audio_sha256=sha256(audio_path),
                   pcm_sha256=hashlib.sha256(audio.astype("<f4").tobytes()).hexdigest())
    return rows


def split_groups(rows: list[dict], seed: int, fraction: float = .25) -> tuple[list, list]:
    parents = list(range(len(rows)))

    def root(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen: dict[str, int] = {}
    for i, row in enumerate(rows):
        for key in sorted(keys(row)):
            if key in seen:
                parents[root(i)] = root(seen[key])
            seen[key] = i
    components: dict[int, list[dict]] = {}
    for i, row in enumerate(rows):
        components.setdefault(root(i), []).append(row)
    groups = sorted(components.values(), key=lambda group: hashlib.sha256(
        f"{seed}:{min(r['ID'] for r in group)}".encode()).hexdigest())
    selected = set()
    for group in groups:
        if len(selected) >= len(rows) * fraction:
            break
        selected.update(r["ID"] for r in group)
    expert = [r for r in rows if r["ID"] not in selected]
    fusion = [r for r in rows if r["ID"] in selected]
    assert_disjoint(expert, fusion)
    return expert, fusion


def mixtures(rows: list[dict], output: Path, seed: int, count: int = 80) -> list[dict]:
    """Only fusion sources; four real/fake combinations and varied mixing levels."""
    rng = np.random.default_rng(seed)
    voice = [[r for r in rows if r["VOICE_PRESENT"] == "1"
              and r["MUSIC_PRESENT"] == "0" and r["FILE_FAKE"] == str(label)] for label in (0, 1)]
    music = [[r for r in rows if r["MUSIC_PRESENT"] == "1"
              and r["MUSIC_FAKE"] == str(label) and r["FILE_FAKE"] == str(label)] for label in (0, 1)]
    if any(not group for group in voice + music):
        raise ValueError("Fusion split requires real/fake speech and music sources")
    output.mkdir(parents=True, exist_ok=True)
    generated = []
    for i in range(count):
        vl, ml = (i // 2) % 2, i % 2
        v = voice[vl][rng.integers(len(voice[vl]))]
        m = music[ml][rng.integers(len(music[ml]))]
        a, _ = librosa.load(v["path"], sr=16000, mono=True)
        b, _ = librosa.load(m["path"], sr=16000, mono=True)
        length = min(max(len(a), len(b)), 16000 * 16)
        a, b = np.resize(a, length), np.resize(b, length)
        a /= max(float(np.sqrt(np.mean(a * a))), 1e-6)
        b /= max(float(np.sqrt(np.mean(b * b))), 1e-6)
        mixed = a + b * 10 ** (float(rng.uniform(-9, 9)) / 20)
        mixed = (mixed * (.8 / max(float(np.abs(mixed).max()), 1e-6))).astype("float32")
        identifier = f"v6_mix_{seed}_{i:04d}"
        path = (output / f"{identifier}.wav").resolve()
        sf.write(path, mixed, 16000, subtype="FLOAT")
        generated.append(dict(ID=identifier, path=str(path),
            group_id="|".join(sorted(set(v["group_id"].split("|") + m["group_id"].split("|")))),
            FILE_FAKE=str(max(vl, ml)), VOICE_FAKE="1" if vl else "",
            MUSIC_FAKE=str(ml), VOICE_PRESENT="1", MUSIC_PRESENT="1", panel="v6_fusion_mix",
            audio_sha256=sha256(path), pcm_sha256=hashlib.sha256(mixed.astype("<f4").tobytes()).hexdigest()))
    return generated


def prepare(train: Path, dev: Path, output: Path, seed: int) -> dict:
    expert, fusion = split_groups(read_manifest(train), seed)
    validation = read_manifest(dev)
    assert_disjoint(expert + fusion, validation)
    fusion += mixtures(fusion, output / "mixtures", seed)
    assert_disjoint(expert, fusion)
    assert_disjoint(fusion, validation)
    audit = dict(seed=seed, purpose="final", train_manifest_sha256=sha256(train),
                 dev_manifest_sha256=sha256(dev), expert=expert, fusion=fusion, dev=validation,
                 dev_status="Previously used development set; not a new untouched holdout")
    write_json(output / "split.json", audit)
    return audit
