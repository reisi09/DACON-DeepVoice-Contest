#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pyarrow", "imageio-ffmpeg"]
# ///
# How to run: uv run prepare_fresh_fma.py [count, default 12, maximum 200]
"""Acquire an artist-disjoint real-song cohort without evaluating any model."""

from __future__ import annotations

import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Final
from urllib.parse import quote, urlparse

from prepare_v2_sources import fetch
from v2_data import decode_audio, pcm_sha256, sha256_file, stable_key

ROOT: Final = Path(__file__).resolve().parent
OUTPUT: Final = ROOT / "_workspace/fresh_fma_20260917"
EVIDENCE: Final = ROOT / "results/leaderboard_audit_20260917"
METADATA: Final = ROOT / "_workspace/v2_discovery/archive_probes/fma_raw_tracks.csv"
PRIOR: Final = (
    ROOT / "_workspace/v2_sources/source_inventory.json",
    ROOT / "recovery_validation/provenance.json",
    ROOT / "v4_validation/provenance.json",
)


def main(count: int = 12) -> None:
    """Retain source files and provenance; keep unverified voice labels unknown."""
    if not 1 <= count <= 200:
        message = "Count must be between 1 and 200"
        raise ValueError(message)
    prior_rows = []
    inputs = {}
    for path in PRIOR:
        prior_rows.extend(json.loads(path.read_text())["sources"])
        inputs[str(path.relative_to(ROOT))] = sha256_file(path)
    initial = ROOT / "_workspace/validation_sources/music_provenance.json"
    prior_rows.extend(json.loads(initial.read_text()))
    inputs[str(initial.relative_to(ROOT))] = sha256_file(initial)
    used_artists = {row["artist_id"] for row in prior_rows if row.get("artist_id")}
    used_artists.update(part.removeprefix("fma_artist_") for row in prior_rows
                        for part in row.get("group_id", "").split("|")
                        if part.startswith("fma_artist_"))
    old_hashes = {row.get("source_sha256", row.get("sha256", "")) for row in prior_rows}
    old_pcm = {row.get(key, "") for row in prior_rows
               for key in ("source_pcm_sha256", "pcm_sha256", "short_pcm_sha256")}
    candidates = []
    with METADATA.open() as stream:
        for row in csv.DictReader(stream):
            license_url = urlparse(row["license_url"])
            permitted = license_url.hostname in {"creativecommons.org", "www.creativecommons.org"}
            permitted = permitted and license_url.path.startswith(("/licenses/by/", "/licenses/by-sa/"))
            duration = row["track_duration"].split(":")
            valid_duration = len(duration) == 2 and all(part.isdigit() for part in duration)
            if not permitted or not valid_duration or not row["track_file"] or not row["artist_id"]:
                continue
            seconds = int(duration[0]) * 60 + int(duration[1])
            if row["artist_id"] in used_artists or not 60 <= seconds <= 360:
                continue
            candidates.append(row)
    candidates.sort(key=lambda row: stable_key(f"new-val|{row['artist_id']}|{row['track_id']}"))
    selected = []
    selected_artists = set()
    for row in candidates:
        if row["artist_id"] not in selected_artists:
            selected.append(row)
            selected_artists.add(row["artist_id"])
        if len(selected) == count:
            break
    if len(selected) != count:
        message = f"Only {len(selected)} eligible distinct artists for requested {count}"
        raise ValueError(message)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    paths = [OUTPUT / "originals" / f"fma_{row['track_id']}.mp3" for row in selected]
    urls = ["https://files.freemusicarchive.org/storage-freemusicarchive-org/"
            + quote(row["track_file"], safe="/") for row in selected]
    # Selection is fixed before downloads; failures remain visible instead of changing the cohort.
    (OUTPUT / "selection.json").write_text(json.dumps(selected, indent=2) + "\n")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, urls, paths))
    rows = []
    pcm_seen = set()
    for item, path, url in zip(selected, paths, urls, strict=True):
        audio = decode_audio(path, limit_seconds=None)
        digest, pcm = sha256_file(path), pcm_sha256(audio)
        # Reject unchanged old source audio and duplicates before admitting new evidence.
        assert digest not in old_hashes, path
        assert pcm not in old_pcm and pcm not in pcm_seen, path
        assert len(audio) >= 64_000, path
        pcm_seen.add(pcm)
        row = {key: item[key] for key in (
            "track_id", "artist_id", "artist_name", "track_title", "track_url", "track_file",
            "license_title", "license_url", "track_instrumental", "track_genres", "track_language_code",
        )}
        row.update(
            ID=f"fresh_fma_{item['track_id']}", path=str(path.relative_to(ROOT)),
            source="FMA", group_id=f"fma_artist_{item['artist_id']}", source_url=url,
            source_sha256=digest, decoded_mono16k_sha256=pcm,
            duration_seconds=len(audio) / 16_000, source_bytes=path.stat().st_size,
            FILE_FAKE="0", MUSIC_FAKE="0", MUSIC_PRESENT="1", VOICE_FAKE="", VOICE_PRESENT="",
            role="new_real_false_positive_cohort", voice_labels_status="not_listened_not_verified",
        )
        rows.append(row)
    assert not selected_artists & used_artists
    with (EVIDENCE / "fresh_fma_manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    receipt = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256_file(Path(__file__)), "prior_source_inputs": inputs,
        "metadata_sha256": sha256_file(METADATA), "excluded_artists": sorted(used_artists),
        "eligible_artists": len({row["artist_id"] for row in candidates}),
        "rows": len(rows), "distinct_artists": len(selected_artists),
        "source_bytes": sum(row["source_bytes"] for row in rows),
        "artist_overlap": 0, "old_source_hash_overlap": 0, "recorded_pcm_hash_overlap": 0,
        "evaluated": False, "trained": False, "official_score": None,
        "limitations": [
            "Real-only cohort cannot estimate EER or official total.",
            "Instrumental metadata is not a verified voice absence label.",
            "Same FMA corpus; artist-disjoint from recorded project sources, not corpus-disjoint.",
            "Pretraining exposure, artist aliases, and partial-content duplicates are not excluded.",
            "Historical PCM hashes mix full audio, crops, and quantized representations; zero hash overlap is not a universal duplicate check.",
        ],
    }
    (EVIDENCE / "fresh_fma_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: receipt[key] for key in
                      ("rows", "distinct_artists", "eligible_artists", "source_bytes", "artist_overlap")}))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
