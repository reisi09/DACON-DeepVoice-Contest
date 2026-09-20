#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "soundfile", "imageio-ffmpeg"]
# ///
# ─── How to run ───
# .venv/bin/python new_panel_data_haim.py
# Uses retained pinned metadata and bounded curl downloads; CPU only.
"""Acquire bounded HAIM full-generation sources for 8-second training views."""
from __future__ import annotations

import csv
import json
import re
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final
from urllib.parse import quote

import numpy as np

from eight_audio import SR, core, load, manifest_row, write
from new_panel_data_inventory import OUT, ROOT
from v2_data import DataIntegrityError, decode_audio, pcm_sha256, sha256_file, stable_key

DATA: Final = ROOT / "_workspace/local_sampler_20260918/new_data/haim"
REVISION: Final = "371be01815b65274762d6d0a33e2cb70e24b9baa"
BASE: Final = f"https://huggingface.co/datasets/mippia/HAIM/resolve/{REVISION}/A_full_generation/A2_fake"
FILE_LIMIT: Final = 5_500_000
TOTAL_LIMIT: Final = 1_500_000_000
LICENSE: Final = "https://creativecommons.org/licenses/by-nc/4.0/"


@dataclass(frozen=True, slots=True)
class Candidate:
    provider: str
    track_id: str
    file_name: str
    metadata: dict[str, str]

    @property
    def identifier(self) -> str:
        return f"haim_{self.provider}_{self.track_id}"

    @property
    def path(self) -> Path:
        return DATA / "originals" / self.provider / self.file_name

    @property
    def url(self) -> str:
        return f"{BASE}/{self.provider}/{quote(self.file_name, safe='/')}"


def fetch(candidate: Candidate) -> None:
    """One bounded request; completed downloads are retained for safe resumption."""
    candidate.path.parent.mkdir(parents=True, exist_ok=True)
    temporary = candidate.path.with_suffix(".part")
    result = subprocess.run(["curl", "-fsSL", "--doh-url", "https://1.1.1.1/dns-query",
                             "--connect-timeout", "15", "--max-time", "180", "--max-filesize", str(FILE_LIMIT),
                             "-o", str(temporary), candidate.url], capture_output=True, text=True)
    if result.returncode:
        raise DataIntegrityError(f"Download failed ({result.returncode}) for {candidate.identifier}: {result.stderr}")
    temporary.replace(candidate.path)


def main() -> None:
    """Freeze selection, enforce acquisition budget and preserve exact provenance."""
    history_path = OUT / "data_preparation_history.json"
    history = json.loads(history_path.read_text())
    old_ids = set(history["recorded_identities"])
    excluded_hashes = set(history["recorded_hashes"])
    excluded_hashes.update(json.loads((OUT / "data_preparation_speech_prior_hashes.json").read_text()))
    locked_paths = [OUT / "data_preparation_music_sources.csv", OUT / "data_preparation_confirmation_manifest.csv"]
    for path in locked_paths:
        with path.open() as stream:
            for row in csv.DictReader(stream):
                old_ids.update(row.get(key, "") for key in ("ID", "parent_id", "group_id"))
                excluded_hashes.update(value for key, value in row.items() if "sha256" in key and value)
                excluded_hashes.add(pcm_sha256(load(ROOT / row["path"])))
    candidates = []
    metadata_hashes = {}
    eligibility = {}
    for provider in ("acestep", "musicgen"):
        metadata_path = DATA / "metadata" / (provider + ".csv")
        metadata_hashes[provider] = sha256_file(metadata_path)
        with metadata_path.open() as stream:
            records = list(csv.DictReader(stream))
        cohort = []
        seen_tracks: set[str] = set()
        for row in sorted(records, key=lambda r: stable_key(f"local-train-20260918|haim|{provider}|{r['track_id']}")):
            track = row["track_id"]
            filename = Path(row["file_name"])
            if not re.fullmatch(r"[A-Za-z0-9_-]+", track) or filename.is_absolute() or ".." in filename.parts:
                raise DataIntegrityError("Unsafe or missing source identity in provider metadata")
            candidate = Candidate(provider, track, row["file_name"], row)
            identities = {track, candidate.identifier, f"reference:haim:{provider}:{track}", row["file_name"]}
            duration_ok = not row.get("payload_duration") or 8 <= float(row["payload_duration"]) <= 120
            if track in seen_tracks or identities & old_ids or not duration_ok:
                continue
            seen_tracks.add(track)
            cohort.append(candidate)
        eligibility[provider] = {"metadata_rows": len(records), "eligible_rows": len(cohort)}
        if len(cohort) < 128:
            raise DataIntegrityError(f"Insufficient eligible {provider} sources")
        candidates.extend(cohort[:128])
    selection = {"revision": REVISION, "metadata_sha256": metadata_hashes,
                 "rule": "SHA256 rank local-train-20260918|haim|provider|track_id; one original per provider track",
                 "ace_duration_bounds_seconds": [8, 120], "rows": [asdict(row) for row in candidates]}
    selection_path = DATA / "selection.json"
    serialized = json.dumps(selection, indent=2) + "\n"
    if selection_path.exists() and selection_path.read_text() != serialized:
        raise DataIntegrityError("Frozen HAIM training selection changed")
    selection_path.write_text(serialized)
    ledger_path = DATA / "download_budget.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {
        "metadata_bytes": sum(path.stat().st_size for path in (DATA / "metadata").iterdir()),
        "reserved_audio_request_bytes": 0, "audio_request_attempts": 0}
    pending = [row for row in candidates if not row.path.exists()]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(0, len(pending), 8):
            batch = pending[start:start + 8]
            reserved = len(batch) * FILE_LIMIT
            if ledger["metadata_bytes"] + ledger["reserved_audio_request_bytes"] + reserved > TOTAL_LIMIT:
                raise DataIntegrityError("Conservative 1.5 GB request budget exhausted")
            ledger["reserved_audio_request_bytes"] += reserved
            ledger["audio_request_attempts"] += len(batch)
            ledger_path.write_text(json.dumps(ledger, indent=2))
            list(pool.map(fetch, batch))
            print(json.dumps({"downloaded": len(candidates) - len(pending) + start + len(batch), "total": 256}), flush=True)
    sources = []
    views = []
    seen_hashes: set[str] = set()
    for candidate in candidates:
        source_hash = sha256_file(candidate.path)
        ffmpeg_pcm = pcm_sha256(decode_audio(candidate.path, limit_seconds=None))
        full = load(candidate.path)
        canonical_pcm = pcm_sha256(full)
        if len(full) < 8 * SR:
            raise DataIntegrityError(f"Source is shorter than natural 8 seconds: {candidate.identifier}")
        if {source_hash, ffmpeg_pcm, canonical_pcm} & (excluded_hashes | seen_hashes):
            raise DataIntegrityError(f"Historical, locked-panel or within-bank source overlap: {candidate.identifier}")
        seen_hashes.update((source_hash, ffmpeg_pcm, canonical_pcm))
        start = (len(full) - 8 * SR) // 2
        segment = full[start:start + 8 * SR]
        gain = min(1.0, .6 / max(float(np.max(np.abs(segment))), 1e-8))
        audio = core(candidate.path, 8)
        if not np.array_equal(audio, segment * gain):
            raise DataIntegrityError("Existing core helper changed the center interval or gain")
        view_pcm = pcm_sha256(audio)
        if view_pcm in excluded_hashes or view_pcm in seen_hashes - {source_hash, ffmpeg_pcm, canonical_pcm}:
            raise DataIntegrityError(f"Historical or within-bank 8-second PCM overlap: {candidate.identifier}")
        seen_hashes.add(view_pcm)
        group = f"reference:haim:{candidate.provider}:{candidate.track_id}"
        source = {"ID": candidate.identifier, "path": str(candidate.path.relative_to(ROOT)),
                  "parent_id": candidate.identifier, "group_id": group, "split": "train", "source": "HAIM",
                  "generator": candidate.provider, "track_id": candidate.track_id, "source_url": candidate.url,
                  "metadata_file_name": candidate.file_name,
                  "generator_subgroup": candidate.file_name.split("/")[0] if candidate.provider == "musicgen" else "",
                  "generator_version": "1.5" if candidate.provider == "acestep" else "",
                  "model_variant": candidate.file_name.split("/")[0] if candidate.provider == "musicgen" else "",
                  "revision": REVISION, "source_sha256": source_hash, "pcm_sha256": ffmpeg_pcm,
                  "canonical_pcm_sha256": canonical_pcm, "duration_seconds": len(full) / SR,
                  "FILE_FAKE": "1", "VOICE_FAKE": "", "MUSIC_FAKE": "1", "VOICE_PRESENT": "", "MUSIC_PRESENT": "1",
                  "license": "CC-BY-NC-4.0", "license_url": LICENSE, "label_status": "source_documented",
                  "independent_listening_review": "false", "human_reference_linkage": "unknown"}
        sources.append(source)
        target = DATA / "audio" / (candidate.identifier + "_base.wav")
        write(target, audio)
        events = [{"component": "music", "start": 0, "end": 8, "label": 1}]
        view = manifest_row(candidate.identifier + "_base", target, [candidate.identifier], {group}, "train", "HAIM",
                            candidate.provider, "music", "base", "core", candidate.identifier,
                            ("1", "", "1", "", "1"), events, ROOT)
        view["source_offsets"] = json.dumps([{"parent_id": candidate.identifier, "start": start / SR,
                                             "end": (start + 8 * SR) / SR, "gain": gain}])
        view["label_status"] = "source_documented; voice_unknown; no_independent_listening"
        view["pcm_sha256"] = view_pcm
        view["generator_version"] = source["generator_version"]
        view["model_variant"] = source["model_variant"]
        if view["audio_sha256"] in excluded_hashes or view["audio_sha256"] in seen_hashes:
            raise DataIntegrityError("Prepared WAV duplicates a historical, locked or within-bank file")
        seen_hashes.add(view["audio_sha256"])
        views.append(view)
        if len(views) % 16 == 0:
            print(json.dumps({"prepared_views": len(views), "total": 256}), flush=True)
    counts = Counter(row["generator"] for row in sources)
    if counts != {"acestep": 128, "musicgen": 128} or len({row["group_id"] for row in sources}) != 256:
        raise DataIntegrityError("Provider or unique-track quota failed")
    paths = {}
    for suffix, records in (("sources", sources), ("manifest", views)):
        path = OUT / f"data_preparation_haim_{suffix}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        paths[suffix] = {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
    actual_bytes = ledger["metadata_bytes"] + sum(row.path.stat().st_size for row in candidates)
    receipt = {"status": "complete", "revision": REVISION, "provider_counts": counts, "source_count": len(sources),
               "view_count": len(views), "view_seconds": 8, "actual_retained_download_bytes": actual_bytes,
               "musicgen_subgroups": Counter(row.file_name.split("/")[0] for row in candidates if row.provider == "musicgen"),
               "distinct_acestep_prompts": len({row.metadata["payload_prompt"] for row in candidates if row.provider == "acestep"}),
               "conservative_network_budget": ledger, "selection_sha256": sha256_file(selection_path),
               "metadata_sha256": metadata_hashes, "license_readme_sha256": sha256_file(DATA / "metadata/README.md"),
               "source_manifest": paths["sources"], "training_manifest": paths["manifest"], "eligibility": eligibility,
               "excluded_known_hashes": len(excluded_hashes), "excluded_identities": len(old_ids),
               "hash_overlap_count": 0, "source_center_replay_checks": len(views), "selection_used_predictions": False,
               "independent_listening_review": False, "label_source": "Pinned HAIM card A2 Full AI category",
               "license_url": LICENSE, "limitations": ["Human-reference links are unknown; no matched human/fake pairing",
                   "MusicGen prompts and lyrics are absent from metadata; ACE-Step lyric provenance is unknown",
                   "Exact hashes do not detect partial copies, aliases or pretraining exposure",
                   "Known generators and source-documented labels; voice labels remain unknown",
                   "ACE-Step original duration restricted to 8-120 seconds for the download budget"]}
    (OUT / "data_preparation_haim.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    import sys

    if sys.argv[1:] in (["--expand-originals"], ["--expand-originals", "--allow-downloads"]):
        from new_panel_data_haim_expansion import main as expand_originals

        expand_originals(allow_downloads="--allow-downloads" in sys.argv[1:])
    elif sys.argv[1:]:
        raise SystemExit("Usage: new_panel_data_haim.py [--expand-originals [--allow-downloads]]")
    else:
        main()
