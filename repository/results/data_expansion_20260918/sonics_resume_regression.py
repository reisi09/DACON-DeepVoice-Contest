#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "imageio-ffmpeg"]
# ///
# How to run: .venv/bin/python results/data_expansion_20260918/sonics_resume_regression.py
"""A rejected duplicate pair must survive resume while retained-source mutation stays fatal."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import new_panel_data_sonics_bank as bank


def check_rejection_resume_and_retained_corruption() -> None:
    """Given real retained bytes, reject one duplicate pair, resume, then reject a changed kept file."""
    originals = [json.loads(line) for line in (bank.OUT / "sonics_retained_sources.jsonl").read_text().splitlines()]
    suno = next(row for row in originals if row["provider"] == "suno")
    udio = [row for row in originals if row["provider"] == "udio"][:2]
    previous = bank.ROOT, bank.DATA, bank.OUT, bank.SELECTION, bank.select_sources
    with tempfile.TemporaryDirectory(prefix="resume_regression_", dir=bank.DATA) as temporary:
        fixture = Path(temporary)
        try:
            bank.ROOT = fixture
            bank.DATA = fixture / "bank"
            bank.OUT = fixture / "receipts"
            bank.SELECTION = bank.OUT / "selection.json"
            for folder in (bank.DATA / "originals", bank.DATA / "metadata", bank.OUT):
                folder.mkdir(parents=True)
            rows = []
            for index, original in enumerate((suno, udio[0], suno, udio[1])):
                path = bank.DATA / "originals" / f"fixture_{index}.mp3"
                shutil.copyfile(REPO / original["path"], path)
                rows.append({**original, "ID": f"fixture_{index}", "source_id": str(index),
                             "archive": "part_fixture.zip", "path": str(path.relative_to(fixture)),
                             "balance_pair": index // 2})
            empty = {key: [] for key in ("source_sha256", "pcm_sha256", "audio_sha256")}
            locks = bank.DATA / "metadata/locked_groups.json"
            bank.save(locks, {"all_prior_seen": empty, "current_locked_eval": empty})
            archive = {"name": "part_fixture.zip", "sha256": "0" * 64, "bytes": 0}
            frozen = {"rows": rows, "archives": [archive], "locks_sha256": bank.sha256_file(locks)}
            bank.save(bank.SELECTION, frozen)
            bank.select_sources = lambda: frozen
            receipt_path = bank.OUT / "sonics_retained_part_fixture.json"
            bank.save(receipt_path, {"rows": rows, "selection_sha256": bank.sha256_file(bank.SELECTION),
                "verifier_sha256": bank.sha256_file(Path(bank.__file__)), "archive_sha256": archive["sha256"]})

            bank.main()
            bank.main()
            resumed = json.loads((bank.OUT / "sonics_retention.json").read_text())
            assert resumed["source_files"] == 2 and resumed["rejected_rows"] == 2
            stored = json.loads(receipt_path.read_text())["rows"]
            assert [row["status"] for row in stored] == ["retained", "retained", "rejected", "rejected"]
            path = fixture / stored[0]["path"]
            with path.open("ab") as stream:
                stream.write(b"corruption regression")
            try:
                bank.main()
            except bank.DataIntegrityError as error:
                assert "Previously retained source changed" in str(error)
            else:
                raise AssertionError("A modified retained source was incorrectly accepted")
        finally:
            bank.ROOT, bank.DATA, bank.OUT, bank.SELECTION, bank.select_sources = previous
    print("PASS: rejected pair resumes without missing-file failure; retained-source corruption is rejected")


if __name__ == "__main__":
    check_rejection_resume_and_retained_corruption()
