"""Integration check for bounded supplemental QA orchestration and fallback provenance."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf


def test_short_real_source_is_explicit_repeated_stress_and_runtime_bound(tmp_path: Path) -> None:
    source = tmp_path / "canonical_real.wav"
    sf.write(source, np.sin(np.arange(16_000 * 2) / 30).astype(np.float32), 16_000)
    with source.open("rb") as stream:
        source_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    package = tmp_path / "package"
    package.mkdir()
    runtime = package / "script.py"
    runtime.write_text(
        """import argparse,csv,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--test-dir',type=Path);p.add_argument('--sample-submission',type=Path);p.add_argument('--output',type=Path);p.add_argument('--device');a=p.parse_args()
with a.sample_submission.open(newline='') as f:rows=list(csv.DictReader(f))
cols=list(rows[0]);
for i,row in enumerate(rows):
    for name in cols[1:]:row[name]=str(0.1+i*0.01)
with a.output.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
with a.output.open('rb') as f:d=hashlib.file_digest(f,'sha256').hexdigest()
(a.output.parent/'runtime_receipt.json').write_text(json.dumps({'files':len(rows),'device':'cuda','submission_sha256':d,'peak_gpu_allocated_bytes':1,'peak_gpu_reserved_bytes':2,'peak_rss_bytes':3}))
""",
        encoding="utf-8",
    )
    output = tmp_path / "qa"
    completed = subprocess.run(
        [sys.executable, "mert12_supplemental_qa.py", "--package-root", str(package),
         "--source-audio", str(source), "--source-sha256", source_sha256,
         "--output", str(output), "--deadline-epoch", str(time.time() + 600),
         "--runtime-python", sys.executable],
        check=True, capture_output=True, text=True,
    )
    report = json.loads((output / "supplemental_qa.json").read_text())
    inputs = json.loads((output / "input_receipt.json").read_text())
    assert json.loads(completed.stdout)["status"] == "passed"
    assert report["long_case"] == "repeated_canonical_real_synthetic_stress"
    assert inputs["source_seconds"] == 2.0 and inputs["quality_claim"] is None
    assert sf.info(output / "test/qa_real_stereo_60s.flac").duration == 60.0
    assert sf.info(output / "test/qa_real_stereo_60s.flac").channels == 2
