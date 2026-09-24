#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
# Run: PYTHONPATH=. python data_v4/audit_feature_backup.py ARCHIVE RECEIPT MANIFEST ASSET_RECEIPT OUTPUT
"""Inspect a hash-attested feature backup on CPU; never promote incomplete features."""
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

import numpy as np

from eight_feature_identity import file_sha256
from eight_feature_schema import validate_feature_arrays


def main(archive: Path, receipt: Path, manifest: Path, assets: Path, output: Path) -> None:
    """Verify all actual arrays and enumerate exact-audio matches without relabeling caches."""
    expected = json.loads(receipt.read_text())
    digest = file_sha256(archive)
    if digest != expected['sha256'] or archive.stat().st_size != expected['bytes']:
        raise ValueError('Feature backup differs from recorded storage receipt')
    if output.exists():
        raise FileExistsError(output)
    known = defaultdict(list)
    for line in manifest.open():
        row = json.loads(line)
        known[row['sha256']].append(row['ID'])
    asset_hashes = {r['path']: r['sha256'] for r in json.loads(assets.read_text())['files']}
    records, contracts, window_count = [], set(), 0
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if len(set(names)) != len(names) or sum(i.file_size for i in bundle.infolist()) > 2**30:
            raise ValueError('Duplicate members or excessive expanded backup size')
        for member in bundle.infolist():
            path = PurePosixPath(member.filename)
            if len(path.parts) != 1 or path.suffix != '.npz' or member.file_size > 8*2**20:
                raise ValueError('Unexpected feature backup member')
            raw = bundle.read(member)
            with zipfile.ZipFile(io.BytesIO(raw)) as inner:
                if sum(i.file_size for i in inner.infolist()) > 32*2**20:
                    raise ValueError('Oversized feature record')
            with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
                identity = json.loads(str(saved['identity_json']))
                arrays = {k: saved[k] for k in saved.files if k != 'identity_json'}
            count = validate_feature_arrays(arrays, identity['feature_profile'])
            if identity['ID'] != path.stem or sorted(arrays) != identity['required_arrays']:
                raise ValueError('Record name/arrays disagree with producer identity')
            if identity['sample_rate'] != 16000 or identity['window_samples'] != 64600 or identity['hop_samples'] != 32300:
                raise ValueError('Unexpected encoder window contract')
            samples = int(arrays['starts'][-1] + arrays['valid_samples'][-1])
            last = max(0, samples - 64600)
            starts = list(range(0, last + 1, 32300))
            if starts[-1] != last:
                starts.append(last)
            if (not np.array_equal(arrays['starts'], starts)
                    or not np.array_equal(arrays['valid_samples'], np.minimum(64600, samples-arrays['starts']))):
                raise ValueError('Incomplete full-duration window coverage')
            if any(arrays[k].dtype != np.float32 for k in ('mert_l6', 'mert_l12', 'mert_l24')):
                raise ValueError('Unexpected MERT precision')
            mismatched_assets = [k for k,v in asset_hashes.items() if identity['baseline_assets_sha256'].get(k) != v]
            producer_changes = [k for k,v in identity['code_sha256'].items()
                                if not (Path(__file__).parents[1]/k).is_file()
                                or file_sha256(Path(__file__).parents[1]/k) != v]
            contract = {k:v for k,v in identity.items() if k not in ('ID', 'source_path', 'audio_sha256', 'parent_sha256')}
            contracts.add(hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest())
            window_count += count
            records.append(dict(ID=identity['ID'], npz_sha256=hashlib.sha256(raw).hexdigest(),
                audio_sha256=identity['audio_sha256'], current_exact_audio_matches=known[identity['audio_sha256']],
                arrays=sorted(arrays), windows=count, samples=samples,
                mismatched_assets=mismatched_assets, producer_code_differences=producer_changes,
                parent_sha256=identity['parent_sha256'],
                mert_l12_array_sha256=hashlib.sha256(arrays['mert_l12'].tobytes()).hexdigest(),
                full_bank_ready=False))
    report = dict(archive_sha256=digest, archive_bytes=archive.stat().st_size,
        verified_records=len(records), verified_windows=window_count, producer_contracts=len(contracts),
        records_matching_current_audio=sum(bool(r['current_exact_audio_matches']) for r in records),
        current_rows_matching_audio=len({rid for r in records for rid in r['current_exact_audio_matches']}),
        asset_mismatch_records=sum(bool(r['mismatched_assets']) for r in records),
        producer_difference_records=sum(bool(r['producer_code_differences']) for r in records),
        available_arrays=dict(Counter(k for r in records for k in r['arrays'])),
        missing_for_full_bank=['voice_embeddings', 'presence', 'current row/supervision binding'],
        cache_promoted=False, gpu_used=False,
        input_sha256={p.name:file_sha256(p) for p in (receipt, manifest, assets)},
        decision='Preserve this music cache; full-bank reuse needs exact audio/frontend compatibility and missing branches.')
    output.mkdir(parents=True)
    (output/'feature_backup_index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    (output/'feature_backup_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main(*(Path(arg) for arg in sys.argv[1:]))
