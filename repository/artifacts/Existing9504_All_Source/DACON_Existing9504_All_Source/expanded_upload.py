# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "soundfile", "imageio-ffmpeg"]
# ///
# Run: .venv/bin/python expanded_upload.py
"""Upload bounded audio shards and read them back before starting GPU compute."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile

import imageio_ffmpeg
import numpy as np
import soundfile as sf

from eight_contract import ManifestRow, sha256_file, write_manifest
from expanded_data import ROOT, OUTPUT, PILOT, Row, audit_groups
import redesign_h100_watch as cloud

VOLUME = 'objvol-a5gxdg7er70i'
PREFIX = 'expanded-h100-data-20260918/'
WORK = ROOT / '_workspace/expanded_h100_20260918'


def crop(row: Row) -> tuple[bytes, float, float]:
    """Center-crop natural audio, with the same 0.6 peak ceiling as the legacy control."""
    path = ROOT / row['path']
    if sha256_file(path) != row['source_sha256']:
        raise ValueError(f'Source changed: {row["ID"]}')
    start = max(0.0, (float(row['duration_seconds']) - 8) / 2)
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-ss', f'{start:.8f}', '-i', str(path),
               '-t', '8', '-f', 'f32le', '-ac', '1', '-ar', '16000', '-']
    audio = np.frombuffer(subprocess.run(command, check=True, capture_output=True).stdout, dtype='<f4').copy()
    if len(audio) < 16000 or not np.isfinite(audio).all() or np.sqrt(np.mean(audio**2)) < 1e-6:
        raise ValueError(f'Unusable deterministic crop: {row["ID"]}')
    gain = min(1.0, .6 / max(float(np.abs(audio).max()), 1e-8))
    audio *= gain
    buffer = io.BytesIO()
    sf.write(buffer, audio, 16000, format='WAV', subtype='FLOAT')
    decoded, rate = sf.read(io.BytesIO(buffer.getvalue()), dtype='float32')
    if rate != 16000 or not np.array_equal(decoded, audio):
        raise ValueError('Float WAV roundtrip changed waveform')
    return buffer.getvalue(), start, gain


def crop_row(row: Row, data: bytes) -> ManifestRow:
    identifier = 'bank_' + hashlib.sha256(row['ID'].encode()).hexdigest()[:24]
    seconds = sf.info(io.BytesIO(data)).duration
    music = row['MUSIC_PRESENT'] == '1'
    events = [{'component': 'music', 'start': 0, 'end': seconds, 'label': int(row['MUSIC_FAKE'])}] if music else []
    return ManifestRow(ID=identifier, path='audio/' + identifier + '.wav', parent_id=row['ID'],
                       parent_ids=json.dumps([row['ID']]), group_id=row['group_id'], split=row['split'],
                       source=row['source'], generator=row['generator'], kind='music' if music else 'speech',
                       variant='base', probe_family='new_generator' if row['split'] == 'confirmation_external' else 'new_bank', pair_id='',
                       FILE_FAKE=row['FILE_FAKE'], VOICE_FAKE=row['VOICE_FAKE'], MUSIC_FAKE=row['MUSIC_FAKE'],
                       VOICE_PRESENT=row['VOICE_PRESENT'], MUSIC_PRESENT=row['MUSIC_PRESENT'],
                       audio_sha256=hashlib.sha256(data).hexdigest(), duration_seconds=f'{seconds:.8f}',
                       event_intervals=json.dumps(events))


def transfer(path: Path, prefix: str) -> dict[str, str | int]:
    """Verify exact remote bytes in a temporary directory, then discard the readback."""
    expected = sha256_file(path)
    cloud._call(['volume', 'upload', VOLUME, str(path), '--remote-prefix', prefix, '--overwrite'], timeout=1800)
    with tempfile.TemporaryDirectory(prefix='readback-', dir=WORK) as temporary:
        cloud._call(['volume', 'download', VOLUME, temporary, '--remote-prefix', prefix, '--overwrite'], timeout=1800)
        found = [p for p in Path(temporary).rglob('*') if p.is_file()]
        if len(found) != 1 or sha256_file(found[0]) != expected or found[0].stat().st_size != path.stat().st_size:
            raise ValueError('Audio shard failed byte-for-byte remote readback')
    return {'prefix': prefix, 'filename': path.name, 'bytes': path.stat().st_size, 'sha256': expected}


def prepare() -> None:
    selection_path = OUTPUT / 'selection.json'
    selection = json.loads(selection_path.read_text())
    WORK.mkdir(parents=True, exist_ok=True)
    dataset = WORK / 'dataset'
    dataset.mkdir(exist_ok=True)
    receipt_path = OUTPUT / 'audio_upload.json'
    selection_hash = sha256_file(selection_path)
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {'selection_sha256': selection_hash, 'shards': [], 'derived_rows': [], 'crops': [], 'gpu_started': False}
    if receipt['selection_sha256'] != selection_hash:
        raise ValueError('Upload resume selection changed')
    selected = selection['new_originals']
    for offset in range(0, len(selected), 128):
        number = offset // 128
        if number < len(receipt['shards']):
            continue
        with tempfile.TemporaryDirectory(prefix='shard-', dir=WORK) as temporary:
            archive = Path(temporary) / f'audio_{number:03d}.zip'
            manifests, crops = [], []
            with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as packed:
                for row in selected[offset:offset + 128]:
                    data, start, gain = crop(row)
                    derived = crop_row(row, data)
                    packed.writestr(derived['path'], data)
                    manifests.append(derived)
                    crops.append({'ID': derived['ID'], 'parent_id': row['ID'], 'start_seconds': start, 'gain': gain,
                                  'source_sha256': row['source_sha256'], 'audio_sha256': derived['audio_sha256']})
            verified = transfer(archive, PREFIX + f'shard_{number:03d}/')
            receipt['shards'].append(verified)
            receipt['derived_rows'].extend(manifests)
            receipt['crops'].extend(crops)
            pending = receipt_path.with_suffix('.pending')
            pending.write_text(json.dumps(receipt, indent=2) + '\n')
            pending.replace(receipt_path)
            print(f'VERIFIED_SHARD {number + 1} rows={len(receipt["derived_rows"])}/{len(selected)} bytes={verified["bytes"]}', flush=True)
    legacy = selection['legacy_rows']
    for row in legacy:
        if sha256_file(PILOT / row['path']) != row['audio_sha256']:
            raise ValueError('Legacy control changed')
    rows = legacy + receipt['derived_rows']
    audit_groups(rows)
    if len({r['audio_sha256'] for r in rows}) != len(rows):
        raise ValueError('Exact derived audio duplicates need explicit review')
    write_manifest(dataset / 'manifest.csv', rows)
    with (PILOT / 'sources.csv').open(newline='') as stream:
        old_sources = list(csv.DictReader(stream))
    parents = {p for row in legacy for p in json.loads(row['parent_ids'])}
    sources = [r for r in old_sources if r['ID'] in parents] + selected
    fields = sorted({k for row in sources for k in row})
    with (dataset / 'sources.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sources)
    (dataset / 'original_music_ids.json').write_text(json.dumps({'train': [r['ID'] for r in legacy if r['ID'].startswith('core_')]}))
    from mert12_submission import window_layout
    windows = sum(len(window_layout(round(float(r['duration_seconds']) * 16000))) for r in rows)
    final = {'rows': len(rows), 'new_originals': len(selected), 'windows': windows,
             'ast_prefix_bytes': windows * 1214 * 768 * 2,
             'manifest_sha256': sha256_file(dataset / 'manifest.csv'), 'source_sha256': sha256_file(dataset / 'sources.csv'),
             'selection_sha256': selection_hash, 'remote_audio_bytes': sum(r['bytes'] for r in receipt['shards']),
             'all_shards_readback_verified': True, 'local_audio_shards_retained': False, 'gpu_started': False}
    (OUTPUT / 'data_preparation.json').write_text(json.dumps(final, indent=2) + '\n')
    print(json.dumps(final), flush=True)


if __name__ == '__main__':
    prepare()
