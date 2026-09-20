"""Protect the selected split and audio identities across real input packaging."""
import csv
import importlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import zipfile

import full_bank_ops as ops
from full_bank_run import extract
from v6_data import sha256


def test_balanced_deployment_preserves_selected_rows_and_reuses_audio():
    # A restart must train the approved subset, never the former full bank.
    try:
        ops.configure_balanced()
        selected = list(csv.DictReader(ops.MANIFEST.open()))
        with tempfile.TemporaryDirectory() as temporary:
            ops.OUTPUT = Path(temporary)/'results'
            ops.WORK = Path(temporary)/'deploy'
            def uploaded(path, key):
                return {'key': key, 'bytes': path.stat().st_size, 'sha256': sha256(path), 'etag': 'offline'}
            with patch.object(ops.expanded_h100_ops, 'configure_profile'), \
                 patch.object(ops.expanded_h100_ops, 'detail'), \
                 patch.object(ops, 'remote_verify'), patch.object(ops, 'upload', side_effect=uploaded):
                ops.prepare()
            with zipfile.ZipFile(ops.WORK/'input.zip') as archive:
                rows = list(csv.DictReader(io.StringIO(archive.read('dataset/manifest.csv').decode())))
                receipt = json.loads(archive.read('audio_upload.json'))
            assert [(r['ID'], r['split'], r['component_id']) for r in rows] == [
                (r['ID'], r['split'], r['component_id']) for r in selected]
            assert {r['ID'] for r in receipt['items']} == {r['ID'] for r in selected}
            for original, runtime in zip(selected, rows, strict=True):
                assert runtime['audio_sha256'] == original['source_sha256']
                assert runtime['original_path'] == original['path']
                for key in ('VOICE_FAKE', 'MUSIC_FAKE', 'VOICE_FAKE_mask', 'MUSIC_FAKE_mask'):
                    assert runtime[key] == original[key]
            deployment = json.loads((ops.OUTPUT/'deployment.json').read_text())
            assert deployment['split_counts'] == {'train': 5716, 'val': 1428}
            assert deployment['runtime_manifest_sha256'] == sha256(ops.WORK/'manifest.csv')
            assert 'FULL_BANK_JOB=/root/balanced7144_h100_20260919' in ops.BOOTSTRAP
            assert b'/shared/balanced7144-h100-20260919/bootstrap.sh' in ops.STARTUP
            audio_zip = Path(temporary)/'audio.zip'
            with zipfile.ZipFile(audio_zip, 'w') as archive:
                archive.writestr('audio/selected.wav', b'approved original bytes')
                archive.writestr('audio/reserved.wav', b'unselected data')
            destination = Path(temporary)/'audio'
            extract(audio_zip, destination, {'audio/selected.wav'})
            assert (destination/'audio/selected.wav').read_bytes() == b'approved original bytes'
            assert not (destination/'audio/reserved.wav').exists()
    finally:
        importlib.reload(ops)
