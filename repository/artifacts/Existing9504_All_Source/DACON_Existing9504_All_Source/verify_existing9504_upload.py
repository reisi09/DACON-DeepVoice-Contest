"""Check that cloud staging preserves the approved split and every label/mask."""
from __future__ import annotations

import csv
import hashlib
import io
import json

from existing9504_upload import OUTPUT, PREFIX, SNAPSHOT, SELECTED_SHA, sha, save, stopped_mount
from full_bank_upload import client


def main() -> None:
    assert sha(SNAPSHOT / 'selected.csv') == SELECTED_SHA
    s3, bucket, base = client()
    response = s3.get_object(Bucket=bucket, Key=base + PREFIX + 'dataset/manifest.csv')
    with response['Body'] as stream:
        raw = stream.read()
    assert hashlib.sha256(raw).hexdigest() == sha(OUTPUT / 'manifest.csv')
    with (SNAPSHOT / 'selected.csv').open(newline='') as stream:
        original = list(csv.DictReader(stream))
    uploaded = list(csv.DictReader(io.StringIO(raw.decode())))
    assert len(uploaded) == len(original) == 9504
    for before, after in zip(original, uploaded, strict=True):
        assert all(after[k] == value for k, value in before.items()
                   if k not in ('path', 'audio_sha256')), before['ID']
        assert after['original_path'] == before['path']
        assert after['audio_sha256'] == before['source_sha256']
        assert not before['audio_sha256'] or before['audio_sha256'] == after['audio_sha256']
        assert after['path'].startswith('audio/' + before['source_sha256'])
        for label in ('FILE_FAKE', 'VOICE_FAKE', 'MUSIC_FAKE', 'VOICE_PRESENT', 'MUSIC_PRESENT'):
            assert bool(after[label]) == (after[label + '_mask'] == '1'), (after['ID'], label)
    groups = {split: {r['component_id'] for r in uploaded if r['split'] == split}
              for split in ('train', 'val', 'test')}
    assert not (groups['train'] & groups['val'] or groups['train'] & groups['test']
                or groups['val'] & groups['test'])
    receipt = json.loads((OUTPUT / 'audio_upload.json').read_text())
    assert {r['ID']: (r['path'], r['audio_sha256']) for r in receipt['items']} == {
        r['ID']: (r['path'], r['audio_sha256']) for r in uploaded}
    stopped_mount()
    report = dict(rows=len(uploaded), remote_manifest_sha256=hashlib.sha256(raw).hexdigest(),
                  all_original_labels_masks_groups_and_splits_preserved=True,
                  all_three_group_intersections_empty=True, gpu_executed=False,
                  absolute_cloud_dataset_root='/root-1/' + PREFIX.rstrip('/'))
    save(OUTPUT / 'cloud_manifest_check.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
